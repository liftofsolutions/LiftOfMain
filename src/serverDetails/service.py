"""Collects the machine vitals that tell whether this server is overloading.

Everything here is non blocking: CPU and network figures are derived from the
gap since the previous call instead of sleeping inside the request.
"""

import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone

import psutil

# Thresholds that decide the reported status. Warning pushes the server to
# "degraded", critical pushes it to "overloaded".
CPU_WARNING_PERCENT = 75.0
CPU_CRITICAL_PERCENT = 90.0
MEMORY_WARNING_PERCENT = 75.0
MEMORY_CRITICAL_PERCENT = 90.0
SWAP_WARNING_PERCENT = 50.0
DISK_WARNING_PERCENT = 80.0
DISK_CRITICAL_PERCENT = 90.0
LOAD_WARNING_PER_CORE = 0.7
LOAD_CRITICAL_PER_CORE = 1.0

BYTES_IN_MB = 1024 * 1024
BYTES_IN_GB = 1024 * 1024 * 1024
BITS_IN_BYTE = 8

STATUS_HEALTHY = "healthy"
STATUS_DEGRADED = "degraded"
STATUS_OVERLOADED = "overloaded"

# Moment the module was imported, i.e. when the API process came up.
SERVER_START_TIME = time.time()

# Last network counter reading, used to derive a throughput rate between calls.
LAST_NETWORK_SAMPLE: dict[str, float] | None = None

# psutil reports CPU usage relative to the previous call, so prime the counters
# once at import time to keep the first request from returning 0.0.
psutil.cpu_percent(interval=None)
psutil.cpu_percent(interval=None, percpu=True)
psutil.Process().cpu_percent(interval=None)


def utcNow() -> datetime:
    """Current time as an aware UTC datetime."""
    return datetime.now(timezone.utc)


def toMb(byteCount: float) -> float:
    """Bytes to mebibytes, rounded for readability."""
    return round(byteCount / BYTES_IN_MB, 2)


def toGb(byteCount: float) -> float:
    """Bytes to gibibytes, rounded for readability."""
    return round(byteCount / BYTES_IN_GB, 2)


def formatDuration(totalSeconds: float) -> str:
    """Seconds as a compact `1d 02h 03m 04s` style string."""
    remaining = int(totalSeconds)
    days, remaining = divmod(remaining, 86400)
    hours, remaining = divmod(remaining, 3600)
    minutes, seconds = divmod(remaining, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m {seconds:02d}s"
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def getUptimeSeconds() -> float:
    """Seconds this API process has been serving."""
    return round(time.time() - SERVER_START_TIME, 2)


def getStartedAt() -> str:
    """ISO timestamp of when this API process came up."""
    return datetime.fromtimestamp(SERVER_START_TIME, tz=timezone.utc).isoformat()


def getSystemRootPath() -> str:
    """Mount point the operating system itself lives on.

    On Windows `os.sep` resolves against the current drive, which is not
    necessarily the system drive, so read the drive from the environment.
    """
    if os.name == "nt":
        return os.environ.get("SystemDrive", "C:") + "\\"
    return "/"


def getCpuVitals() -> dict:
    """Processor pressure: overall usage, per core usage and load average."""
    logicalCores = psutil.cpu_count(logical=True)

    try:
        frequency = psutil.cpu_freq()
        frequencyMhz = round(frequency.current, 2) if frequency else None
    except (NotImplementedError, AttributeError, OSError):
        frequencyMhz = None

    try:
        loadAverage = [round(value, 2) for value in psutil.getloadavg()]
    except (OSError, AttributeError):
        loadAverage = None

    loadPerCore = None
    if loadAverage and logicalCores:
        loadPerCore = round(loadAverage[0] / logicalCores, 2)

    return {
        "usagePercent": psutil.cpu_percent(interval=None),
        "perCorePercent": psutil.cpu_percent(interval=None, percpu=True),
        "physicalCores": psutil.cpu_count(logical=False),
        "logicalCores": logicalCores,
        "frequencyMhz": frequencyMhz,
        "loadAverage": loadAverage,
        "loadPerCore": loadPerCore,
    }


def getMemoryVitals() -> dict:
    """RAM and swap consumption."""
    virtualMemory = psutil.virtual_memory()
    swapMemory = psutil.swap_memory()

    return {
        "totalMb": toMb(virtualMemory.total),
        "usedMb": toMb(virtualMemory.total - virtualMemory.available),
        "availableMb": toMb(virtualMemory.available),
        "usagePercent": virtualMemory.percent,
        "swapTotalMb": toMb(swapMemory.total),
        "swapUsedMb": toMb(swapMemory.used),
        "swapUsagePercent": swapMemory.percent,
    }


def getDiskVitals() -> dict:
    """Space consumption per partition plus disk I/O counters."""
    partitions = []
    for partition in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except (PermissionError, OSError):
            # Empty optical drives and unreadable mounts are simply skipped.
            continue
        partitions.append(
            {
                "mountPoint": partition.mountpoint,
                "totalGb": toGb(usage.total),
                "usedGb": toGb(usage.used),
                "freeGb": toGb(usage.free),
                "usagePercent": usage.percent,
            }
        )

    rootPath = getSystemRootPath()
    try:
        rootUsagePercent = psutil.disk_usage(rootPath).percent
    except OSError:
        rootUsagePercent = max((item["usagePercent"] for item in partitions), default=0.0)

    try:
        ioCounters = psutil.disk_io_counters()
    except (RuntimeError, OSError):
        ioCounters = None

    return {
        "rootUsagePercent": rootUsagePercent,
        "partitions": partitions,
        "readMb": toMb(ioCounters.read_bytes) if ioCounters else None,
        "writeMb": toMb(ioCounters.write_bytes) if ioCounters else None,
    }


def getNetworkVitals() -> dict:
    """Traffic counters plus throughput measured against the previous call."""
    global LAST_NETWORK_SAMPLE

    counters = psutil.net_io_counters()
    sampledAt = time.time()

    sentRateKbps = None
    receivedRateKbps = None
    sampleWindowSeconds = None

    if LAST_NETWORK_SAMPLE is not None:
        elapsed = sampledAt - LAST_NETWORK_SAMPLE["sampledAt"]
        sentDelta = counters.bytes_sent - LAST_NETWORK_SAMPLE["bytesSent"]
        receivedDelta = counters.bytes_recv - LAST_NETWORK_SAMPLE["bytesReceived"]
        # A counter reset (interface restart) shows up as a negative delta.
        if elapsed > 0 and sentDelta >= 0 and receivedDelta >= 0:
            sampleWindowSeconds = round(elapsed, 2)
            sentRateKbps = round(sentDelta * BITS_IN_BYTE / elapsed / 1000, 2)
            receivedRateKbps = round(receivedDelta * BITS_IN_BYTE / elapsed / 1000, 2)

    LAST_NETWORK_SAMPLE = {
        "sampledAt": sampledAt,
        "bytesSent": counters.bytes_sent,
        "bytesReceived": counters.bytes_recv,
    }

    try:
        establishedConnections = [
            item for item in psutil.net_connections(kind="inet") if item.status == "ESTABLISHED"
        ]
        activeConnections = len(establishedConnections)
    except (psutil.AccessDenied, PermissionError, OSError):
        activeConnections = None

    return {
        "sentMb": toMb(counters.bytes_sent),
        "receivedMb": toMb(counters.bytes_recv),
        "sentRateKbps": sentRateKbps,
        "receivedRateKbps": receivedRateKbps,
        "sampleWindowSeconds": sampleWindowSeconds,
        "errorsIn": counters.errin,
        "errorsOut": counters.errout,
        "dropIn": counters.dropin,
        "dropOut": counters.dropout,
        "activeConnections": activeConnections,
    }


def getProcessVitals() -> dict:
    """Footprint of this API process inside the machine."""
    currentProcess = psutil.Process()

    try:
        openFiles = len(currentProcess.open_files())
    except (psutil.AccessDenied, OSError):
        openFiles = None

    return {
        "pid": currentProcess.pid,
        "memoryMb": toMb(currentProcess.memory_info().rss),
        "memoryPercent": round(currentProcess.memory_percent(), 2),
        "cpuPercent": currentProcess.cpu_percent(interval=None),
        "threads": currentProcess.num_threads(),
        "openFiles": openFiles,
        "totalProcesses": len(psutil.pids()),
    }


def getHostVitals() -> dict:
    """Static identity of the machine plus how long the API has been up."""
    uptimeSeconds = getUptimeSeconds()

    return {
        "hostname": socket.gethostname(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "pythonVersion": sys.version.split()[0],
        "bootTime": datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc).isoformat(),
        "uptimeSeconds": uptimeSeconds,
        "uptimeHuman": formatDuration(uptimeSeconds),
    }


def assessOverload(cpuVitals: dict, memoryVitals: dict, diskVitals: dict) -> tuple[str, list[str]]:
    """Turn raw vitals into an overall status and the reasons behind it.

    Returns the status alongside one message per breached threshold so a caller
    can see exactly which resource is under pressure.
    """
    warnings: list[str] = []
    isCritical = False

    cpuUsage = cpuVitals["usagePercent"]
    if cpuUsage >= CPU_CRITICAL_PERCENT:
        warnings.append(f"CPU usage at {cpuUsage}% (critical above {CPU_CRITICAL_PERCENT}%)")
        isCritical = True
    elif cpuUsage >= CPU_WARNING_PERCENT:
        warnings.append(f"CPU usage at {cpuUsage}% (warning above {CPU_WARNING_PERCENT}%)")

    loadPerCore = cpuVitals["loadPerCore"]
    if loadPerCore is not None:
        if loadPerCore >= LOAD_CRITICAL_PER_CORE:
            warnings.append(f"Load average per core at {loadPerCore}, run queue is saturated")
            isCritical = True
        elif loadPerCore >= LOAD_WARNING_PER_CORE:
            warnings.append(f"Load average per core at {loadPerCore}, processes are queueing")

    memoryUsage = memoryVitals["usagePercent"]
    if memoryUsage >= MEMORY_CRITICAL_PERCENT:
        warnings.append(
            f"Memory usage at {memoryUsage}% (critical above {MEMORY_CRITICAL_PERCENT}%)"
        )
        isCritical = True
    elif memoryUsage >= MEMORY_WARNING_PERCENT:
        warnings.append(f"Memory usage at {memoryUsage}% (warning above {MEMORY_WARNING_PERCENT}%)")

    swapUsage = memoryVitals["swapUsagePercent"]
    if swapUsage >= SWAP_WARNING_PERCENT:
        warnings.append(f"Swap usage at {swapUsage}%, the host is paging to disk")

    for partition in diskVitals["partitions"]:
        partitionUsage = partition["usagePercent"]
        mountPoint = partition["mountPoint"]
        freeGb = partition["freeGb"]
        if partitionUsage >= DISK_CRITICAL_PERCENT:
            warnings.append(f"Disk {mountPoint} at {partitionUsage}% ({freeGb} GB free, critical)")
            isCritical = True
        elif partitionUsage >= DISK_WARNING_PERCENT:
            warnings.append(f"Disk {mountPoint} at {partitionUsage}% ({freeGb} GB free, warning)")

    if isCritical:
        return STATUS_OVERLOADED, warnings
    if warnings:
        return STATUS_DEGRADED, warnings
    return STATUS_HEALTHY, warnings


def collectServerVitals() -> dict:
    """Assemble every vital plus the overload verdict drawn from them."""
    cpuVitals = getCpuVitals()
    memoryVitals = getMemoryVitals()
    diskVitals = getDiskVitals()
    networkVitals = getNetworkVitals()
    processVitals = getProcessVitals()
    hostVitals = getHostVitals()

    status, warnings = assessOverload(cpuVitals, memoryVitals, diskVitals)

    return {
        "status": status,
        "healthy": status != STATUS_OVERLOADED,
        "timestamp": utcNow().isoformat(),
        "warnings": warnings,
        "host": hostVitals,
        "cpu": cpuVitals,
        "memory": memoryVitals,
        "disk": diskVitals,
        "network": networkVitals,
        "process": processVitals,
    }


def getServerStatus() -> dict:
    """Liveness reply: the server is up, and since when."""
    uptimeSeconds = getUptimeSeconds()

    return {
        "status": "up",
        "message": "Server is up and running",
        "startedAt": getStartedAt(),
        "uptimeSeconds": uptimeSeconds,
        "uptimeHuman": formatDuration(uptimeSeconds),
        "timestamp": utcNow().isoformat(),
    }
