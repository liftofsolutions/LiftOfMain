"""Response models for the server details endpoints."""

from pydantic import BaseModel, Field


class CpuVitals(BaseModel):
    """Processor pressure of the machine."""

    usagePercent: float = Field(description="Overall CPU utilisation since the last sample.")
    perCorePercent: list[float] = Field(description="Utilisation of every logical core.")
    physicalCores: int | None = Field(description="Physical cores, None when undetectable.")
    logicalCores: int | None = Field(description="Logical cores (threads).")
    frequencyMhz: float | None = Field(description="Current clock speed in MHz.")
    loadAverage: list[float] | None = Field(
        description="1/5/15 minute load average, None when unsupported."
    )
    loadPerCore: float | None = Field(
        description="1 minute load average divided by the logical core count."
    )


class MemoryVitals(BaseModel):
    """RAM and swap consumption."""

    totalMb: float
    usedMb: float
    availableMb: float
    usagePercent: float
    swapTotalMb: float
    swapUsedMb: float
    swapUsagePercent: float


class DiskPartitionVitals(BaseModel):
    """Space consumption of a single mounted partition."""

    mountPoint: str
    totalGb: float
    usedGb: float
    freeGb: float
    usagePercent: float


class DiskVitals(BaseModel):
    """Space consumption plus I/O counters of the storage layer."""

    rootUsagePercent: float
    partitions: list[DiskPartitionVitals]
    readMb: float | None = Field(description="Bytes read since boot, None when unavailable.")
    writeMb: float | None = Field(description="Bytes written since boot, None when unavailable.")


class NetworkVitals(BaseModel):
    """Traffic counters and the throughput seen since the previous health call."""

    sentMb: float = Field(description="Total sent since boot.")
    receivedMb: float = Field(description="Total received since boot.")
    sentRateKbps: float | None = Field(description="Send rate since the previous health call.")
    receivedRateKbps: float | None = Field(description="Receive rate since the previous health call.")
    sampleWindowSeconds: float | None = Field(description="Gap the rates were measured over.")
    errorsIn: int
    errorsOut: int
    dropIn: int
    dropOut: int
    activeConnections: int | None = Field(
        description="Established sockets, None when the process lacks permission."
    )


class ProcessVitals(BaseModel):
    """Footprint of this API process inside the machine."""

    pid: int
    memoryMb: float
    memoryPercent: float
    cpuPercent: float
    threads: int
    openFiles: int | None
    totalProcesses: int


class HostVitals(BaseModel):
    """Static identity of the machine serving the request."""

    hostname: str
    platform: str
    pythonVersion: str
    bootTime: str
    uptimeSeconds: float
    uptimeHuman: str


class HealthResponse(BaseModel):
    """Full vitals payload used to judge whether the server is overloading."""

    status: str = Field(description="One of healthy, degraded or overloaded.")
    healthy: bool
    timestamp: str
    warnings: list[str] = Field(description="Human readable reason for every breached threshold.")
    host: HostVitals
    cpu: CpuVitals
    memory: MemoryVitals
    disk: DiskVitals
    network: NetworkVitals
    process: ProcessVitals


class StatusResponse(BaseModel):
    """Minimal liveness reply."""

    status: str
    message: str
    startedAt: str
    uptimeSeconds: float
    uptimeHuman: str
    timestamp: str
