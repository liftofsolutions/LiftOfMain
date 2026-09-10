"""Btest for the serverDetails component.

Logic only: every test builds the data the component is supposed to receive,
calls the processing function directly and checks the result matches what was
expected. Nothing here goes through a route or an HTTP client - that belongs in
the Ptest suite.
"""

from types import SimpleNamespace

import pytest

from serverDetails import service
from serverDetails.schemas import HealthResponse, StatusResponse

# Baseline values that sit comfortably below every threshold, so a test only has
# to override the one field it is actually exercising.
IDLE_CPU_PERCENT = 12.0
IDLE_MEMORY_PERCENT = 40.0
IDLE_DISK_PERCENT = 35.0


def buildCpuVitals(**overrides) -> dict:
    """A healthy CPU reading shaped exactly like `getCpuVitals` returns one."""
    cpuVitals = {
        "usagePercent": IDLE_CPU_PERCENT,
        "perCorePercent": [10.0, 14.0],
        "physicalCores": 2,
        "logicalCores": 4,
        "frequencyMhz": 2400.0,
        "loadAverage": [0.4, 0.5, 0.6],
        "loadPerCore": 0.1,
    }
    cpuVitals.update(overrides)
    return cpuVitals


def buildMemoryVitals(**overrides) -> dict:
    """A healthy memory reading shaped exactly like `getMemoryVitals` returns one."""
    memoryVitals = {
        "totalMb": 8192.0,
        "usedMb": 3276.8,
        "availableMb": 4915.2,
        "usagePercent": IDLE_MEMORY_PERCENT,
        "swapTotalMb": 2048.0,
        "swapUsedMb": 102.4,
        "swapUsagePercent": 5.0,
    }
    memoryVitals.update(overrides)
    return memoryVitals


def buildPartition(**overrides) -> dict:
    """One partition entry shaped exactly like `getDiskVitals` produces one."""
    partition = {
        "mountPoint": "/",
        "totalGb": 100.0,
        "usedGb": 35.0,
        "freeGb": 65.0,
        "usagePercent": IDLE_DISK_PERCENT,
    }
    partition.update(overrides)
    return partition


def buildDiskVitals(partitions: list[dict] | None = None, **overrides) -> dict:
    """A healthy disk reading shaped exactly like `getDiskVitals` returns one."""
    diskVitals = {
        "rootUsagePercent": IDLE_DISK_PERCENT,
        "partitions": partitions if partitions is not None else [buildPartition()],
        "readMb": 1024.0,
        "writeMb": 512.0,
    }
    diskVitals.update(overrides)
    return diskVitals


def buildNetworkCounters(**overrides) -> SimpleNamespace:
    """Stand-in for the counter object `psutil.net_io_counters` hands back."""
    counters = {
        "bytes_sent": 0,
        "bytes_recv": 0,
        "errin": 0,
        "errout": 0,
        "dropin": 0,
        "dropout": 0,
    }
    counters.update(overrides)
    return SimpleNamespace(**counters)


class TestFormatDuration:
    """Uptime seconds turned into the human string."""

    def testSecondsOnly(self):
        assert service.formatDuration(45) == "45s"

    def testMinutesAndSeconds(self):
        assert service.formatDuration(125) == "2m 05s"

    def testHoursMinutesSeconds(self):
        assert service.formatDuration(3725) == "1h 02m 05s"

    def testDaysAreLeadingUnit(self):
        assert service.formatDuration(93784) == "1d 02h 03m 04s"

    def testFractionalSecondsAreTruncated(self):
        assert service.formatDuration(59.9) == "59s"


class TestUnitConversion:
    """Byte counters converted to the units the response advertises."""

    def testBytesToMebibytes(self):
        assert service.toMb(service.BYTES_IN_MB * 3) == 3.0

    def testBytesToGibibytes(self):
        assert service.toGb(service.BYTES_IN_GB * 2.5) == 2.5

    def testConversionIsRoundedToTwoDecimals(self):
        assert service.toMb(1_234_567) == 1.18


class TestSystemRootPath:
    """The mount point checked for space differs between dev and production."""

    def testWindowsUsesTheSystemDriveNotTheCurrentDrive(self, monkeypatch):
        # Development happens on Windows, where the working drive may be D:.
        monkeypatch.setattr(service.os, "name", "nt")
        monkeypatch.setenv("SystemDrive", "C:")
        assert service.getSystemRootPath() == "C:\\"

    def testWindowsFallsBackToCDriveWhenEnvironmentIsMissing(self, monkeypatch):
        monkeypatch.setattr(service.os, "name", "nt")
        monkeypatch.delenv("SystemDrive", raising=False)
        assert service.getSystemRootPath() == "C:\\"

    def testLinuxUsesFilesystemRoot(self, monkeypatch):
        # Production runs on AWS Linux.
        monkeypatch.setattr(service.os, "name", "posix")
        assert service.getSystemRootPath() == "/"


class TestAssessOverload:
    """The verdict drawn from a set of vitals - the component's core logic."""

    def testIdleMachineIsHealthyWithNoWarnings(self):
        status, warnings = service.assessOverload(
            buildCpuVitals(), buildMemoryVitals(), buildDiskVitals()
        )

        assert status == service.STATUS_HEALTHY
        assert warnings == []

    def testCpuAboveWarningIsDegraded(self):
        status, warnings = service.assessOverload(
            buildCpuVitals(usagePercent=80.0), buildMemoryVitals(), buildDiskVitals()
        )

        assert status == service.STATUS_DEGRADED
        assert len(warnings) == 1
        assert "CPU usage at 80.0%" in warnings[0]

    def testCpuAtWarningThresholdCounts(self):
        status, _ = service.assessOverload(
            buildCpuVitals(usagePercent=service.CPU_WARNING_PERCENT),
            buildMemoryVitals(),
            buildDiskVitals(),
        )

        assert status == service.STATUS_DEGRADED

    def testCpuAboveCriticalIsOverloaded(self):
        status, warnings = service.assessOverload(
            buildCpuVitals(usagePercent=95.0), buildMemoryVitals(), buildDiskVitals()
        )

        assert status == service.STATUS_OVERLOADED
        assert "critical" in warnings[0]

    def testSaturatedRunQueueIsOverloaded(self):
        status, warnings = service.assessOverload(
            buildCpuVitals(loadAverage=[4.8, 4.2, 3.9], loadPerCore=1.2),
            buildMemoryVitals(),
            buildDiskVitals(),
        )

        assert status == service.STATUS_OVERLOADED
        assert any("Load average per core at 1.2" in warning for warning in warnings)

    def testQueueingProcessesAreDegraded(self):
        status, warnings = service.assessOverload(
            buildCpuVitals(loadAverage=[3.2, 3.0, 2.8], loadPerCore=0.8),
            buildMemoryVitals(),
            buildDiskVitals(),
        )

        assert status == service.STATUS_DEGRADED
        assert any("processes are queueing" in warning for warning in warnings)

    def testMissingLoadAverageIsIgnored(self):
        # Windows may report no load average at all; it must not crash the check.
        status, warnings = service.assessOverload(
            buildCpuVitals(loadAverage=None, loadPerCore=None),
            buildMemoryVitals(),
            buildDiskVitals(),
        )

        assert status == service.STATUS_HEALTHY
        assert warnings == []

    def testMemoryAboveCriticalIsOverloaded(self):
        status, warnings = service.assessOverload(
            buildCpuVitals(), buildMemoryVitals(usagePercent=93.0), buildDiskVitals()
        )

        assert status == service.STATUS_OVERLOADED
        assert any("Memory usage at 93.0%" in warning for warning in warnings)

    def testSwapPressureIsOnlyDegraded(self):
        status, warnings = service.assessOverload(
            buildCpuVitals(), buildMemoryVitals(swapUsagePercent=65.0), buildDiskVitals()
        )

        assert status == service.STATUS_DEGRADED
        assert any("paging to disk" in warning for warning in warnings)

    def testFullPartitionIsOverloadedAndNamesTheMountPoint(self):
        diskVitals = buildDiskVitals(
            partitions=[buildPartition(mountPoint="/var", usagePercent=95.0, freeGb=5.0)]
        )

        status, warnings = service.assessOverload(
            buildCpuVitals(), buildMemoryVitals(), diskVitals
        )

        assert status == service.STATUS_OVERLOADED
        assert "/var" in warnings[0]
        assert "5.0 GB free" in warnings[0]

    def testOnlyTheBreachedPartitionIsReported(self):
        diskVitals = buildDiskVitals(
            partitions=[
                buildPartition(mountPoint="/", usagePercent=20.0),
                buildPartition(mountPoint="/data", usagePercent=85.0, freeGb=15.0),
            ]
        )

        status, warnings = service.assessOverload(
            buildCpuVitals(), buildMemoryVitals(), diskVitals
        )

        assert status == service.STATUS_DEGRADED
        assert len(warnings) == 1
        assert "/data" in warnings[0]

    def testOneCriticalAmongWarningsStillMeansOverloaded(self):
        status, warnings = service.assessOverload(
            buildCpuVitals(usagePercent=80.0),
            buildMemoryVitals(usagePercent=95.0, swapUsagePercent=60.0),
            buildDiskVitals(),
        )

        assert status == service.STATUS_OVERLOADED
        assert len(warnings) == 3


class TestNetworkRateMath:
    """Throughput derived from the gap between two counter readings."""

    def primeSample(self, monkeypatch, bytesSent, bytesReceived, sampledAt):
        """Pretend a previous health call already recorded these counters."""
        monkeypatch.setattr(
            service,
            "LAST_NETWORK_SAMPLE",
            {"sampledAt": sampledAt, "bytesSent": bytesSent, "bytesReceived": bytesReceived},
        )

    def patchReadings(self, monkeypatch, counters, now):
        """Point the service at fixed counters and a fixed clock."""
        monkeypatch.setattr(service.psutil, "net_io_counters", lambda: counters)
        monkeypatch.setattr(service, "time", SimpleNamespace(time=lambda: now))
        monkeypatch.setattr(service.psutil, "net_connections", lambda kind: [])

    def testFirstCallHasNoRateToReportYet(self, monkeypatch):
        monkeypatch.setattr(service, "LAST_NETWORK_SAMPLE", None)
        self.patchReadings(monkeypatch, buildNetworkCounters(bytes_sent=100), now=1000.0)

        networkVitals = service.getNetworkVitals()

        assert networkVitals["sentRateKbps"] is None
        assert networkVitals["receivedRateKbps"] is None
        assert networkVitals["sampleWindowSeconds"] is None

    def testRateIsBytesDeltaConvertedToKilobitsPerSecond(self, monkeypatch):
        self.primeSample(monkeypatch, bytesSent=0, bytesReceived=0, sampledAt=1000.0)
        counters = buildNetworkCounters(bytes_sent=1_000_000, bytes_recv=500_000)
        self.patchReadings(monkeypatch, counters, now=1002.0)

        networkVitals = service.getNetworkVitals()

        # 1_000_000 bytes * 8 bits / 2 seconds / 1000 = 4000 kbps
        assert networkVitals["sentRateKbps"] == 4000.0
        assert networkVitals["receivedRateKbps"] == 2000.0
        assert networkVitals["sampleWindowSeconds"] == 2.0

    def testCounterResetIsReportedAsNoRateRatherThanNegative(self, monkeypatch):
        # An interface restart makes the counters go backwards.
        self.primeSample(monkeypatch, bytesSent=9_000_000, bytesReceived=9_000_000, sampledAt=1000.0)
        self.patchReadings(monkeypatch, buildNetworkCounters(bytes_sent=10), now=1002.0)

        networkVitals = service.getNetworkVitals()

        assert networkVitals["sentRateKbps"] is None
        assert networkVitals["receivedRateKbps"] is None

    def testSampleIsStoredForTheNextCall(self, monkeypatch):
        monkeypatch.setattr(service, "LAST_NETWORK_SAMPLE", None)
        counters = buildNetworkCounters(bytes_sent=77, bytes_recv=88)
        self.patchReadings(monkeypatch, counters, now=1500.0)

        service.getNetworkVitals()

        assert service.LAST_NETWORK_SAMPLE == {
            "sampledAt": 1500.0,
            "bytesSent": 77,
            "bytesReceived": 88,
        }

    def testTotalsAndErrorCountersArePassedThrough(self, monkeypatch):
        monkeypatch.setattr(service, "LAST_NETWORK_SAMPLE", None)
        counters = buildNetworkCounters(
            bytes_sent=service.BYTES_IN_MB * 4,
            bytes_recv=service.BYTES_IN_MB * 6,
            errin=1,
            errout=2,
            dropin=3,
            dropout=4,
        )
        self.patchReadings(monkeypatch, counters, now=1000.0)

        networkVitals = service.getNetworkVitals()

        assert networkVitals["sentMb"] == 4.0
        assert networkVitals["receivedMb"] == 6.0
        assert networkVitals["errorsIn"] == 1
        assert networkVitals["errorsOut"] == 2
        assert networkVitals["dropIn"] == 3
        assert networkVitals["dropOut"] == 4

    def testDeniedConnectionListingLeavesTheCountUnknown(self, monkeypatch):
        monkeypatch.setattr(service, "LAST_NETWORK_SAMPLE", None)
        self.patchReadings(monkeypatch, buildNetworkCounters(), now=1000.0)

        def raiseAccessDenied(kind):
            raise service.psutil.AccessDenied()

        monkeypatch.setattr(service.psutil, "net_connections", raiseAccessDenied)

        assert service.getNetworkVitals()["activeConnections"] is None


class TestCollectServerVitals:
    """The assembled payload, checked against the model it must satisfy."""

    def testPayloadMatchesTheHealthResponseModel(self):
        vitals = service.collectServerVitals()

        # Pydantic validates every field name and type in one shot; a mismatch
        # between the logic and the declared model raises here.
        healthResponse = HealthResponse(**vitals)

        assert healthResponse.status in {
            service.STATUS_HEALTHY,
            service.STATUS_DEGRADED,
            service.STATUS_OVERLOADED,
        }
        assert healthResponse.host.hostname
        assert healthResponse.cpu.logicalCores >= 1
        assert healthResponse.memory.totalMb > 0
        assert healthResponse.disk.partitions
        assert healthResponse.process.pid > 0

    def testVerdictAndWarningsComeFromTheAssessment(self, monkeypatch):
        monkeypatch.setattr(
            service,
            "assessOverload",
            lambda cpuVitals, memoryVitals, diskVitals: (
                service.STATUS_DEGRADED,
                ["swap is filling up"],
            ),
        )

        vitals = service.collectServerVitals()

        assert vitals["status"] == service.STATUS_DEGRADED
        assert vitals["warnings"] == ["swap is filling up"]

    @pytest.mark.parametrize(
        "assessedStatus, expectedHealthy",
        [
            ("healthy", True),
            ("degraded", True),
            ("overloaded", False),
        ],
    )
    def testHealthyFlagIsFalseOnlyWhenOverloaded(
        self, monkeypatch, assessedStatus, expectedHealthy
    ):
        monkeypatch.setattr(
            service,
            "assessOverload",
            lambda cpuVitals, memoryVitals, diskVitals: (assessedStatus, []),
        )

        assert service.collectServerVitals()["healthy"] is expectedHealthy


class TestServerStatus:
    """The liveness payload, checked against the model it must satisfy."""

    def testPayloadMatchesTheStatusResponseModel(self):
        statusResponse = StatusResponse(**service.getServerStatus())

        assert statusResponse.status == "up"
        assert statusResponse.message == "Server is up and running"
        assert statusResponse.uptimeSeconds >= 0

    def testUptimeIsMeasuredFromTheProcessStartTime(self, monkeypatch):
        monkeypatch.setattr(service, "SERVER_START_TIME", 1000.0)
        monkeypatch.setattr(service, "time", SimpleNamespace(time=lambda: 1000.0 + 93784))

        serverStatus = service.getServerStatus()

        assert serverStatus["uptimeSeconds"] == 93784.0
        assert serverStatus["uptimeHuman"] == "1d 02h 03m 04s"
