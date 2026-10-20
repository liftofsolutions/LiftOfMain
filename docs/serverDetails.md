# serverDetails

Operational endpoints: is the server up, and is it in trouble.

```
src/serverDetails/
  routes.py     GET /health, GET /status
  service.py    collection and assessment logic
  model.py      response models
```

`routes.py` does nothing but delegate - `collectServerVitals()` and
`getServerStatus()` in `service.py` hold everything, and neither imports FastAPI.

---

## `GET /status`

The cheap one. Answers "is the process alive, and since when", and touches
nothing outside the process to do it. Suitable for a load balancer target or an
uptime pinger polling every few seconds.

```json
{
  "status": "up",
  "message": "Server is up and running",
  "startedAt": "2026-09-05T09:14:02.117446+00:00",
  "uptimeSeconds": 4517.83,
  "uptimeHuman": "1h 15m 17s",
  "timestamp": "2026-09-05T10:29:19.951203+00:00"
}
```

| Field | Meaning |
| --- | --- |
| `status` | Always `"up"` - if the process were not, there would be no reply |
| `startedAt` | UTC ISO timestamp of when this process came up |
| `uptimeSeconds` | Seconds since then, as a number for machines |
| `uptimeHuman` | The same, as `1d 02h 03m 04s`, for people |
| `timestamp` | When this reply was produced |

`startedAt` is worth watching: an uptime that keeps resetting means the process is
restarting, which a plain "up" would hide.

---

## `GET /health`

The full picture. Everything needed to judge whether the machine is overloading,
plus the verdict itself.

### The verdict

Three fields summarise the rest, so a caller does not have to re-derive them:

| Field | Meaning |
| --- | --- |
| `status` | `healthy`, `degraded`, or `overloaded` |
| `healthy` | `false` only when `overloaded` |
| `warnings` | One human-readable line per breached threshold |

A warning names the resource, the reading and the limit, so it can be alerted on
as-is:

```json
{
  "status": "overloaded",
  "healthy": false,
  "warnings": [
    "CPU usage at 94.2% (critical above 90.0%)",
    "Disk /var at 95.0% (5.0 GB free, critical)"
  ]
}
```

### The thresholds

All of them are module constants at the top of `service.py`; tune them there.

| Reading | Warning → `degraded` | Critical → `overloaded` |
| --- | --- | --- |
| CPU usage | 75% | 90% |
| Load average per core | 0.7 | 1.0 |
| Memory usage | 75% | 90% |
| Swap usage | 50% | *(never critical)* |
| Any partition | 80% | 90% |

Comparisons are `>=`, so a reading exactly at a threshold counts as breached.
One critical reading anywhere makes the whole response `overloaded`, however
many mere warnings sit alongside it.

Swap deliberately has no critical level. Swapping means the host is paging to
disk and everything on it has become slow, which is worth surfacing, but the
memory percentage that caused it is what should escalate.

### The vitals

**`host`** - hostname, platform string, Python version, machine boot time, and
this process's uptime in both numeric and human form.

**`cpu`** - `usagePercent`, `perCorePercent` (one entry per logical core, so a
single pinned core is visible even when the average looks calm), `physicalCores`,
`logicalCores`, `frequencyMhz`, `loadAverage` (1/5/15 minute), and `loadPerCore`.

`loadPerCore` is the interesting one: raw load average means nothing without
knowing the core count, and dividing by it gives a figure where 1.0 means "the
run queue is exactly full" on any size of machine.

**`memory`** - `totalMb`, `usedMb`, `availableMb`, `usagePercent`, and the same
for swap.

`usedMb` is total minus *available*, not the kernel's "used". Available accounts
for cache and buffers that would be reclaimed under pressure; the other figure
makes a healthy Linux box look permanently full.

**`disk`** - `rootUsagePercent` for the drive the OS lives on, `partitions[]` with
`mountPoint` / `totalGb` / `usedGb` / `freeGb` / `usagePercent` for each mounted
partition, and cumulative `readMb` / `writeMb` since boot.

Every partition is assessed, not just the root one - a full `/var` will take the
service down as surely as a full `/`. Partitions that cannot be read (empty
optical drives, permission-denied mounts) are skipped rather than raising.

**`network`** - `sentMb` / `receivedMb` since boot, `sentRateKbps` /
`receivedRateKbps` for live throughput, `sampleWindowSeconds`, the `errorsIn` /
`errorsOut` / `dropIn` / `dropOut` counters, and `activeConnections`.

Rising error and drop counters point at a saturated link or a failing NIC, which
CPU and memory figures will not show. `activeConnections` counts established
inbound and outbound sockets and is `null` when the process lacks the privilege
to enumerate them.

**`process`** - this API's own `pid`, `memoryMb`, `memoryPercent`, `cpuPercent`,
`threads`, `openFiles`, and `totalProcesses` on the box.

This separates "the server is busy" from "*we* are the reason it is busy". A
climbing `threads` or `openFiles` with flat traffic is a leak.

---

## Design decisions

### Nothing blocks

The obvious way to measure CPU percentage is `psutil.cpu_percent(interval=1)`,
which sleeps for a second and reports what happened during it. Inside a request
handler that is a second of the event loop held hostage, on the endpoint most
likely to be polled frequently - a health check that degrades the health of the
server is self-defeating.

Instead both CPU and network figures are measured **between calls**:

- `psutil.cpu_percent(interval=None)` reports usage since the previous call. The
  counters are primed once at import so the first request is not comparing
  against nothing.
- Network throughput keeps the previous byte counters and timestamp in
  `LAST_NETWORK_SAMPLE`, and divides the delta by the elapsed time.

The cost is that the very first call after startup returns `0.0` CPU and `null`
rates - there is no window to measure yet. From the second call on, both are
real, and `sampleWindowSeconds` says how wide the window was so a reader can tell
a 200ms sample from a 5-minute one.

A counter going backwards - an interface restart resets it - is reported as
`null` rather than as a negative rate.

### Always HTTP 200

The response is `200` even when `status` is `overloaded`. The endpoint's job is
to report; deciding what an overloaded box means is the caller's.

If a load balancer should pull the instance out of rotation on `overloaded`, that
is a one-line change in `routes.py` to return `503` on that status. It was left
out because it is a policy decision, and because returning non-200 from `/health`
makes simple monitoring unable to distinguish "overloaded" from "down".

### Degrade, never raise

Everything platform-dependent is guarded and falls back to `null`: `loadAverage`
and `loadPerCore` where the platform has no load average, `frequencyMhz` where
the clock is not readable, `openFiles` and `activeConnections` where the process
lacks permission, disk I/O counters where they are unavailable.

A health endpoint that returns a 500 because one optional field could not be read
has failed at the one thing it exists to do.

### Cross-platform disk root

`getSystemRootPath()` branches on `os.name` rather than using
`os.path.abspath(os.sep)`. See [architecture.md](architecture.md#platform-windows-in-development-aws-linux-in-production)
for why the portable-looking version is wrong on Windows.

---

## Tests

`Btest/serverDetails/ServerDetailsTest.py` - 36 tests, logic only, no route
calls. See [testing.md](testing.md).
