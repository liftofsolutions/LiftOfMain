# Testing

Two suites that do not overlap.

| Suite | Tests | Rule |
| --- | --- | --- |
| **Btest** | A component's logic | Call the functions directly. **Never call a route.** |
| **Ptest** | The product | Drive the app end to end through its routes. |

```
Btest/<componentName>/<ComponentName>Test.py
Ptest/<componentName>/<ComponentName>Test.py
```

The split is the point. When a Btest fails, the logic is wrong - there is no HTTP
layer in the way to be the culprit. When a Ptest fails and the Btests pass, the
wiring is wrong: a router not included, a `response_model` that does not match, a
path typo. Testing logic through the HTTP layer collapses those two signals into
one and makes the fast suite slow.

## Running

```bash
uv run pytest Btest -q      # logic
uv run pytest Ptest -q      # product
uv run pytest -q            # both
```

Discovery is configured in `pyproject.toml`, because the project's camelCase
naming does not match pytest's snake_case defaults:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["Btest", "Ptest"]
python_files = ["*Test.py"]
python_classes = ["Test*"]
python_functions = ["test*"]
```

`pythonpath = ["src"]` is what lets a test write `from serverDetails import
service` - the project is not installed into the venv, so without it there would
be nothing to import.

## Writing a Btest

Build the data the component is supposed to receive, call the processing function
directly, assert the result matches.

**1. Give every fixture a healthy baseline with overrides.** A test should state
only the one thing it is exercising; everything else stays valid and quiet.

```python
def buildCpuVitals(**overrides) -> dict:
    """A healthy CPU reading shaped exactly like `getCpuVitals` returns one."""
    cpuVitals = {
        "usagePercent": IDLE_CPU_PERCENT,
        "perCorePercent": [10.0, 14.0],
        "logicalCores": 4,
        "loadAverage": [0.4, 0.5, 0.6],
        "loadPerCore": 0.1,
        # ...
    }
    cpuVitals.update(overrides)
    return cpuVitals
```

**2. Call the real function.** Not an endpoint, not a client.

```python
def testCpuAboveCriticalIsOverloaded(self):
    status, warnings = service.assessOverload(
        buildCpuVitals(usagePercent=95.0), buildMemoryVitals(), buildDiskVitals()
    )

    assert status == service.STATUS_OVERLOADED
    assert "critical" in warnings[0]
```

**3. Check a payload by instantiating its model.** Where a function produces a
response body, constructing the pydantic model validates every field name and
type at once, so logic drifting away from the declared schema fails here rather
than in production:

```python
def testPayloadMatchesTheHealthResponseModel(self):
    healthResponse = HealthResponse(**service.collectServerVitals())
    assert healthResponse.cpu.logicalCores >= 1
```

**4. Isolate the outside world with `monkeypatch`,** never with a running server -
the clock, `psutil`, `os.name`, and module-level state such as
`LAST_NETWORK_SAMPLE`.

```python
def testRateIsBytesDeltaConvertedToKilobitsPerSecond(self, monkeypatch):
    self.primeSample(monkeypatch, bytesSent=0, bytesReceived=0, sampledAt=1000.0)
    self.patchReadings(monkeypatch, buildNetworkCounters(bytes_sent=1_000_000), now=1002.0)

    # 1_000_000 bytes * 8 bits / 2 seconds / 1000 = 4000 kbps
    assert service.getNetworkVitals()["sentRateKbps"] == 4000.0
```

Patching the clock is what makes that assertion an exact number instead of an
approximate range.

**No `TestClient`, `httpx`, `requests` or `app` import belongs in a Btest.** If a
test needs the HTTP layer, it is a Ptest.

## What gets a test

Anything with a decision in it. For `serverDetails` that meant:

- **Threshold logic** - every branch of `assessOverload`, including boundaries
  (exactly at the limit counts), one critical among several warnings, and only
  the breached partition being named.
- **Arithmetic** - byte conversion, duration formatting, the kbps rate.
- **Platform branches** - the Windows and Linux paths of `getSystemRootPath`, both
  exercised by patching `os.name`, plus the missing-`%SystemDrive%` fallback.
- **Degradation paths** - a missing load average, a counter reset, a permission
  denial. Each must produce `None` rather than an exception.
- **Payload shape** - both response models constructed from real output.

## Confirm the tests can fail

A green suite proves nothing until you have seen it go red. Before calling work
done, break the logic on purpose and check the right tests catch it.

For this suite, flipping `isCritical = True` to `False` in `assessOverload` failed
exactly the three tests that assert on critical escalation, and nothing else -
which is the shape you want: the tests fail, and they point at the change.
