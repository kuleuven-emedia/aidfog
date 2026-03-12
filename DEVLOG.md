# Development Log -- Audio Cueing for PineBuds Pro

All changes made to this repository, in chronological order.

---

## 2026-03-06 -- Initial firmware commit

**Commit:** `a3e65e2` -- Add PineBuds Pro cueing firmware: BLE GATT service and profile

**What was done:**
Copied the custom BLE cueing firmware files into `buds_firmware/`. These are the C source files that add a new Bluetooth Low Energy service to the PineBuds Pro earbuds, allowing a computer to wirelessly send "play sound" and "stop sound" commands.

**Files added:**
- `buds_firmware/services/ble_profiles/cueing/` -- GATT service definition (4 files)
- `buds_firmware/services/ble_app/app_cueing/` -- Application logic (2 files)
- `buds_firmware/services/ble_app/app_main/app.c`, `app_task.c` -- Modified registration files
- `buds_firmware/services/ble_profiles/prf/prf.c` -- Modified profile registry
- `buds_firmware/services/ble_stack/ble_ip/` -- Modified BLE stack config (4 files)
- `buds_firmware/services/ble_app/Makefile`, `buds_firmware/services/ble_profiles/Makefile`

---

## 2026-03-12 -- HERMES pipeline implementation

**Commit:** `8fa04e9` -- Implement HERMES cueing pipeline: BLE backend, handler, types, stream

**What was done:**
Replaced the exoskeleton boilerplate in `hermes/aidfog/` with the actual audio cueing implementation. The boilerplate was set up by the supervisor with Nicla IMU sensors, CAN bus motors, and exoskeleton-specific logic. All of that was replaced with cueing-specific code that talks to the PineBuds Pro earbuds over BLE.

**Why these files were changed (not just buds_firmware/):**
The `hermes/aidfog/` files are the Python-side HERMES pipeline components. The supervisor created them as a starting template. They contained exoskeleton code (Nicla sensors, servo motors, CAN bus) that had to be replaced with earbud cueing code. Without these changes, the HERMES pipeline cannot communicate with the earbuds.

**Files changed:**

| File | Before (boilerplate) | After (cueing) |
|------|---------------------|----------------|
| `utils/types.py` | ServoMotor, NiclaData, ExoNiclaMapping, CAN bus types | CueingConfig, CueState, CueEvent, BLE UUIDs, command constants |
| `controller/buds_facade.py` | NiclaBleBackend (connects to Nicla IMU sensors) | BudsBleBackend (connects to PineBuds Pro, sends start/stop/configure) |
| `controller/buds_handler.py` | BudsHandler polling Nicla sensor data | BudsHandler processing cueing commands via multiprocessing queues |
| `pipeline.py` | BudsPipeline forwarding Nicla/motor data | BudsPipeline with FoG threshold logic, cueing command dispatch |
| `stream.py` | BudsStream with Nicla/motor/power HDF5 streams | BudsStream with cueing status HDF5 streams |
| `utils/utilities.py` | CAN bus config, angle wrapping | Running stats, percentile computation |
| `utils/__init__.py` | Empty | Exports cueing types and constants |
| `pyproject.toml` | No bleak dependency | Added `bleak>=0.21.0` |

**Nothing outside `hermes/aidfog/` and `buds_firmware/` was touched.** The HERMES framework itself (`hermes/base/`, `hermes/utils/`, etc.) is unchanged.

---

## 2026-03-12 -- BLE service confirmed working (in OpenPineBuds repo)

**Not a commit in this repo** -- this happened in the separate [OpenPineBuds](https://github.com/pine64/OpenPineBuds) firmware repo.

**Milestone:** Connected to the earbud at `12:34:56:C2:A2:30` from a Python script and confirmed the full GATT service tree is visible, including the custom Audio Cueing Service with all three characteristics (Command, Status, Config).

**Latency benchmark results (100 iterations):**

| Metric | Value |
|--------|-------|
| Mean round-trip | 28.36 ms |
| Median round-trip | 25.01 ms |
| P95 | 43.72 ms |
| P99 | 52.76 ms |
| Timeout rate | 0% (200/200 succeeded) |
