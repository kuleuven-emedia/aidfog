# Audio Cueing System for Parkinson's Patients -- Project Overview

This document explains what this project does, how it works, and what has been achieved, written for people who are not deeply familiar with embedded hardware or Bluetooth programming.

---

## What is the problem?

People with Parkinson's disease sometimes experience "freezing of gait" (FoG) -- their feet suddenly feel glued to the floor and they cannot walk. Research has shown that playing a rhythmic sound (like a metronome click) through earbuds can help them "unfreeze" and resume walking. This is called **audio cueing**.

The challenge is doing this **automatically and in real time**. Body-worn sensors detect that the patient is about to freeze (using AI), and the system needs to immediately tell the earbuds to start playing a sound. This requires the earbuds to accept wireless commands from a computer -- something normal earbuds cannot do.

---

## What are the PineBuds Pro?

The [PineBuds Pro](https://wiki.pine64.org/wiki/PineBuds_Pro) are open-source wireless earbuds made by Pine64. Unlike AirPods or Galaxy Buds, their firmware (the software running inside the earbuds) is open-source and can be modified. They use a BES2300YP chip -- a small computer with two processors, less than 1 MB of memory, and a Bluetooth radio.

Out of the box, these earbuds only understand standard audio protocols (playing music, phone calls). There is no way for a computer program to tell them "play this specific sound right now."

---

## What did we build?

We added a **custom wireless interface** to the earbuds. Think of it like adding a new "remote control channel" that a laptop can use to send commands to the earbuds.

This was done in two parts:

### Part 1: Inside the earbuds (firmware, written in C)

We added a new Bluetooth Low Energy (BLE) service to the earbuds' firmware. A BLE service is like a small API that other devices can interact with wirelessly. Our service has three "endpoints":

| Endpoint | What it does | Direction |
|----------|-------------|-----------|
| **Command** | Receives instructions: "start playing," "stop playing," or "change settings" | Laptop → Earbud |
| **Status** | Reports what the earbud is doing: idle, playing a sound, or error | Earbud → Laptop |
| **Config** | Stores and retrieves settings: which sound, how loud, how long, burst patterns | Both ways |

This required writing approximately 1,500 lines of new C code and modifying 18 existing firmware files.

### Part 2: On the laptop (Python scripts and HERMES integration)

On the computer side, we wrote Python code that uses a Bluetooth library called `bleak` to wirelessly talk to the earbuds. This code is structured to fit into the **HERMES framework** -- a real-time data processing system developed at KU Leuven that handles sensor data, AI inference, and actuation in a coordinated pipeline.

The pipeline works like this:

```
Body Sensors → AI Model (detects freezing) → Cueing Controller → Earbuds (play sound)
                                                    ↑
                                              This is our part
```

When the AI model outputs a high probability of freezing, our controller sends a "start" command to the earbuds over Bluetooth. When the probability drops, it sends "stop."

---

## What challenges did we face?

### Memory constraints
The chip inside the earbuds has only 992 KB of RAM -- shared between the operating system, audio processing, and Bluetooth. Enabling the BLE stack required disabling features we don't need (noise cancellation, a high-quality audio codec) to free up enough memory.

### Boot crash
After enabling BLE, the earbuds crashed with a red LED when powered on inside the charging case. We traced this to a timing issue: the firmware tried to use BLE data structures before they were initialized. Fixed by moving the initialization earlier in the startup sequence.

### Invisible to scanners
Even after the crash was fixed, computers couldn't "see" the earbuds when scanning for Bluetooth devices. The firmware had three separate checks that blocked the earbuds from announcing their presence -- all designed for a two-earbud pairing scenario that doesn't apply to our use case. We bypassed all three.

### Device name mismatch
The earbuds advertise their name as "D&D TECH" (the factory-programmed name), not "PineBuds Pro" as we expected. Our scanning scripts were looking for the wrong name. Once we figured this out and pointed the scripts at the correct address (`12:34:56:C2:A2:30`), everything connected immediately.

---

## What has been achieved?

### The system works end-to-end

On March 12, 2026, we confirmed full end-to-end operation:
1. A Python script on a Windows laptop connects to the earbud over BLE
2. It sends a "start cueing" command
3. The earbud acknowledges with a "CUEING" status notification
4. After 500ms the earbud auto-stops and sends an "IDLE" notification
5. The script sends "stop" and gets confirmation

### Latency is well within requirements

We ran a benchmark of 100 start/stop cycles with zero failures:

| Metric | Value | What it means |
|--------|-------|---------------|
| **Median response time** | **25 ms** | Half of all commands get a response in under 25 milliseconds |
| Mean response time | 28 ms | The average across all 200 operations |
| Worst case (P99) | 53 ms | 99% of commands complete within 53 ms |
| **Reliability** | **100%** | All 200 operations succeeded, zero timeouts |

For context, real-time FoG cueing typically requires a response time under 100-500 ms. Our 25 ms median is well within this range -- the patient would hear the sound almost instantly after the AI detects freezing.

### Integration with HERMES framework

The Python cueing code has been integrated into the HERMES pipeline structure in this repository:
- `buds_facade.py` handles the BLE connection (with automatic reconnection)
- `buds_handler.py` runs in a background process, receiving commands from the pipeline
- `pipeline.py` implements the control logic (when to start/stop cueing based on FoG probability)
- `stream.py` defines the data format for logging cueing events during experiments

---

## What's left to do?

| Task | Description |
|------|-------------|
| **Stability test** | Run the system for several hours to verify the Bluetooth connection stays stable |
| **Custom sounds** | Replace the default alert beeps with proper metronome clicks |
| **AI integration** | Connect the cueing pipeline to Alex's FoG detection model |
| **Experiments** | Collect latency data under different conditions for the thesis |
| **Thesis** | Write up methodology, results, and analysis |

---

## File structure in this repository

```
buds_firmware/                          ← Earbud firmware (C code)
  services/ble_profiles/cueing/         ← BLE service definition (GATT database)
  services/ble_app/app_cueing/          ← Application logic (command handling, audio)
  services/ble_app/app_main/            ← Modified registration files
  services/ble_stack/ble_ip/            ← Modified BLE stack configuration

hermes/aidfog/                          ← HERMES pipeline (Python code)
  controller/buds_facade.py             ← BLE backend (connect, send commands, reconnect)
  controller/buds_handler.py            ← Background process managing the earbuds
  pipeline.py                           ← HERMES Pipeline node (FoG → cueing logic)
  stream.py                             ← HDF5 data stream definitions
  utils/types.py                        ← UUIDs, constants, data structures
  utils/utilities.py                    ← Statistical helper functions
```
