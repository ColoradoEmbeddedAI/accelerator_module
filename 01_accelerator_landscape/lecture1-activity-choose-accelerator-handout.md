# Choose the Accelerator — Handout
### Embedded AI — The Accelerator Landscape

Each card describes a product. Your group's job is to pick the class of hardware that runs its neural network, the way an engineer would: **eliminate first, then choose, then say what you gave up.**

The device classes: **CPU + SIMD** (Cortex-M with CMSIS-NN, Helium) · **DSP** · **GPU** (embedded, e.g. Jetson) · **FPGA** · **microNPU** (Ethos-U55/U65/U85, ST Neural-ART) · **edge/client NPU** (Edge TPU, Hexagon, Apple Neural Engine, AMD XDNA2) · **analog / in-memory compute**.

The axes: performance per watt (energy per inference) · flexibility and programmability · latency determinism · memory capacity and bandwidth · toolchain maturity and development effort · unit cost · power envelope.

## How to work each card (about 7 minutes per card)

1. **Eliminate.** Which classes are ruled out, and by which requirement? Usually the power envelope or the latency does most of the work.
2. **Estimate.** Do one back-of-envelope number: MACs per second needed, or energy per inference allowed. The rough figures below are enough.
3. **Choose** one class and a representative part. Justify it with at least three axes.
4. **Give up.** What does your choice cost you? What is the runner-up, and why did it lose?
5. **Verify.** What is the first thing you would measure to check the choice?

## Rough figures you may use

All order-of-magnitude; state any other assumption you make.

| | Throughput (int8) | Power while running | Notes |
|---|---|---|---|
| Cortex-M4 @ 168 MHz, CMSIS-NN | ~5 cycles/MAC measured (Lecture 9) → ~35 M MAC/s | ~100 mW (core + memories) | our board |
| Cortex-M55 + Helium | ~8 MACs/cycle peak | ~tens of mW | |
| Ethos-U55-256 @ 500 MHz | 256 MACs/cycle peak (128 G MAC/s); assume ~50% utilization | ~tens of mW (NPU + SRAM) | needs a Cortex-M host |
| ST STM32N6 (Neural-ART) | 600 GOPS peak (~300 G MAC/s) | ~hundreds of mW | Cortex-M55 host |
| Edge TPU | 4 TOPS peak | ~2 W | USB/PCIe module |
| Client NPU (XDNA2, ANE, Hexagon) | 40–50 TOPS peak | ~1–5 W | in a laptop/phone SoC |
| Embedded GPU (Jetson Orin Nano) | tens of TOPS (sparse) | 7–25 W | Linux, CUDA |
| FPGA (small/medium) | depends on the design; one result per clock is typical | 1–10 W | hours per compile |

Useful constants: a CR2032 coin cell holds about 225 mAh at 3 V ≈ **2,400 J**. A year is about **31.5 million seconds**. A typical GPU kernel launch costs **microseconds**.

---

## Card 1 — Keyword spotter on a coin cell

A wireless button listens for "hey device". It must run **for one year on one CR2032 coin cell**.

| Requirement | Value |
|---|---|
| Model | a small CNN, about **5 M MACs** per inference (keyword spotting in MLPerf Tiny is this size) |
| Rate | **2 inferences per second**, always on |
| Latency | under 200 ms from end of word |
| Power envelope | the whole device: microwatts on average |
| Cost / volume | under $3 of silicon, millions of units |

Hint: how many joules per inference does the battery allow, after leaving half of it for sleep current and the microphone?

## Card 2 — 30 fps drone vision

A quadcopter detects people and obstacles from a forward camera.

| Requirement | Value |
|---|---|
| Model | an object detector at 320×320, about **1 G MACs per frame** |
| Rate | **30 frames per second** |
| Latency | under 50 ms per frame |
| Power envelope | the drone's motors draw ~100 W; the compute module may draw a few watts; weight and heat matter |
| Flexibility | the model is retrained every few months, same architecture |

## Card 3 — A sub-microsecond trigger

A detector produces a new event every **25 ns**. A small neural network must decide, for every event, whether to keep it.

| Requirement | Value |
|---|---|
| Model | a 4-layer MLP, 16 → 64 → 32 → 32 → 5, about **5 k MACs** |
| Rate | one inference every 25 ns (40 M per second), every event |
| Latency | the decision must be ready **within 200 ns**, always |
| Power envelope | a rack of electronics: watts are fine |
| Flexibility | the model changes a few times a year |

## Card 4 — Battery doorbell camera

A doorbell wakes on motion, then checks for a person before notifying the owner's phone.

| Requirement | Value |
|---|---|
| Model | a person detector at 192×192, about **100 M MACs per frame** |
| Rate | **5 fps for 10 s** per motion event; about **20 events a day** |
| Latency | notify within 1 s |
| Power envelope | a 5,000 mAh Li-ion pack, **6 months** between charges; asleep almost all the time |
| Cost / volume | consumer product, $10–20 of electronics for compute |

## Card 5 — Laptop video-call effect

Background blur and noise suppression during video calls, on a thin laptop.

| Requirement | Value |
|---|---|
| Model | segmentation (~2 G MACs per frame at 30 fps) + audio denoiser (~20 M MACs every 10 ms) |
| Rate | sustained for **hours** of calls |
| Latency | under one frame (33 ms) |
| Power envelope | the laptop's battery life must not drop noticeably; a few watts at most for this feature |
| Flexibility | the OS vendor ships model updates |

## Card 6 — A research robot

A university lab's mobile robot runs perception and a small vision-language model for task planning.

| Requirement | Value |
|---|---|
| Model | changes **every week**; new operators appear (attention variants, custom layers) |
| Rate | 10–30 Hz perception, planning a few times per second |
| Power envelope | a 24 V battery pack; **20–30 W** for compute is acceptable |
| Team | graduate students who write PyTorch, not hardware |

## Card 7 — Vibration-anomaly sensor node

A sensor bolted to a factory motor flags unusual vibration.

| Requirement | Value |
|---|---|
| Pipeline | 1 kHz accelerometer → 256-point FFT once per second → a small autoencoder, about **50 k MACs** |
| Rate | **once per second** |
| Power envelope | energy-harvesting supply: a few hundred microwatts on average |
| Cost / volume | cheapest possible MCU, thousands of units |

## Card 8 — Hearing aid

A hearing aid that suppresses background noise in speech.

| Requirement | Value |
|---|---|
| Model | a recurrent/convolutional denoiser, about **2 M MACs per 8 ms audio frame** (250 M MACs/s) |
| Latency | **under 10 ms** end to end, every frame: a late frame is an audible glitch |
| Power envelope | a tiny zinc-air or Li-ion cell, a full day of use: **about 1 mW** for all signal processing |
| Also needed | filter banks, feedback cancellation, compression: classic DSP work |

---

## Worksheet (one per card)

| Step | Your answer |
|---|---|
| Ruled out, and by what | |
| Back-of-envelope number | |
| Choice (class + representative part) | |
| Three axes that justify it | |
| What you give up; the runner-up and why it lost | |
| First thing you'd measure | |
