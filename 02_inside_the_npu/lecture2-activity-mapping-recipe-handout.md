# Reading and Explaining a Simulator's Report — In-Class Exercise
### Embedded AI — Accelerators 2: Inside the NPU

We'll work through this together in class, in pairs, about 20 minutes. Everything you need to recreate it on your own, or to go further, is below: the setup, the commands, and the formulas.

**The loop from the lecture:** run the simulator, read its report, explain the slow layers with the four questions, change something, run again.

**The four questions:** (1) What is the GEMM? (2) What does the hardware hold still? (3) How does it tile: how many passes, and how long is each? (4) Where do partial sums live? Then: why it's idle points to what to change.

---

## Setup (to recreate this on your own)

```bash
python3 -m venv ~/scalesim-env                  # tested on Python 3.12
source ~/scalesim-env/bin/activate
pip install scalesim==3.0.0 numpy==2.3.5        # SCALE-Sim 3.0.0 breaks on numpy 2.4+
```

If the install or a run fails, install Python 3.12 and create the venv with `python3.12 -m venv ~/scalesim-env` instead. Python 3.10 won't work: numpy 2.3.5 doesn't support it.

| Command | What it gives you | Time |
|---|---|---|
| `python3 what_if.py report` | SCALE-Sim's compute report for the CIFAR-10 CNN on 16×16, weight-stationary | ~1 min |
| `python3 what_if.py array` | per-layer cycles on 16×16 vs 32×32 | ~2 min |
| `python3 what_if.py dataflow` | per-layer cycles on 16×16, weight- vs output- vs input-stationary, best per layer | ~3 min |
| `python3 mapping_recipe.py M N K [--transpose] [--rows R --cols C]` | the four questions worked out for one GEMM on a weight-stationary array (formula only, instant) | — |

The model and the hardware are plain text files in `scalesim_demo/raw/`: `cifar_cnn_conv.csv` (one row per layer) and `conv2_ws.cfg` (array size, dataflow, buffers). The README in `scalesim_demo/` has the details.

## Reference card: an R × C array (16 × 16 unless stated)

| Dataflow | On the R rows | On the C columns | In time | Passes (folds) | Cycles per pass |
|---|---|---|---|---|---|
| Weight-stationary (WS) | K | N | M | ⌈K/R⌉ × ⌈N/C⌉ | M + 2R + C − 2 (**M + 46** on 16×16) |
| Output-stationary (OS) | M | N | K | ⌈M/R⌉ × ⌈N/C⌉ | K + R + C − 2 (**K + 30** on 16×16) |

Cycles = passes × cycles per pass. Ideal cycles = MACs ÷ (R × C). Utilization = ideal ÷ actual. Transposing a GEMM swaps M and N.

The formulas count cycles from 1; SCALE-Sim's report counts from 0, so it shows one cycle less.

## The report: the CIFAR-10 CNN on 16×16, weight-stationary

Lab 3's model: six 3×3 convolutions c1–c6 and a classifier fc. The GEMM shapes and the columns that matter from `what_if.py report`:

| Layer | Output | M × N × K | Total Cycles | Overall Util % | Mapping Efficiency % |
|---|---|---|---:|---:|---:|
| c1 | 32×32 | 1024 × 64 × 27 | 8,559 | 80.8 | 84.4 |
| c2 | 32×32 | 1024 × 64 × 576 | 154,079 | 95.7 | 100.0 |
| c3 | 16×16 | 256 × 128 × 576 | 86,975 | 84.8 | 100.0 |
| c4 | 16×16 | 256 × 128 × 1152 | 173,951 | 84.8 | 100.0 |
| c5 | 8×8 | 64 × 256 × 1152 | 126,719 | 58.2 | 100.0 |
| c6 | 8×8 | 64 × 256 × 2304 | 253,439 | 58.2 | 100.0 |
| fc | 1×1 | 1 × 10 × 256 | 751 | 1.3 | 62.5 |

---

## Problem 1 — Explain c5

c5 has every pass full (mapping efficiency 100%) but only 58% utilization. Use the four questions to explain why.

| Question | Your answer |
|---|---|
| 1. GEMM (M, N, K) and MACs | |
| 2. What's held still; what's on the rows, the columns, in time | |
| 3. Passes, and cycles per pass | |
| Cycles (check against the report) | |
| Ideal cycles, utilization | |
| What fraction of each pass is load and drain? | |

## Problem 2 — Explain c1

c1 is the opposite case: utilization 81%, but mapping efficiency only 84%.

a. Passes and cycles? (Check against the report.)

b. Why isn't every pass full? Where does the 84.4% come from?

c. Which kind of waste is this, compared with c5?

## Problem 3 — Change c5

a. **Lever 2, a different dataflow.** Predict c5's cycles **output-stationary** on 16×16. Faster or slower than weight-stationary? Why?

b. **Lever 3, reshape.** Predict c5's cycles **transposed** (swap M and N), still weight-stationary on 16×16. Compare with (a).

c. **Lever 1, more hardware.** Predict c5's cycles weight-stationary on a **32×32** array. What speedup is that for 4× the PEs, and why not 4×?

Check: `mapping_recipe.py 64 256 1152 [--transpose] [--rows 32 --cols 32]` for the weight-stationary cases; `what_if.py dataflow` and `what_if.py array` for SCALE-Sim's numbers.

## Problem 4 — Grouping a depthwise layer (on a 32×32 array)

A depthwise 3×3 layer filters each channel on its own: one GEMM per channel. Take a made-up layer: **64 channels on an 8×8 feature map**, on a **32×32** weight-stationary array (cycles per pass = M + 94).

a. One channel as a GEMM: M × N × K = ? How many GEMMs? Cycles for the layer? How many of the 1,024 PEs does each pass use?

b. **Group** g channels into one GEMM with a block-diagonal weight matrix (each channel's 9 weights on the diagonal, zeros elsewhere): M × g × 9g, run 64/g times. Fill in:

| g | GEMM (M × N × K) | passes per GEMM | cycles per GEMM | GEMMs | total cycles |
|---:|---|---:|---:|---:|---:|
| 1 | 64 × 1 × 9 | | | 64 | |
| 4 | | | | 16 | |
| 8 | | | | 8 | |

c. What does grouping cost that the cycle count doesn't show? And what starts to go wrong once g is larger than the array is wide?

## Problem 5 — The same layers on other NPUs (discussion)

a. **Ethos-U55-256** reaches 256 MACs/cycle on a convolution only if (TRM table 4-128, 8-bit activations) either the input depth is a multiple of 32 (depth-first), or the kernel area is a multiple of 4 and the input depth a multiple of 8 (kernel-first); otherwise "the hardware effectively rounds up" and the extra MACs are lost. Estimate the best efficiency for **c1** (3×3, 3 input channels).

b. **XDNA2** has 8 columns of 4 compute tiles, 64 KB of data memory per tile, and a 512 KB memory tile per column. c6 has 3 × 3 × 256 × 256 int8 weights. Can every tile hold all of them? Splitting K down a column? Splitting N across the columns? Propose a placement that fits.

---

## Going further

- **Your own model:** put your pruned Lab 3 channel counts into `raw/cifar_cnn_conv.csv` (the `Channels` and `Num Filter` columns) and rerun `what_if.py report`. Did pruning make the model fit the array better or worse?
- **Other arrays:** try `mapping_recipe.py` with `--rows 8 --cols 8` or `--rows 64 --cols 64` on c2 and c6. Where does each layer stop benefiting from a bigger array?
- **Change the config:** copy `raw/conv2_ws.cfg`, change one line (array size, dataflow), and run SCALE-Sim on the model yourself: `python3 -m scalesim.scale -c <your.cfg> -t raw/cifar_cnn_conv.csv -l raw/layout.csv -p out/`. The command-line tool also writes large per-cycle trace files.
