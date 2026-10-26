#!/usr/bin/env python3
"""
featured_layer.py -- conv2 on a 16x16 systolic array, three dataflows: the formula vs SCALE-Sim.

    # in a venv with scalesim==3.0.0 and numpy==2.3.5, from this directory:
    python3 featured_layer.py

Prints, for each dataflow, the folds and cycles the formula predicts and the
compute cycles SCALE-Sim 3.0.0 reports. SCALE-Sim runs on the files in raw/
(the config's dataflow is swapped per run), with no DRAM limit, so its cycle
count is pure compute.
"""
import contextlib
import csv
import io
import math
import os
import re
import shutil
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

R = C = 16
M, N, K = 256, 32, 144          # conv2 as a GEMM
MACS = M * N * K

# Which GEMM dimension goes on the array's rows, which on its columns, which is streamed in time.
MAPPING = {"ws": ("K", "N", "M"), "os": ("M", "N", "K"), "is": ("K", "M", "N")}


def recipe(df):
    dims = {"M": M, "N": N, "K": K}
    rows, cols, time = (dims[d] for d in MAPPING[df])
    folds = math.ceil(rows / R) * math.ceil(cols / C)
    # Per fold, SCALE-Sim's weight/input-stationary pass: load the stationary tile (R cycles),
    # stream `time` rows, drain the skew (R + C - 2). Output-stationary: no preload (R + C - 2 skew
    # in and out).
    per_fold = time + (2 * R + C - 2 if df in ("ws", "is") else R + C - 2)
    return folds, per_fold, folds * per_fold


def simulate(df):
    """Compute cycles for conv2 in one dataflow, counted from 1 (SCALE-Sim's reports count from 0)."""
    from scalesim.scale_sim import scalesim
    d = tempfile.mkdtemp(prefix="featured_layer_")
    try:
        cfg = open(os.path.join(RAW, "conv2_ws.cfg")).read()
        cfg = re.sub(r"(?m)^Dataflow\s*:.*$", f"Dataflow : {df}", cfg)
        cfg = re.sub(r"(?m)^run_name\s*=.*$", "run_name = run", cfg)
        with open(os.path.join(d, "c.cfg"), "w") as f:
            f.write(cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            s = scalesim(save_disk_space=True, verbose=False, config=os.path.join(d, "c.cfg"),
                         topology=os.path.join(RAW, "conv2_conv.csv"), layout=os.path.join(RAW, "layout.csv"))
            s.run_scale(top_path=d)
        rows = list(csv.reader(open(os.path.join(d, "run", "COMPUTE_REPORT.csv"))))
        head = [h.strip() for h in rows[0]]
        total = int(float(rows[1][head.index("Total Cycles")]))
        stall = int(float(rows[1][head.index("Stall Cycles")]))
        return total - stall + 1
    finally:
        shutil.rmtree(d, ignore_errors=True)


print(f"conv2: M={M} N={N} K={K}, {MACS:,} MACs, ideal on {R}x{C}: {MACS // (R * C):,} cycles\n")
print(f"{'df':3s} {'rows,cols,time':>16s} {'folds':>6s} {'per fold':>9s} {'predicted':>10s} {'SCALE-Sim':>10s} {'util':>6s}")
for df in ("ws", "os", "is"):
    folds, per_fold, pred = recipe(df)
    cyc = simulate(df)
    print(f"{df:3s} {'/'.join(MAPPING[df]):>16s} {folds:>6d} {per_fold:>9d} {pred:>10,d} {cyc:>10,d} "
          f"{100 * MACS / (R * C * cyc):>5.1f}%")
