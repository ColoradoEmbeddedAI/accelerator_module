#!/usr/bin/env python3
"""
what_if.py -- what-if questions about a CIFAR-10 CNN on a systolic array, answered with SCALE-Sim.

    # in a venv with scalesim==3.0.0 and numpy==2.3.5, from this directory:
    python3 what_if.py report      # the baseline: SCALE-Sim's compute report (16x16, weight-stationary)
    python3 what_if.py array       # A: would a 32x32 array (4x the PEs) be 4x faster?
    python3 what_if.py dataflow    # B: same 16x16 array, best dataflow per layer

The model (raw/cifar_cnn_conv.csv) is a VGG-style CNN: six 3x3 convolutions (64, 64, 128,
128, 256, 256 channels at 32x32, 16x16 and 8x8) and a 256 -> 10 classifier. Every run starts
from the same config (raw/conv2_ws.cfg) and changes one or two lines. No DRAM limit, so the
cycles are pure compute. The model is large: each run takes 30-60 s.
"""
import contextlib
import csv
import functools
import io
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
TOPOLOGY = os.path.join(RAW, "cifar_cnn_conv.csv")
LAYERS = ["c1", "c2", "c3", "c4", "c5", "c6", "fc"]


@functools.lru_cache(maxsize=None)
def report(size=16, df="ws"):
    """SCALE-Sim's COMPUTE_REPORT for the model on a size x size array: one dict per layer."""
    from scalesim.scale_sim import scalesim
    d = tempfile.mkdtemp(prefix="what_if_")
    try:
        cfg = open(os.path.join(RAW, "conv2_ws.cfg")).read()
        for key in ("ArrayHeight", "ArrayWidth"):
            cfg = re.sub(rf"(?m)^{key}:.*$", f"{key}:    {size}", cfg)
        cfg = re.sub(r"(?m)^Dataflow\s*:.*$", f"Dataflow : {df}", cfg)
        cfg = re.sub(r"(?m)^run_name\s*=.*$", "run_name = run", cfg)
        with open(os.path.join(d, "c.cfg"), "w") as f:
            f.write(cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            s = scalesim(save_disk_space=True, verbose=False, config=os.path.join(d, "c.cfg"),
                         topology=TOPOLOGY, layout=os.path.join(RAW, "layout.csv"))
            s.run_scale(top_path=d)
        rows = list(csv.reader(open(os.path.join(d, "run", "COMPUTE_REPORT.csv"))))
        head = [h.strip() for h in rows[0]]
        return tuple({h: float(v) for h, v in zip(head, r) if h} for r in rows[1:])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def cycles(size=16, df="ws"):
    """Per-layer cycles ("Total Cycles" in the report: compute plus stalls; no stalls here)."""
    return {n: int(r["Total Cycles"]) for n, r in zip(LAYERS, report(size, df))}


def scenario_array():
    return cycles(16, "ws"), cycles(32, "ws"), None


def scenario_dataflow():
    runs = {df: cycles(16, df) for df in ("ws", "os", "is")}
    best = {n: min(runs, key=lambda df: runs[df][n]) for n in LAYERS}
    return runs["ws"], {n: runs[best[n]][n] for n in LAYERS}, best


def show(title, base, new, best):
    print(title + "\n")
    print(f"{'layer':6s} {'16x16 WS':>9s} {'what-if':>9s} {'speedup':>8s}" + ("  dataflow" if best else ""))
    for n in LAYERS:
        print(f"{n:6s} {base[n]:>9,d} {new[n]:>9,d} {base[n] / new[n]:>7.2f}x" + (f"  {best[n]}" if best else ""))
    tb, tn = sum(base.values()), sum(new.values())
    print(f"{'total':6s} {tb:>9,d} {tn:>9,d} {tb / tn:>7.2f}x")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "report":                    # SCALE-Sim's COMPUTE_REPORT.csv, percentages rounded
        rep = report(16, "ws")
        print(", ".join(rep[0]))
        for r in rep:
            print(", ".join(f"{v:.0f}" if "%" not in h else f"{v:.1f}" for h, v in r.items()))
    elif which == "array":
        show("A: 16x16 -> 32x32 (4x the PEs), weight-stationary", *scenario_array())
    elif which == "dataflow":
        show("B: 16x16, each layer picks its best dataflow", *scenario_dataflow())
    else:
        sys.exit(__doc__)
