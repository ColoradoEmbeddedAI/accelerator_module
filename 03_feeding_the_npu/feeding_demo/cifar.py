"""
cifar.py -- Lecture 3's demo model, the CIFAR-10 CNN from Lab 3 and Lecture 2, run through the lab's helper.

The lab is TinyCNN; the lecture demonstrates on a different model so that its numbers don't answer
the lab's questions (Lecture 2 does the same). The helper takes any model in TinyCNN's CSV format,
so every number here has the same three memory-model fixes as the lab's.

    # from anywhere, in the lab venv: the same report as the lab's `accel.py layer`
    python3 path/to/feeding_demo/cifar.py layer c1 --df ws --bw 8 --sram 32,16,16

    # or from Python, with this directory on the path
    import cifar
    cifar.accel.evaluate_layer("c1", "ws", (32, 16, 16), 8)
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tools"))
os.environ.setdefault("ACCEL_CACHE", os.path.join(HERE, ".accel_cache"))
import accel  # noqa: E402

MODEL = accel.load_model(os.path.join(HERE, "cifar_cnn.csv"))
NAMES = [L["name"] for L in MODEL]
accel.BY_NAME.update({L["name"]: L for L in MODEL})   # names don't collide with TinyCNN's


def folds(name):
    """Weight-stationary folds on the 16 x 16 array: (K-folds, column-folds)."""
    L = accel.BY_NAME[name]
    return math.ceil(L["K"] / accel.ARRAY_ROWS), math.ceil(L["N"] / accel.ARRAY_COLS)


def main():
    ap = argparse.ArgumentParser(description="One CIFAR CNN layer on the lab's hardware (same report as accel.py layer).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("layer")
    pl.add_argument("name", choices=NAMES)
    pl.add_argument("--df", default="ws", choices=accel.DATAFLOWS)
    pl.add_argument("--bw", type=int, default=None, help="DRAM words/cycle (default: unlimited)")
    pl.add_argument("--sram", default="32,16,16", help="ifmap,filter,ofmap KB (default 32,16,16)")
    a = ap.parse_args()
    r = accel.evaluate_layer(a.name, a.df, tuple(int(x) for x in a.sram.split(",")), a.bw)
    for k, v in r.items():
        print(f"  {k:18s} {v:,.1f}" if isinstance(v, float) else f"  {k:18s} {v:,}" if isinstance(v, int)
              else f"  {k:18s} {v}")


if __name__ == "__main__":
    main()
