#!/usr/bin/env python3
"""
mapping_recipe.py -- the four-step mapping recipe for a weight-stationary R x C array, and its result.

    python3 mapping_recipe.py 256 32 144            # conv2
    python3 mapping_recipe.py 64 256 2304           # c6 of the CIFAR-10 CNN in what_if.py
    python3 mapping_recipe.py 64 256 2304 --transpose
    python3 mapping_recipe.py 64 256 2304 --rows 32 --cols 32

No simulator needed: this is the arithmetic that SCALE-Sim's compute-cycle
count follows (featured_layer.py checks it for conv2).
"""
import argparse
import math

ap = argparse.ArgumentParser()
ap.add_argument("M", type=int)
ap.add_argument("N", type=int)
ap.add_argument("K", type=int)
ap.add_argument("--rows", type=int, default=16)
ap.add_argument("--cols", type=int, default=16)
ap.add_argument("--transpose", action="store_true", help="swap the roles of M and N")
a = ap.parse_args()
M, N, K, R, C = a.M, a.N, a.K, a.rows, a.cols
if a.transpose:
    M, N = N, M

print(f"1. The GEMM: M={M} (streamed), N={N} (columns), K={K} (rows); {M * N * K:,} MACs")
print("2. Held still: the weights (a K x N tile per pass)")
rf, cf = math.ceil(K / R), math.ceil(N / C)
print(f"3. Tiling: ceil({K}/{R}) x ceil({N}/{C}) = {rf} x {cf} = {rf * cf} folds (passes)")
print(f"4. Partial sums: in the PEs during a pass; across the {rf} K-folds they leave the array "
      f"and must be accumulated outside (SCALE-Sim writes them to the OFMAP buffer)")
per = M + 2 * R + C - 2
cyc = rf * cf * per
ideal = M * N * K / (R * C)
used = (K / (rf * R)) * (N / (cf * C))
print(f"Result: per pass {M} streaming + {2 * R + C - 2} fill/drain = {per} cycles; "
      f"{rf * cf} x {per} = {cyc:,} cycles")
print(f"   ideal {ideal:,.0f} cycles -> utilization {100 * ideal / cyc:.1f}% "
      f"(PEs mapped: {100 * used:.1f}%, fill/drain: {100 * (2 * R + C - 2) / per:.1f}% of each pass)")
