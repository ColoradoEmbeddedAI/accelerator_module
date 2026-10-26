#!/usr/bin/env python3
"""
feeding_tables.py -- every simulated number in Lecture 3, from SCALE-Sim.

    # in the lab venv, from this directory (several minutes the first time; cached after):
    python3 feeding_tables.py

The lecture's demo model is the CIFAR-10 CNN (cifar_cnn.csv, loaded by cifar.py) on the lab's
hardware; the lab itself is TinyCNN. Uses the lab's helper
(tools/accel.py). Results are cached in ./.accel_cache.
"""
import cifar

a = cifar.accel
SRAM = (32, 16, 16)
E = {"mac": 1, "sram": 6, "dram": 200}
KB = [8, 16, 24, 32, 40, 48]
KEYS = ("compute_cycles", "stall_cycles", "prefetch_cycles", "cycles",
        "dram_reads_ifmap", "dram_reads_filter", "dram_writes_ofmap")


def run(specs):
    """specs: [(layer, dataflow, sram, bw)] -> results, simulated in parallel."""
    res = a.run_jobs([a.layer_job(n, df, s, bw) for n, df, s, bw in specs])
    return [a._layer_result(n, df, r) for (n, df, _, _), r in zip(specs, res)]


print("== Slide 13: c1's GEMM and folds")
L = a.BY_NAME["c1"]
print(f"  M {L['M']}, N {L['N']}, K {L['K']}; K-folds x column-folds = {cifar.folds('c1')}")
for kb, r in zip((8, 16, 32), run([("c1", "ws", (kb, 16, 16), 8) for kb in (8, 16, 32)])):
    print(f"  IFMAP buffer {kb:2d} KB: input reads {r['dram_reads_ifmap']:,} (im2col {L['M'] * L['K']:,})")

print("\n== Slides 15, 17: c1, weight-stationary, 8 words/cycle, buffers 32/16/16 KB (slide 15 reads demand off this report)")
r = run([("c1", "ws", SRAM, 8)])[0]
for k in KEYS:
    print(f"  {k:18s} {r[k]:>8,d}")
for k, port in (("dram_reads_ifmap", "IFMAP"), ("dram_reads_filter", "filter"), ("dram_writes_ofmap", "OFMAP")):
    print(f"  {port:6s} port: {r[k] / r['compute_cycles']:5.1f} words per compute cycle; "
          f"{r[k] / 8:,.0f} cycles at 8 words/cycle")

print("\n== Slide 16: c1 cycles vs bandwidth")
bws = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
rs = run([("c1", df, SRAM, bw) for bw in bws for df in a.DATAFLOWS])
print("  bw  " + "".join(f"{df:>10s}" for df in a.DATAFLOWS))
for i, bw in enumerate(bws):
    print(f"  {bw:<3d} " + "".join(f"{rs[3 * i + j]['cycles']:>10,d}" for j in range(3)))
fl = run([("c1", df, SRAM, None) for df in a.DATAFLOWS])
print("  floor" + "".join(f"{r['compute_cycles']:>10,d}" for r in fl))

print("\n== Slides 18, 44: the CIFAR CNN, weight-stationary, 8 words/cycle")
print(f"  {'layer':6s}{'K-f x C-f':>10s}{'compute':>10s}{'stall':>10s}{'fill':>8s}{'in reads':>11s}{'psum wr':>11s}")
for n, r in zip(cifar.NAMES, run([(n, "ws", SRAM, 8) for n in cifar.NAMES])):
    kf, cf = cifar.folds(n)
    print(f"  {n:6s}{f'{kf} x {cf}':>10s}{r['compute_cycles']:>10,d}{r['stall_cycles']:>10,d}"
          f"{r['prefetch_cycles']:>8,d}{r['dram_reads_ifmap']:>11,d}{r['dram_writes_ofmap']:>11,d}")

print("\n== Slide 46: c3, output-stationary, 8 words/cycle: splitting 64 KB")
SPLITS = [(48, 8, 8), (32, 16, 16), (24, 32, 8), (8, 48, 8)]
for sp, r in zip(SPLITS, run([("c3", "os", sp, 8) for sp in SPLITS])):
    print(f"  {str(sp):13s} weight reads {r['dram_reads_filter']:>9,d}  input reads {r['dram_reads_ifmap']:>9,d}"
          f"  total {r['cycles']:>8,d}")

print("\n== Slides 21, 30, 43: c3, output-stationary, 8 words/cycle: grow one buffer (past the 64 KB budget)")
for which, sram in (("filter", lambda kb: (16, kb, 16)), ("input", lambda kb: (kb, 16, 16))):
    print(f"  grow the {which} buffer:")
    sizes = KB + ([64, 96, 128, 192, 256, 320] if which == "input" else [])
    for kb, r in zip(sizes, run([("c3", "os", sram(kb), 8) for kb in sizes])):
        print(f"    {kb:3d} KB: weight reads {r['dram_reads_filter']:>9,d}  input reads {r['dram_reads_ifmap']:>9,d}"
              f"  compute+stall {r['compute_cycles'] + r['stall_cycles']:>8,d}  fill {r['prefetch_cycles']:>6,d}"
              f"  total {r['cycles']:>8,d}")

print("\n== Slides 23, 27: c1 at 8 words/cycle, by dataflow (traffic, cycles, energy)")
for df, r in zip(a.DATAFLOWS, run([("c1", df, SRAM, 8) for df in a.DATAFLOWS])):
    sram = r["sram_reads_ifmap"] + r["sram_reads_filter"] + r["sram_writes_ofmap"]
    e = r["macs"] * E["mac"] + sram * E["sram"] + r["dram_traffic"] * E["dram"]
    print(f"  {df}: in {r['dram_reads_ifmap']:>7,d}  w {r['dram_reads_filter']:>6,d}  out {r['dram_writes_ofmap']:>7,d}"
          f"  cycles {r['cycles']:>7,d}  energy {e / 1e6:.1f} M (DRAM {100 * r['dram_traffic'] * E['dram'] / e:.0f}%)")

print("\n== Lab 4 overview, appendix slides A1-A3: c5 at 16 words/cycle (the worked example)")
for s in [(32, 16, 16), (16, 40, 8), (24, 32, 8)]:
    for df, r in zip(a.DATAFLOWS, run([("c5", df, s, 16) for df in a.DATAFLOWS])):
        print(f"  {str(s):12s} {df}: compute {r['compute_cycles']:>7,d}  stall {r['stall_cycles']:>6,d}"
              f"  total {r['cycles']:>7,d}  in {r['dram_reads_ifmap']:>9,d}  w {r['dram_reads_filter']:>9,d}")
