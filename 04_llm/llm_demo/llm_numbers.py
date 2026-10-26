#!/usr/bin/env python3
"""
llm_numbers.py -- every derived number in the LLM decks (4a and 4b), from published configs and benchmarks.

    python3 llm_numbers.py            # all sections
    python3 llm_numbers.py count kv   # just some: count, vocab, lora, ops, kv, flm, prefill,
                                      #   xdna, ladder, why, batch

Standard library only. The lecture's diagrams script imports the data and the
fit from here, so a figure and its slide can't disagree.

Sources (Known = published; Inferred = derived here; Assumed = not published):
  Llama 3.2 1B config: Hugging Face meta-llama/Llama-3.2-1B config.json (Known).
  stories260K: Karpathy, llama2.c model card (Known).
  FastFlowLM: Llama 3 benchmarks, FLM v0.9.30, Ryzen AI 7 350, 32 GB, Performance mode,
      https://fastflowlm.com/docs/benchmarks/llama3_results/ (Known, fetched 2026-10-23).
  Q4NX: int4 in groups of 32 + bf16 scale + bf16 minimum = 5 bits/weight (arXiv 2602.06063 v2 §3.1.1; Known).
  XDNA2: Linux amdxdna docs; AMD AM027 (AIE-ML v2); arXiv 2602.06063 v2 §2.1, §3.2.3, §5; destevez.net,
      "Getting peak TOPS on a Ryzen AI 7 350 NPU" (2026-05); xdna-engine and ryzen-ai-arch on GitHub (dispatch cost).
  Training-set sizes: GPT-3, Llama 2, Llama 3 papers/blog; Llama 4 blog (2025); Qwen3 report (2025) (Known).
"""
import sys

# ---- model configs (Known) -------------------------------------------------
LLAMA_1B = dict(name="Llama 3.2 1B", layers=16, dim=2048, hidden=8192, heads=32, kv_heads=8, head_dim=64,
                vocab=128256, tied=True)
STORIES_260K = dict(name="stories260K", layers=5, dim=64, hidden=172, heads=8, kv_heads=4, head_dim=8,
                    vocab=512, tied=True, max_ctx=512)
F4 = dict(flash_kb=1024, sram_kb=128, ccm_kb=64)

# FastFlowLM decode tok/s by context (Known); prefill tok/s by prompt length (Known)
FLM_CTX_K = [1, 2, 4, 8, 16, 32, 64, 128]
FLM_DECODE = {"Llama 3.2 1B": [64.5, 62.2, 58.9, 53.9, 45.5, 35.0, 24.1, 13.6],
              "Llama 3.2 3B": [26.3, 25.5, 24.1, 21.7, 18.0, 13.6, 9.0, None],
              "Llama 3.1 8B": [12.8, 12.6, 12.2, 11.5, 10.2, 8.5, None, None]}
FLM_PREFILL_CTX_K = [1, 2, 4, 8, 16, 32]
FLM_PREFILL_1B = [1686, 2136, 2339, 2212, 1706, 1157]
# parameters read per decode token (Inferred): 1B/3B tie the embedding to the output layer, which is read
# in full; the 8B model's separate 128,256 x 4,096 input table is excluded (decode reads one row of it).
FLM_PARAMS_READ = {"Llama 3.2 1B": 1.24e9, "Llama 3.2 3B": 3.21e9, "Llama 3.1 8B": 7.5e9}
Q4NX_BYTES_PER_PARAM = 5 / 8

# XDNA2 (Ryzen AI 300) for the design ladder (4b)
#   Known: 8 columns x 4 compute tiles, 64 KB each; 8 memory tiles, 512 KB each (Linux amdxdna docs;
#     arXiv 2602.06063 v2 §2.1); 1.8 GHz (§2.1.1; measured 1.808 GHz by destevez.net, 2026-05); the LLM
#     kernels compute in bf16 (§3.1.1); decode GEMVs use 4 columns (§3.2.3).
#   Derived: 256 bf16 multipliers per tile (AM027, AIE-ML v2); one column's shim tile has 2 MM2S channels of
#     64-bit streams (AM027), so 2 x 8 B x 1.8 GHz into a column.
#   Assumed: the DDR bandwidth the NPU gets, 40-50 GB/s (the paper's "capped below 40 GB/s" (§5) vs >= 50
#     implied by FastFlowLM's 1B rate); one NPU dispatch costs 80 us (Inferred: xrt-smi's latency test
#     reports 72 us; xdna-engine cut 196 dispatches per token and saved 19.8 ms, ~100 us each).
XDNA2 = dict(cols=8, rows=4, clk=1.8e9, bf16_macs=256, l1_kb=64, l2_kb=512, mm2s_per_col=2, stream_bytes=8)
GEMV_COLS = 4
NPU_BW = (40e9, 50e9)
KERNEL_BW = 60e9            # Known lower bound: the decode kernels' read requirement "exceeds 60 GB/s at
                            #   1.8 GHz" (arXiv 2602.06063 v2 §5), i.e. compute keeps up with more than DDR gives
DISPATCH_S = 80e-6

# training-set sizes, tokens (Known; papers cited in the 4a deck)
# Llama 4: "more than 30 trillion" (Meta blog, 2025-04-05); Qwen3: 36T, every size 0.6B-235B (arXiv 2505.09388)
TRAINING_TOKENS = [("GPT-3", 2020, 499e9), ("Llama 2", 2023, 2.0e12), ("Llama 3", 2024, 15e12),
                   ("Llama 4", 2025, 30e12), ("Qwen3", 2025, 36e12)]


# ---- derived ---------------------------------------------------------------
def params(c):
    d, h, kvd = c["dim"], c["hidden"], c["kv_heads"] * c["head_dim"]
    attn = d * d * 2 + d * kvd * 2                 # Q and O are d x d; K and V are d x (kv_heads * head_dim)
    mlp = 3 * d * h                                # gate, up, down
    norms = 2 * d
    emb = c["vocab"] * d
    total = c["layers"] * (attn + mlp + norms) + d + emb * (1 if c["tied"] else 2)
    return {"per_layer": attn + mlp + norms, "attn": attn, "mlp": mlp, "embedding": emb, "total": total}


def kv_bytes_per_token(c, bytes_per=2):
    return c["layers"] * c["kv_heads"] * c["head_dim"] * 2 * bytes_per          # K and V


def fit(xs, ys):
    """Least-squares line y = a + b x."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def flm_fit():
    """1/(tok/s) = a + b * context for FastFlowLM's 1B decode. Returns (a s, b s/token, crossover tokens)."""
    xs = [k * 1024 for k in FLM_CTX_K]
    ys = [1 / t for t in FLM_DECODE["Llama 3.2 1B"]]
    a, b = fit(xs, ys)
    return a, b, a / b


def col_bw():
    x = XDNA2
    return x["mm2s_per_col"] * x["stream_bytes"] * x["clk"]


def block_ops(ctx=1024):
    """One decoder block in decode, in order: (name, DDR bytes, kind). kind is 'w' (a matrix multiply that
    streams 5-bit weights), 'kv' (a matrix multiply against this block's KV cache: Q·Kᵀ reads K, × V reads V)
    or 'op' (everything else: no DDR traffic to speak of, and on the CPU in v0-v2)."""
    c = LLAMA_1B
    d, h, kvd = c["dim"], c["hidden"], c["kv_heads"] * c["head_dim"]
    q = Q4NX_BYTES_PER_PARAM
    kv = ctx * kv_bytes_per_token(c) / c["layers"] / 2                   # K or V, one block
    return [("RMSNorm", 0, "op"), ("Q proj", d * d * q, "w"), ("K proj", d * kvd * q, "w"),
            ("V proj", d * kvd * q, "w"), ("RoPE", 0, "op"), ("Q·Kᵀ", kv, "kv"), ("softmax", 0, "op"),
            ("× V", kv, "kv"), ("O proj", d * d * q, "w"), ("add, RMSNorm", 0, "op"), ("gate proj", d * h * q, "w"),
            ("up proj", d * h * q, "w"), ("SiLU, multiply", 0, "op"), ("down proj", h * d * q, "w"), ("add", 0, "op")]


def npu_calls(version):
    """NPU calls per block, as lists of indices into block_ops(). v0/v1: one call per matrix multiply;
    v2: neighboring matrix multiplies merged (the CPU ops are the boundaries); v3: one program."""
    ops = block_ops()
    mm = [i for i, (_, _, k) in enumerate(ops) if k != "op"]
    if version in ("v0", "v1"):
        return [[i] for i in mm]
    if version == "v2":
        groups, cur = [], []
        for i, (_, _, k) in enumerate(ops):
            if k == "op":
                if cur:
                    groups.append(cur)
                cur = []
            else:
                cur.append(i)
        return groups + ([cur] if cur else [])
    return [list(range(len(ops)))]


def ladder():
    """The decode design ladder for Llama 3.2 1B on XDNA2: (key, label, weight bytes, calls per token, context,
    KV bytes per token) per version: the calls per block from npu_calls() x the layers, plus one for the
    output layer. "32k" is v3 at 32k tokens of context (a stress test, not a design step); "32k8" adds an
    8-bit KV cache."""
    c = LLAMA_1B
    p = params(c)["total"]
    L = c["layers"]
    bf16, q4 = p * 2, p * Q4NX_BYTES_PER_PARAM
    kv = kv_bytes_per_token(c, 2)
    n = {v: len(npu_calls(v)) * L + 1 for v in ("v0", "v2", "v3")}
    return [("v0", "bf16, one call per matrix multiply", bf16, n["v0"], 1024, kv),
            ("v1", "5-bit weights", q4, n["v0"], 1024, kv),
            ("v2", "neighboring calls merged", q4, n["v2"], 1024, kv),
            ("v3", "one program per block", q4, n["v3"], 1024, kv),
            ("32k", "v3 at 32k context", q4, n["v3"], 32768, kv),
            ("32k8", "v3 at 32k, 8-bit KV cache", q4, n["v3"], 32768, kv // 2)]


def step_time(wbytes, disp, ctx, kvb, bw, batch=1):
    """Seconds per decode step and its parts: (total, weights, KV, dispatch, compute). One step serves
    `batch` tokens: the weights are read once, each token reads its own KV cache."""
    macs = params(LLAMA_1B)["total"] * batch
    tw, tk, td = wbytes / bw, batch * ctx * kvb / bw, disp * DISPATCH_S
    tc = macs / (XDNA2["bf16_macs"] * XDNA2["clk"] * GEMV_COLS * XDNA2["rows"])
    return tw + tk + td + tc, tw, tk, td, tc


def weights_per_s(bw, bytes_per_weight):
    """Weights per second that a bw bytes/s stream delivers."""
    return bw / bytes_per_weight


def spec_tokens(alpha, k):
    """Expected tokens kept per verification pass with k drafts, each right with probability alpha."""
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def gb(x):
    return x / 1e9


# ---- printing --------------------------------------------------------------
def sec_count():
    print("== Count it (4a): bytes and FLOPs per token")
    for n, p in [("7B", 7e9), ("Llama 3.2 1B", params(LLAMA_1B)["total"]), ("TinyLlama 1.1B", 1.1e9),
                 ("stories260K", params(STORIES_260K)["total"])]:
        print(f"  {n:15s} {p:>14,.0f} params: fp32 {gb(p * 4):8.3f} GB, fp16 {gb(p * 2):7.3f} GB, "
              f"int8 {gb(p):7.3f} GB, 4-bit {gb(p / 2):7.3f} GB; ~{2 * p / 1e9:,.1f} GFLOP/token")
    p7 = 7e9
    print(f"  7B at 4-bit vs the F4's 1 MB of Flash: {p7 / 2 / (F4['flash_kb'] * 1024):,.0f}x too big")
    # F4 compute: ~2 MAC/cycle peak with SMLAD at 168 MHz (Lecture 1's roofline); ~5 cycles/MAC measured
    for label, macs_per_s in [("peak, 2 MAC/cycle", 2 * 168e6), ("measured CMSIS-NN, ~5 cycles/MAC", 168e6 / 5)]:
        print(f"  7B on the F4 ({label}): {p7 / macs_per_s:,.0f} s per token if it fit")


def sec_vocab():
    print("== Vocabulary -> embedding table and output layer (4a)")
    for c, gpt in [(LLAMA_1B, False), (None, True)]:
        if gpt:
            v, d, total = 50257, 768, 124e6
            name = "GPT-2 small"
        else:
            v, d, total, name = c["vocab"], c["dim"], params(c)["total"], c["name"]
        print(f"  {name}: vocab {v:,} x width {d:,} = {v * d:,} params = {100 * v * d / total:.0f}% of "
              f"{total:,.0f} (tied: one table is both)")


def sec_lora():
    print("== LoRA (4a): trainable params for one d x d projection")
    for d, r in [(2048, 8), (4096, 8), (4096, 16)]:
        print(f"  d={d}, rank {r}: full {d * d:,}; LoRA 2*d*r = {2 * d * r:,} ({100 * 2 * d * r / d / d:.2f}%)")


def sec_ops():
    print("== One Llama 3.2 1B decoder block as GEMMs (4b): K x N per token")
    c = LLAMA_1B
    d, h, kvd = c["dim"], c["hidden"], c["kv_heads"] * c["head_dim"]
    gemms = [("Q", d, d), ("K", d, kvd), ("V", d, kvd), ("O", d, d), ("gate", d, h), ("up", d, h), ("down", h, d)]
    tot = sum(k * n for _, k, n in gemms)
    for n, k, nn in gemms:
        print(f"  {n:5s} {k:>5,} x {nn:>5,} = {k * nn:>11,} weights ({100 * k * nn / tot:4.1f}% of the block)")
    p = params(c)
    print(f"  block: {p['per_layer']:,} params; x{c['layers']} layers = {c['layers'] * p['per_layer']:,}; "
          f"embedding/output {p['embedding']:,}; total {p['total']:,}")
    print(f"  output layer per token: {d:,} x {c['vocab']:,} = {p['embedding']:,} MACs "
          f"({100 * p['embedding'] / p['total']:.0f}% of all weights read per token)")


def sec_kv():
    print("== KV cache (4b)")
    c = LLAMA_1B
    kv = kv_bytes_per_token(c, 2)
    no_gqa = c["layers"] * c["heads"] * c["head_dim"] * 2 * 2
    w = params(c)["total"] * Q4NX_BYTES_PER_PARAM
    print(f"  {c['name']}: bf16 KV = {kv:,} B/token ({kv // 1024} KB); without GQA (32 KV heads) {no_gqa // 1024} KB")
    print(f"     weights at 5-bit {gb(w):.2f} GB; KV equals that at {w / kv:,.0f} tokens; "
          f"KV at 128k = {gb(kv * 131072):.1f} GB")
    s = STORIES_260K
    p = params(s)["total"]
    kv4 = kv_bytes_per_token(s, 4)
    print(f"  {s['name']}: {p:,} params = {p * 4 / 1024:,.0f} KB fp32; fp32 KV = {kv4:,} B/token; "
          f"at {s['max_ctx']} tokens {kv4 * s['max_ctx'] // 1024} KB vs F4 SRAM {F4['sram_kb']} KB "
          f"+ CCM {F4['ccm_kb']} KB")


def sec_flm():
    print("== FastFlowLM decode (4b): (a) memory-bound, (b) the fit")
    for m, p in FLM_PARAMS_READ.items():
        t = FLM_DECODE[m][0]
        by = p * Q4NX_BYTES_PER_PARAM
        print(f"  (a) {m}: {gb(p):.2f}B params read -> {gb(by):.2f} GB at 5-bit; {t} tok/s -> {gb(by * t):.0f} GB/s")
    a, b, x = flm_fit()
    print(f"  (b) 1/(tok/s) = {a * 1e3:.2f} ms + {b * 1e9:.0f} ns x context; weights = KV cost at {x:,.0f} tokens")
    worst = 0
    for k, t in zip(FLM_CTX_K, FLM_DECODE["Llama 3.2 1B"]):
        pred = 1 / (a + b * k * 1024)
        worst = max(worst, abs(pred / t - 1))
        print(f"      {k:>3}k: measured {t:5.1f}, fit {pred:5.1f} ({100 * (pred / t - 1):+.1f}%)")
    print(f"      worst miss {100 * worst:.1f}%")
    print("  Optional: test the KV-precision assumption")
    for label, kvb in [("bf16 KV (32 KB/token)", 32768), ("8-bit KV (16 KB/token)", 16384)]:
        bw = kvb / b
        print(f"      {label}: bandwidth {gb(bw):.0f} GB/s, weight bytes/token {gb(a * bw):.2f} GB")
    print(f"      all-Q4NX weights need >= {gb(1.24e9 * Q4NX_BYTES_PER_PARAM):.2f} GB/token; "
          f"bf16 output layer: {gb((1.24e9 - params(LLAMA_1B)['embedding']) * Q4NX_BYTES_PER_PARAM + params(LLAMA_1B)['embedding'] * 2):.2f} GB")


def sec_prefill():
    print("== FastFlowLM prefill, 1B (4b optional)")
    for k, t in zip(FLM_PREFILL_CTX_K, FLM_PREFILL_1B):
        print(f"  {k:>2}k prompt: {t:,} tok/s ({t / FLM_DECODE['Llama 3.2 1B'][0]:.0f}x decode at 1k)")


def sec_xdna():
    print("== XDNA2 as the design target (4b)")
    x, c = XDNA2, LLAMA_1B
    tiles = x["cols"] * x["rows"]
    onchip = tiles * x["l1_kb"] + x["cols"] * x["l2_kb"]
    p = params(c)
    print(f"  {tiles} tiles x {x['bf16_macs']} bf16 MACs x {x['clk'] / 1e9:g} GHz = "
          f"{tiles * x['bf16_macs'] * x['clk'] / 1e12:.1f} T MAC/s bf16")
    print(f"  on chip: {onchip / 1024:.0f} MB; weights {gb(p['total'] * 2):.2f} GB bf16, "
          f"{gb(p['total'] * Q4NX_BYTES_PER_PARAM):.2f} GB 5-bit; one bf16 up projection "
          f"{c['dim'] * c['hidden'] * 2 / 2**20:.0f} MB; KV tokens that fit in all of it: "
          f"{onchip * 1024 // kv_bytes_per_token(c):,}")
    print(f"  one column's DMA: {gb(col_bw()):.1f} GB/s; {GEMV_COLS} columns: {gb(GEMV_COLS * col_bw()):.0f} GB/s; "
          f"DDR to the NPU (Assumed): {gb(NPU_BW[0]):.0f}-{gb(NPU_BW[1]):.0f} GB/s")


def sec_why():
    print("== The 'why' slides (4b)")
    for bw in NPU_BW:
        print(f"  {gb(bw):.0f} GB/s delivers {weights_per_s(bw, 2) / 1e9:.0f} G bf16 weights/s or "
              f"{weights_per_s(bw, Q4NX_BYTES_PER_PARAM) / 1e9:.0f} G 5-bit weights/s")
    print(f"  the kernels keep up with > {gb(KERNEL_BW):.0f} GB/s of 5-bit weights = "
          f"{weights_per_s(KERNEL_BW, Q4NX_BYTES_PER_PARAM) / 1e9:.0f} G weights/s: "
          f"{KERNEL_BW / NPU_BW[1]:.1f}-{KERNEL_BW / NPU_BW[0]:.1f}x headroom over DDR")
    bw = sum(NPU_BW) / 2
    ops = block_ops()
    stream = sum(b for _, b, _ in ops) / bw
    print(f"  one block at {gb(bw):.0f} GB/s: DDR streams {stream * 1e3:.2f} ms, plus calls:")
    for v in ("v1", "v2", "v3"):
        n = len(npu_calls(v))
        print(f"      {v}: {n:2d} calls = {n * DISPATCH_S * 1e3:.2f} ms -> {(stream + n * DISPATCH_S) * 1e3:.2f} ms per block")
    for n, b, k in ops:
        if b:
            print(f"      {n:10s} {b / 1e6:6.2f} MB  {b / bw * 1e6:6.1f} us")


def sec_ladder():
    print(f"== The decode ladder (4b): Llama 3.2 1B on XDNA2, {DISPATCH_S * 1e6:.0f} us per dispatch")
    for key, label, w, d, ctx, kvb in ladder():
        lo, hi = (step_time(w, d, ctx, kvb, bw) for bw in NPU_BW)
        parts = ", ".join(f"{n} {a * 1e3:.1f}-{b * 1e3:.1f}" for n, a, b in
                          zip(("weights", "KV", "dispatch", "compute"), lo[1:], hi[1:]))
        print(f"  {key:4s} {label:36s} {d:3d} calls, {ctx // 1024:2d}k: {parts} ms -> "
              f"{1 / lo[0]:.0f}-{1 / hi[0]:.0f} tok/s")
    print(f"  FastFlowLM measured (Known): {FLM_DECODE['Llama 3.2 1B'][0]} at 1k, {FLM_DECODE['Llama 3.2 1B'][5]} at 32k")
    print("  Calibrate on the 1B (weights only, 1k), predict the others:")
    bw = FLM_PARAMS_READ["Llama 3.2 1B"] * Q4NX_BYTES_PER_PARAM * FLM_DECODE["Llama 3.2 1B"][0]
    print(f"      bandwidth = 64.5 tok/s x {gb(FLM_PARAMS_READ['Llama 3.2 1B'] * Q4NX_BYTES_PER_PARAM):.2f} GB "
          f"= {gb(bw):.0f} GB/s")
    for m, p in FLM_PARAMS_READ.items():
        pred, meas = bw / (p * Q4NX_BYTES_PER_PARAM), FLM_DECODE[m][0]
        print(f"      {m}: predicted {pred:.1f}, measured {meas} ({100 * (pred / meas - 1):+.0f}%)")


def sec_batch():
    print("== Make M > 1 (4b): batching and speculative decoding, v3 at 50 GB/s")
    _, _, w, d, _, kvb = ladder()[3]
    for ctx in (1024, 32768):
        for b in (1, 8):
            t = step_time(w, d, ctx, kvb, NPU_BW[1], batch=b)
            print(f"  {ctx // 1024:2d}k context, {b} requests: weights {t[1] * 1e3:.1f}, KV {t[2] * 1e3:.1f}, "
                  f"compute {t[4] * 1e3:.2f} ms -> {b / t[0]:.0f} tok/s total, {1 / t[0]:.0f} per request")
    for k in (4, 16, 64):
        t = step_time(w, d, 1024, kvb, NPU_BW[1], batch=k)
        print(f"  verify {k:2d} tokens in one pass: compute {t[4] * 1e3:.1f} ms vs weights {t[1] * 1e3:.1f} ms")
    for a, k in ((0.6, 4), (0.8, 4), (0.8, 8)):
        print(f"  speculative, {k} drafts right with p={a}: {spec_tokens(a, k):.1f} tokens per pass")


SECTIONS = {"count": sec_count, "vocab": sec_vocab, "lora": sec_lora, "ops": sec_ops, "kv": sec_kv,
            "flm": sec_flm, "prefill": sec_prefill, "xdna": sec_xdna, "ladder": sec_ladder, "why": sec_why,
            "batch": sec_batch}

if __name__ == "__main__":
    for s in sys.argv[1:] or SECTIONS:
        SECTIONS[s]()
        print()
