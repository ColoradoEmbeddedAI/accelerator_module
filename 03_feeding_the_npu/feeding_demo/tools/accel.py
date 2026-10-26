#!/usr/bin/env python3
"""
accel.py -- run SCALE-Sim on TinyCNN and get the numbers back as Python dicts.

This is the lab's only interface to SCALE-Sim. You never write a SCALE-Sim
config file or parse a report CSV: build a Part 1 config or a Part 2
schedule (plain dicts, or the JSON files in part1/ and part2/), and call
the functions below.

    import accel
    r = accel.evaluate_part1(accel.load_json("part1/config.json"))
    print(r["total"]["cycles"], r["meets_deadline"])
    for row in r["layers"]:
        print(row["name"], row["cycles"], row["stall_cycles"], row["dram_reads_filter"])

    s = accel.evaluate_part2({"conv7": {"transpose": True}})
    print(s["total_cycles"])

Command line (run from the lab root):

    python3 tools/accel.py model                         # TinyCNN, its GEMM shapes, MACs
    python3 tools/accel.py part1 part1/config.json       # per-layer table + deadline check
    python3 tools/accel.py part2 part2/schedule.json     # per-layer table, total cycles
    python3 tools/accel.py layer conv2 --df os --bw 8 --sram 32,16,16   # one layer, Part 1 hardware
    python3 tools/accel.py gemm 16 128 576               # one GEMM on the Part 2 array

Every result is cached in .accel_cache/ (SCALE-Sim is deterministic), so
re-running a configuration you have already simulated is instant. Runs use
all CPU cores; set ACCEL_JOBS=N to use fewer.

The SCALE-Sim model, and the changes this file makes to it
----------------------------------------------------------
SCALE-Sim 3.0.0 (pip install scalesim==3.0.0) models one systolic array of
multiply-accumulate units (PEs), fed from three double-buffered SRAMs
(IFMAP, filter, OFMAP), which are filled from and drained to DRAM. It
simulates one layer at a time, as a GEMM. Read README.md section 2 for what
it does and doesn't model. This file fixes three problems in its memory
model and sets one configuration detail. None of them changes compute cycles;
they change how much DRAM traffic and how many stall cycles SCALE-Sim
reports, so that those behave the way the lectures describe:

  1. Weight-stationary weight order. SCALE-Sim prefetches each tile of
     weights top row first but the array consumes it bottom row first, so its
     streaming buffer misses and re-reads the weight matrix many times over.
     The fix prefetches each tile in the order the array consumes it: each
     weight is read from DRAM once, as on real weight-stationary hardware.
  2. Read-buffer misses. SCALE-Sim's IFMAP/filter buffer holds a sliding
     window of the operand's access stream. When the array asks for data the
     window has already passed, 3.0.0 streams through the whole rest of the
     operand and wraps around to reach it (so a slightly different buffer size
     can cost several times more traffic), double-counts stall cycles, and
     charges a full half-buffer of DRAM traffic even when less data is left.
     It also counts the empty (padding) slots of its fetch matrix as DRAM
     words in the first buffer fill. The fix keeps the same model (a
     double-buffered window, a serial DRAM channel at your bandwidth) but on a
     miss re-fetches only what's needed, from the nearer of the two
     directions; counts each stall once; charges only real data; and lets a
     fetch start at most one half-buffer of transfer time before its data is
     needed (the lookahead double buffering actually provides).
  3. Bandwidth. In USER bandwidth mode, SCALE-Sim 3.0.0 reads the IFMAP and
     filter DRAM bandwidths from the [layout] section of the config and only
     the OFMAP bandwidth from "Bandwidth". This file sets all three to the
     bandwidth you choose (words per cycle, per operand).

The grader applies exactly the same changes (it uses a copy of this file).
"""

import argparse
import concurrent.futures as cf
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile

LAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_CSV = os.path.join(LAB_DIR, "model", "tinycnn.csv")
CACHE_DIR = os.environ.get("ACCEL_CACHE", os.path.join(LAB_DIR, ".accel_cache"))

# ---- The hardware ---------------------------------------------------------
ARRAY_ROWS = 16
ARRAY_COLS = 16
DATAFLOWS = ("ws", "os", "is")

# Part 1: what you may choose, and the deadline.
SRAM_BUDGET_KB = 64          # IFMAP + filter + OFMAP buffers, total
SRAM_STEP_KB = 8             # each buffer is a multiple of 8 KB ...
SRAM_MIN_KB = 8              # ... and at least 8 KB
BW_MIN, BW_MAX = 1, 64       # DRAM bandwidth, words (bytes) per cycle, per operand
DEADLINE_CYCLES = 60_000     # Part 1: the model must finish within this many cycles

# Part 2: fixed hardware, you restructure the work.
PART2_DATAFLOW = "ws"
PART2_MAX_LAYERS = 3         # restructure at most this many layers

PATCH_ID = "ws-filter-order+read-buffer-v3"   # part of every cache key


# ---- TinyCNN --------------------------------------------------------------
def load_model(path=MODEL_CSV):
    """TinyCNN as a list of layer dicts, with output sizes and GEMM shapes added.

    GEMM shape, for one output image (batch 1):
      M = output pixels (out_h * out_w), N = output channels (filters),
      K = window size (kernel * kernel * in_c), the length of each dot product.
    A depthwise layer is in_c independent GEMMs, one per channel, each with
    N = 1 and K = kernel * kernel ('per_channel': True).
    """
    layers = []
    with open(path) as f:
        for row in csv.DictReader(f):
            L = {k: (v if k in ("name", "type") else int(v)) for k, v in row.items()}
            k, s, p = L["kernel"], L["stride"], L["pad"]
            L["out_h"] = (L["in_h"] + 2 * p - k) // s + 1
            L["out_w"] = (L["in_w"] + 2 * p - k) // s + 1
            # SCALE-Sim has no padding field: give it the padded input rows/cols the
            # windows actually touch, so its output size matches.
            L["sim_h"] = (L["out_h"] - 1) * s + k
            L["sim_w"] = (L["out_w"] - 1) * s + k
            M = L["out_h"] * L["out_w"]
            if L["type"] == "dw":
                L.update(M=M, N=1, K=k * k, channels=L["in_c"], per_channel=True)
                L["weights"] = k * k * L["in_c"]
            else:
                L.update(M=M, N=L["out_c"], K=k * k * L["in_c"], channels=1, per_channel=False)
                L["weights"] = k * k * L["in_c"] * L["out_c"]
            L["macs"] = L["M"] * L["N"] * L["K"] * L["channels"]
            layers.append(L)
    return layers


MODEL = load_model()
LAYER_NAMES = [L["name"] for L in MODEL]
BY_NAME = {L["name"]: L for L in MODEL}


# ---- SCALE-Sim, patched ---------------------------------------------------
def _patch_scalesim():
    import numpy as np
    from scalesim.compute import systolic_compute_ws as ws
    from scalesim.memory import read_buffer as rb
    if getattr(ws.systolic_compute_ws, "_accel_patched", False):
        return

    # --- 1. weight-stationary weight order ---------------------------------
    orig_filter_prefetch = ws.systolic_compute_ws.create_filter_prefetch_mat

    def create_filter_prefetch_mat(self):
        # Prefetch each arr_row-row weight tile in the order the demand matrix
        # consumes it (create_filter_demand_mat flips each tile: bottom row first).
        orig_filter_prefetch(self)
        m = self.filter_prefetch_matrix
        K, R = self.filter_op_mat.shape[0], self.arr_row
        blocks = []
        for fc in range(self.col_fold):
            col_block = m[fc * K:(fc + 1) * K]
            for r0 in range(0, K, R):
                blocks.append(np.flip(col_block[r0:r0 + R], 0))
        self.filter_prefetch_matrix = np.concatenate(blocks, axis=0)

    ws.systolic_compute_ws.create_filter_prefetch_mat = create_filter_prefetch_mat

    # --- 2. read-buffer misses --------------------------------------------
    orig_prepare = rb.read_buffer.prepare_hashed_buffer
    orig_service = rb.read_buffer.service_reads

    def prepare_hashed_buffer(self):
        # The original splits the operand's fetch stream into 'sets' of
        # ceil(buffer elements / 100) elements; the buffer holds a window of sets.
        # Record each set's size and which sets hold each address.
        orig_prepare(self)
        per_set = math.ceil(self.total_size_elems / 100)
        counts, where = [], {}
        sid = ctr = 0
        for v in self.fetch_matrix.ravel():
            if v == -1:
                continue
            where.setdefault(v, []).append(sid)
            ctr += 1
            if ctr >= per_set:
                counts.append(ctr)
                sid, ctr = sid + 1, 0
        counts.append(ctr)
        self._set_counts = counts
        self._addr_sets = {k: sorted(set(v)) for k, v in where.items()}

    def _fetch(self, elems, now):
        # A serial DRAM channel: each fetch starts after the previous one ends, and
        # no more than one prefetch half-buffer of transfer time before it's needed.
        if elems <= 0:
            return
        lines = math.ceil(elems / self.req_gen_bandwidth)
        lookahead = math.ceil(self.prefetch_buf_size / self.req_gen_bandwidth)
        start = max(self.last_prefetch_cycle + 1, now - lookahead)
        self.last_prefetch_cycle = start + lines - 1 + self.backing_buffer.get_latency()
        self.num_access += elems
        t = self.trace_matrix[:, 0]
        self.trace_matrix = np.array([[min(t.min(), start)], [max(t.max(), self.last_prefetch_cycle)]])

    def _bring_in(self, addr, now):
        sets = self._addr_sets.get(addr)
        if not sets:
            return
        n, A = self.num_lines, self.num_active_buf_lines
        start, end = self.active_buffer_set_limits
        ahead = min(sets, key=lambda x: (x - end) % n)
        f = (ahead - end) % n                    # sets beyond the window's end
        cost_f = sum(self._set_counts[(end + j) % n] for j in range(f + 1)) if f + 1 <= A else None
        behind = min(sets, key=lambda x: (start - x) % n)
        b = (start - behind) % n                 # sets before the window's start
        cost_b = sum(self._set_counts[(behind + j) % n] for j in range(min(b, A)))
        if cost_f is not None and cost_f <= cost_b:
            s = (start + f + 1) % n              # slide forward by f + 1 sets
            self.active_buffer_set_limits = [s, (s + A) % n]
            self._fetch(cost_f, now)
        else:                                    # re-fetch a window at the nearest copy
            jump = sum(self._set_counts[(ahead + j) % n] for j in range(A))
            if jump < cost_b:
                behind, cost_b = ahead, jump
            self.active_buffer_set_limits = [behind, (behind + A) % n]
            self._fetch(cost_b, now)

    orig_prefill = rb.read_buffer.prefetch_active_buffer

    def prefetch_active_buffer(self, start_cycle):
        # The first fill: the active window's data, before the first compute cycle. The original
        # also charges the null (-1) padding slots of the fetch matrix as DRAM words and time.
        if self.enable_layout_evaluation:
            return orig_prefill(self, start_cycle)
        A = self.num_active_buf_lines
        elems = sum(self._set_counts[:A])
        lines = max(1, math.ceil(elems / self.req_gen_bandwidth))
        self.num_access += elems
        self.last_prefetch_cycle = int(start_cycle)
        self.trace_matrix = np.array([[start_cycle - lines], [start_cycle]])
        self.trace_valid = True
        self.active_buffer_set_limits = [0, A]
        self.prefetch_buffer_set_limits = [A, A + self.num_prefetch_buf_lines]
        self.active_buf_full_flag = True
        return lines

    def service_reads(self, incoming_requests_arr_np, incoming_cycles_arr):
        if self.enable_layout_evaluation:
            return orig_service(self, incoming_requests_arr_np, incoming_cycles_arr)
        if not self.active_buf_full_flag:
            self.prefetch_active_buffer(start_cycle=incoming_cycles_arr[0][0])
        out = []
        offset = self.hit_latency
        for i in range(incoming_requests_arr_np.shape[0]):
            cycle = float(np.asarray(incoming_cycles_arr[i]).ravel()[0])
            for addr in incoming_requests_arr_np[i]:
                if addr == -1 or self.active_buffer_hit(addr):
                    continue
                self._bring_in(addr, int(cycle + offset))
                stall = self.last_prefetch_cycle - (cycle + offset)
                if stall > 0:
                    offset += stall
            out.append(cycle + offset)
        return np.asarray(out, dtype=float).reshape((len(out), 1))

    rb.read_buffer.prepare_hashed_buffer = prepare_hashed_buffer
    rb.read_buffer.prefetch_active_buffer = prefetch_active_buffer
    rb.read_buffer._fetch = _fetch
    rb.read_buffer._bring_in = _bring_in
    rb.read_buffer.service_reads = service_reads
    ws.systolic_compute_ws._accel_patched = True


_CFG = """[general]
run_name = run

[architecture_presets]
ArrayHeight:    {rows}
ArrayWidth:     {cols}
IfmapSramSzkB:   {ib}
FilterSramSzkB:  {fb}
OfmapSramSzkB:   {ob}
IfmapOffset:    0
FilterOffset:   10000000
OfmapOffset:    20000000
Bandwidth : {bw}
Dataflow : {df}
ReadRequestBuffer: 32
WriteRequestBuffer: 32

[layout]
IfmapCustomLayout: False
IfmapSRAMBankBandwidth: {bw}
IfmapSRAMBankNum: 1
IfmapSRAMBankPort: 2
FilterCustomLayout: False
FilterSRAMBankBandwidth: {bw}
FilterSRAMBankNum: 1
FilterSRAMBankPort: 2

[sparsity]
SparsitySupport : false
SparseRep : ellpack_block
OptimizedMapping : false
BlockSize : 8
RandomNumberGeneratorSeed : 40

[run_presets]
InterfaceBandwidth: {mode}
UseRamulatorTrace: False
"""

# Column names of SCALE-Sim's reports that we keep, in our own names.
_COMPUTE_COLS = {"Total Cycles (incl. prefetch)": "cycles", "Total Cycles": "compute_plus_stall",
                 "Stall Cycles": "stall_cycles"}
_DETAIL_COLS = {"SRAM IFMAP Reads": "sram_reads_ifmap", "SRAM Filter Reads": "sram_reads_filter",
                "SRAM OFMAP Writes": "sram_writes_ofmap", "DRAM IFMAP Reads": "dram_reads_ifmap",
                "DRAM Filter Reads": "dram_reads_filter", "DRAM OFMAP Writes": "dram_writes_ofmap"}


def _run_scalesim(job):
    """One SCALE-Sim invocation. job: dict with kind ('conv'|'gemm'), rows (topology
    rows), df, ib, fb, ob (KB), bw (words/cycle, or None for no DRAM limit).
    Returns a list of per-row dicts (a depthwise row expands to one per channel)."""
    _patch_scalesim()
    from scalesim.scale_sim import scalesim
    d = tempfile.mkdtemp(prefix="accel_")
    try:
        bw = job["bw"]
        with open(os.path.join(d, "c.cfg"), "w") as f:
            f.write(_CFG.format(rows=ARRAY_ROWS, cols=ARRAY_COLS, ib=job["ib"], fb=job["fb"], ob=job["ob"],
                                df=job["df"], bw=bw if bw else 10, mode="USER" if bw else "CALC"))
        gemm = job["kind"] == "gemm"
        with open(os.path.join(d, "t.csv"), "w") as f:
            f.write("Layer, M, N, K,\n" if gemm else
                    "Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, "
                    "Num Filter, Strides,\n")
            for r in job["rows"]:
                f.write(", ".join(str(x) for x in r) + ",\n")
        with open(os.path.join(d, "l.csv"), "w") as f:
            f.write("unused,\n")
        with contextlib.redirect_stdout(io.StringIO()):
            s = scalesim(save_disk_space=True, verbose=False, config=os.path.join(d, "c.cfg"),
                         topology=os.path.join(d, "t.csv"), layout=os.path.join(d, "l.csv"),
                         input_type_gemm=gemm)
            s.run_scale(top_path=d)
        out = []
        rep = os.path.join(d, "run")
        comp = list(csv.reader(open(os.path.join(rep, "COMPUTE_REPORT.csv"))))
        det = list(csv.reader(open(os.path.join(rep, "DETAILED_ACCESS_REPORT.csv"))))
        ch = [h.strip() for h in comp[0]]
        dh = [h.strip() for h in det[0]]
        for crow, drow in zip(comp[1:], det[1:]):
            r = {}
            for col, key in _COMPUTE_COLS.items():
                r[key] = int(round(float(crow[ch.index(col)])))
            for col, key in _DETAIL_COLS.items():
                r[key] = int(round(float(drow[dh.index(col)])))
            out.append(r)
        return out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _key(job):
    blob = json.dumps({"job": job, "patch": PATCH_ID, "array": [ARRAY_ROWS, ARRAY_COLS],
                       "scalesim": _scalesim_version()}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


def _scalesim_version():
    try:
        from importlib.metadata import version
        return version("scalesim")
    except Exception:
        return "unknown"


def _cached(job):
    path = os.path.join(CACHE_DIR, _key(job) + ".json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    return None


def _store(job, result):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, _key(job) + ".json")
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, path)


def run_jobs(jobs, n_jobs=None):
    """Run a list of SCALE-Sim jobs (cached, in parallel). Returns results in order."""
    results = [None] * len(jobs)
    todo = []
    for i, job in enumerate(jobs):
        hit = _cached(job)
        if hit is None:
            todo.append(i)
        else:
            results[i] = hit
    if todo:
        n = n_jobs or int(os.environ.get("ACCEL_JOBS", "0")) or os.cpu_count() or 1
        # one job per distinct key, even if a list repeats a job
        uniq = {}
        for i in todo:
            uniq.setdefault(_key(jobs[i]), []).append(i)
        if n == 1 or len(uniq) == 1:
            for k, idx in uniq.items():
                r = _run_scalesim(jobs[idx[0]])
                _store(jobs[idx[0]], r)
                for i in idx:
                    results[i] = r
        else:
            with cf.ProcessPoolExecutor(max_workers=min(n, len(uniq))) as ex:
                futs = {ex.submit(_run_scalesim, jobs[idx[0]]): idx for idx in uniq.values()}
                for fut in cf.as_completed(futs):
                    idx = futs[fut]
                    r = fut.result()
                    _store(jobs[idx[0]], r)
                    for i in idx:
                        results[i] = r
    return results


# ---- Part 1: feed the array -----------------------------------------------
def _conv_row(L):
    """SCALE-Sim conv-topology row for a TinyCNN layer. A layer name containing
    'DP' makes SCALE-Sim run it as depthwise: one single-channel layer per channel."""
    name = ("DP_" + L["name"]) if L["per_channel"] else L["name"]
    nfilt = 1 if L["per_channel"] else L["out_c"]
    return [name, L["sim_h"], L["sim_w"], L["kernel"], L["kernel"], L["in_c"], nfilt, L["stride"]]


def layer_job(name, df, sram_kb, bw):
    ib, fb, ob = sram_kb
    return {"kind": "conv", "rows": [_conv_row(BY_NAME[name])], "df": df, "ib": ib, "fb": fb, "ob": ob,
            "bw": bw}


def _sum_rows(rows):
    out = {}
    for r in rows:
        for k, v in r.items():
            out[k] = out.get(k, 0) + v
    return out


def _layer_result(name, df, rows):
    L = BY_NAME[name]
    r = _sum_rows(rows)
    # SCALE-Sim counts compute cycles from 0; count them from 1 (as Part 2 does), taking the
    # cycle from the first-fill share so that compute + stall + fill = cycles still holds.
    n = len(rows)
    r["compute_cycles"] = r["compute_plus_stall"] - r["stall_cycles"] + n
    r["prefetch_cycles"] = max(0, r["cycles"] - r["compute_plus_stall"] - n)
    r["cycles"] = r["compute_cycles"] + r["stall_cycles"] + r["prefetch_cycles"]
    del r["compute_plus_stall"]
    r["dram_traffic"] = r["dram_reads_ifmap"] + r["dram_reads_filter"] + r["dram_writes_ofmap"]
    r["macs"] = L["macs"]
    r["util_pct"] = 100.0 * L["macs"] / (ARRAY_ROWS * ARRAY_COLS * max(1, r["cycles"]))
    return {"name": name, "dataflow": df, **r}


def check_part1_config(cfg):
    """Return a list of problems with a Part 1 config (empty if it's valid)."""
    errs = []
    if not isinstance(cfg, dict):
        return ["config must be a JSON object"]
    extra = set(cfg) - {"sram_kb", "bandwidth", "dataflow", "comment"}
    if extra:
        errs.append(f"unknown keys: {sorted(extra)}")
    s = cfg.get("sram_kb")
    if not (isinstance(s, dict) and set(s) == {"ifmap", "filter", "ofmap"}):
        errs.append('sram_kb must be {"ifmap": KB, "filter": KB, "ofmap": KB}')
    else:
        for k, v in s.items():
            if not (isinstance(v, int) and not isinstance(v, bool)) or v < SRAM_MIN_KB or v % SRAM_STEP_KB:
                errs.append(f"sram_kb.{k} = {v!r}: must be an integer multiple of {SRAM_STEP_KB}, "
                            f">= {SRAM_MIN_KB}")
        if all(isinstance(v, int) for v in s.values()) and sum(s.values()) > SRAM_BUDGET_KB:
            errs.append(f"sram_kb total {sum(s.values())} KB exceeds the {SRAM_BUDGET_KB} KB budget")
    b = cfg.get("bandwidth")
    if not (isinstance(b, int) and not isinstance(b, bool)) or not BW_MIN <= b <= BW_MAX:
        errs.append(f"bandwidth = {b!r}: must be an integer in [{BW_MIN}, {BW_MAX}] words/cycle")
    d = cfg.get("dataflow")
    if not (isinstance(d, dict) and set(d) == set(LAYER_NAMES)):
        errs.append(f"dataflow must give one of {DATAFLOWS} for every layer: {LAYER_NAMES}")
    else:
        for k, v in d.items():
            if v not in DATAFLOWS:
                errs.append(f"dataflow.{k} = {v!r}: must be one of {DATAFLOWS}")
    return errs


def evaluate_part1(cfg, n_jobs=None):
    """Simulate TinyCNN under a Part 1 config. Returns
    {"layers": [per-layer dict], "total": {...}, "deadline": D, "meets_deadline": bool}.
    Per-layer keys: name, dataflow, cycles (incl. the first buffer fill), compute_cycles,
    stall_cycles, prefetch_cycles, util_pct, macs, sram_reads_ifmap/_filter,
    sram_writes_ofmap, dram_reads_ifmap/_filter, dram_writes_ofmap, dram_traffic."""
    errs = check_part1_config(cfg)
    if errs:
        raise ValueError("invalid Part 1 config:\n  " + "\n  ".join(errs))
    s = cfg["sram_kb"]
    sram = (s["ifmap"], s["filter"], s["ofmap"])
    jobs = [layer_job(n, cfg["dataflow"][n], sram, cfg["bandwidth"]) for n in LAYER_NAMES]
    res = run_jobs(jobs, n_jobs)
    layers = [_layer_result(n, cfg["dataflow"][n], r) for n, r in zip(LAYER_NAMES, res)]
    total = _sum_rows([{k: v for k, v in L.items() if isinstance(v, int)} for L in layers])
    total.pop("util_pct", None)
    return {"layers": layers, "total": total, "deadline": DEADLINE_CYCLES,
            "meets_deadline": total["cycles"] <= DEADLINE_CYCLES}


def evaluate_layer(name, df, sram_kb=(32, 16, 16), bw=None):
    """One TinyCNN layer on the Part 1 hardware. bw=None: no DRAM limit (no stalls)."""
    return _layer_result(name, df, run_jobs([layer_job(name, df, tuple(sram_kb), bw)])[0])


# ---- Part 2: shape the work ----------------------------------------------
def gemm_job(M, N, K):
    return {"kind": "gemm", "rows": [["g", M, N, K]], "df": PART2_DATAFLOW, "ib": 64, "fb": 64, "ob": 64,
            "bw": None}


def gemm_folds(M, N, K):
    """Passes ('folds') a weight-stationary array needs: K rows x N columns per pass."""
    return math.ceil(K / ARRAY_ROWS) * math.ceil(N / ARRAY_COLS)


def check_part2_schedule(sched):
    """Return a list of problems with a Part 2 schedule (empty if it's valid)."""
    errs = []
    if not isinstance(sched, dict):
        return ["schedule must be a JSON object: {layer_name: {transform...}}"]
    layers = {k: v for k, v in sched.items() if k != "comment"}
    for name, t in layers.items():
        if name not in BY_NAME:
            errs.append(f"{name}: not a TinyCNN layer ({LAYER_NAMES})")
            continue
        if not isinstance(t, dict):
            errs.append(f"{name}: transform must be an object")
            continue
        L = BY_NAME[name]
        extra = set(t) - {"transpose", "split", "group"}
        if extra:
            errs.append(f"{name}: unknown keys {sorted(extra)} (allowed: transpose, split, group)")
        if "transpose" in t and not isinstance(t["transpose"], bool):
            errs.append(f"{name}: transpose must be true or false")
        g = t.get("group")
        if g is not None:
            if not L["per_channel"]:
                errs.append(f"{name}: group applies only to the depthwise layer")
            elif not (isinstance(g, int) and not isinstance(g, bool)) or g < 1 or L["channels"] % g:
                errs.append(f"{name}: group must be a positive integer dividing {L['channels']}")
        sp = t.get("split", {})
        if not isinstance(sp, dict) or set(sp) - {"M", "N", "K"}:
            errs.append(f"{name}: split must be an object with keys among M, N, K")
            continue
        if all(not e.startswith(name + ":") for e in errs):
            base = _base_gemm(L, t)
            for dim, parts in sp.items():
                full = dict(zip("MNK", base))[dim]
                if not (isinstance(parts, list) and parts and
                        all(isinstance(p, int) and not isinstance(p, bool) and p > 0 for p in parts)):
                    errs.append(f"{name}: split.{dim} must be a list of positive integers")
                elif sum(parts) != full:
                    errs.append(f"{name}: split.{dim} sums to {sum(parts)}, but {dim} = {full}")
    if len(layers) > PART2_MAX_LAYERS:
        errs.append(f"{len(layers)} layers restructured; at most {PART2_MAX_LAYERS} allowed")
    return errs


def _base_gemm(L, t):
    """The GEMM a layer is split from (before transposing): (M, N, K)."""
    if L["per_channel"]:
        g = t.get("group") or 1
        return (L["M"], g, L["K"] * g)
    return (L["M"], L["N"], L["K"])


def expand_layer(name, t=None):
    """The list of GEMMs a (possibly restructured) layer runs as, as dicts with
    M, N, K, count (how many identical copies run) and zero_macs (MACs that multiply
    zero weights: grouping a depthwise layer computes a block-diagonal GEMM)."""
    L = BY_NAME[name]
    t = t or {}
    M0, N0, K0 = _base_gemm(L, t)
    copies = L["channels"] // (t.get("group") or 1) if L["per_channel"] else 1
    sp = t.get("split", {})
    gemms = []
    for m in sp.get("M", [M0]):
        for n in sp.get("N", [N0]):
            for k in sp.get("K", [K0]):
                mm, nn = (n, m) if t.get("transpose") else (m, n)
                gemms.append((mm, nn, k))
    merged = {}
    for g in gemms:
        merged[g] = merged.get(g, 0) + copies
    out = []
    g_size = t.get("group") or 1
    for (M, N, K), c in merged.items():
        zero = 0
        if L["per_channel"] and g_size > 1:
            zero = M * N * K * c * (g_size - 1) // g_size
        out.append({"M": M, "N": N, "K": K, "count": c, "zero_macs": zero})
    return out


def evaluate_part2(sched=None, n_jobs=None):
    """Simulate TinyCNN on the Part 2 array (16x16, weight-stationary, no DRAM limit)
    with some layers restructured. Returns {"layers": [...], "total_cycles": int}.
    Per layer: name, restructured, gemms (each with M, N, K, count, folds, cycles
    per copy), cycles, folds, macs (useful), array_macs, util_pct."""
    sched = sched or {}
    errs = check_part2_schedule(sched)
    if errs:
        raise ValueError("invalid Part 2 schedule:\n  " + "\n  ".join(errs))
    plan = {n: expand_layer(n, sched.get(n)) for n in LAYER_NAMES}
    jobs, where = [], []
    for n in LAYER_NAMES:
        for i, g in enumerate(plan[n]):
            jobs.append(gemm_job(g["M"], g["N"], g["K"]))
            where.append((n, i))
    res = run_jobs(jobs, n_jobs)
    for (n, i), r in zip(where, res):
        g = plan[n][i]
        g["cycles"] = r[0]["compute_plus_stall"] + 1   # SCALE-Sim counts cycles from 0
        g["folds"] = gemm_folds(g["M"], g["N"], g["K"])
    layers = []
    for n in LAYER_NAMES:
        L = BY_NAME[n]
        gs = plan[n]
        cyc = sum(g["cycles"] * g["count"] for g in gs)
        folds = sum(g["folds"] * g["count"] for g in gs)
        array_macs = sum(g["M"] * g["N"] * g["K"] * g["count"] for g in gs)
        layers.append({"name": n, "restructured": n in sched, "gemms": gs, "cycles": cyc, "folds": folds,
                       "macs": L["macs"], "array_macs": array_macs,
                       "util_pct": 100.0 * L["macs"] / (ARRAY_ROWS * ARRAY_COLS * cyc)})
    return {"layers": layers, "total_cycles": sum(L["cycles"] for L in layers)}


def evaluate_gemm(M, N, K):
    """Cycles for one GEMM on the Part 2 array."""
    c = run_jobs([gemm_job(M, N, K)])[0][0]["compute_plus_stall"] + 1   # SCALE-Sim counts from 0
    return {"M": M, "N": N, "K": K, "cycles": c, "folds": gemm_folds(M, N, K),
            "util_pct": 100.0 * M * N * K / (ARRAY_ROWS * ARRAY_COLS * c)}


# ---- printing -------------------------------------------------------------
def load_json(path):
    with open(path) as f:
        return json.load(f)


def print_model():
    print(f"TinyCNN: {len(MODEL)} layers, {sum(L['weights'] for L in MODEL):,} weights, "
          f"{sum(L['macs'] for L in MODEL):,} MACs (batch 1)\n")
    print(f"{'layer':7s} {'type':5s} {'input':>12s} {'output':>12s} {'k/s':>5s} {'M':>6s} {'N':>5s} "
          f"{'K':>6s} {'x':>4s} {'MACs':>10s}")
    for L in MODEL:
        cpy = f"x{L['channels']}" if L["per_channel"] else ""
        print(f"{L['name']:7s} {L['type']:5s} {L['in_h']:>3d}x{L['in_w']:<3d}x{L['in_c']:<4d} "
              f"{L['out_h']:>3d}x{L['out_w']:<3d}x{L['out_c']:<4d} {L['kernel']}/{L['stride']:<3d} "
              f"{L['M']:>6d} {L['N']:>5d} {L['K']:>6d} {cpy:>4s} {L['macs']:>10,d}")
    print("\nGEMM: M = output pixels, N = output channels, K = kernel*kernel*in_c. "
          "The depthwise layer is one GEMM per channel (x32).")


def print_part1(r):
    print(f"{'layer':7s} {'df':3s} {'cycles':>9s} {'compute':>9s} {'stall':>9s} {'prefetch':>8s} "
          f"{'util%':>6s} {'DRAM rd ifmap':>13s} {'rd filter':>10s} {'wr ofmap':>9s}")
    for L in r["layers"]:
        print(f"{L['name']:7s} {L['dataflow']:3s} {L['cycles']:>9,d} {L['compute_cycles']:>9,d} "
              f"{L['stall_cycles']:>9,d} {L['prefetch_cycles']:>8,d} {L['util_pct']:>6.1f} "
              f"{L['dram_reads_ifmap']:>13,d} {L['dram_reads_filter']:>10,d} {L['dram_writes_ofmap']:>9,d}")
    t = r["total"]
    print(f"{'total':7s} {'':3s} {t['cycles']:>9,d} {t['compute_cycles']:>9,d} {t['stall_cycles']:>9,d} "
          f"{t['prefetch_cycles']:>8,d} {'':>6s} {t['dram_reads_ifmap']:>13,d} {t['dram_reads_filter']:>10,d} "
          f"{t['dram_writes_ofmap']:>9,d}")
    ok = "MEETS" if r["meets_deadline"] else "MISSES"
    print(f"\n{ok} the deadline: {t['cycles']:,} cycles vs {r['deadline']:,}")


def print_part2(r):
    print(f"{'layer':7s} {'cycles':>8s} {'folds':>6s} {'util%':>6s}  GEMMs (M,N,K)x copies")
    for L in r["layers"]:
        g = ", ".join(f"({x['M']},{x['N']},{x['K']})x{x['count']}" for x in L["gemms"])
        star = "*" if L["restructured"] else " "
        print(f"{L['name']:6s}{star} {L['cycles']:>8,d} {L['folds']:>6d} {L['util_pct']:>6.1f}  {g}")
    print(f"\ntotal cycles: {r['total_cycles']:,}    (* = restructured)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("model")
    p1 = sub.add_parser("part1")
    p1.add_argument("config")
    p1.add_argument("--json", action="store_true", help="print the full result as JSON")
    p2 = sub.add_parser("part2")
    p2.add_argument("schedule", nargs="?")
    p2.add_argument("--json", action="store_true")
    pl = sub.add_parser("layer")
    pl.add_argument("name", choices=LAYER_NAMES)
    pl.add_argument("--df", default="ws", choices=DATAFLOWS)
    pl.add_argument("--bw", type=int, default=None, help="DRAM words/cycle (default: unlimited)")
    pl.add_argument("--sram", default="32,16,16", help="ifmap,filter,ofmap KB (default 32,16,16)")
    pg = sub.add_parser("gemm")
    pg.add_argument("M", type=int)
    pg.add_argument("N", type=int)
    pg.add_argument("K", type=int)
    a = ap.parse_args()

    if a.cmd == "model":
        print_model()
    elif a.cmd == "part1":
        r = evaluate_part1(load_json(a.config))
        print(json.dumps(r, indent=1)) if a.json else print_part1(r)
    elif a.cmd == "part2":
        r = evaluate_part2(load_json(a.schedule) if a.schedule else {})
        print(json.dumps(r, indent=1)) if a.json else print_part2(r)
    elif a.cmd == "layer":
        r = evaluate_layer(a.name, a.df, tuple(int(x) for x in a.sram.split(",")), a.bw)
        for k, v in r.items():
            print(f"  {k:18s} {v:,.1f}" if isinstance(v, float) else f"  {k:18s} {v:,}" if isinstance(v, int)
                  else f"  {k:18s} {v}")
    elif a.cmd == "gemm":
        print(evaluate_gemm(a.M, a.N, a.K))


if __name__ == "__main__":
    main()
