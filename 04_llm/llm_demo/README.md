# llm_demo

Numbers behind the LLM decks (lecture 4a, LLM background; lecture 4b, LLMs on an NPU).

| File | What it does | Needs |
|---|---|---|
| `llm_numbers.py [section ...]` | every derived number in both decks: bytes and FLOPs per token, vocabulary tables, LoRA sizes, Llama 3.2 1B's GEMMs, KV cache sizes, the FastFlowLM fit and assumption test, prefill, and lecture 4b's design ladder on XDNA2 (the hardware numbers, the per-version time per token, the "why" slides, batching and speculative decoding). Sections: `count vocab lora ops kv flm prefill xdna ladder why batch` | Python standard library |
| `demo_prompt.txt` | a ~500-token prompt for lecture 4a's opening `ollama run tinyllama --verbose` demo | Ollama |

```bash
python3 llm_numbers.py                     # everything
python3 llm_numbers.py xdna ladder why     # just lecture 4b's design ladder
```

## Notes

- **The opening demo depends on the laptop.** With a short prompt, Ollama's two rates look similar; use `demo_prompt.txt`. Re-running the same prompt reports a prefill rate in the thousands, because Ollama reuses the prompt's KV cache: use a fresh prompt, or restart the model, to measure. On a CPU-only machine (i7-1255U under WSL, phi4-mini) the rates were 19.5 and 8.5 tokens/s (2.3×); a laptop with a GPU or Apple silicon shows a much bigger gap.
- **The ladder's soft numbers** are the DDR bandwidth the NPU gets (40–50 GB/s, assumed) and the cost of one NPU call (80 µs, inferred); both are constants at the top of the XDNA2 section, with their sources.
- **Data sources** (in `llm_numbers.py`'s header): Llama 3.2 1B config (Hugging Face); FastFlowLM's Llama 3 benchmarks (FLM v0.9.30, fetched 2026-10-23); arXiv 2602.06063 v2 for Q4NX, the XDNA2 clock and decode layout, and the kernels' bandwidth; AMD AM027 and the Linux amdxdna docs for the array. Re-fetch the FastFlowLM table before each offering: FLM is at v1.0.x and may republish.
