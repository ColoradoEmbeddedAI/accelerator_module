# feeding_demo — Lecture 3's numbers

`feeding_tables.py` prints every simulated number in `lecture3-feeding-the-npu-slides.md`. The lecture's demo model is the CIFAR-10 CNN (`cifar_cnn.csv`, loaded by `cifar.py`) on the lab's hardware; the lab itself is TinyCNN, so the deck shows no simulated TinyCNN numbers. Everything is computed with SCALE-Sim 3.0.0 through the lab's helper; `tools/accel.py` is a copy of it, with the model it loads by default (`model/tinycnn.csv`).

```bash
source <the lab venv>/bin/activate      # or a fresh venv: pip install -r requirements.txt
python3 feeding_tables.py               # several minutes the first time; cached in .accel_cache/ after
python3 cifar.py layer c1 --df ws --bw 8 --sram 32,16,16   # one layer, same report as the lab's accel.py layer
```

The helper patches three problems in SCALE-Sim 3.0.0's memory model at run time (documented at the top of `accel.py`, and on Lecture 3's backup slide 42). Raw SCALE-Sim gives different DRAM traffic and stall numbers for the same configurations; compute cycles are identical.
