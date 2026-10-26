# scalesim_demo

Using [SCALE-Sim](https://github.com/scalesim-project/SCALE-Sim) to answer what-if questions about a systolic array. `what_if.py` runs a VGG-style CIFAR-10 CNN (six 3×3 convolutions c1–c6 with 64–256 channels, then a classifier fc) on a 16×16 weight-stationary array, prints SCALE-Sim's per-layer compute report, and asks two questions: would a 4× bigger array (32×32) make the model 4× faster, and how much does letting each layer pick its own dataflow help on the original array? Each changes one line of the same config. The model is large (153 M MACs), so each run takes about a minute.

`featured_layer.py` is a smaller check: it predicts one layer's cycles by hand and compares with SCALE-Sim. The layer is `conv2` of a smaller CNN: a 3×3, stride-2 convolution, 16 → 32 channels, 32×32 → 16×16. As a matrix multiply it is M = 256 (output pixels), N = 32 (output channels), K = 144 (3 × 3 × 16, one dot product). The array is 16×16. For each dataflow (weight-, output- and input-stationary), the formula counts the passes ("folds") the array needs and the cycles per pass, and SCALE-Sim's cycle count should match it exactly.

## Prerequisites and install

- Python: tested on 3.12
- `scalesim==3.0.0` and `numpy==2.3.5`. SCALE-Sim 3.0.0 fails with numpy 2.4 or later ("only 0-dimensional arrays can be converted to Python scalars"), so use a fresh venv:

```bash
python3 -m venv ~/scalesim-env              # tested on Python 3.12
source ~/scalesim-env/bin/activate
pip install scalesim==3.0.0 numpy==2.3.5
```

- If the install or a run fails, install Python 3.12 and create the venv with `python3.12 -m venv ~/scalesim-env` instead. Python 3.10 won't work: numpy 2.3.5 doesn't support it.

- `mapping_recipe.py` needs only the Python standard library.

## Files

| File | What it does |
|---|---|
| `what_if.py report` / `array` / `dataflow` | the CIFAR-10 CNN: SCALE-Sim's compute report on 16×16 weight-stationary; per-layer and total speedup on a 32×32 array, or with the best dataflow per layer (one to four minutes each) |
| `featured_layer.py` | conv2 on a 16×16 array in all three dataflows: folds, the formula's cycle prediction, and SCALE-Sim's cycles side by side; runs SCALE-Sim on the files in `raw/`, once per dataflow (about 1 s) |
| `mapping_recipe.py M N K [--transpose]` | the same arithmetic for any GEMM on a weight-stationary array (default 16×16; `--rows`, `--cols` to change it); no simulator needed |
| `raw/` | raw SCALE-Sim input files: the config (`conv2_ws.cfg`, also the base config for `what_if.py`), topologies for the CIFAR-10 CNN (`cifar_cnn_conv.csv`) and for conv2 alone (`conv2_conv.csv`, `conv2_gemm.csv`), the layout placeholder (`layout.csv`), and conv2's reports (`expected_output/`) |

## Running

```bash
python3 what_if.py report
python3 what_if.py array
python3 what_if.py dataflow
python3 featured_layer.py
python3 mapping_recipe.py 256 32 144                 # conv2
python3 mapping_recipe.py 64 256 2304 --transpose    # the CIFAR CNN's c6, transposed

cd raw
python3 -m scalesim.scale -c conv2_ws.cfg -t conv2_conv.csv -l layout.csv -p out/   # ~2 s
python3 -m scalesim.scale -c conv2_ws.cfg -t conv2_gemm.csv -l layout.csv -i gemm -p out_gemm/
```

## Notes on raw SCALE-Sim 3.0.0

- The layout file (`-l`) is required even though custom layouts are off; a one-line placeholder is enough.
- The CLI always writes per-cycle traces (its `-s` flag is ignored): about 2.8 MB for conv2.
- Cycle counts in the reports start from 0, so "Total Cycles" is 5,435, one less than the 5,436 the formula gives. `featured_layer.py` adds the 1 back.
- The conv topology format has no padding field: give the padded input size the windows touch ((16 − 1)·2 + 3 = 33 for conv2). There is no groups field either; a layer name containing `DP` is run as depthwise.
- `InterfaceBandwidth: CALC` in the config means no DRAM limit, so there are no stall cycles and "Total Cycles" is pure compute. Set it to `USER` to make SCALE-Sim model a bandwidth limit and the stalls it causes. In `USER` mode, 3.0.0 takes the IFMAP and filter DRAM bandwidths from `IfmapSRAMBankBandwidth` and `FilterSRAMBankBandwidth` in the `[layout]` section, and only the OFMAP bandwidth from `Bandwidth`, so set all three. Expect its DRAM traffic and stall numbers to be pessimistic: 3.0.0's read buffers re-fetch far more data than needed on a miss. Compute cycles are unaffected.
