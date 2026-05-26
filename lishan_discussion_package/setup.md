# Setup Guide — Lishan Resilience POC

Environment: Python 3.9+ on macOS/Linux. No CUDA required.

## Step 1: Clone and checkout

```bash
git clone https://github.com/bistrulli/SOGA.git
cd SOGA
git checkout feat/lishan-resilience-poc
```

## Step 2: Install dependencies

```bash
pip install numpy scipy matplotlib ipywidgets jupyter
```

Or if using the venv:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib ipywidgets jupyter
```

## Step 3: Setup matrices

```bash
python3 experiments/lishan_resilience_2026-05-25/setup_matrices.py
```

This generates `experiments/lishan_resilience_2026-05-25/results/A_kernel.npz`
(identity matrix I_32, matching the `lishan_2mm_32x32.soga` kernel).

## Step 4: Run sweeps

```bash
# Step 1: resilience vs input value scale (bit-exact primary, default)
python3 experiments/lishan_resilience_2026-05-25/run_sweep.py

# Step 3: resilience vs bimodal mixing weight p (bit-exact primary)
python3 experiments/lishan_resilience_2026-05-25/run_step3.py

# To use legacy 5-class model (historical reference):
python3 experiments/lishan_resilience_2026-05-25/run_sweep.py --mode 5_class
python3 experiments/lishan_resilience_2026-05-25/run_step3.py --mode 5_class
```

**Note**: The default mode changed in v2 (config_version=2) from `5_class` to `bit_exact`.
This is a deliberate behavioral change — bit-exact results are within 4.2% of MC reference
(was ~38x off with 5-class). A `[MODE]` banner is printed at script start.

## Step 5: Generate plots

```bash
# Single-model comparison (MC vs bit-exact)
python3 experiments/lishan_resilience_2026-05-25/plot_step1.py
python3 experiments/lishan_resilience_2026-05-25/plot_step3.py

# Three-model comparison (MC | 5-class legacy | bit-exact primary)
python3 experiments/lishan_resilience_2026-05-25/plot_step1_3panel_comparison.py
```

Figures saved to `experiments/lishan_resilience_2026-05-25/figures/`.
Key output: `figures/resilience_vs_input_value_3panel.png` — shows the improvement
from 5-class (38x off) to bit-exact (< 4.2% off) at a glance.

## Step 6: Interactive notebook

```bash
jupyter notebook lishan_discussion_package/lishan_pitch.ipynb
```

Or in JupyterLab:
```bash
jupyter lab lishan_discussion_package/lishan_pitch.ipynb
```

## Step 7: Run tests

```bash
python3 -m pytest experiments/lishan_resilience_2026-05-25/tests/ -v
```

Expected: ~116 passed, 1 xfailed, 2 xpassed. Runtime < 60s (MC 5000-sample tests included).

## Verification

```bash
# Verify input matrix hashes
sha256sum experiments/lishan_resilience_2026-05-25/results/A_kernel.npz
# Expected: ce3b79ddf4453d4ce23c36beac62de6a0dd8b3cf2340d041984b087fc3535d1b
```

## Known limitations

- Input-side fault model (NOT register-level SASSIFI/NVBitFI injection)
- Bit-exact model (default) matches MC at < 4.2% relative error; the 5-class legacy
  model (--mode 5_class) overestimates SDC by ~38x and is provided for historical reference only
- See `experiments/lishan_resilience_2026-05-25/lib/DESIGN_BIT_EXACT.md` for the
  IEEE 754 bit-flip implementation rationale and all bug fixes applied during development
