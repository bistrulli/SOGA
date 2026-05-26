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
# Step 1: resilience vs input value scale
python3 experiments/lishan_resilience_2026-05-25/run_sweep.py

# Step 3: resilience vs bimodal mixing weight p
python3 experiments/lishan_resilience_2026-05-25/run_step3.py
```

## Step 5: Generate plots

```bash
python3 experiments/lishan_resilience_2026-05-25/plot_step1.py
python3 experiments/lishan_resilience_2026-05-25/plot_step3.py
```

Figures saved to `experiments/lishan_resilience_2026-05-25/figures/`.

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

Expected: 72 passed, 2 xfailed, 1 xpassed. Runtime < 10s.

## Verification

```bash
# Verify input matrix hashes
sha256sum experiments/lishan_resilience_2026-05-25/results/A_kernel.npz
# Expected: ce3b79ddf4453d4ce23c36beac62de6a0dd8b3cf2340d041984b087fc3535d1b
```

## Known limitations

- Input-side fault model (NOT register-level SASSIFI/NVBitFI injection)
- SOGA mantissa approximation overestimates SDC by ~38x for A=I_32
- See `experiments/lishan_resilience_2026-05-25/LIMITATIONS.md` for details
