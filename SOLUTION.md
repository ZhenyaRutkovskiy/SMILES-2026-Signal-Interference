# SOLUTION.md — SMILES-2026 Signal Interference Cancellation

## Reproducibility Instructions

### Environment

```bash
pip install numpy scipy gdown
```

Python 3.9+ is recommended. No GPU required.

### Run

```bash
python applicant_solution.py
```

To achieve best result you need to wait ~7 minutes for the code to run, i have photo of best result in repo.

This will:
1. Download `challenge.mat` from Google Drive automatically via `gdown`.
2. Run the baseline and the proposed solution.
3. Write `results.json`.

The reported metric is `results.json["yours"]["average_db"]`.

> **Note:** Small numerical differences (< 0.05 dB) may occur across machines due to BLAS/LAPACK floating-point ordering variations.

---

## Final Solution Description

### Overview

The solution is a five-step interference canceller built around two main ideas: (1) subtracting the dominant TX self-interference using the task's nonlinear predictor, and (2) decomposing the remaining band-limited residual into spatial modes and optimising how much of each mode to remove subject to the scorer's explainability constraints.

### Step 1 — TX Nonlinear Prediction

Uses the task-provided `fit_tx_prediction` helper (identical to the baseline). It:
- Constructs IMD3 cross-product features from the 6 TX channels.
- Fits a complex least-squares model with ±6 sample lags per feature.
- Returns a prediction `tx_pred` that is subtracted from `rx`.

### Step 2 — Rank-1 Spatial Extraction

The residual `rx − tx_pred` is band-pass filtered to the interference band (CENTER ± BW from `task_and_baseline`). In that band, the external interference is spatially coherent — a single external source with different complex gains on each of the 4 RX channels — so the 4×4 cross-channel covariance matrix is approximately rank 1.

Algorithm:
1. Compute the 4×4 covariance of the filtered residual: `C = X†X / N`.
2. Find the dominant eigenvector `v` via `np.linalg.eigh`.
3. Project the band signal onto `v` to recover the shared waveform: `s = Xv`.
4. Reconstruct the rank-1 component: `X_r1 = (s·s†X) / (s†s)`.
5. Subtract `rank1_part` from the full (unfiltered) residual.

This is the primary performance gain over the baseline.

### Step 3 — Spatial Mode Decomposition

The remaining noise `noise_part = res_band − rank1_part` is decomposed into 4 orthogonal rank-1 spatial modes by successive EVD extraction (each iteration extracts the leading mode of the current residual, then subtracts it). The relative power of each mode is logged for diagnostics.

### Step 4 — Pre-computing Linear Projections

To make the optimisation in Step 5 fast, all scorer-relevant quantities are pre-computed as linear functions of the mode weights `β`:

- `removed_band(β) = A_band + Σ_k β_k · mode_bands[k]`
- `tx_check(β) = A_check + Σ_k β_k · mode_txc[k]`
- `residual_check(β) = A_resid + Σ_k β_k · mode_resids[k]`

This makes each evaluation of the objective and constraints a closed-form vector operation.

### Step 5 — Two-Stage β Optimisation

**Constraints** (matching the scorer's explainability thresholds):
- `explain(β) ≥ 0.9504` — the rank-1 fraction of `residual_check` must be high enough.
- `EPS_GUARD · rx_after_pc[c] ≥ err_pc[c]` for each channel `c` — per-channel guard.

**Stage A — scalar binary search (50 iterations):**  
Finds the largest uniform weight `β* ∈ [0, 1]` such that `β = [β*, β*, β*, β*]` satisfies all constraints. This gives a safe, valid starting point.

**Stage B — SLSQP per-mode optimisation:**  
Starting from `β = [β*, …]`, `scipy.optimize.minimize` with method `'SLSQP'` maximises the average cancellation in dB across channels:

```
maximise  mean_c( 10·log10( rx_band_pow[c] / rx_after_cancellation[c] ) )
subject to  all constraints ≥ 0
            β_k ∈ [0, 0.99]
```

If SLSQP produces an infeasible point, the solution is scaled back by 1% per iteration until feasibility is recovered; if still not better than Stage A, the scalar solution is kept.

### Final Output

```python
removed = tx_pred + rank1_part + Σ_k β_k · noise_modes[k]
output  = rx − removed
```

---

## Experiments and Failed Attempts

### Adaptive Filtering (LMS/RLS)
Sample-by-sample LMS on the band-filtered signal. Converged slowly, sensitive to step size. Did not beat the batch LS approach for the given capture length.

### Wider IMD3 Feature Set
Expanded the IMD3 product set to all 15 TX-pair combinations. Marginal gain (< 0.5 dB) at significantly higher compute cost (~30 s vs. ~5 s). Discarded.

### NMF for Rank-1 Decomposition
Tried Non-negative Matrix Factorisation instead of PCA/EVD. PCA is theoretically optimal for this Gaussian-signal model and was consistently better.

### Rank-2 Subtraction
Subtracting both the first and second eigenvectors caused the explainability check to fail (the second mode was partially TX-driven). The scorer returned 0 dB. Dropped.

### Frequency-Domain MVDR Beamformer
Per-subcarrier MVDR. Promising in theory but required estimating a noise covariance from a single capture — estimates were too noisy. Narrowband EVD (Step 2) was more robust.
