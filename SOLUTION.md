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

This will:
1. Download `challenge.mat` from Google Drive automatically via `gdown`.
2. Run the baseline and the proposed solution.
3. Write `results.json`.

The reported metric is `results.json["yours"]["average_db"]`.

> **Note:** Small numerical differences (< 0.05 dB) may occur across machines due to BLAS/LAPACK floating-point ordering variations.

---

## Final Solution Description

### Overview

The solution is a three-stage interference canceller. Each stage targets a distinct component of the interference model described in the task.

### Stage 1 — TX Nonlinear Cancellation

Identical to the provided baseline. Uses the task's `fit_tx_prediction` helper, which:
- Constructs IMD3 cross-product features from the 6 TX channels (e.g. `tx[:,0]² · conj(tx[:,1])`).
- Fits a complex least-squares model with ±6 sample lags per feature.
- Subtracts the prediction from `rx`.

This stage handles the dominant transmitter self-interference.

**Contribution:** ~4 dB (matches baseline).

### Stage 2 — Rank-1 Spatial Cancellation

The external interference `E[n, c]` is described in the task as **spatially coherent** — a single external source appearing on all 4 RX channels with different complex gains. This means in the interference band, the 4×4 cross-channel covariance matrix of the residual is **rank 1**.

Algorithm:
1. Band-pass filter the stage-1 residual (same 1.9 MHz ± 0.3 MHz band used by the scorer).
2. Compute the 4×4 spatial covariance matrix.
3. Extract the leading eigenvector `v` (dominant spatial mode) via `np.linalg.eigh`.
4. Project the band signal onto `v` to recover the shared waveform.
5. Estimate per-channel complex gains and subtract the rank-1 component from the full (unfiltered) residual.
6. Repeat for 3 iterations to handle residual coupling between stages.

This is the **key improvement** over the baseline. The external interference is spatially coherent by construction, so PCA/EVD in the signal band isolates it cleanly.

**Contribution:** ~4–8 additional dB.

### Stage 3 — Extra TX Feature Cleanup

After the first two stages, weak residual TX-driven products remain:
- Fifth-order IMD products: `tx[:,0]³ · conj(tx[:,1])²`, etc.
- Self-cubic amplitude terms: `tx[:,c] · |tx[:,c]|²`.
- Cross-pair 5th-order terms.

These are fit with a **wider lag window** (±10 samples vs. the baseline's ±6) on the stage-2 residual using the same regularised least-squares approach.

**Contribution:** ~1–3 additional dB depending on the capture.

### Why These Choices

- **Stage 2 is the dominant gain.** The rank-1 spatial structure is given explicitly in the problem statement; PCA in the narrow band is the natural, signal-theoretically sound estimator.
- **Iterating stages 1 & 2** avoids over-subtraction that would fail the explainability check.
- **Keeping stage 3 lightweight** (small feature set, per-channel independent fit) avoids the risk of the explainability validator rejecting the solution.

---

## Experiments and Failed Attempts

### Adaptive Filtering (LMS/RLS)
Tried sample-by-sample LMS on the band-filtered signal. Converged slowly and was sensitive to step size. Did not beat the batch LS approach for the given capture length.

### Wider IMD3 Feature Set (all pairs)
Expanded the IMD3 product set to all 15 pair combinations of 6 TX channels. Marginal gain (<0.5 dB) but significantly increased compute time (~30s vs ~5s). Discarded.

### Non-negative Matrix Factorization for Rank-1
Tried NMF as an alternative to PCA for spatial decomposition. PCA (EVD) is theoretically optimal for this Gaussian-signal model and was consistently better.

### Second Rank-1 Component (Rank-2)
Attempted to subtract the second eigenvector as well. This caused the explainability check to fail (the second mode was partially TX-driven, not purely external), forcing the score to 0 dB. Dropped.

### Frequency-Domain MMSE
Built a per-subcarrier minimum variance distortionless response (MVDR) beamformer. Promising in theory but required estimating a noise covariance; with only one capture, estimates were noisy. Narrowband EVD (stage 2) was more robust.
