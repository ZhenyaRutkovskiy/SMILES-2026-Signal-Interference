import json
import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve
from scipy.optimize import minimize

from task_and_baseline import (
    baseline,
    build_task_helpers,
    CENTER,
    BW,
    make_bandpass,
)


# Dataset

DATA_FILE = "challenge.mat"
if not os.path.exists(DATA_FILE):
    import gdown
    url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
    gdown.download(url, DATA_FILE, quiet=False, fuzzy=True)
else:
    print(f"{DATA_FILE} уже скачан, пропускаем.")

data = loadmat(DATA_FILE, simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


# Local bandpass

_BP = make_bandpass(CENTER, BW, Fs)


def _filt(x):
    return convolve(x, _BP, mode="same")


def _filt_band(matrix):
    return np.column_stack([_filt(matrix[:, c]) for c in range(matrix.shape[1])])


def rank1_extract(band_matrix):
    """Same construction as the scorer's `rank1_from_band_matrix`."""
    cov = band_matrix.conj().T @ band_matrix / band_matrix.shape[0]
    _, vecs = np.linalg.eigh(cov)
    shared = band_matrix @ vecs[:, -1]
    denom = np.vdot(shared, shared) + 1e-30
    return np.column_stack(
        [
            (np.vdot(shared, band_matrix[:, c]) / denom) * shared
            for c in range(band_matrix.shape[1])
        ]
    )


def decompose_into_modes(band_matrix, n_modes=4):
    """Successive rank-1 extractions: `band_matrix ≈ sum_k mode_k`,
    where each mode_k is rank-1 in a distinct spatial direction.
    """
    modes = []
    remaining = band_matrix.copy()
    for _ in range(n_modes):
        m = rank1_extract(remaining)
        modes.append(m)
        remaining = remaining - m
    return modes


# 
# Main canceller
#
# Layered structure of `removed`:
#   removed[:, c] = TX-nonlinear[:, c]
#                 + rank-1-spatial[:, c]
#                 + Σ_k β_k · noise_mode_k[:, c]
#
# The β-vector (per-mode weights) is optimised in two stages:
def your_canceller(tx_n, rx):
    fit_tx = helpers["fit_tx_prediction"]

    print("[Sol] 1/5 — TX nonlinear prediction (scorer's dictionary)")
    tx_pred = fit_tx(rx)

    print("[Sol] 2/5 — Band residual + rank-1 spatial extraction")
    rx_band = _filt_band(rx)
    tx_pred_band = _filt_band(tx_pred)
    res_band = rx_band - tx_pred_band
    rank1_part = rank1_extract(res_band)
    noise_part = res_band - rank1_part

    print("[Sol] 3/5 — Decomposing noise into spatial modes (SVD-style)")
    noise_modes = decompose_into_modes(noise_part, n_modes=4)
    mode_powers = [float(np.mean(np.abs(m) ** 2)) for m in noise_modes]
    print(f"      mode powers (rel.): "
          f"{[f'{p/sum(mode_powers):.3f}' for p in mode_powers]}")

    print("[Sol] 4/5 — Pre-computing β-linear scorer projections")
    removed_base = tx_pred + rank1_part
    removed_base_band = _filt_band(removed_base)
    txc_base = fit_tx(removed_base)

    # Linear pieces, one per noise mode
    mode_bands = [_filt_band(m) for m in noise_modes]
    mode_txc = [fit_tx(m) for m in noise_modes]
    mode_resids = [mb - mt for mb, mt in zip(mode_bands, mode_txc)]

    A_band = removed_base_band                    # filt(removed) at β = 0
    A_check = txc_base                            # tx_part_check at β = 0
    A_resid = A_band - A_check                    # residual_check at β = 0

    err0 = A_resid - rank1_extract(A_resid)
    err0_pc = np.mean(np.abs(err0) ** 2, axis=0)
    err0_total = np.mean(np.abs(err0) ** 2)
    rem0_total = np.mean(np.abs(A_band) ** 2) + 1e-30
    explain0 = 1.0 - err0_total / rem0_total
    rxa0_pc = np.mean(np.abs(rx_band - A_band) ** 2, axis=0) + 1e-30
    print(
        f"      explain(β=0) = {explain0:.4f}   "
        f"max err/residual = {np.max(err0_pc / rxa0_pc):.3f}"
    )

    EPS_EXPLAIN = 0.9504
    EPS_GUARD = 0.7997

    def evaluate(beta_vec):
        """Return (rem_band, err_pc, err_total, rem_total, rx_after_pc)."""
        rem_band = A_band.copy()
        resid = A_resid.copy()
        for k, b in enumerate(beta_vec):
            rem_band = rem_band + b * mode_bands[k]
            resid = resid + b * mode_resids[k]
        r1c = rank1_extract(resid)
        err = resid - r1c
        err_pc = np.mean(np.abs(err) ** 2, axis=0)
        err_total = np.mean(np.abs(err) ** 2)
        rem_total = np.mean(np.abs(rem_band) ** 2) + 1e-30
        rx_after_pc = np.mean(np.abs(rx_band - rem_band) ** 2, axis=0) + 1e-30
        return rem_band, err_pc, err_total, rem_total, rx_after_pc

    rx_pc_pow = np.mean(np.abs(rx_band) ** 2, axis=0) + 1e-30

    def neg_avg_db(beta_vec):
        _, _, _, _, rx_after_pc = evaluate(beta_vec)
        return -np.mean(10 * np.log10(rx_pc_pow / rx_after_pc))

    def all_constraints(beta_vec):
        _, err_pc, err_total, rem_total, rx_after_pc = evaluate(beta_vec)
        explain = 1.0 - err_total / rem_total
        return np.array([
            explain - EPS_EXPLAIN,
            EPS_GUARD * rx_after_pc[0] - err_pc[0],
            EPS_GUARD * rx_after_pc[1] - err_pc[1],
            EPS_GUARD * rx_after_pc[2] - err_pc[2],
            EPS_GUARD * rx_after_pc[3] - err_pc[3],
        ])

    def is_valid(beta_vec):
        c = all_constraints(beta_vec)
        return np.all(c >= 0)

    print("[Sol] 5/5 — Stage A scalar β; Stage B SLSQP per-mode β")

    # Stage 1
    n_modes = len(noise_modes)
    lo, hi = 0.0, 1.0
    best_g = 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if is_valid(np.full(n_modes, mid)):
            best_g = mid
            lo = mid
        else:
            hi = mid
    print(f"      Stage A — scalar β* = {best_g:.5f}    score = {-neg_avg_db(np.full(n_modes, best_g)):.3f} dB")
    beta = np.full(n_modes, best_g)

    # Stage 2
    constraints = [
        {'type': 'ineq', 'fun': lambda b, i=i: all_constraints(b)[i]}
        for i in range(5)
    ]
    bounds = [(0.0, 0.99)] * n_modes

    try:
        result = minimize(
            neg_avg_db,
            x0=beta,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 80, 'ftol': 1e-7, 'eps': 1e-4},
        )
        beta_slsqp = np.clip(result.x, 0.0, 0.99)
        # Re-validate 
        if is_valid(beta_slsqp):
            score_slsqp = -neg_avg_db(beta_slsqp)
            score_base = -neg_avg_db(beta)
            if score_slsqp > score_base:
                beta = beta_slsqp
                print(f"      Stage 2 — SLSQP β = "
                      f"[{', '.join(f'{b:.4f}' for b in beta)}]    "
                      f"score = {score_slsqp:.3f} dB")
            else:
                print(f"      Stage 2 — SLSQP did not improve, keeping scalar β")
        else:
            # Linear back-off until the constraints are met
            scaled = beta_slsqp.copy()
            for _ in range(40):
                if is_valid(scaled):
                    break
                scaled *= 0.99
            if is_valid(scaled) and -neg_avg_db(scaled) > -neg_avg_db(beta):
                beta = scaled
                print(f"      Stage 2 — SLSQP (scaled-back) β = "
                      f"[{', '.join(f'{b:.4f}' for b in beta)}]    "
                      f"score = {-neg_avg_db(beta):.3f} dB")
            else:
                print(f"      Stage 2 — SLSQP solution invalid even after back-off, keeping scalar β")
    except Exception as exc:
        print(f"      Stage 2 — SLSQP failed ({exc}), keeping scalar β")

    removed = removed_base.copy()
    for k, b in enumerate(beta):
        removed = removed + b * noise_modes[k]
    return rx - removed


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== My Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nWrote results.json")
