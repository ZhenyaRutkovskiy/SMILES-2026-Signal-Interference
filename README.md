# SMILES-2026 Signal Interference Cancellation

Repository layout:

- `task_and_baseline.py` contains the task-side logic, shared evaluation helpers, and baseline.
- `applicant_solution.py` must contain the applicant's solution and write `results.json`.

## Overview

A physical device simultaneously transmits and receives signals across multiple
channels. The received signal is corrupted by a structured interference term that
arises from the device's own transmission — plus an additional external source.
Your goal is to estimate and subtract this interference, recovering a cleaner
received signal.

This is a real-world problem. The data comes from actual hardware
measurements. No domain knowledge is required — the problem is fully self-contained.

---

## Setup

The device operates as follows:

```
                    ┌──────────────────────────┐
  TX[n] ──────────▶│                          │──────▶  (transmitted)
                    │      Physical Device      │
  RX[n] ◀──────────│                          │◀──────  (received)
                    └──────────────────────────┘
                              ↑
                    Interference leaks into RX
```

- **6 transmit channels** — complex samples, 3 pairs at different power levels
- **4 receive channels** — complex samples, corrupted by interference
- **Sample rate**: Fs = 7.68 MHz
- **Capture length**: N = 2,457,600 samples (~320 ms)

All signals are complex. TX channels come in pairs — each pair uses
a different operating point of the same device.

---

## Signal Model

The received signal on channel `c` at time `n` is:

```
rx  = s  +  I  +  η
```

where:

- **s[n, c]** — desired signal (what you want to keep)
- **I[n, c]** — structured interference (what you want to remove)
- **η[n, c]** — background noise

The interference **I[n, c]** has two components:

```
I = F_c( TX )  +  E
```

- **F_c(·)** is an unknown nonlinear function of **all transmitted signals jointly**.
  In particular, the interference on channel `c` can depend on cross-products
  between different TX channels.
- **E[n, c]** is an external interference term. It is **not** a function of
  `tx` but it is **spatially coherent** — the same source appears (with
  different amplitude and phase) across all 4 receive channels.

---

## Dataset

### File

| File | Contents |
|---|---|
| `challenge.mat` | `tx` (N×6), `rx` (N×4), `Fs`, `FC_TX` |

### Variables

| Variable | Shape | Description |
|---|---|---|
| `tx` | (N, 6) complex128 | Transmitted signals — 6 channels |
| `rx` | (N, 4) complex128 | Received signals — 4 channels |
| `Fs` | scalar | Sample rate: 7.68 × 10⁶ Hz |
| `FC_TX` | scalar | Reference carrier frequency: 1987.5 MHz |

### TX column layout

The 6 TX columns come from 3 operating points of the same device. Within each
pair, column `2k` and `2k+1` carry signals on two different carrier frequencies:

| Column | Carrier | Power level |
|---|---|---|
| `tx[:, 0]` | Carrier A | High |
| `tx[:, 1]` | Carrier B | High |
| `tx[:, 2]` | Carrier A | Medium |
| `tx[:, 3]` | Carrier B | Medium |
| `tx[:, 4]` | Carrier A | Low |
| `tx[:, 5]` | Carrier B | Low |

---

## Task

Given `tx` and `rx`, produce a corrected signal `rx_hat = rx - predicted_interference`
such that interference power is minimised across all 4 channels.

In this starter repository, applicants should implement their method inside
`your_canceller(tx_n, rx)` in `applicant_solution.py`. The function must return
an array `rx_hat` with the same shape as `rx`.

### Scoring

The score is computed inside a narrow band where the interference is concentrated.
The scoring function applies this band filter internally:

```
score = (1/4) · Σ_{c=0}^{3}  reduction(c)

reduction(c) = 10 · log₁₀ ( E[|filt(rx[:,c])|²] / E[|filt(rx_hat[:,c])|²] )   [dB]
```

**Higher is better.**

A solution that subtracts nothing scores **0 dB**.
A solution can also be marked **invalid**. In the official scorer, a valid
removed component must be largely explainable as:

- a TX-driven nonlinear component, plus
- a spatially coherent rank-1 component shared across RX channels.

If this explainability check fails, the score is forced to **0 dB**.

### Performance tiers

These numbers are only rough guidance for the provided capture:

| Tier | Score |
|---|---|
| No cancellation | 0 dB |
| Baseline (provided) | ~4 dB |
| Good | > 8 dB |

## Repository layout

- `task_and_baseline.py` - shared task helpers, scorer, and baseline method
- `applicant_solution.py` - the file applicants are expected to modify
- `results.json` - expected output file written by `applicant_solution.py`

## How to run

Create an environment with Python 3 and install the required packages:

```bash
pip install numpy scipy
```

Then run:

```bash
python applicant_solution.py
```

This command:

- loads `challenge.mat`
- computes the provided baseline
- runs your implementation from `your_canceller(...)`
- writes `results.json`

## Expected `results.json`

Running `python applicant_solution.py` produces a file with the following structure:

```json
{
  "baseline": {
    "per_channel_db": [0.0, 0.0, 0.0, 0.0],
    "average_db": 0.0
  },
  "yours": {
    "per_channel_db": [0.0, 0.0, 0.0, 0.0],
    "average_db": 0.0
  }
}
```

The main number of interest is `results.json["yours"]["average_db"]`.

# What is expected from the applicant of SMILES-2026 ?

**Q1:** What must the applicant submit in the application form ?<br>
**A1:** Submit:
1. A link to your GitHub repository.

**Q2:** What must the applicants include in the GitHub repository ?<br>
**A2:** Your repository must contain:
1. Your solution code, with `applicant_solution.py` remaining runnable as the main entrypoint.
2. `results.json` produced by running `python applicant_solution.py`.
3. A report file in Markdown format named `SOLUTION.md`.

**Q3:** Report requirements (`SOLUTION.md`)<br>
**A3:** Your report must include:
- Reproducibility instructions: exact commands to run your solution and acquire the same `results.json`, required environment (if any), and any important implementation details needed to reproduce your result.
- Final solution description: What components you modified ? What your final approach is ? Why you made these choices ? What contributed most to improving the metric ?
- Experiments and failed attempts: What ideas you tried but did not include in the final solution ? Why they did not work or were discarded ?

**Q4:** Reproducibility<br>
**A4:** The repository must be self-contained and runnable with `python applicant_solution.py`. Your solution must not require changes to the fixed task files such as `task_and_baseline.py` or the dataset file. Running `python applicant_solution.py` from a clean checkout must generate your final `results.json`. The reported metric of record is `results.json["yours"]["average_db"]`, which should be reproducible up to small numerical differences caused by environment or BLAS variations.
