# Neural Network Surrogates for Heston Model Calibration

**Deep Differential Networks (DML) as fast, arbitrage-free, gradient-stable surrogates for semi-analytical option pricing.**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![JAX](https://img.shields.io/badge/JAX-autodiff-9cf)
![License](https://img.shields.io/badge/License-MIT-green)

> BSc thesis, Chalmers University of Technology (2026) — *Neural Network Surrogates for Optimizing Heston Model Calibration: A Comparative Study on Design Choices for Deep Differential Pricing Surrogates.*

---

## TL;DR

Calibrating the Heston stochastic-volatility model to market data means solving an expensive inverse problem: repeatedly pricing thousands of options across strikes and maturities, for which there is no closed-form solution. The Fourier-based **COS method** is the standard semi-analytical workhorse, but it is slow and numerically fragile in the tails — exactly where a gradient-based calibrator needs stable derivatives.

This project trains neural networks to **replace the pricing engine entirely**, and shows that *how you frame the output matters as much as the network itself*. The best model — a Deep Differential Network trained on **implied volatilities** (`DDN_IV`) — delivers:

- ⚡ **~764× higher throughput** than the COS benchmark (1.23 × 10⁶ samples/s)
- 🎯 **~440× lower pricing MSE** (8.0 × 10⁻⁹ vs. 3.6 × 10⁻⁶ for COS at N=512)
- 📐 **Stable Greeks in the extreme tails**, where price-trained networks and COS both misbehave
- ✅ **0.00% no-arbitrage bound violations** — arbitrage-free *by construction*

The core idea: by predicting IV and mapping back to price through Black–Scholes, the surrogate inherits the theoretical guarantees of the pricing formula while sidestepping the numerical singularities that plague the COS method at low vega.

---

## Background

The Heston model captures volatility clustering and the volatility smile by treating variance as a mean-reverting stochastic process. That realism comes at a cost: option prices must be recovered numerically, and calibration — finding the parameters (κ, θ, σ, ρ, v₀) that reproduce observed prices — becomes a computational bottleneck. In latency-sensitive settings like real-time risk and high-frequency trading, this is prohibitive.

**Surrogate modelling** replaces the numerical pricer with a trained network: pricing collapses to a single forward pass with 𝒪(1) inference cost, and — crucially — the network is differentiable, so a calibration optimizer can obtain parameter gradients directly instead of via finite differences.

## Approach

### Differential Machine Learning (DML)

Rather than fitting prices alone, the differential networks are trained with **Sobolev loss**: the exact parameter derivatives (Greeks) are added as supervision targets. This acts as a powerful regularizer, teaching the network the *shape* of the pricing manifold, not just its values — which is what makes the learned gradients trustworthy enough to calibrate against.

### Data pipeline

- **Sampling** — Heston parameters and market state (maturity, rate, log-moneyness) drawn via **Sobol sequences** for low-discrepancy coverage of the 8-dimensional input space.
- **Ground truth** — prices *and* exact parameter Greeks computed with the **COS method, implemented in JAX** so the Greeks come from automatic differentiation rather than finite differences.
- **IV targets** — implied volatilities and IV Greeks obtained via Jäckel's **"Let's Be Rational"** algorithm, which inverts Black–Scholes robustly without the divergence risk of Newton–Raphson.

### The four surrogates compared

| Model | Target | Idea |
|-------|--------|------|
| `MLP_P` | Price | Standard supervised baseline — **no** Greeks, no DML |
| `DDN_P` | Price | Differential network on raw prices + Greeks |
| `DDN_PdivK` | Price / Strike | Normalizes the vast price scale into a bounded [0, 1] target |
| **`DDN_IV`** ⭐ | Implied vol | Predicts IV, prices via Black–Scholes → arbitrage-free by construction |

All networks use **SiLU** activations (smooth, continuously differentiable — essential for clean Greeks) and a **Softplus** output head to guarantee non-negative predictions.

## Results

Benchmarked on a held-out test set against the COS method (N = 512 expansion terms).

| Metric | `MLP_P` | `DDN_P` | `DDN_PdivK` | **`DDN_IV`** | COS |
|---|---|---|---|---|---|
| Speedup vs. COS | 323× | 687× | 664× | **764×** | 1× |
| Pricing MSE | 1.76×10⁻⁷ | 1.94×10⁻⁸ | — | **8.04×10⁻⁹** | 3.55×10⁻⁶ |
| Price-bound violations | 2.77% | — | — | **0.00%** | 1.18% |
| Calendar-spread violations | — | — | — | **0.31%** | 1.21% |

### Seeing the difference

<p align="center">
  <img src="assets/iv_surface_cos_short.png" width="49%" alt="Implied-volatility surface recovered from COS prices at short maturities, showing numerical blow-up" />
  <img src="assets/iv_surface_ddn_short.png" width="49%" alt="Smooth, bounded implied-volatility surface produced by the DDN_IV surrogate" />
</p>

<p align="center"><em><b>Left — COS.</b> Inverting COS prices to implied vol at short maturities detonates in the low-vega wings: IV spikes past 20 where the numerical integration loses precision. <b>Right — DDN_IV.</b> The surrogate learns a smooth, bounded, artifact-free volatility surface across the whole domain — the same structural robustness that drives its near-zero arbitrage violations.</em></p>

**What the numbers say:**

- **Output representation dominates.** The plain MLP is the weakest network by a wide margin; adding differential training helps, but *reframing the target as implied volatility* is what unlocks the best accuracy, stability, and arbitrage compliance simultaneously.
- **Homogeneous errors.** Price-trained models have errors that scale with option magnitude (biased toward in-the-money contracts). `DDN_IV`'s error distribution is flat across the whole domain.
- **Tail stability.** Using an empirical Lipschitz analysis (Monte Carlo over 20k pairs), the differential networks damp the high-frequency numerical artifacts that COS produces in low-vega tail regions — e.g. `DDN_IV`'s 99th-percentile sensitivity ratio for κ is 0.41× that of COS.
- **Extrapolation.** Out-of-distribution, `DDN_P` and `MLP_P` diverge and even produce invalid negative Gamma; `DDN_IV` generalizes gracefully beyond its training bounds.

A full quantitative breakdown — including Lipschitz analytics, error heatmaps, and the assumptions behind each claim — is in [`project_information.md`](project_information.md).

## Repository structure

```
.
├── notebooks/
│   ├── 1_data_generation/     # Sobol sampling, COS+JAX ground truth, IV transform, arbitrage post-processing
│   ├── 2_model_training/      # One notebook per surrogate (MLP_P, DDN_P, DDN_PdivK, DDN_IV, ...)
│   └── 3_evaluation/          # final_eval.ipynb — the full head-to-head benchmark
├── src/heston_project/
│   ├── pricing/
│   │   ├── COS_nojax.py                # Reference COS implementation
│   │   └── COS_with_jax_gradients.py   # JAX COS → exact Greeks via autodiff
│   ├── models/
│   │   ├── DDN.py             # Deep Differential Network
│   │   ├── MLP.py             # Baseline MLP
│   │   └── saved/             # Trained checkpoints (.pth)
│   └── utils.py              # Parameter bounds & input/target scaling
├── tests/                    # COS validation, Sobol-vs-LHS study, "Let's Be Rational" checks
├── data/                     # Datasets available on request (see data/info.md)
└── pyproject.toml
```

## Getting started

```bash
git clone https://github.com/mans-westman/DML-HestonCalibration.git
cd DML-HestonCalibration

python -m venv .venv && source .venv/bin/activate
pip install -e .          # installs the heston_project package
```

Then open the notebooks in order — `1_data_generation` → `2_model_training` → `3_evaluation`. The pretrained checkpoints in `src/heston_project/models/saved/` let you reproduce the evaluation in `notebooks/3_evaluation/final_eval.ipynb` without retraining. The synthetic datasets are large and are **available on request** (see [`data/info.md`](data/info.md)).

## Tech stack

**PyTorch** (surrogate training) · **JAX** (differentiable COS pricer / exact Greeks) · **NumPy / SciPy** · Sobol sequencing · "Let's Be Rational" for implied-vol inversion.

## Authors

Alex Andersson · Linnea Davidson · Markus Eliassen · Jacob Hansén · Isak Stridh · **Måns Westman**
Bachelor's thesis, Chalmers University of Technology, 2026.

## License

Released under the [MIT License](LICENSE).
