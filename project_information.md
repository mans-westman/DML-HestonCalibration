# Project Information: Neural Network Surrogates for Optimizing Heston Model Calibration

## 1. Project Overview

- **Title:** Neural Network Surrogates for Optimizing Heston Model Calibration: A Comparative Study on Design Choices for Deep Differential Pricing Surrogates.
- **Authors:** Alex Andersson, Linnea Davidson, Markus Eliassen, Jacob Hansén, Isak Stridh, Måns Westman.
- **Primary objective:** Test whether neural network surrogates can match the accuracy and stability of semi-analytical option pricing while running much faster, in the setting of Heston model calibration.
- **Secondary objective:** Look at how design choices — output representation in particular — affect pricing accuracy, speed, and derivative stability, relative to standard (non-differential) networks and the COS method.

---

## 2. Background and Problem Statement

Calibrating a stochastic volatility model like Heston means finding the parameters that let the model reproduce observed market option prices.

Because Heston has a stochastic variance component, there's no closed-form solution for option prices in the price domain. Calibration therefore leans on numerical methods — the Fourier-based COS method here — to evaluate the semi-analytical pricing formula across thousands of strikes and maturities. That is slow and expensive. In settings that need speed, such as real-time risk management and high-frequency trading, it becomes a genuine bottleneck, which is what motivates building a fast surrogate.

---

## 3. Methodology and Data Pipeline

We generated a synthetic dataset over a wide parameter space covering a range of market regimes.

- **Sampling.** Inputs were drawn with Sobol sequences to get uniform, low-discrepancy coverage of the high-dimensional parameter space.
- **Parameter bounds.** The dataset samples the Heston parameters — mean-reversion speed $\kappa$, long-run variance $\theta$, volatility of variance $\sigma$, correlation $\rho$, and initial variance $v_0$ — along with the market state variables: time to maturity, risk-free rate, and log-moneyness.
- **Ground truth.** Prices and exact parameter Greeks come from the COS method, implemented in JAX so the Greeks are obtained by automatic differentiation.
- **Implied volatility.** IVs and IV Greeks were computed with the "Let's Be Rational" algorithm, which inverts Black–Scholes robustly and avoids the divergence you can hit with plain Newton–Raphson.

---

## 4. Evaluated Surrogate Architectures

We implemented four networks to test the effect of differential machine learning (DML) and of output representation. DML adds the derivatives of the outputs to the loss function (Sobolev training), which acts as a regularizer during training.

- `MLP_P` (baseline): a standard MLP trained only on prices, with ordinary supervised learning and no Greeks.
- `DDN_P`: a Deep Differential Network trained on raw prices and their parameter Greeks.
- `DDN_PdivK`: a Deep Differential Network trained on prices normalized by strike ($P/K$), plus Greeks. Dividing by strike flattens the huge absolute price scale and bounds the target between 0 and 1.
- `DDN_IV`: a Deep Differential Network trained on implied volatilities and IV Greeks. This targets the volatility structure directly and avoids the large scale gap between deep in-the-money and out-of-the-money options.

All networks use SiLU activations in the hidden layers (for continuous differentiability) and Softplus in the output layer (to keep predictions non-negative).

---

## 5. Core Findings and Results

We benchmarked the models against COS on a held-out test set, looking at speed, pricing accuracy, derivative stability, and no-arbitrage compliance.

### 5.1 Computational efficiency

Every network was two to three orders of magnitude faster than COS. `DDN_IV` reached 764× the throughput of the semi-analytical baseline across the full IV and IV-Greek pipeline. Throughput also climbed sharply with batch size once large batches were run in parallel on the GPU.

### 5.2 Pricing and prediction accuracy

`DDN_IV` was the most accurate model, beating both COS and the other networks on MSE and MAE for almost every target. The `MLP_P` baseline was the least accurate of the networks — a sign that plain supervised learning struggles with stochastic-volatility dynamics without the regularization that differential training provides.

The error heatmaps also show a difference in *where* the errors land. The price-trained models (`DDN_P`, `MLP_P`) have errors that scale with the absolute size of the option, which biases them toward in-the-money errors. `DDN_IV`'s error distribution is close to uniform across the domain.

### 5.3 Greek stability and extrapolation

Stable parameter sensitivities matter because a gradient-based optimizer relies on them to move through the loss landscape during calibration.

In the extreme tails, the differential networks produced noticeably smoother, more stable derivatives than COS, cutting down the high-frequency numerical artifacts. Under out-of-distribution (OOD) testing, `DDN_P` and `MLP_P` diverged and even returned negative Gamma (which is not valid), while `DDN_IV` stayed stable and generalized well beyond the training bounds.

### 5.4 Arbitrage compliance

`DDN_IV` had near-zero arbitrage violations across the empirical checks. Since it gets prices by pushing its predicted IV through Black–Scholes, it satisfies the theoretical bounds and monotonicity constraints by construction. It also learned a consistent volatility structure, with fewer calendar-spread and total-variance violations than COS.

---

## 6. Conclusions

Differential neural networks make strong surrogates for Heston pricing, and the output representation turns out to matter a lot. `DDN_IV`, trained on implied volatilities, was the best of the four: it avoids the numerical instabilities COS has in the tail regions, gives stable gradients, and stays close to arbitrage-free. That combination makes it the most robust and accurate of the surrogates we tested for gradient-based Heston calibration.

---

## 7. Quantitative Appendix and Supplementary Analysis

This section gives the quantitative detail behind the findings above, and tries to be clear about which results are well-supported empirically and which rest on assumptions.

### 7.1 Computational throughput and hardware assumptions

The throughput numbers show a large speedup for every surrogate relative to COS:

- `DDN_IV`: $1.23 \times 10^6$ samples/second, a 764× speedup over the COS stage-3 benchmark.
- `DDN_P` and `DDN_PdivK`: 687× and 664× over the COS price-and-Greek baseline.
- `MLP_P` (baseline): 323×.

One caveat worth being upfront about: network inference ran on an NVIDIA RTX 2070 GPU, while the COS benchmark ran on an Apple M2 CPU. So these exact multipliers shouldn't be read as universal — the speedup mixes algorithmic efficiency with hardware-specific parallelization, and on identical hardware the relative gap would likely look different. The $\mathcal{O}(1)$ inference cost of a trained network should still outscale the numerical integration COS needs, but the specific factor is hardware-dependent.

### 7.2 Error diagnostics and scale-dependence

Judging surrogates by MSE alone carries a hidden assumption: that every absolute error matters equally. Since option prices span several orders of magnitude across regimes, MSE on its own can misrepresent how useful a model is in deep out-of-the-money (OTM) states.

- Pricing MSE: `DDN_IV` reached $8.038 \times 10^{-9}$, ahead of `DDN_P` ($1.942 \times 10^{-8}$) and the `MLP_P` baseline ($1.760 \times 10^{-7}$). COS at $N=512$ expansion terms gave a higher MSE ($3.551 \times 10^{-6}$), mostly because of extreme outliers in the tails.
- IV mapping: by targeting IV directly, `DDN_IV` sidesteps the scale-dependence of raw prices and ends up with a much more uniform error distribution. The price-trained models (`MLP_P`, `DDN_P`) have errors that grow with the option price, which builds in a bias toward in-the-money contracts.

### 7.3 Gradient stability (Lipschitz analysis)

Gradient-based optimizers need smooth, stable parameter derivatives. We estimated the empirical Lipschitz constant $\hat{L}$ by Monte Carlo over 20,000 pairs.

- Tail behavior: the median Lipschitz ratios of the DDN models sat close to COS (ratios $\approx 1.0$), but their 99th-percentile ratios were much lower. `DDN_IV`, for example, gave an extreme-tail ratio of $0.4136$ for $\kappa$ (mean-reversion speed) and $0.5630$ for $\sigma$ (volatility of variance) relative to COS.
- A limit worth flagging: computing the exact global Lipschitz constant of a deep network is NP-hard, so the $\hat{L}$ values here are stochastic empirical lower bounds. The evidence that the DDNs damp high-frequency noise in the tails (getting past the low-vega singularities) is solid, but any claim about the *global* smoothness of the learned manifold stays theoretical.

### 7.4 Arbitrage compliance

A sound pricing surrogate shouldn't hand a calibration optimizer arbitrage opportunities to exploit.

- Price bounds: `DDN_IV` had $0.00\%$ bound breaches. The `MLP_P` baseline breached in $2.77\%$ of cases, and COS in $1.18\%$ (mostly from numerical truncation).
- Calendar spreads: `DDN_IV`'s violation rate was $0.31\% (\pm 0.084\%)$, well below COS ($1.21\%$).
- Why: because `DDN_IV` gets prices by running its predicted IV back through Black–Scholes, the output is bounded and monotonicity holds by construction.

### 7.5 Generalization and the synthetic-data assumption

The results make a clear case that `DDN_IV` beats the standard MLP and the semi-analytical baseline on speed and stability. But those results rest on one assumption about the data.

The networks were trained on synthetic data from Sobol sequences over wide theoretical bounds (for instance $\kappa \in [0.05, 5.0]$ and $\sigma \in [0.1, 1.5]$). They clearly learn this clean, noiseless space well. Whether that carries over to live market order books — which are noisy, discrete, and driven by microstructure — is still an open question. The real test would be running these surrogates inside a live, end-to-end gradient-based calibration loop.
