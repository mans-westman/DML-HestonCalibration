# Project Information: Neural Network Surrogates for Optimizing Heston Model Calibration

## 1. Project Overview

* **Title:** Neural Network Surrogates for Optimizing Heston Model Calibration: A Comparative Study on Design Choices for Deep Differential Pricing Surrogates.


* **Authors:** Alex Andersson, Linnea Davidson, Markus Eliassen, Jacob Hansén, Isak Stridh, Måns Westman.


* **Primary Objective:** To investigate if neural network surrogate models can match the accuracy and stability of semi-analytical option pricing while offering substantial computational speedups for Heston model calibration.


* **Secondary Objective:** To examine how neural network design choices, particularly output representations, affect pricing performance, computational efficiency, and derivative stability compared to standard non-differential networks and the COS method benchmark.



---

## 2. Background and Problem Statement

Calibrating stochastic volatility models, such as the Heston model, requires finding parameters that allow the model to accurately reproduce observed market option prices.

* Because the Heston model includes a stochastic variance feature, there is no closed-form solution for option prices in the original price domain.


* Consequently, calibration relies heavily on numerical evaluation methods, such as the Fourier-based COS method, to evaluate semi-analytical pricing formulas across thousands of option strikes and maturities.


* This process is computationally expensive and slow. In practical settings requiring high computational speed—such as real-time risk management and high-frequency trading—this bottleneck severely limits the utility of traditional methods, driving the need for highly accurate, accelerated surrogate models.



---

## 3. Methodology and Data Pipeline

To train and evaluate the surrogate models, a comprehensive synthetic dataset was generated across a wide parameter space representing various market regimes.

* **Sampling Strategy:** Synthetic input parameters were generated using Sobol sequencing to prevent the curse of dimensionality and ensure uniform coverage of the multidimensional parameter space.


* **Parameter Bounds:** The dataset sampled Heston parameters—mean reversion speed $\kappa$, long-term mean-variance $\theta$, volatility of variance $\sigma$, correlation $\rho$, and initial variance $v_{0}$—alongside market state variables including time to maturity, risk-free rate, and log-moneyness.


* **Ground Truth Generation:** Option prices and exact parameter derivatives (Greeks) were computed via the semi-analytical COS method. The COS algorithm was implemented using the JAX Python library to obtain analytically exact parameter Greeks via automatic differentiation.


* **Implied Volatility Transformation:** Corresponding implied volatilities (IV) and IV Greeks were computed using the "Let's Be Rational" algorithm, which robustly inverts the Black-Scholes formula without the divergence risks common to standard Newton-Raphson methods.



---

## 4. Evaluated Surrogate Architectures

The project implemented four distinct neural network architectures to test the impact of differential machine learning (DML) and output representation. DML improves network training by incorporating the derivatives of the outputs directly into the loss function (Sobolev training), acting as a regularizer.

* **MLP_P (Baseline):** A standard multilayer perceptron trained purely on option prices using standard supervised learning without parameter Greeks.


* **DDN_P:** A Deep Differential Network trained on raw option prices and their corresponding parameter Greeks.


* **DDN_PdivK:** A Deep Differential Network trained on option prices normalized by the strike price ($P/K$) and their corresponding Greeks. This division flattens the vast absolute price scale and bounds the target variable strictly between 0 and 1.


* **DDN_IV:** A Deep Differential Network trained on Implied Volatilities (IV) and IV Greeks. This targets the underlying volatility structure directly, bypassing the large scale differences between deep in-the-money and out-of-the-money options.



*Network specifics:* All networks utilized SiLU activation functions in hidden layers to ensure continuous differentiability and Softplus in the output layer to enforce strictly non-negative predictions.

---

## 5. Core Findings and Results

The models were benchmarked against the COS method on a separate test set, evaluating computational speed, pricing precision, derivative stability, and no-arbitrage compliance.

### 5.1. Computational Efficiency

* All tested neural network architectures achieved computational speedups of two to three orders of magnitude compared to the COS benchmark.


* The `DDN_IV` model recorded a 764 times higher throughput across the full IV and IV-Greeks pipeline compared to the semi-analytical baseline.


* Batch size scaling demonstrated that surrogate throughput increased substantially when executing large parallelized batches on GPU hardware.



### 5.2. Pricing and Prediction Accuracy

* The `DDN_IV` model demonstrated the highest precision, outperforming both the standard COS benchmark and all other neural network variants on Mean Squared Error (MSE) and Mean Absolute Error (MAE) for nearly all targets.


* The `MLP_P` baseline exhibited the worst accuracy among the networks, confirming that standard supervised learning struggles to map the complex dynamics of stochastic volatility models without the regularization provided by differential training.


* Heatmap analysis revealed that price-trained models (`DDN_P`, `MLP_P`) produced errors that scaled proportionally with the absolute magnitude of the options, creating a bias toward in-the-money errors. The `DDN_IV` model produced a highly homogeneous error distribution across the entire domain.



### 5.3. Greek Stability and Extrapolation

* Stable parameter sensitivities are critical for gradient-based optimizers to successfully traverse the loss landscape during calibration.


* The differential networks (`DDN`) produced significantly smoother and more stable parameter derivatives in the extreme tails of the distribution compared to the COS benchmark, successfully reducing high-frequency numerical artifacts.


* While `DDN_P` and `MLP_P` severely diverged and produced theoretically invalid negative Gamma values in out-of-distribution (OOD) testing, `DDN_IV` maintained exceptional extrapolative stability and generalized well outside the explicit training bounds.



### 5.4. Arbitrage Compliance

* `DDN_IV` produced near-zero arbitrage violations across all empirical checks.


* Because `DDN_IV` derives output prices by passing its predicted IV through the Black-Scholes formula, it inherently satisfies theoretical bounds and monotonicity constraints by construction.


* Furthermore, `DDN_IV` successfully learned a coherent volatility structure, exhibiting significantly fewer calendar spread and total-variance violations than the COS method.



---

## 6. Conclusions

The study concludes that differential neural networks are highly capable surrogates for Heston model pricing. Specifically, the choice of the output representation is highly influential. The `DDN_IV` model, trained on implied volatilities, proved to be the superior architecture. By bypassing the numerical instabilities inherent to the COS method's tail regions and providing reliable, stable gradient updates alongside near-perfect arbitrage compliance, `DDN_IV` stands out as the most robust and accurate surrogate model for optimizing gradient-based Heston calibration.

## 7. Quantitative Appendix and Supplementary Analysis

The following section provides a rigorous quantitative breakdown of the core findings, explicitly delineating between well-supported empirical results and the mathematical assumptions inherent to the methodology.

### 7.1. Computational Throughput and Hardware Assumptions

The reported throughput highlights a dramatic acceleration for all surrogate models relative to the semi-analytical baseline.

* **DDN_IV** achieved a throughput of $1.23 \times 10^6$ samples/second (a 764x speedup over the COS stage-3 benchmark).
* **DDN_P** and **DDN_PdivK** achieved speedups of 687x and 664x, respectively, over the COS price and Greek baseline.
* **MLP_P** (Baseline) achieved a 323x speedup.

*Methodological Caveat & Hardware Assumption:* It is necessary to state explicitly that the neural network inference was conducted on an NVIDIA RTX 2070 GPU, whereas the COS benchmark was executed on an Apple M2 CPU. The assumption that these specific multiplier metrics translate directly to all production environments is speculative. The observed acceleration is a combination of algorithmic efficiency and hardware-specific parallelization. In a purely homogeneous hardware setting, the relative magnitude of this speedup may differ, although the $\mathcal{O}(1)$ inference complexity of a trained neural network will fundamentally outscale the numerical integration required by the COS method.

### 7.2. Error Diagnostics and Scale-Dependence

Evaluating surrogate models strictly via Mean Squared Error (MSE) introduces a hidden assumption: that all absolute errors are equally financially meaningful. Because option prices span multiple orders of magnitude across different market regimes, relying solely on MSE can misrepresent model utility in deep out-of-the-money (OTM) states.

* **Pricing MSE:** DDN_IV achieved a pricing MSE of $8.038 \times 10^{-9}$, outperforming DDN_P ($1.942 \times 10^{-8}$) and the MLP_P baseline ($1.760 \times 10^{-7}$). The COS benchmark evaluated at $N=512$ expansion terms yielded a higher MSE ($3.551 \times 10^{-6}$), driven largely by extreme numerical outliers in the tail regimes.
* **Implied Volatility Mapping:** DDN_IV’s direct targeting of IV bypasses the scale-dependence of raw prices. By flattening the vast absolute price scale, it achieves a highly homogeneous error distribution. Models trained directly on option prices (MLP_P, DDN_P) exhibit errors that scale proportionally with the absolute magnitude of the option price, generating an inherent bias toward in-the-money contracts.

### 7.3. Gradient Stability (Lipschitz Analytics)

Gradient-based optimizers require smooth, stable parameter derivatives to reliably traverse the loss landscape. The empirical Lipschitz constant ($\hat{L}$) was evaluated via Monte Carlo sampling across $20,000$ pairs to measure this stability.

* **Tail Stabilization:** While the median Lipschitz ratios of the DDN models closely tracked the COS benchmark (ratios $\approx 1.0$), their 99th percentile ratios were significantly lower. For instance, DDN_IV yielded an extreme-tail ratio of $0.4136$ for $\kappa$ (mean-reversion speed) and $0.5630$ for $\sigma$ (volatility of variance) compared to COS.
* *Uncertainty in Global Bounds:* It must be noted that computing the exact global Lipschitz constant for deep neural networks is an NP-hard problem. The $\hat{L}$ values computed here are stochastic empirical lower bounds. While it is well-supported that the DDN architectures dampen high-frequency numerical noise in the extreme tails (effectively bypassing low-Vega singularities), any absolute claims regarding the *global* smoothness of the learned manifold remain theoretical rather than proven certainties.

### 7.4. Arbitrage Compliance

A theoretically sound pricing surrogate must not introduce arbitrage opportunities that a calibration optimizer might exploit.

* **Price Bounds:** DDN_IV generated $0.00\%$ total bound breaches. The MLP_P baseline breached bounds in $2.77\%$ of cases, and the COS benchmark generated $1.18\%$ bound breaches (principally driven by numerical truncation artifacts).
* **Calendar Spreads:** DDN_IV exhibited a calendar spread violation rate of just $0.31\% (\pm 0.084\%)$, significantly outperforming COS ($1.21\%$).
* *Structural Advantage:* DDN_IV achieves this near-perfect compliance because its output (IV) is passed back through the Black-Scholes formula to obtain prices. This mathematically bounds the predictions and structurally enforces monotonicity by construction.

### 7.5. Generalization vs. The Synthetic Data Assumption

While the empirical results firmly establish the superiority of the DDN_IV architecture over standard MLPs and semi-analytical baselines in terms of speed and stability, these conclusions rest on a foundational assumption regarding data generation.

The networks were trained on a synthetic dataset constructed via Sobol sequencing (spanning wide theoretical bounds such as $\kappa \in [0.05, 5.0]$ and $\sigma \in [0.1, 1.5]$). It is well-supported that the DDN surrogates map this idealized, noiseless parameter space effectively. However, the assumption that this performance will generalize seamlessly when exposed to the noisy, discrete, and micro-structure-driven realities of live market order books remains speculative. The true robustness of these deep differential surrogates must ultimately be verified by deploying them within a live, end-to-end gradient-based calibration loop.