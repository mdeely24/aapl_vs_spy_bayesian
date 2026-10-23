
# Does Apple Beat the Market? A Bayesian Analysis of Monthly Excess Returns


## Question
1. In a given month, what is the probability that AAPL outperforms the S&P 500 (SPY)?
2. Is AAPL's average monthly excess return over SPY actually above zero?

A Bayesian approach answers both with probabilities and credible intervals rather than a single pass/fail p-value, which is useful here because the sample is small and returns are noisy.

## Data
- Source: Yahoo Finance via `yfinance` (dividend- and split-adjusted prices).
- Period: February 2015 to December 2025 (131 monthly returns).
- Monthly return = percent change in month-end adjusted close.
- Excess return = AAPL monthly return minus SPY monthly return.

## Models

### Model 1: Beta-Binomial (win probability)
Each month is a win (AAPL beats SPY) or a loss.
- Likelihood: `wins ~ Binomial(n, p)`
- Prior: `p ~ Beta(2, 2)`, weakly informative and centered on a coin flip.
- Verified against the exact conjugate posterior, Beta(2 + wins, 2 + losses).

### Model 2: Student-t (average excess return)
- Likelihood: `excess ~ StudentT(nu, mu, sigma)`, chosen to allow for fat tails.
- Priors (monthly decimal returns):
  - `mu ~ Normal(0, 0.02)`: skeptical of large alpha, but open to it.
  - `sigma ~ HalfNormal(0.05)`
  - `nu ~ Exponential(1/30)`
- Sampled with NUTS in PyMC (4 chains, 2,000 draws, 1,000 tuning steps, fixed seed).

## Results

| Quantity | Posterior mean | 94% credible interval | Key probability |
|---|---|---|---|
| P(AAPL beats SPY in a month) | 0.54 | [0.46, 0.62] | P(p > 0.5) = 0.83 |
| Mean monthly excess return (`mu`) | +0.91% | [-0.09%, +1.8%] | P(mu > 0) = 0.96 |

- AAPL beat SPY in 71 of 131 months (54.2%).
- The posterior mean excess return is roughly 11% per year (simple annualization, 12 x monthly).
- Monthly excess-return volatility is about 5.6%, which is why the intervals are wide.
- The fitted degrees of freedom (`nu`, posterior mean about 31) suggest tails are only mildly heavier than normal in this sample.

**Interpretation:** The data lean toward AAPL outperforming, but not decisively. There is about an 83% chance that AAPL's true monthly win rate is above 50%, and about a 96% chance its mean excess return is positive, but the 94% credible interval for `mu` just includes zero and the win-rate interval comfortably includes 0.5.

## Validation
- **Convergence:** R-hat = 1.00, large effective sample sizes, and no divergences (sampler run with `target_accept=0.95`).
- **Prior predictive check:** the priors imply monthly returns in a plausible range (roughly -20% to +20% at the 1st and 99th percentiles).
- **Posterior predictive check:** simulated data closely match the observed distribution of excess returns.
- **Prior sensitivity:** P(mu > 0) was 0.90, 0.97 and 0.97 under priors on `mu` with standard deviation 0.5%, 2% and 10%. The conclusion is stable under moderately wide priors and weaker only under a very tight, skeptical prior.
- **Frequentist comparison:** a one-sample t-test gives t = 1.79, two-sided p = 0.076 (one-sided about 0.04), consistent with the Bayesian result. The two approaches answer different questions, so the numbers are not directly interchangeable.

## Limitations
- **Selection bias:** AAPL was chosen in advance as a well-known strong performer, so these results likely overstate what a randomly chosen stock would show.
- **Not risk-adjusted:** excess return versus SPY ignores that AAPL is more volatile and has a higher market beta. Part of the outperformance may be compensation for risk.
- **Small sample:** 131 monthly observations give wide posteriors.
- **Descriptive, not predictive:** this estimates past relative performance and does not forecast future returns.
- **Simplifications:** returns are treated as independent and identically distributed; no time variation or autocorrelation is modeled.

## Possible extensions
- Bayesian CAPM regression (`alpha` and `beta` against SPY) for a risk-adjusted estimate.
- Hierarchical model across many stocks with partial pooling, to reduce selection bias.

## How to run
1. Open https://colab.research.google.com/drive/1ug1SByfaYJVVXwOL8VojxcIAHj-ZD45t?usp=sharing in Google Colab.
2. Run the first cell to install dependencies: `pymc`, `arviz`, `yfinance`.
3. Run all cells. Sampling takes a few minutes.

Dependencies: `pymc`, `arviz`, `yfinance`, `numpy`, `pandas`, `matplotlib`, `scipy`.
