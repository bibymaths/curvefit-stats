# Kinetics and asymmetric errors

Michaelis–Menten kinetics describes reaction velocity at substrate concentration
$S$:

$$
v(S)=\frac{V_{\max}S}{K_m+S}.
$$

Use $S$ and $K_m$ in mM and $v$ and $V_{\max}$ in µmol/min. Low-substrate data
mainly inform $V_{\max}/K_m$; high correlation is a local experimental-design
diagnostic, not proof of structural non-identifiability.

## Fit, profile and resample

This experiment has IID Gaussian errors with an unknown common variance.
Each block continues the previous one:

```python
import numpy as np
from scipy.optimize import curve_fit
from curvefit_stats import (
    bootstrap_curve_fit, collinearity_diagnostics,
    fit_summary, profile_likelihood_errors,
)

def kinetics(substrate, vmax, km):
    return vmax * substrate / (km + substrate)

substrate = np.geomspace(0.15, 8, 35)
rng = np.random.default_rng(314)
velocity = kinetics(substrate, 3.0, 1.2) + rng.normal(0, 0.06, substrate.size)
bounds = ([0.001, 0.001], [np.inf, np.inf])
popt, pcov = curve_fit(kinetics, substrate, velocity, [2.5, 1.0], bounds=bounds)
print(fit_summary(kinetics, substrate, velocity, popt, pcov))
diagnostics = collinearity_diagnostics(pcov)
assert diagnostics["rank"] == 2

errors = profile_likelihood_errors(
    kinetics, substrate, velocity, popt, bounds=bounds, pcov=pcov,
    sigma_level=1.95996398454,
)
profile_endpoints = np.array([
    [estimate + error["down"], estimate + error["up"]]
    for estimate, error in zip(popt, errors, strict=True)
])
assert np.isfinite(profile_endpoints).all()
print(profile_endpoints)
```

With an unknown common scale, the profile statistic is
$N\log(Q_{\mathrm{profile}}/Q_{\mathrm{best}})$, not an SSE difference.
The threshold is `sigma_level**2`, giving approximately 95% one-parameter
coverage here under regular Wilks assumptions. Nuisance parameters are
reoptimized at every trial. This is not an exact Student-t interval; bounds,
small samples, weak identification and local minima can invalidate calibration.
An unresolved endpoint is a warned NaN, not a bound reported as a crossing.

```python
mean, covariance, samples = bootstrap_curve_fit(
    kinetics, substrate, velocity, p0=popt,
    bounds=bounds, n_resamples=150, random_state=2718,
)
percentile_endpoints = np.quantile(samples, [0.025, 0.975], axis=0).T
assert covariance.shape == (2, 2)
assert samples.shape == (150, 2)
print(percentile_endpoints)

efficiency_samples = samples[:, 0] / samples[:, 1]
efficiency_interval = np.quantile(efficiency_samples, [0.025, 0.975])
assert efficiency_interval[0] < efficiency_interval[1]
```

Bootstrap `mean` is the empirical refit mean, not the original optimum. Its
covariance uses successful samples and divisor B_success-1. In this seeded
well-conditioned example every attempt succeeds; in general inspect the
successful count and warnings. Failed-fit selection can bias the result.
Residual bootstrap assumes IID observation errors at fixed predictors and
does not apply a leverage correction or support weighted/correlated errors.

The empirical efficiency interval retains the sampled dependence between
parameters. Treating the bootstrap covariance as Gaussian is a further
approximation that loses asymmetry and tails. Compare methods by their
assumptions, not by expecting identical endpoint numbers.

See [profile assumptions](../user-guide.md#asymmetric-profile-likelihood),
[bootstrap limitations](../user-guide.md#residual-bootstrap-covariance) and
[local identifiability](../user-guide.md#collinearity-and-local-identifiability).
