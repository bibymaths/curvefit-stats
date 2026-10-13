# Errors in both axes

An uncertain sampling time changes the statistical model: vertical residual
least squares alone treats the observed predictor as exact. EIV fits latent
adjusted coordinates with independent uncertainties on both axes.

$$
Q_{\mathrm{ODR}}=\sum_i (\delta_i/s_{x,i})^2
                 +\sum_i (\varepsilon_i/s_{y,i})^2,
$$

where $\delta$ is adjusted minus observed x and $\varepsilon$ is fitted minus
observed y. The predictor term vanishes in exact-x mode (`sx=None`).

## Fit and inspect the backend result

```python
import warnings
import numpy as np
from scipy.optimize import curve_fit
from curvefit_stats import eiv_curve_fit, generate_error_bands

def decay(time, amplitude, rate):
    return amplitude * np.exp(-rate * time)

rng = np.random.default_rng(82)
true_time = np.linspace(0, 4, 70)
sx, sy = 0.15, 0.08
observed_time = true_time + rng.normal(0, sx, true_time.size)
concentration = decay(true_time, 3.0, 0.7) + rng.normal(0, sy, true_time.size)
ordinary, _ = curve_fit(decay, observed_time, concentration, [2.5, 0.5], sigma=sy)
with warnings.catch_warnings():
    warnings.filterwarnings("default", message="scipy.odr is deprecated", category=FutureWarning)
    popt, pcov, diagnostic = eiv_curve_fit(
        decay, observed_time, concentration, [2.5, 0.5],
        sx=sx, sy=sy, absolute_sigma=True, full_output=True,
    )
assert diagnostic.success and diagnostic.identifiable
assert pcov.shape == (2, 2)
print("Vertical LS:", ordinary, "EIV:", popt)
print(diagnostic.stopreason)

# Coordinates here are exact evaluation points, not noisy future measurements.
grid = np.linspace(0, 4, 80)
lower, upper = generate_error_bands(decay, grid, popt, pcov, reference="normal")
assert np.isfinite(lower).all() and np.isfinite(upper).all()
```

This single experiment illustrates the distinction; it does not prove that EIV
universally outperforms ordinary least squares. The internal adapter accepts
ordinary `func(x, *parameters)` models and handles ODR's parameter-first convention.

## Covariance and failure semantics

`absolute_sigma=True` returns ODR's unscaled `cov_beta`. Relative errors return
`cov_beta * res_var`, with identified `res_var=Q_ODR/(N-p)` and one common unknown
multiplier for both axis variances. Delta bands at exact coordinates are a local
approximation; future uncertain x requires an additional observation model.

By default, non-convergence raises `RuntimeError`. With `full_output=True`, the
function can return current estimates with a failure diagnostic and **NaN
covariance**, plus an explicit warning. Never proceed to uncertainty propagation
without checking convergence and covariance validity.

!!! warning "Backend and statistical limits"

    Supported SciPy is >=1.12,<1.19. ODR is deprecated from 1.17 and removed in
    1.19. One predictor/response and scalar/vector axis errors are supported;
    bounds, correlated axis errors, fixed parameters and analytic ODR Jacobians
    are not. Unsupported `curve_fit` keywords are rejected, not silently ignored.

Response-only `audit_fit`, profile likelihood, bootstrap, bounded covariance and
Student-t summaries are not ODR methods. Feeding them ODR parameters does not
change their likelihood or calibration. Read [EIV integration](../integration.md#errors-in-both-axes)
and [`eiv_curve_fit`][curvefit_stats.estimators.eiv_curve_fit] for the full contract.
