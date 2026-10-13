# Combining estimators and inference

All ten features are available from `curvefit_stats`. Direct imports from
`curvefit_stats.inference` and `curvefit_stats.estimators` are identical objects,
with identical signatures. There are no other functional submodules.

## SciPy curve fitting

For independent homoscedastic Gaussian residuals with unknown variance:

```python
import numpy as np
from scipy.optimize import curve_fit
from curvefit_stats import audit_fit, fit_summary, generate_error_bands

model = lambda x, slope, intercept: slope*x + intercept
x = np.linspace(0, 4, 30)
y = model(x, 2, 1) + np.random.default_rng(4).normal(0, 0.1, x.size)
popt, pcov = curve_fit(model, x, y)
summary = fit_summary(model, x, y, popt, pcov)
audit = audit_fit(model, x, y, popt)
grid = np.linspace(0, 5, 80)
lower, upper = generate_error_bands(model, grid, popt, pcov, y, x_fit=x)
```

The bands are pointwise, not simultaneous. `x_fit` and `y` determine residual
degrees of freedom; the prediction grid does not. Residual-based prediction
intervals assume independent common-variance noise. For known absolute Gaussian
variances, choose `reference="normal"`. `fit_summary` uses approximate Student-t
Wald statistics; it has no automatic known-scale or estimator selection.

## Global fits

Use the actual global parameter count, not the number of local model arguments.
For example, one shared slope and two local intercepts estimate three parameters
across both datasets.

```python
from curvefit_stats import GlobalFitGroup, generate_error_bands

x_b = np.linspace(0, 5, 37)
y_b = model(x_b, 2, 2) + np.random.default_rng(6).normal(0, 0.1, x_b.size)
group = GlobalFitGroup(model, {
    "a": {"x": x, "y": y},
    "b": {"x": x_b, "y": y_b},
}, shared_parameters={"slope"})
result = group.fit([1, 0, 0])
indices = group.parameter_indices["a"]
local_popt = result.popt[indices]
local_pcov = result.pcov[np.ix_(indices, indices)]
lower, upper = generate_error_bands(
    model, grid, local_popt, local_pcov, dof=result.diagnostics.dof,
)
```

Pass the explicit global `dof` with no per-dataset `y`. Supplying local `y` would
assert an ordinary fit to that dataset with local `N-p`; mismatched explicit
values are rejected. Shared aliases may repeat indices: projection must retain
those repetitions because they encode tied local parameters.

For homoscedastic unweighted prediction, supply the global residual variance as
`prediction_variance=result.diagnostics.weighted_sse/result.diagnostics.dof`.
For relative heteroscedastic weights, multiply that global scale by each future
observation's specified variance. For known scales, use the specified future
variance and normal-reference bands. Correlated noise and dependence between
future observations and fitted parameters need an explicit supported model.

`audit_fit` and `fit_summary` can analyze the complete joint response-only model
if its parameter vector and observations contain all global parameters and all
datasets. Per-dataset audit statistics use local counts and are not global AIC,
BIC, or reduced chi-square. Different independently estimated residual scales
per dataset and cross-dataset error covariance are not implemented.

## Errors in both axes

```python
from curvefit_stats import eiv_curve_fit, propagate_uncertainty

x_observed = x + np.random.default_rng(5).normal(0, 0.03, x.size)
popt, pcov, odr = eiv_curve_fit(
    model, x_observed, y, [2, 1], sx=0.03, sy=0.1,
    absolute_sigma=True, full_output=True,
)
uncertainty = propagate_uncertainty(
    lambda slope, intercept: slope + intercept,
    popt, pcov, random_state=7,
)
```

ODR covariance accounts for fitted latent coordinates. For known absolute error
scales, `absolute_sigma=True` returns unscaled `cov_beta`; otherwise it returns
`cov_beta * res_var`. The latter assumes one common unknown multiplier for the
specified variances of both axes. Omitting `sx` fixes all predictors exactly.

For mean predictions at *exact* evaluation coordinates, Delta Method bands can
use ODR covariance (`reference="normal"` for known scales). This does not include
uncertainty in evaluation coordinates. Future uncertain x and y require a
separate observation model; response-only residual variance is not a valid
substitute. Gaussian parameter propagation is only a local approximation.

`audit_fit` evaluates a vertical-response Gaussian likelihood, not the ODR
objective; its AIC/BIC must not be presented as ODR likelihood criteria.
Likewise, `profile_likelihood_errors`, `bootstrap_curve_fit`, and
`robust_bounded_pcov` reoptimize response-only fits, not latent-x ODR fits.
`fit_summary` does not automatically select an EIV reference distribution.
These mismatches cannot be detected from plain arrays, so retain estimator
metadata and choose the correct method rather than mixing objectives.

SciPy ODR is supported on versions 1.12–1.18, deprecated from 1.17, and removed
in 1.19. EIV calls warn on deprecated versions and reject unsupported versions.

## Bootstrap and profile intervals

Bootstrap mean/covariance can initialize Gaussian derived-quantity propagation,
but that Gaussian approximation discards asymmetry and heavy tails. To obtain a
bootstrap percentile interval, evaluate the derived function on the returned
parameter samples and take empirical quantiles directly. Residual resampling
requires exchangeable errors; weighting and correlated residuals are unsupported.
A normal denominator allows ratios with undefined population moments even when
finite Monte Carlo samples yield finite summary statistics.

Profile likelihood and Wald covariance intervals use different reference laws.
Known-scale regular linear models agree at a common normal-equivalent coverage.
For unknown scale, profile likelihood uses `N*log(SSE_profile/SSE_best)` and a
chi-square likelihood-ratio threshold; it is not an exact Student-t interval.
Near bounds or strong nonlinearity, disagreement can be scientifically expected.
Unresolved/bound-limited profile endpoints are warned `NaN` values.

## Active bounds and singularity

`robust_bounded_pcov` and bound-active `GlobalFitGroup` results return `NaN` active
rows/columns and conditional covariance on the free block. Bands and Monte Carlo
propagation reject the full non-finite covariance. Collinearity diagnostics warn
that full covariance is unreliable; they do not fill unknown active uncertainty
with zeros. To analyze the finite free block, hold active parameters fixed in the
model or derived function and use the conditional covariance degrees of freedom.
Label those results as conditional, not full parameter confidence statements.

A full-rank Jacobian does not prove structural identifiability, solver convergence
does not establish a global optimum, and covariance correlation flags describe
local dependence. Rank-deficient covariance remains undefined rather than being
silently replaced with a finite pseudoinverse.
