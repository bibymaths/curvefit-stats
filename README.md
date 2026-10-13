# curvefit-stats

[Documentation website](https://bibymaths.github.io/curvefit-stats/) ·
[Quickstart](docs/quickstart.md) · [Build the docs](docs/documentation.md)

Statistical inference for SciPy-style nonlinear models, plus shared-parameter
fitting and errors-in-variables regression. Use ordinary functions written as
`model(x, *parameters)` with NumPy arrays.

The package contains exactly two functional modules:

- `curvefit_stats.inference`: bands, likelihood diagnostics, parameter summaries,
  profile likelihood, bootstrap covariance, collinearity diagnostics,
  bound-aware covariance, and derived-quantity propagation.
- `curvefit_stats.estimators`: `GlobalFitGroup` and `eiv_curve_fit`.

All ten features are also available directly from `curvefit_stats`; these imports
refer to the same objects, without wrappers.

## Run from this checkout

Runtime dependencies are **NumPy >=1.26.4,<3 and SciPy >=1.12,<1.19**,
plus Python's standard library.
The examples require no plotting package, downloaded data, or notebook runtime.
Python 3.11–3.14 is tested. Install locally from this checkout; no PyPI
publication is implied by these instructions.

```bash
python -m venv .venv
# Activate .venv (Windows: .venv\Scripts\activate).
source .venv/bin/activate
python -m pip install .
python examples/workflows.py
python examples/workflows.py profile
```

The SciPy upper limit applies to the EIV backend: `scipy.odr` is deprecated in
SciPy 1.17 and removed in 1.19. EIV calls on deprecated versions emit a
`FutureWarning`; calls outside the supported range fail explicitly. See the
[ODR compatibility notes](docs/user-guide.md#errors-in-both-axes).

## Quick start: concentration decay

Fit concentration in mg/L against time in hours with
`C(t) = amplitude * exp(-rate * t)`. This example assumes independent Gaussian
measurement errors with an unknown common variance.

```python
import numpy as np
from scipy.optimize import curve_fit
from curvefit_stats import generate_error_bands, fit_summary, audit_fit


def decay(time, amplitude, rate):
    return amplitude * np.exp(-rate * time)


rng = np.random.default_rng(2026)
time = np.linspace(0, 5, 45)
concentration = decay(time, 3.0, 0.7) + rng.normal(0, 0.08, time.size)
popt, pcov = curve_fit(decay, time, concentration, p0=[2.5, 0.5])

grid = np.linspace(0, 5, 101)
lower, upper = generate_error_bands(
    decay, grid, popt, pcov, concentration, x_fit=time,
)
print(fit_summary(decay, time, concentration, popt, pcov))
print(audit_fit(decay, time, concentration, popt))
```

The bands have approximate **pointwise 95% confidence coverage for the mean**.
They do not cover an entire curve simultaneously or include the noise of a new
measurement. The original fitting sample determines degrees of freedom, not
`grid`. For known absolute Gaussian errors, fit with `absolute_sigma=True` and
choose `reference="normal"` for bands.

## Eleven executable workflows

[examples/workflows.py](examples/workflows.py) provides seeded, independently
selectable examples. Run `PYTHONPATH=src python examples/workflows.py NAME`.

| Name | Workflow | Main feature |
| --- | --- | --- |
| `confidence` | Nonlinear decay fitting and mean-concentration bands | `generate_error_bands` |
| `intervals` | Mean confidence versus future-observation prediction | `generate_error_bands` |
| `diagnostics` | Gaussian fit diagnostics and AIC/BIC comparison | `audit_fit` |
| `summary` | R-inspired approximate nonlinear parameter table | `fit_summary` |
| `profile` | Asymmetric Michaelis–Menten profile errors | `profile_likelihood_errors` |
| `bootstrap` | Residual-bootstrap covariance and percentile interval | `bootstrap_curve_fit` |
| `collinearity` | Unsaturated kinetics and local parameter dependence | `collinearity_diagnostics` |
| `bounds` | Decay baseline at an active lower bound | `robust_bounded_pcov` |
| `propagation` | Concentration and half-life uncertainty | `propagate_uncertainty` |
| `global` | Shared decay rate, separate initial concentrations | `GlobalFitGroup` |
| `eiv` | Uncertain time and concentration measurements | `eiv_curve_fit` |

The collinearity and bounds examples intentionally emit statistical warnings;
EIV also emits its compatibility warning on SciPy 1.17–1.18. Numerical output can
vary slightly across NumPy/SciPy versions. Fixed seeds make each experiment
reproducible within an environment, without requiring bit-for-bit agreement
across platforms.

## Choose the statistical convention explicitly

`sigma` supplies standard deviations, not variances, for scalar/vector errors.
A two-dimensional `sigma` supplies the full observation covariance. With
`absolute_sigma=True`, these are known absolute error scales. With `False`, they
supply relative weighting, and one unknown common variance multiplier is fitted.
These choices affect covariance scaling and likelihood parameter counting.

A covariance array does not retain its estimator, noise model, constraints, or
reference distribution. Preserve that context. Ordinary response-only
likelihood methods cannot automatically become ODR methods; conditional
covariance at active bounds is not full marginal uncertainty; and empirical
bootstrap covariance does not establish a Student-t pivot.

Read the [user guide](docs/user-guide.md) for equations, assumptions, failure
handling, and workflow choices. The
[API reference](https://bibymaths.github.io/curvefit-stats/reference/) is generated
directly from Python source and NumPy-style docstrings on every site build. A
[generated offline snapshot](docs/api.md) is retained for repository readers. The
[integration guide](docs/integration.md) explains covariance projection between
modules, and the [stress-test report](docs/stress-testing.md) records independent
numerical validation and remaining scientific limitations.

## Reproducible development and packaging

The [development guide](docs/development.md) describes the hash-locked Python
3.12 development environment, strict quality checks, compatibility matrix,
wheel/source builds, and isolated installation checks. `py.typed` is included
in both distributions. Development tools live in optional extras and are not
base runtime dependencies. CI tests built wheels and direct source-distribution
installation; it does not publish releases.

The [pre-release scientific review](docs/review.md) records independently
reproduced defects, regression fixes, scientific limitations, and release gates.
