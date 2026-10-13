# Final release-readiness verification

Verified locally on 2026-10-13 for version 0.1.0. **Recommendation: NOT READY
for public release until the configured GitHub Actions workflow passes,
including Windows and macOS.** All executed Linux checks passed. No new
release-blocking implementation defect was reproduced in this final pass.
This report records finite benchmark evidence, not a guarantee for arbitrary
models, callbacks, or numerical platforms.

## 1. Feature-completion matrix

| Feature | Module | Completion and independent validation |
| --- | --- | --- |
| `generate_error_bands` | inference | Complete; analytical Delta Jacobians, pointwise confidence/prediction coverage, normal versus Student-t calibration, output grids, covariance scaling and underflow. |
| `audit_fit` | inference | Complete; normalized known/unknown-scale Gaussian likelihoods, weighted covariance determinants, AIC/BIC parameter counts, residual dof, zero variance and redundant parameters. |
| `fit_summary` | inference | Complete; parameter standard errors, Student-t tail probabilities and intervals; undefined inference for singular covariance or model Jacobians. |
| `profile_likelihood_errors` | inference | Complete; independent variable-projection endpoints, nuisance reoptimization, known/unknown variance thresholds, asymmetry, bounds and unresolved endpoints. |
| `bootstrap_curve_fit` | inference | Complete; independent residual resampling/refits, empirical covariance, controlled seeds, fixed attempts, failed/invalid replicate accounting and response scaling. |
| `collinearity_diagnostics` | inference | Complete; independently calculated correlation, inverse-correlation diagonals, rank and conditioning; singular and invalid covariance cases. |
| `robust_bounded_pcov` | inference | Complete; independent free-Jacobian information, active lower/upper/fixed bounds, relative variance scaling and rank failure. |
| `propagate_uncertainty` | inference | Complete; Gaussian linear transformations, independent ratio integration, seeded Monte Carlo tolerances, singular support, invalid output accounting and large offsets. |
| `GlobalFitGroup` | estimators | Complete; independent joint weighted regression, unequal grids/lengths, shared/tied degrees of freedom, joint Jacobian, bounds, scaling, rank and solver failures. |
| `eiv_curve_fit` | estimators | Complete; Deming regression, explicit latent-coordinate optimization and covariance, negligible/exact x-error limits, ODR scaling, backend controls and non-convergence. |

## 2. Public API and architecture

There are exactly two functional submodules: `curvefit_stats.inference` and
`curvefit_stats.estimators`. `__init__.py` re-exports the same ten objects without
wrappers. Direct module imports remain available. There are no additional
functional modules in the source or wheel. Typed result dataclasses remain
inside `estimators.py`; `py.typed` is included in both distribution types.

The signature/docstring consistency tests passed for all ten root exports and
`GlobalFitGroup.fit`. No signature or runtime source changed in this final pass.
The exact signatures, annotations, NumPy-style Parameters/Returns/Raises/Notes/
Examples sections and return shapes are in [the API reference](api.md).

Bands return lower/upper arrays; audit and collinearity return metric dictionaries;
summary returns a table string; profiles return per-parameter error dictionaries;
bootstrap returns mean parameters, empirical covariance and successful samples;
bounded covariance returns a parameter covariance matrix; propagation returns
sample summaries. `GlobalFitGroup(...).fit(...)` returns a typed joint-fit result.
EIV returns `(popt, pcov)`, or a third typed diagnostic result with
`full_output=True`.

Runtime source import inspection and built metadata confirm only the standard
library, NumPy and SciPy. Ruff found no unnecessary imports. Source inspection
found no unfinished implementation, debug statements or duplicate public
implementations. Ellipsis-only bodies are intentional typing overloads for EIV.

## 3. Tests, coverage and quality checks

The hash-locked Linux Python 3.12 development environment passed **405 tests**
with **95.27% combined statement/branch coverage**, exceeding the 90% gate.
The five installed-wheel suites below each passed all 405 tests, with no skips
or expected failures. All use strict pytest configuration and warning handling.

- Ruff lint passed; formatting passed for all 15 Python files.
- Strict mypy passed for all 15 source/test/example/tool files.
- Thirteen doctests passed in the source environment and each wheel environment.
  One explicitly expected bound-active conditional-covariance warning was emitted.
- All eleven executable workflows passed, as did README, user-guide and integration
  Python snippets. Tests distinguish narrowly expected warnings from failures.
- Actionlint 1.7.7 and `git diff --check` passed. This validates workflow syntax;
  it does not establish successful execution on GitHub's runners.
- Isolated PEP 517 sdist and wheel builds passed. Strict Twine validation and
  archive-content/metadata checks passed. The wheel was built from the sdist.
- Isolated installed-package imports and all-ten-feature smoke checks passed in
  each Linux wheel environment and separate runtime-only wheel/sdist environments.
  Direct sdist installation used an isolated backend. `pip check` passed in all
  eight checked environments. Runtime-only distributions were exactly pip,
  curvefit-stats, NumPy and SciPy, with no development tools.
- Two independent fixed-epoch wheel builds were byte-identical. This result is
  specific to this backend/platform/epoch; sdist gzip or cross-platform numerical
  byte reproducibility is not asserted.

## 4. Numerical validation

Executed independent references include linear regression, exponential decay,
Michaelis–Menten kinetics, logistic growth and correlated nonlinear models.
The stress/regression suites exercise heteroscedastic errors, very small/large
uncertainties and response units, large parameter-unit differences, nearly
singular Jacobians, active constraints and large floating-point offsets.
Pathological uncertainties, covariance matrices, observation shapes, insufficient
dof and failed optimizations are separately tested. Stochastic comparisons use
controlled local generators and sampling tolerances; repeatability does not
imply identical optimizer/LAPACK results across platforms.

Covariance and identifiability claims were checked against noise provenance:
shared parameters occupy one joint optimization column, conditional bound
covariance leaves active rows/columns NaN, and EIV covariance does not turn
response-only likelihood methods into ODR methods. Failed optimization and
unidentified models do not receive fabricated usable uncertainty. See the
[scientific review](review.md) for the nine previously corrected defects and
their regression cases; all 46 review regression cases passed again.

## 5. Executed package compatibility matrix

| Platform / CPython | NumPy | SciPy | Installed-wheel suite | Coverage | Doctests / isolated smoke |
| --- | --- | --- | --- | --- | --- |
| Linux / 3.11.17 | 1.26.4 | 1.12.0 | 405 passed | 95.22% | Passed |
| Linux / 3.12.14 | 1.26.4 | 1.12.0 | 405 passed | 95.22% | Passed |
| Linux / 3.12.14 | 2.5.4 | 1.18.1 | 405 passed | 95.27% | Passed |
| Linux / 3.13.5 | 2.5.4 | 1.18.1 | 405 passed | 95.27% | Passed |
| Linux / 3.14.8 | 2.5.4 | 1.18.1 | 405 passed | 95.27% | Passed |
| Windows / 3.12 | 2.5.4 | 1.18.1 | Configured; not executed | Unverified | Unverified |
| macOS / 3.12 | 2.5.4 | 1.18.1 | Configured; not executed | Unverified | Unverified |

Metadata specifies Python >=3.11, NumPy >=1.26.4,<3 and SciPy >=1.12,<1.19.
These ranges are compatibility policy, not exhaustive testing of every version
combination. The reference development extra/lock uses Python 3.12 because of
the official SciPy stubs; runtime/test-extra compatibility includes Python 3.11.

## 6. Remaining documented scientific limitations

- Bands are pointwise; nonlinear Delta/Wald inference is local and approximate.
  `fit_summary` always uses Student-t with N-p; known-error normal inference is
  available for bands, not that table. Covariance arrays cannot encode provenance.
- Regular AIC/BIC assume maximum-likelihood estimates and valid parameter counts;
  rank screening does not establish optimality, structural identification or
  boundary regularity. Profile thresholds use asymptotic Wilks calibration and
  local nuisance optimization; boundaries can require separate calibration.
- Residual bootstrap assumes IID observation errors at fixed predictors; fitted
  residuals are leverage-dependent. It provides neither an unbiased finite-sample
  covariance guarantee nor automatic corrections for leverage or failed fits.
- Bound-active covariance holds selected parameters fixed. It is not unrestricted
  marginal covariance, active-parameter uncertainty or selection-calibrated inference.
- Gaussian propagation may violate bounds; rejecting derived outputs changes the
  summarized distribution. Finite ratio samples do not imply finite ratio moments.
- Global fitting estimates one common unknown noise multiplier; separate unknown
  dataset scales and cross-dataset error covariance are not modeled.
- EIV supports one predictor/response and scalar/vector axis uncertainties, without
  bounds, correlated axis errors or analytic ODR Jacobians. SciPy ODR is deprecated
  from 1.17 and removed in 1.19; calls emit an explicit compatibility warning on
  deprecated versions and the dependency upper bound excludes removal.
- Float64 range, numerical rank thresholds and model scaling limit arbitrary
  callbacks. Numerical convergence does not establish a global optimum.

These are scientific/API limits, distinct from reproduced software defects.
They are explained in [the user guide](user-guide.md) and
[integration guidance](integration.md).

## 7. Release blockers and 8. recommendation

**NOT READY:** the candidate's actual GitHub Actions workflow, including its
Windows and macOS matrix jobs, has not executed. Local Linux equivalents and
Actionlint are insufficient evidence for that platform release gate. No further
release-blocking code defect was reproduced. The next checkpoint is successful
execution of the existing workflow on the candidate revision before public
release. No push, workflow dispatch or publication was performed.

Only this report was added in the final verification phase; runtime APIs and
implementations were preserved. Distribution artifacts were regenerated locally.

## Executed commands

Commands used the repository cwd `/workspace/curvefit-stats`, except isolated
smokes and dependency checks, which used `/tmp`. Logs and coverage XML files
are under `/workspace/setup-checks/curvefit-stats`.

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
COVERAGE_FILE=/workspace/setup-checks/curvefit-stats/.coverage-release /workspace/setup-checks/curvefit-stats/locked-dev/bin/python -m pytest -q --cov-report=xml:/workspace/setup-checks/curvefit-stats/coverage-release.xml
.venv/bin/python -m pytest -q --no-cov --doctest-modules src/curvefit_stats examples -W 'default:Bound-active parameter rows:UserWarning'
.venv/bin/python examples/workflows.py
SOURCE_DATE_EPOCH=1767225600 PIP_CACHE_DIR=/workspace/.cache/pip .venv/bin/python -m build
SOURCE_DATE_EPOCH=1767225600 PIP_CACHE_DIR=/workspace/.cache/pip .venv/bin/python -m build --wheel --outdir /workspace/setup-checks/curvefit-stats/release-repeat-dist
.venv/bin/python -m twine check --strict dist/*
.venv/bin/python tools/check_distributions.py
/workspace/setup-checks/curvefit-stats/actionlint/actionlint
git diff --check
```

For each ENV in `py311`, `compat-venv`, `wheel312`, `py313`, and `py314`,
PYTHON was `/workspace/setup-checks/curvefit-stats/ENV/bin/python`. Each ran:

```text
UV_CACHE_DIR=/workspace/.cache/uv uv pip install --python PYTHON --reinstall --no-deps dist/curvefit_stats-0.1.0-py3-none-any.whl
COVERAGE_FILE=/workspace/setup-checks/curvefit-stats/.coverage-release-ENV PYTHON -m pytest -q --cov-report=xml:/workspace/setup-checks/curvefit-stats/coverage-release-ENV.xml
PYTHON -m pytest -q -c pyproject.toml --no-cov --doctest-modules INSTALLED_PACKAGE_PATH examples -W 'default:Bound-active parameter rows:UserWarning'
PYTHON -I /workspace/curvefit-stats/tools/check_install.py
PYTHON -m pip check
```

The installed package path was obtained from that interpreter's
`curvefit_stats.__file__`, never by setting `PYTHONPATH=src` for artifact checks.
The separate runtime-only checks ran:

```bash
UV_CACHE_DIR=/workspace/.cache/uv uv pip install --python /workspace/setup-checks/curvefit-stats/wheel-runtime/bin/python --reinstall --no-deps dist/curvefit_stats-0.1.0-py3-none-any.whl
PIP_CACHE_DIR=/workspace/.cache/pip /workspace/setup-checks/curvefit-stats/sdist312/bin/python -m pip install --no-cache-dir --force-reinstall --no-deps dist/curvefit_stats-0.1.0.tar.gz
/workspace/setup-checks/curvefit-stats/wheel-runtime/bin/python -I /workspace/curvefit-stats/tools/check_install.py
/workspace/setup-checks/curvefit-stats/sdist312/bin/python -I /workspace/curvefit-stats/tools/check_install.py
```

Both runtime interpreters and `locked-dev` also passed `python -m pip check`.
Additional stdlib probes verified module enumeration, signatures, AST import
roots/intentional overload declarations, runtime distribution inventory, artifact
source equality and fixed-epoch wheel-byte equality.
