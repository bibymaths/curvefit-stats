# Independent pre-release scientific review

The review inspected both functional modules, root exports, typing, numerical
helpers, reference tests, examples, documentation, metadata, packaging, and CI.
Previous successful tests were not treated as proof of correctness. New probes
identified six major software/statistical defects, two moderate numerical
defects, and one minor documentation defect. No critical defect was reproduced.
All reproduced defects below were corrected without changing public signatures
or adding functional modules. `tests/test_review.py` adds 46 regression cases.

## Resolved defects

| ID | Severity | Reproduction and correction |
| --- | --- | --- |
| R1 | Major | Multiplying all relative global uncertainties by `1e5` could leave `[0.5,0]` unchanged instead of the independent WLS optimum near `[2.02,1.00]`, while reporting success and finite covariance. Normalize the optimizer objective by a common initial residual scale; restore the Jacobian, cost, and optimality to original units. Regression cases span `1e-100` to `1e100`, numerical/analytic Jacobians, and both covariance conventions. |
| R2 | Major | Trust-region bootstrap accepted the finite pseudoinverse covariance of `(a+b)*x`, and ignored each replicate's returned covariance unless SciPy warned. Validate base and replicate Jacobian rank and covariance; reject an unidentified base model and exclude unreliable replicates with explicit counts. |
| R3 | Major | Trust-region bootstrap with responses multiplied by `1e-8` returned the unchanged guesses and zero empirical covariance. Center/scale responses and analytic Jacobians internally; restore known-unit covariance and preserve robust-loss `f_scale` in physical units. Independent linear resampling and a soft-L1 score-equation solver verify the corrected samples. |
| R4 | Major | A singular fit covariance for `(a+b)*x` generated individual p-values around `1e-27` and `1e-36`, while only warning about joint inference. Singular covariance or an unvalidated/rank-deficient Jacobian now produces NaN standard errors, tests, and intervals. A full-rank supplied covariance cannot override the data/model rank check. |
| R5 | Major | A valid but oversized covariance search seed (`pcov=[[1e40]]`) made profile root tolerance so large that zero displacement was accepted as a threshold crossing. Use displacement-relative accuracy independent of seed magnitude, up to 1000 root iterations, and verify the returned likelihood-ratio threshold. Independent known/unknown-scale linear endpoints are recovered. |
| R6 | Major | `(a+b)*x` received an extra AIC penalty of two compared with its identical one-parameter model family. Reject numerically rank-deficient whitened Jacobians rather than presenting regular criteria or residual degrees of freedom with redundant parameter counts. The reduced model is verified against its independently computed Gaussian maximum likelihood. |
| R7 | Moderate | Large-offset Monte Carlo means could round to `1e16+4` instead of the represented sample mean `1e16+2`. Compute means and order statistics in translated coordinates; bootstrap means receive the same correction. Independent represented-value summaries verify the arithmetic. |
| R8 | Moderate | Positive Delta uncertainty could underflow to a zero-width band. Absolute finite-difference probes could also miss a small derivative despite representable output uncertainty. Use covariance-informed parameter units for band derivatives and reject provable uncertainty/width underflow. An independent analytical case resolves nonzero bands at `x=1e-320`. |
| R9 | Minor | Bootstrap documentation assumed fitted residuals themselves were IID. Correct this to IID observation errors: fitted residuals are correlated and leverage-dependent. Explain asymptotic design requirements, finite population moments, finite-sample bias, and success-selection limitations. Existing independent linear bootstrap references explicitly use empirical residual variance with divisor N. |

Validation is stricter; warnings do not substitute for defining uncertainty.
Unidentified/bootstrap-invalid bases raise errors, unusable Wald results are
NaN, invalid regular likelihood audits are rejected, unresolved profile
thresholds are NaN, and unrepresentable bands raise numerical errors.

## Function-by-function mathematical conclusions

| Feature | Checked formulation and interpretation |
| --- | --- |
| `generate_error_bands` | Delta mean variance `j V j^T`; prediction adds independent future measurement variance. Student-t with unknown common variance versus normal with known Gaussian scales; original fitting dof, not grid length. Pointwise coverage only. |
| `audit_fit` | Known deviance `N log(2π)+logdet(C)+Q`, K=p; unknown common scale uses `Q/N`, deviance `N[log(2π)+1+log(Q/N)]+logdet(C)`, K=p+1. AIC/BIC retain normalization. Maximum likelihood and identified regular models are necessary. |
| `fit_summary` | `SE=sqrt(diag(V))`, t=estimate/SE, two-sided Student-t tails and intervals with N-p. These are approximate nonlinear zero-parameter tests, not model-adequacy tests. Unrestricted rank failures no longer produce finite significance. |
| `profile_likelihood_errors` | Reoptimize nuisance parameters. Known scale uses SSE differences; unknown common scale uses `N log(Q_profile/Q_best)`. Threshold is `sigma_level^2`; finite endpoints must resolve it. |
| `bootstrap_curve_fit` | Centered residual resampling at fixed x, fixed attempt count, empirical covariance with B_success-1. Point-estimator refits, their rank/covariance, response-unit conventions, and reproducible RNG are checked independently. |
| `collinearity_diagnostics` | Correlation normalization; inverse-correlation diagonals and correlation-eigenvalue conditioning. VIF-like diagnostics are not generally classical design VIFs or structural identification proofs. |
| `robust_bounded_pcov` | Inverse free-parameter information with active values held fixed; relative scaling uses N-p_free. Active rows/columns remain NaN. It is not unrestricted covariance or a calibrated active-set selection distribution. |
| `propagate_uncertainty` | Correlation-unit PSD validation and a physically scaled Gaussian factor preserve parameter units. Seeded samples and translated empirical summaries are checked. Singular support and covariance repair are explicit. |
| `GlobalFitGroup` | Concatenate whitened residuals and their joint Jacobian. Shared variables occur once; repeated local aliases add derivative columns into the same global column. Covariance comes from the joint Jacobian with global parameter counts. |
| `eiv_curve_fit` | ODR parameter-first adapter, independent-axis objective, latent-coordinate Schur complement, and Deming references. Absolute covariance is `cov_beta`; relative covariance is `cov_beta*res_var`, where identified `res_var=Q_ODR/(N-p)`. |

## Remaining scientific and compatibility limitations

- Plain covariance arrays do not encode estimator/noise provenance. Callers must
  select compatible reference laws, degrees of freedom, and future noise models.
  EIV covariance cannot turn response-only likelihood methods into ODR methods.
- Nonlinear Wald/Delta and Gaussian parameter propagation are local
  approximations. Numerical rank is not structural identification. Conservative
  finite-difference screening can decline inference for poorly scaled or
  domain-limited models; reparameterize/rescale or supply analytic derivatives
  where that API accepts them.
- Audits check numerical rank, not maximum-likelihood optimality or boundary
  regularity. Fixed/tied parameters require a reduced model/vector. Profile
  chi-square calibration is asymptotic and nonregular/boundary models need
  separate calibration; nuisance fits are local, not global searches.
- Residual bootstrap is an empirical approximation with no leverage or residual
  degrees-of-freedom inflation. Failures can select/bias the retained samples;
  successful fits and zero empirical spread do not establish population
  uncertainty or finite population moments.
- Conditional bound covariance holds values fixed in the model. It does not
  provide active-parameter one-sided uncertainty, unrestricted marginal
  uncertainty, or automatic calibration for selecting that active set. Activity
  itself depends on numerical tolerances, and an inactive mask does not prove
  that boundary effects are negligible.
- Gaussian sampling can violate bounds; finite sampled ratios do not imply
  finite population moments. Rejected outputs change the summarized distribution.
  Float64 and native backend range/conditioning limitations remain.
- Global fitting uses one common unknown variance multiplier across datasets;
  cross-dataset error covariance and separate unknown dataset scales are absent.
- ODR supports SciPy >=1.12,<1.19, one predictor/response, scalar/vector axis
  errors, and N>p. Bounds, correlated axis errors, selected fixed predictors,
  and analytic ODR derivatives are unsupported. Its deprecation/removal policy
  is explicitly documented and enforced by the runtime dependency upper bound.

## Release disposition

The reproduced critical/major code blockers are resolved after the corrections.
This is finite scientific benchmark evidence, not proof for arbitrary callbacks
or statistical models. Linux Python/numerical compatibility is tested locally.
The actual GitHub workflow has not been triggered, so configured Windows/macOS
jobs remain unverified; their execution is the remaining platform release gate.
No publication or repository push was performed.

Final executed commands, compatibility results, and artifact checks follow below.

## Final verification outcomes

| Environment | NumPy / SciPy | Full suite | Statement/branch coverage |
| --- | --- | --- | --- |
| Linux CPython 3.11.17 | 1.26.4 / 1.12.0 | 405 passed | 95.22% |
| Linux CPython 3.12.14 | 1.26.4 / 1.12.0 | 405 passed | 95.22% |
| Linux CPython 3.12.14 | 2.5.4 / 1.18.1 | 405 passed | 95.27% |
| Linux CPython 3.13.5 | 2.5.4 / 1.18.1 | 405 passed | 95.27% |
| Linux CPython 3.14.8 | 2.5.4 / 1.18.1 | 405 passed | 95.27% |

The main source/editable environment also passed 405 tests with 95.27% coverage.
No skips or expected failures occurred. Each suite executes all eleven example
workflows, documentation/signature checks, installed metadata tests, and the
independent scientific reference/stress tests. Ruff lint/format checks and strict
mypy passed for all 15 Python files. Thirteen doctests passed with one expected
bound-active conditional-covariance warning. Actionlint and `git diff --check`
passed. Wheel/sdist builds, strict Twine validation, archive checks, and isolated
runtime-only wheel/sdist feature smoke tests passed; reviewed artifacts were
rebuilt. NumPy and SciPy remain the only mandatory third-party dependencies.

Commands executed from `/workspace/curvefit-stats`:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
COVERAGE_FILE=/workspace/setup-checks/curvefit-stats/.coverage-review-final .venv/bin/python -m pytest -q --cov-report=xml:/workspace/setup-checks/curvefit-stats/coverage-review-final.xml
.venv/bin/python -m pytest -q --no-cov --doctest-modules src/curvefit_stats examples -W 'default:Bound-active parameter rows:UserWarning'
SOURCE_DATE_EPOCH=1767225600 PIP_CACHE_DIR=/workspace/.cache/pip .venv/bin/python -m build
.venv/bin/python -m twine check --strict dist/*
.venv/bin/python tools/check_distributions.py
/workspace/setup-checks/curvefit-stats/actionlint/actionlint .github/workflows/ci.yml
git diff --check
```

Each wheel environment below installed the rebuilt wheel with
`uv pip install --python PYTHON --reinstall --no-deps dist/curvefit_stats-0.1.0-py3-none-any.whl`, then ran
`COVERAGE_FILE=/workspace/setup-checks/curvefit-stats/.coverage-review-ENV PYTHON -m pytest -q --cov-report=xml:/workspace/setup-checks/curvefit-stats/coverage-review-ENV.xml`.

| ENV | PYTHON |
| --- | --- |
| py311 | `/workspace/setup-checks/curvefit-stats/py311/bin/python` |
| compat-venv | `/workspace/setup-checks/curvefit-stats/compat-venv/bin/python` |
| wheel312 | `/workspace/setup-checks/curvefit-stats/wheel312/bin/python` |
| py313 | `/workspace/setup-checks/curvefit-stats/py313/bin/python` |
| py314 | `/workspace/setup-checks/curvefit-stats/py314/bin/python` |

Clean runtime-only checks used:

```bash
UV_CACHE_DIR=/workspace/.cache/uv uv pip install --python /workspace/setup-checks/curvefit-stats/wheel-runtime/bin/python --reinstall --no-deps dist/curvefit_stats-0.1.0-py3-none-any.whl
/workspace/setup-checks/curvefit-stats/sdist312/bin/python -m pip install --no-cache-dir --force-reinstall --no-deps dist/curvefit_stats-0.1.0.tar.gz
/workspace/setup-checks/curvefit-stats/wheel-runtime/bin/python -I /workspace/curvefit-stats/tools/check_install.py
/workspace/setup-checks/curvefit-stats/sdist312/bin/python -I /workspace/curvefit-stats/tools/check_install.py
```

The final two smoke commands ran from `/tmp`, preventing checkout imports.
Files changed in this review: `src/curvefit_stats/inference.py`,
`src/curvefit_stats/estimators.py`, `tests/test_review.py`, `docs/api.md`,
`docs/user-guide.md`, `docs/review.md`, and `README.md`, plus regenerated artifacts
under the ignored `dist/` directory. No public signature changed. The main
compatibility clarification is that optimizer gtol applies to the internally
normalized objective; original-unit diagnostics and covariance are restored.
Numerically singular ordinary inference is now refused instead of presented
as usable significance/model criteria.
