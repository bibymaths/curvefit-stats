# Scientific user guide

The examples model first-order concentration decay,
`C(t) = A exp(-k t)`, and Michaelis–Menten enzyme kinetics,
`v(S) = Vmax S/(Km + S)`. Time is in hours, concentration in mg/L, substrate
in mM, and velocity in µmol/min. Rate constants have units h⁻¹; half-life is
`log(2)/k`. Always choose uncertainty units consistent with the observations.

Run all workflows with `PYTHONPATH=src python examples/workflows.py` from the
checkout. Each section below names its selectable workflow. The
[API reference](reference/index.md) describes all arguments, output shapes, exceptions,
notes, and doctest examples. All features can be imported from the package root
or directly from the two functional modules.

## Weighting and covariance conventions

Let `r = y - f(x, θ)`, N be the observation count, p the number of estimated
parameters, and J the model parameter Jacobian. For observation covariance C,
weighted least squares minimizes

$$
Q=r^T C^{-1}r=\sum_i (r_i/\sigma_i)^2,
\qquad C=\operatorname{diag}(\sigma_i^2)
$$

for independent observations with standard deviations $\sigma_i$.

- Scalar `sigma` means the same **standard deviation** for every observation.
- Vector `sigma` supplies one positive, finite standard deviation per observation.
- Matrix `sigma` is the positive-definite observation **covariance**, including
  off-diagonal correlations. This is supported by response-only inference and
  global fitting; EIV accepts scalar/vector `sx` and `sy` only.
- `sigma=None` gives equal unit weighting. With `absolute_sigma=True`, this
  asserts known unit variance; with `False`, it estimates an unknown common scale.

With known absolute covariance (`absolute_sigma=True`), the interior local
approximation is

$$
V\approx (J^T C^{-1}J)^{-1}.
$$

With `False`, C specifies relative weights and the returned covariance is
approximately

$$
V\approx \frac{Q}{N-p}(J^T C^{-1}J)^{-1}.
$$
Multiplying all relative standard deviations by a common factor changes neither
the fit nor its scaled covariance. Multiplying known absolute standard deviations
changes the absolute parameter covariance by the square of that factor.

For example, a relative sigma vector `[1, 2]` weights the first observation four
times as strongly as the second; it does not establish measurement errors of
one and two physical units. Relative weighting is valid, and `audit_fit` emits
an explanatory warning when it is used. Do not present its weighted Q as a
known-scale chi-square goodness-of-fit calibration without knowing the scale.

These formulas need local regularity and adequate degrees of freedom. A singular
Jacobian does not support ordinary inverse-information uncertainty. Active
bounds require the separate conditional convention below.

## Ordinary nonlinear fitting and confidence bands

Workflow: `confidence`.

First call SciPy `curve_fit`, then pass the fitted parameters and covariance to
`generate_error_bands`. At evaluation point x, linearization gives

$$
\operatorname{Var}[f(x,\hat\theta)]\approx j(x)Vj(x)^T,
\qquad f(x,\hat\theta)\pm q\sqrt{j(x)Vj(x)^T}.
$$

For unknown common Gaussian variance, the default q is the Student-t upper-tail
quantile at `alpha/2` with residual degrees of freedom N-p. Supply the original
fitting observations through `y` and their coordinates through `x_fit` when
evaluating on a new grid, or supply `dof` explicitly without y. Grid length is
irrelevant to fitting degrees of freedom. Supplying inconsistent y and dof is
rejected.

For known absolute Gaussian error scales, choose `reference="normal"`.
No degrees of freedom are needed for that reference. Strong nonlinearity,
small samples, non-identifiability, and active constraints can invalidate
nominal coverage. An analytic `jac(x, *parameters)` of shape `(n_eval, p)` is
recommended for poorly scaled or numerically sensitive models.

## Confidence versus prediction intervals

Workflow: `intervals`.

A confidence interval concerns the fitted **mean**. A prediction interval
concerns a **new scalar observation** and adds its measurement variance:

$$
\operatorname{Var}[\text{prediction error}]\approx jVj^T+\sigma_{\mathrm{future}}^2.
$$

This assumes the future observation noise is independent of parameter estimation.

For an ordinary unweighted unknown-scale fit, supplying y lets the prediction
option estimate common future noise variance as `RSS/(N-p)`. This assumes
independent homoscedastic Gaussian errors. For known or heteroscedastic noise,
supply `prediction_variance` explicitly as a scalar or one variance per
evaluation point. It takes **variances**, unlike `sigma`, which takes standard
deviations. For example, known errors of 0.08 mg/L imply variance `0.08**2`:

```python
popt, pcov = curve_fit(
    decay, time, concentration, p0=[2.5, 0.5],
    sigma=0.08, absolute_sigma=True,
)
prediction_lower, prediction_upper = generate_error_bands(
    decay, grid, popt, pcov,
    interval_type="prediction", reference="normal",
    prediction_variance=0.08**2,
)
```

This snippet continues the README quick start and explicitly refits with the
known-error convention before selecting the normal reference.

Both types of interval are **pointwise**: approximately 95% coverage at one
specified coordinate does not imply 95% coverage of all coordinates at once.
Simultaneous bands and simultaneous multiple-parameter regions are not provided.
Future predictors with measurement error require more than adding y-noise.

## Fit diagnostics and information criteria

Workflow: `diagnostics`.

`audit_fit` returns N, p, residual dof, raw RSS, RMSE,
`residual_variance=RSS/(N-p)`, chi-square when defined, reduced chi-square,
AIC, and BIC. Raw residual variance is in response units and is not an estimated
heteroscedastic covariance matrix. Counts are returned as floats.

For known C, the normalized Gaussian deviance is

$$
D=-2\log L=N\log(2\pi)+\log\det C+Q,\qquad K=p.
$$
For relative C with one unknown multiplier s², its maximum-likelihood estimate
is `s²_ML=Q/N`, and

$$
D=N[\log(2\pi)+1+\log(Q/N)]+\log\det C,\qquad K=p+1,
$$

$$
\mathrm{AIC}=D+2K,\qquad\mathrm{BIC}=D+K\log N.
$$

The divisor N in the likelihood estimate differs deliberately from N-p in
covariance/residual-variance scaling. Gaussian normalization is retained;
negative AIC/BIC values are valid. Compare criteria for the same observations,
response units, and compatible likelihood definitions. Do not compare values
from different datasets or label a vertical-response audit as an ODR likelihood.
AICc is not implemented. Criteria assume maximum-likelihood estimates in an
identified regular model and count every supplied parameter. Use a reduced
parameter vector/model for fixed or tied parameters; singular and boundary
models require separate calibration. The audit rejects a numerically
rank-deficient whitened Jacobian rather than computing criteria with a redundant
parameter count; it does not verify that supplied estimates maximize likelihood.

No residual dof makes residual variance and reduced chi-square undefined.
Zero residual variance with an unknown scale makes Gaussian likelihood unbounded;
AIC/BIC become NaN with a warning rather than finite evidence for a perfect model.

## R-inspired parameter summaries

Workflow: `summary`.

`fit_summary` returns an ASCII table with parameter estimates,
`SEᵢ=sqrt(pcovᵢᵢ)`, `tᵢ=θ̂ᵢ/SEᵢ`, two-sided Student-t tail probabilities,
and `θ̂ᵢ ± t_(1-alpha/2,N-p) SEᵢ` endpoints. Names come from the model signature
or an explicit `param_names` list. The resemblance to an R summary concerns
the table layout; these nonlinear Wald results are approximate.

The tests concern a zero-valued parameter. They are not model-adequacy tests.
Student-t calibration is exact only in the appropriate regular linear Gaussian
setting; nonlinear models rely on local approximation. Known absolute errors
would ordinarily motivate normal-reference Wald analysis, but this function
always uses Student-t with N-p and has no alternative reference switch.
Do not use it as an automatically calibrated known-scale or EIV summary.
Singular covariance and rank-deficient or unverifiable parameter Jacobians
produce NaN standard errors and Wald results, preventing false marginal
significance. Reduce fixed/tied models before requesting this unrestricted
summary. Zero SE, nonpositive dof, and unusable covariance produce warnings and NaN
inference rather than spuriously infinite significance.

## Asymmetric profile likelihood

Workflow: `profile`.

`profile_likelihood_errors` fixes one parameter at a trial value and refits the
remaining nuisance parameters within their bounds. It searches for a
likelihood-ratio threshold rather than holding the nuisance estimates fixed.
For known scale, the ratio is `Q_profile-Q_best`. For one unknown common scale,
it is `N log(Q_profile/Q_best)`.

The threshold is `sigma_level**2`. `sigma_level=1` corresponds to normal-equivalent
central coverage about 68.27%; `1.95996398454` gives about 95% under the regular
one-parameter chi-square/Wilks calibration. This is not an exact Student-t
interval for unknown variance. `pcov`, if supplied, selects initial search
steps; it does not determine endpoints.

Each output dictionary contains signed `down` and `up` deviations from its fitted
estimate. Add them to the estimate to obtain endpoints. Search directions
limited by a bound or failed nuisance optimization return warned NaNs; a bound
is not falsely reported as a likelihood crossing. Starting estimates must be
locally optimal. Nuisance reoptimization is local, so disconnected solutions,
boundaries, weak identification, and small samples may require specialized
coverage calibration or a more complete search.

## Residual-bootstrap covariance

Workflow: `bootstrap`.

`bootstrap_curve_fit` centers the original residuals, samples them with
replacement at fixed x, adds them to the fitted mean, and refits. It returns
`(mean, covariance, samples)` with shapes `(p,)`, `(p,p)`, and `(B_success,p)`.
Base and replicate fits are checked for numerical Jacobian rank and reliable
covariance, including finite trust-region pseudoinverses. Response centering
and scaling protect against false trust-region convergence in tiny units.
Covariance has divisor `B_success-1`; the returned mean is the bootstrap mean,
not the original best fit. `n_resamples` counts attempts, not required successes.

This assumes independent identically distributed observation errors and an
adequate mean model. Fitted residuals themselves are correlated and can have
unequal leverage-dependent variances. Their empirical distribution approximates
the error law; consistency needs regular fitting and an appropriate asymptotic
design, such as vanishing maximum leverage. Finite-sample covariance is not
generally unbiased.
It does not support sigma weighting, correlated errors, or x measurement error,
and applies no leverage correction. It does not automatically correct outliers
or model misspecification. Failed replicates are excluded with warnings;
conditioning on successful fits can itself bias inference. Fewer than two
successes raise an error; a success fraction below 90% warns separately.

Use empirical quantiles of samples for bootstrap percentile intervals. Gaussian
propagation based on bootstrap covariance is a further approximation that can
lose asymmetry and tails. Reproducibility uses a local seeded Generator;
the global NumPy random state is not modified.

## Collinearity and local identifiability

Workflow: `collinearity`.

Given covariance V with positive diagonal, parameter correlation is
`Rᵢⱼ=Vᵢⱼ/sqrt(Vᵢᵢ Vⱼⱼ)`. `collinearity_diagnostics` reports R,
`VIF-likeᵢ=(R⁻¹)ᵢᵢ`, flagged zero-based pairs, rank, and correlation condition
number `lambda_max(R)/lambda_min(R)`. It diagnoses correlation-unit conditioning,
not the dimension-dependent condition number of the raw covariance.

Unsaturated Michaelis–Menten measurements can constrain approximately `Vmax/Km`
while leaving Vmax and Km individually imprecise. High correlation indicates
local dependence under this experimental design. It does **not** prove structural
non-identifiability, which concerns whether distinct parameters generate the same
ideal observable behavior. These VIF-like values need not equal classical
linear-design VIFs. Singular valid correlation produces infinite VIF/condition
values; non-finite covariance and zero-variance directions are flagged rather
than silently repaired with a pseudoinverse.

## Covariance at active bounds

Workflow: `bounds`.

`robust_bounded_pcov` marks bound-active parameter rows and columns NaN, holds
those values fixed, and estimates only the free-parameter block. With unknown
noise scale that block uses `Q/(N-p_free)`. This is **conditional covariance**,
not the full unrestricted marginal covariance of all parameters and not a
one-sided uncertainty estimate for an active parameter. Here conditional means
the model holds those values fixed; it does not calibrate inference conditional
on discovering an active set from the same data. Constraint-selection and
boundary coverage require separate analysis.

The example forces a nonnegative background to its lower bound. It extracts the
finite amplitude/rate block and uses a model with the background held fixed.
Passing the full non-finite matrix to bands or Gaussian propagation is rejected.
Its conditional intervals explicitly use N-p_free. This deliberately conflicting
synthetic baseline also illustrates that numerical conditional covariance does
not establish an adequate model.

Activity uses a documented scale-relative proximity tolerance. Supply an analytic
Jacobian when appropriate. A rank-deficient free Jacobian cannot supply ordinary
conditional uncertainty. Global fits use the same conditional interpretation;
`diagnostics.covariance_dof` describes the free block, while `diagnostics.dof`
counts all optimized parameters.

## Derived physical quantities

Workflow: `propagation`.

`propagate_uncertainty` draws `θ ~ Normal(popt, pcov)` and evaluates a scalar or
fixed-length vector `derived_func(*θ)` for each draw. It returns empirical mean,
median, sample standard deviation (divisor M-1), central quantiles, and attempted,
valid, and invalid counts. Scalar outputs produce Python floats; vector outputs
produce one-dimensional arrays. An invalid vector component rejects the whole
sample. Callable exceptions propagate; they are not silently counted as missing.

Covariance must be finite positive semidefinite. Small numerical PSD errors may
be repaired with warnings; singularity warns. Non-finite derived outputs warn,
and too few valid results raise an error. Accepted summaries describe the valid
subset and can be biased by rejection. Parameter uncertainty below float64
resolution raises an error; recentering/rescaling may be necessary.

The Gaussian approximation can violate physical bounds and discard likelihood
asymmetry. Finite ratio samples do not guarantee finite population moments:
a Gaussian rate permits values arbitrarily close to zero in `log(2)/rate`.
The example's half-life statistics are empirical local summaries, not a claim
of finite exact Gaussian ratio moments. For constrained or highly asymmetric
parameters, consider directly transforming appropriate profile or bootstrap
samples, with their own assumptions preserved.

## Shared-parameter global fitting

Workflow: `global`.

`GlobalFitGroup` accepts named dictionaries with x, y, and optional sigma. One
common model serves all datasets, including unequal lengths and different grids.
A set identifies shared local argument names; a mapping can rename them into
global aliases. Unlisted parameters become dataset-specific. Ambiguous or
variadic model signatures require explicit `parameter_names`.

Shared variables follow model argument order, then dataset-specific variables
follow lexically sorted dataset names and model order. The example has
`('rate', 'control.amplitude', 'treated.amplitude')`: the shared rate is optimized
once, not duplicated and later averaged. Supply p0 and bounds in that order or
use mappings keyed by these names. `group.parameter_indices[name]` maps local
model arguments to the global vector, and may contain repeated indices for
explicitly tied aliases.

`.fit()` uses SciPy `least_squares` on concatenated whitened residuals. Covariance
comes from the joint Jacobian, never concatenated independent covariance blocks.
For interior relative-weight fits, global dof is total N minus the number of
optimized global degrees of freedom. One common unknown variance multiplier
covers all datasets; separate unknown dataset scales and cross-dataset error
covariance are unsupported. Only Gaussian linear-loss covariance semantics are
supported.

The result provides `popt`, `pcov`, `parameter_names`, per-dataset estimates,
predictions and residuals, the global Jacobian, active masks, solver diagnostics,
and statistical diagnostics. Check `result.success` and covariance validity:
convergence does not imply identifiability or a global optimum. Failure or rank
deficiency leaves covariance undefined.

Project covariance with `pcov[np.ix_(indices, indices)]`. For relative-scale
per-dataset bands, pass the **global** dof without supplying local y; local data
must not redefine global degrees of freedom. The executable example uses known
errors and normal-reference bands. Per-dataset `audit_fit` or `fit_summary`
cannot produce global AIC/BIC or globally calibrated Student-t summaries merely
by receiving projected arrays. See [cross-module integration](integration.md).

## Errors in both axes

Workflow: `eiv`.

`eiv_curve_fit` adapts `func(x, *parameters)` internally to ODR's parameter-first
convention. It accepts one-dimensional aligned observations, positive
scalar/vector `sx` and `sy`, and initial guesses `p0`. Omitting sx fixes every x
exactly; zero sx values are rejected. Omitting sy means unit y weighting.

For independent axis errors, ODR fits latent adjusted coordinates and minimizes
`Q_ODR = sum((delta/sx)²) + sum((eps/sy)²)`, where
`delta = adjusted_x-x` and `eps = fitted_y-y`. Absolute scale returns ODR's
unscaled `cov_beta`; relative scale returns `cov_beta * res_var`, with
`res_var=Q_ODR/(N-p)` for identified fits. One unknown scale multiplies both
axis variance specifications. At least one residual degree of freedom is
required, even for known scales.

The default returns `(popt, pcov)`. `full_output=True` adds a typed diagnostic
object with success, stop reasons, rank, objective, coordinate corrections,
residual variance, and covariance convention. Default nonconvergence raises
`RuntimeError`; detailed mode warns and returns current parameters, NaN covariance,
and `success=False`. Check that before interpreting uncertainties. Rank-deficient
fits also have undefined covariance even when the solver converges.

ODR covariance can support local Delta analysis of the fitted mean at **exact**
evaluation coordinates. Future uncertain predictors need an additional
observation model. Vertical-residual likelihood, profile, residual bootstrap,
bounded covariance, and Student-t summary functions are not automatically ODR
methods. In particular, response-only `audit_fit` AIC/BIC are not ODR likelihood
criteria.

Compatibility is explicitly SciPy `>=1.12,<1.19`: scipy.odr is deprecated starting
in 1.17 and removed in 1.19. No replacement dependency is configured. This is
not a constrained `curve_fit` replacement: bounds, method selection, multiple
predictors, correlated axis errors, fixed parameters, and analytic ODR Jacobians
are unsupported. Only `maxit`, `sstol`, `partol`, `ndigit`, and `taufac` backend
controls are accepted; other keyword arguments are rejected. Optimization is
local, so a single synthetic comparison with ordinary least squares does not
prove universal EIV superiority.

## Warnings, numerical limits, and verification

Invalid shapes, non-finite observations, invalid covariance, and nonpositive
measurement uncertainties are rejected. Undefined statistical quantities are
reported as warned NaNs or infinities where documented; optimization failures
must not produce fabricated finite uncertainties. Finite values outside usable
float64 numerical range can raise `FloatingPointError` or invalidate covariance.
Read each [API exception contract](reference/index.md) before handling failures.

Examples intentionally retain warnings for strong correlation and conditional
bound covariance. EIV retains its SciPy compatibility warning. The package has
independent analytical/reference tests and seeded numerical stress tests; see
[the stress report](stress-testing.md). These checks establish behavior for tested
models and scales, not universal coverage or convergence for arbitrary models.

## Documentation verification

The documentation website renders API signatures and NumPy-style docstrings
from source and includes the original example file directly. Its strict build
and validation job executes all eleven workflows, authored Python tutorial
blocks, and checks local links, assets and API anchors. The generated offline
API snapshot remains available for repository readers and existing regression
checks; it is excluded from the site to avoid duplication.

See [documentation development](documentation.md) for reproducible uv commands
and deployment, and [release verification](release-readiness.md) for the
scientific suite, coverage and compatibility evidence.
