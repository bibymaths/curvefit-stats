# Executable scientific examples

Eleven seeded workflows model concentration decay and Michaelis–Menten kinetics.
They execute with only the package, standard library, NumPy and SciPy. No external
data or plotting dependency is required.

## Run one or run them all

From an installed checkout:

```bash
python examples/workflows.py
python examples/workflows.py profile
```

From a source checkout in the locked documentation environment:

```bash
uv run --project requirements/docs --locked python docs/_tools/validate.py --examples-only
```

| Command name | Scientific question | Further reading |
| --- | --- | --- |
| `confidence` | How uncertain is a fitted decay curve? | [Quickstart](quickstart.md) |
| `intervals` | How do mean and new-measurement intervals differ? | [Fit to uncertainty](tutorials/fitting.md) |
| `diagnostics` | What do Gaussian diagnostics and AIC/BIC measure? | [Likelihood conventions](user-guide.md#fit-diagnostics-and-information-criteria) |
| `summary` | How should a nonlinear parameter table be read? | [Wald limitations](user-guide.md#r-inspired-parameter-summaries) |
| `profile` | Are kinetic parameter errors asymmetric? | [Kinetics tutorial](tutorials/kinetics.md) |
| `bootstrap` | What covariance do empirical residual refits give? | [Bootstrap assumptions](user-guide.md#residual-bootstrap-covariance) |
| `collinearity` | Do unsaturated measurements separate Vmax and Km? | [Local identifiability](user-guide.md#collinearity-and-local-identifiability) |
| `bounds` | What uncertainty remains with an active baseline bound? | [Conditional covariance](user-guide.md#covariance-at-active-bounds) |
| `propagation` | How uncertain is a half-life or concentration ratio? | [Derived quantities](user-guide.md#derived-physical-quantities) |
| `global` | Can two experiments share one decay rate? | [Global tutorial](tutorials/global.md) |
| `eiv` | What if both time and concentration have error? | [EIV tutorial](tutorials/eiv.md) |

## Reproducibility and warnings

Each experiment controls its random generator. Seeds reproduce stochastic
experiments within an environment; optimizers and numerical backends can still
differ across platforms. Assertions and reference tests use scale-aware and
sampling-aware tolerances rather than cross-platform byte equality.

The correlation and active-bound examples deliberately emit interpretation
warnings. EIV emits a compatibility warning on SciPy 1.17–1.18. The validation
script permits these specific messages and treats unexpected warnings as errors.
Failed fits must not be interpreted as successful uncertainty estimates.

## Complete, maintained source

The code below is included directly from the repository's executable example
file at build time; it is not a separately maintained copy.

??? example "All eleven workflows"

    ```python
    --8<-- "examples/workflows.py"
    ```

[View or download the original file](https://github.com/bibymaths/curvefit-stats/blob/main/examples/workflows.py).
