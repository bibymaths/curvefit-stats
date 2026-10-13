# Development, packaging, and CI

`curvefit-stats` is a pure-Python, MIT-licensed distribution. The initial package
version is 0.1.0; no registry publication is performed by these instructions or
by CI. Only NumPy and SciPy are mandatory third-party runtime dependencies.
The package retains exactly two functional modules, `inference` and `estimators`.

## Supported versions

Python 3.11–3.14 is explicitly tested. Package metadata requires Python >=3.11;
future Python releases are not claimed as tested merely because that lower-bound
specifier permits them. Numerical dependency constraints are:

```text
numpy >=1.26.4,<3
scipy >=1.12,<1.19
```

The SciPy upper bound is intentional: the EIV backend uses scipy.odr, deprecated
in 1.17 and removed in 1.19. The package's explicit compatibility warning remains
visible. NumPy's major-version upper bound avoids an untested future API break.
These broad runtime ranges accommodate compatible installers; the exact matrix
below supplies reproducible test points rather than claiming every version pair
inside the ranges has been tested.

| OS | Python | NumPy | SciPy |
| --- | --- | --- | --- |
| Linux | 3.11 | 1.26.4 | 1.12.0 |
| Linux | 3.12 | 1.26.4 | 1.12.0 |
| Linux | 3.12 | 2.5.4 | 1.18.1 |
| Linux | 3.13 | 2.5.4 | 1.18.1 |
| Linux | 3.14 | 2.5.4 | 1.18.1 |
| Windows | 3.12 | 2.5.4 | 1.18.1 |
| macOS | 3.12 | 2.5.4 | 1.18.1 |

All five Linux combinations are validated locally during packaging preparation.
Windows/macOS are configured CI targets and require actual GitHub runner execution
before claiming platform validation. The workflow itself is syntax-checked with
actionlint. Python setup, checkout, and artifact actions are pinned to verified
commit SHAs. No CI job publishes artifacts to a package registry.

## Reproduce the reference development environment

Use Python 3.12 on Linux for the hash-locked reference environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements/dev-py312.lock
python -m pip install --no-deps -e .
python -m pip check
```

The lock includes exact transitive dependencies and SHA256 hashes. It is specific
to the Linux Python 3.12 development environment; it is not an installation
constraint imposed on library users. The numerical stack is constrained by
`requirements/dev-constraints.txt`. Build backend requirements are separately
pinned in `pyproject.toml` and installed in PEP 517 isolated build environments.

Alternatively, `python -m pip install -e '.[dev]'` installs the optional development
extra. It pins direct tools but can resolve newer transitive dependencies, so it
is not the hash-locked route. The current SciPy stub version requires Python 3.12+
and matches the reference numerical stack. For runtime compatibility testing on
Python 3.11, use the `test` extra instead of the full developer/type-checking extra.
Type checking targets Python 3.11 language compatibility and runs in Python 3.12.

To intentionally refresh the reference lock with uv:

```bash
uv pip compile pyproject.toml --extra dev --python-version 3.12 \
  --generate-hashes --constraint requirements/dev-constraints.txt \
  --output-file requirements/dev-py312.lock
```

Review version and hash changes, reinstall from the new lock, and rerun quality
and compatibility checks. Do not freeze the development environment into the
mandatory runtime dependency list.

## Quality gates

Configuration lives in `pyproject.toml`; no ad hoc isolated flags are needed:

```bash
ruff check .
ruff format --check .
mypy
python -m pytest
python -m pytest --no-cov --doctest-modules src/curvefit_stats examples \
  -W 'default:Bound-active parameter rows:UserWarning'
python examples/workflows.py
```

Ruff retains E/F/I/B/UP rules and NumPy docstring checks. Mypy is strict, checks
source, tests, examples, and packaging tools, and does not ignore missing imports
or exclude difficult functions. Official SciPy stubs are development-only.
Pytest uses strict configuration/markers and treats warnings as errors. Existing
narrowly targeted expected compatibility/statistical warnings remain explicitly
checked or handled; numerical failures are not suppressed to make CI pass.

Pytest-cov runs by default, emits terminal and XML reports, and requires **90%
combined statement/branch coverage**. Branch measurement strengthens the prior
line-only gate. Use `--no-cov` for doctests so a small example-only invocation
does not masquerade as full-suite coverage. Do not lower the full-suite threshold.

## Build and inspect artifacts

```bash
python -m build
python -m twine check --strict dist/*
python tools/check_distributions.py
```

Default `python -m build` first creates the sdist, then builds its wheel from that
sdist. The backend is `setuptools.build_meta` with pinned setuptools/wheel build
requirements. Artifacts are `dist/curvefit_stats-0.1.0-py3-none-any.whl` and
`dist/curvefit_stats-0.1.0.tar.gz`. The archive check expects one wheel and one
sdist; use a clean output directory when changing versions.

The wheel contains only `__init__.py`, `inference.py`, `estimators.py`, and
`py.typed` in its functional package, plus distribution/license metadata. The
sdist includes source, typing data, license, README, pyproject/manifest, scientific
tests, documentation, examples, packaging tools, the development lock, and CI.
It excludes virtual environments and generated build/cache contents.

Pinned dependencies improve reproducibility, but archive timestamps and build
platforms can still affect byte hashes. `SOURCE_DATE_EPOCH` can fix wheel timestamps
for release builds. Do not equate a pure-Python wheel with bit-for-bit numerical
reproducibility across NumPy/LAPACK platforms.

## Install and test the artifacts

Use clean environments outside the checkout. On Windows, replace `bin/python`
with `Scripts/python.exe`; the following commands illustrate POSIX systems:

```bash
python -m venv /tmp/curvefit-wheel-check
/tmp/curvefit-wheel-check/bin/python -m pip install dist/curvefit_stats-0.1.0-py3-none-any.whl
/tmp/curvefit-wheel-check/bin/python -m pip check
/tmp/curvefit-wheel-check/bin/python -I tools/check_install.py

python -m venv /tmp/curvefit-sdist-check
/tmp/curvefit-sdist-check/bin/python -m pip install dist/curvefit_stats-0.1.0.tar.gz
/tmp/curvefit-sdist-check/bin/python -m pip check
/tmp/curvefit-sdist-check/bin/python -I tools/check_install.py
```

The stdlib/NumPy/SciPy smoke script checks the actual import location, installed
metadata, runtime-only requirements, typing marker, architecture, and all ten
features. `-I` disables environment/current-directory source import shortcuts.
The smoke rejects an editable source import; an installed wheel must satisfy it.
These environments install no pytest, typing, lint, or build-validation tools
as runtime requirements. Sdist installation uses pip's isolated build backend.

For full wheel compatibility testing, install the test extra and exact numerical
versions, then execute tests from the repository with the wheel interpreter:

```bash
/tmp/curvefit-wheel-check/bin/python -m pip install \
  'dist/curvefit_stats-0.1.0-py3-none-any.whl[test]' \
  'numpy==1.26.4' 'scipy==1.12.0'
/tmp/curvefit-wheel-check/bin/python -m pytest
```

Do not set `PYTHONPATH=src` for artifact verification. Documentation subprocesses
also deliberately test source examples; the separate isolated smoke ensures
those cannot hide missing or broken wheel contents. When running environments
concurrently, give each coverage run a unique `COVERAGE_FILE` and XML output path.

## CI structure

`.github/workflows/ci.yml` has three stages: hash-locked strict quality checks;
wheel/sdist generation, metadata/archive inspection, and direct sdist installation;
then the explicit wheel-based compatibility matrix. Each matrix job checks
installation, the complete scientific suite and coverage, doctests, and all
examples. Tests on different Python versions do not share a coverage database.
Build artifacts are retained through GitHub Actions artifact storage.

No GitHub workflow has been triggered from this local preparation session.
Local equivalents and their exact outcomes are recorded below after verification.

## Packaging preparation results

All local checks below passed. The exact Linux compatibility results, using
installed wheels rather than an editable package, were:

| Python | NumPy | SciPy | Full suite | Statement/branch coverage | Doctests |
| --- | --- | --- | --- | --- | --- |
| 3.11.17 | 1.26.4 | 1.12.0 | 359 passed | 95.70% | 13 passed |
| 3.12.14 | 1.26.4 | 1.12.0 | 359 passed | 95.70% | 13 passed |
| 3.12.14 | 2.5.4 | 1.18.1 | 359 passed | 95.76% | 13 passed |
| 3.13.5 | 2.5.4 | 1.18.1 | 359 passed | 95.76% | 13 passed |
| 3.14.8 | 2.5.4 | 1.18.1 | 359 passed | 95.76% | 13 passed |

Each doctest run emitted one deliberately expected conditional bound-covariance
warning. Each full suite includes execution of all eleven documented workflows,
source/API consistency checks, installed metadata/typing checks, and the existing
independent mathematical and numerical regression tests. No tests were skipped
or marked expected-failure in these runs.

A fresh Linux Python 3.12 environment was installed directly from the hash lock
with pip's `--require-hashes`; editable installation and `pip check` passed.
Its full suite passed 359 tests at 95.76% combined coverage. Ruff lint passed;
Ruff formatting passed for all 14 Python files; strict mypy passed for all 14
source/test/example/tool files. Mypy language compatibility is set to Python 3.11.
Array conversions were made explicit where SciPy/NumPy stubs under that target
needed unambiguous array return types; sparse solver Jacobians are converted via
CSR before densification. Existing sparse-Jacobian tests pass on both stacks.
Numerical methods, public signatures, and test tolerances were preserved.

Both built archives passed `twine check --strict` and the stdlib archive checker.
Isolated `python -I` wheel smoke checks passed for all five Linux combinations.
Separate clean wheel and direct-sdist environments passed every feature smoke
check and `pip check`; installed distributions were exactly pip, curvefit-stats,
NumPy, and SciPy. `py.typed` was present after both installation routes.
Actionlint 1.7.7 passed for the CI workflow; its downloaded binary was verified
against the release SHA256 checksum. `git diff --check` passed.

The exact commands below use this workspace's interpreter locations and unique
coverage outputs. They supplement the portable commands above:

```bash
PIP_CACHE_DIR=/workspace/.cache/pip /workspace/setup-checks/curvefit-stats/locked-dev/bin/python -m pip install --require-hashes -r requirements/dev-py312.lock
/workspace/setup-checks/curvefit-stats/locked-dev/bin/ruff check .
/workspace/setup-checks/curvefit-stats/locked-dev/bin/ruff format --check .
/workspace/setup-checks/curvefit-stats/locked-dev/bin/mypy
COVERAGE_FILE=/workspace/setup-checks/curvefit-stats/.coverage-locked /workspace/setup-checks/curvefit-stats/locked-dev/bin/python -m pytest -q --cov-report=xml:/workspace/setup-checks/curvefit-stats/coverage-locked.xml
/workspace/setup-checks/curvefit-stats/actionlint/actionlint .github/workflows/ci.yml
```

For the wheel suites, each interpreter ran
`COVERAGE_FILE=/workspace/setup-checks/curvefit-stats/.coverage-TAG PYTHON -m pytest -q --cov-report=xml:/workspace/setup-checks/curvefit-stats/coverage-TAG.xml`:

| TAG | PYTHON |
| --- | --- |
| 311 | `/workspace/setup-checks/curvefit-stats/py311/bin/python` |
| 312-low | `/workspace/setup-checks/curvefit-stats/compat-venv/bin/python` |
| 312 | `/workspace/setup-checks/curvefit-stats/wheel312/bin/python` |
| 313 | `/workspace/setup-checks/curvefit-stats/py313/bin/python` |
| 314 | `/workspace/setup-checks/curvefit-stats/py314/bin/python` |

Each ran `PYTHON -I /workspace/curvefit-stats/tools/check_install.py` from `/tmp`.
Installed-package doctests ran
`PYTHON -m pytest -q -c pyproject.toml --no-cov --doctest-modules PACKAGE_DIR examples -W 'default:Bound-active parameter rows:UserWarning'`, with PACKAGE_DIR set
to that environment's installed `site-packages/curvefit_stats` directory.

Windows/macOS runner execution and registry publication remain unperformed.
Scientific limitations are unchanged; see the [user guide](user-guide.md).

Two independent isolated wheel builds were also compared with
`SOURCE_DATE_EPOCH=1767225600`: the default sdist-to-wheel route and a direct
wheel build. Their wheel bytes were identical. This verifies wheel reproducibility
for this source, backend, platform, and fixed epoch; it does not assert identical
sdist gzip timestamps or cross-platform scientific numerical results.

```bash
SOURCE_DATE_EPOCH=1767225600 PIP_CACHE_DIR=/workspace/.cache/pip .venv/bin/python -m build
SOURCE_DATE_EPOCH=1767225600 PIP_CACHE_DIR=/workspace/.cache/pip .venv/bin/python -m build --wheel --outdir /workspace/setup-checks/curvefit-stats/repeat-dist
```
