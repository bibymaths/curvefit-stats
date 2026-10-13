# Building and publishing this documentation

The site uses **Zensical 0.0.70**, its modern theme and supported native API
navigation, plus **mkdocstrings-python 2.0.9**. Configuration is in
[`zensical.toml`](https://github.com/bibymaths/curvefit-stats/blob/main/zensical.toml),
not a legacy `mkdocs.yml`. Documentation dependencies are separate from runtime
dependencies and the package's development extras.

## Reproduce a build with uv

Use uv 0.12.23 and Python 3.12, matching CI. From the repository root:

```bash
uv sync --project requirements/docs --locked --python 3.12
uv run --project requirements/docs --locked python docs/_tools/build.py
uv run --project requirements/docs --locked python docs/_tools/validate.py
```

The standalone `requirements/docs/pyproject.toml` and committed `uv.lock` pin
direct/transitive dependencies with hashes and public PyPI URLs. A stale lock
fails `uv sync --locked`. The base package still installs only NumPy and SciPy;
documentation tools are not runtime dependencies. API collection reads `src`,
and executable validation uses that same source without requiring an installed
package in this environment.

The build refreshes a generated repository-only `docs/api.md` snapshot for
existing offline readers and regression tests. It is excluded from the website,
preventing duplicate API content. Website reference pages and navigation are
generated directly from source signatures and NumPy-style docstrings on every
build. Never hand-edit copied signatures or docstrings.

Validation checks built HTML links/assets and anchors, search/API coverage, all
ten public objects and `GlobalFitGroup.fit`, all eleven maintained workflows,
and each authored Python tutorial block. Unexpected warnings fail; only specific
documented statistical/ODR warnings are permitted. Scientific examples require
no network or external data.

## Preview locally

```bash
uv run --project requirements/docs --locked python docs/_tools/build.py --serve
```

Open the address printed by Zensical. Source in the configured mkdocstrings
`paths` is watched along with Markdown. You can also inspect a built site with
`python -m http.server 8000 --directory site`. Production uses the canonical
URL `https://bibymaths.github.io/curvefit-stats/`; preview does not publish it.

## GitHub Pages deployment

`.github/workflows/docs.yml` separates validation from deployment:

| Event | Strict build, links/API and executable examples | Pages deployment |
| --- | --- | --- |
| Pull request, including forks | Yes; read-only repository permissions | No |
| Push/merge to `main` | Yes | After validation passes |
| Manual dispatch on `main` | Yes | After validation passes |
| Manual dispatch on another branch | Yes | No |

The build uploads `site/` using `actions/upload-pages-artifact`. Deployment uses
`actions/configure-pages`, `actions/deploy-pages`, the `github-pages` environment
and OIDC. `pages: write` and `id-token: write` are limited to the deployment job.
Actions are pinned to verified commit SHAs. PR code never runs with deployment
privileges; there is no `pull_request_target`, custom token or `gh-pages` branch push.
Only uv's dependency cache is reused; Zensical's build cache is not persisted.

**Repository prerequisite:** in **Settings → Pages → Build and deployment →
Source**, select **GitHub Actions**. Allow the `github-pages` environment to
deploy from `main`. These repository settings cannot be verified or enabled by
a local build. Once configured, merging this site and workflow to `main`
automatically publishes the validated artifact. The deployment output records
the actual URL in GitHub's environment panel; no remote deployment is asserted here.

## Branding and mathematical rendering

Local SVG branding depicts a curve and uncertainty band. The landing illustration
is calculated from the seeded quickstart fit using its analytical Delta gradient
and Student-t quantile; it illustrates that experiment, not universal coverage.
Supported Zensical theme controls provide system/light/dark selection, search,
syntax highlighting and copyable code. Custom CSS provides responsive layout,
local system fonts, focus indicators and reduced-motion handling.

KaTeX 0.16.22 is vendored under `docs/assets/vendor/katex` with its MIT license,
source URL and tarball SHA256 in `PROVENANCE.txt`. Renderer, CSS and fonts are
local, so math needs no CDN. The official `document$` observable rerenders math
after instant navigation. Write equations using `$…$` or `$$…$$` with the
supported `pymdownx.arithmatex` extension.

## Update documentation dependencies

Change exact versions in `requirements/docs/pyproject.toml`, then execute:

```bash
uv lock --project requirements/docs --default-index https://pypi.org/simple
uv sync --project requirements/docs --locked
uv run --project requirements/docs --locked python docs/_tools/build.py
uv run --project requirements/docs --locked python docs/_tools/validate.py
```

Review the lock diff against [current official Zensical conventions](https://zensical.org/docs/).
Recheck theme, API and workflow compatibility. Do not replace locked builds with
unbounded installation commands in CI.

## Verification of this site

The implemented site passed a clean, strict Zensical build with no warnings.
The validator checked **21 HTML pages**, **1,325 internal links/assets**, all
ten generated API entries and `GlobalFitGroup.fit`. All **eleven original
workflows** and **15 authored Python documentation blocks** executed successfully.
The narrowly expected SciPy ODR compatibility warnings remained visible.

Local Chromium checks passed for system/light/dark selection and live system
preference changes, a 390-pixel mobile layout without horizontal page overflow,
source API rendering, API-name search, and mathematical rendering after instant
navigation. The concepts page rendered nine formulas without KaTeX errors.
Math and fonts were served locally; the theme's optional GitHub repository
metadata requests are separate from those assets. No JavaScript or local
resource errors were observed in these browser checks.

The existing full scientific suite also passed **405 tests**, with **95.27%
combined statement/branch coverage**. Ruff lint/format and strict mypy passed
for the existing 15 Python files and the three documentation helper scripts.
Actionlint validated the deployment workflow. Scientific source, tests,
executable example source, runtime dependencies and public APIs were unchanged.

These are executed local checks. The actual Pages workflow and repository
settings remain to be verified on GitHub; local success does not assert a live
deployment or remove the separately documented cross-platform release gate.
