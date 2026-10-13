"""Validate built-site links, source API coverage and executable documentation."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import warnings
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
EXPECTED_WARNINGS = [
    "Strong local parameter correlation",
    "Bound-active parameter rows",
    "scipy.odr is deprecated in SciPy",
]


class Page(HTMLParser):
    """Collect HTML anchors and links without a browser dependency."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.feed(path.read_text())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record local destinations and fragment targets."""
        data = dict(attrs)
        for key in ["id", "name"]:
            if data.get(key):
                self.ids.add(str(data[key]))
        for key in ["href", "src"]:
            if data.get(key):
                self.links.append(str(data[key]))


def validate_site() -> None:
    """Fail on broken internal destinations, fragments or missing public API."""
    pages = {path.resolve(): Page(path) for path in SITE.rglob("*.html")}
    assert pages, "Build the site before validating it."
    errors = []
    checked = 0
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path and not url.fragment:
                continue
            relative = unquote(url.path)
            if relative.startswith("/"):
                assert relative.startswith("/curvefit-stats/"), (path, link)
                target = SITE / relative.removeprefix("/curvefit-stats/")
            else:
                target = path.parent / relative if relative else path
            if target.is_dir():
                target /= "index.html"
            target = target.resolve()
            checked += 1
            if not target.is_file():
                errors.append(f"{path.relative_to(SITE)}: {link} (missing file)")
            elif url.fragment and target in pages:
                if unquote(url.fragment) not in pages[target].ids:
                    errors.append(f"{path.relative_to(SITE)}: {link} (missing anchor)")
    assert not errors, "\n".join(errors)
    sys.path.insert(0, str(ROOT / "src"))
    import curvefit_stats

    for name in curvefit_stats.__all__:
        obj = getattr(curvefit_stats, name)
        key = obj.__module__ + "." + name
        path = (
            SITE / "reference" / Path(*obj.__module__.split(".")) / "index.html"
        ).resolve()
        assert key in pages[path].ids, f"Missing generated API: {key}"
    estimators = (SITE / "reference/curvefit_stats/estimators/index.html").resolve()
    assert "curvefit_stats.estimators.GlobalFitGroup.fit" in pages[estimators].ids
    search = json.loads((SITE / "search.json").read_text())
    text = json.dumps(search)
    assert all(name in text for name in curvefit_stats.__all__)
    home = (SITE / "index.html").read_text()
    assert "assets/brand/logo.svg" in home and "assets/brand/favicon.svg" in home
    assert "Switch to system preference" in home
    assert "arithmatex" in (SITE / "quickstart/index.html").read_text()
    print(
        f"Validated {len(pages)} HTML pages, {checked} local links/assets "
        "and all 10 generated APIs."
    )


def validate_examples() -> None:
    """Execute maintained workflows and every authored Python tutorial block."""
    sys.path.insert(0, str(ROOT / "src"))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    command = [sys.executable, "-W", "error"]
    for message in EXPECTED_WARNINGS:
        category = "FutureWarning" if "odr" in message else "UserWarning"
        command.extend(["-W", f"default:{message}:{category}"])
    command.append(str(ROOT / "examples/workflows.py"))
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    names = [
        "confidence",
        "intervals",
        "diagnostics",
        "summary",
        "profile",
        "bootstrap",
        "collinearity",
        "bounds",
        "propagation",
        "global",
        "eiv",
    ]
    assert all(f"\n{name.upper()}\n" in result.stdout for name in names)
    contexts = [
        ["README.md", "docs/user-guide.md"],
        ["docs/integration.md"],
        ["docs/index.md"],
        ["docs/quickstart.md"],
        *[
            [str(p.relative_to(ROOT))]
            for p in sorted((ROOT / "docs/tutorials").glob("*.md"))
        ],
    ]
    count = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for message in EXPECTED_WARNINGS:
            warning_type = FutureWarning if "odr" in message else UserWarning
            warnings.filterwarnings("default", message=message, category=warning_type)
        for context in contexts:
            namespace: dict[str, object] = {}
            for relative in context:
                blocks = re.findall(
                    r"```python[^\n]*\n(.*?)```",
                    (ROOT / relative).read_text(),
                    re.DOTALL,
                )
                for index, block in enumerate(blocks, 1):
                    exec(compile(block, f"{relative}:block-{index}", "exec"), namespace)
                    count += 1
    print(
        f"Executed all 11 workflows and {count} authored Python documentation blocks."
    )


def main() -> None:
    """Run build-output validation and executable example checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-only", action="store_true")
    args = parser.parse_args()
    if not args.examples_only:
        validate_site()
    validate_examples()


if __name__ == "__main__":
    main()
