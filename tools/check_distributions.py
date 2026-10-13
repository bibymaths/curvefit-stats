"""Inspect wheel and source archives for packaging contracts using the stdlib."""

from __future__ import annotations

import argparse
import re
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def main() -> None:
    """Check metadata, runtime dependencies, typing marker, and sdist contents.

    Returns
    -------
    None
        Print validated artifact names and raise on missing or unexpected contents.

    Raises
    ------
    AssertionError
        Artifact counts, package layout, metadata, or source contents are invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("dist"))
    directory = parser.parse_args().directory
    wheels = sorted(directory.glob("*.whl"))
    sources = sorted(directory.glob("*.tar.gz"))
    assert len(wheels) == len(sources) == 1, (wheels, sources)
    expected = {"__init__.py", "inference.py", "estimators.py", "py.typed"}
    with zipfile.ZipFile(wheels[0]) as wheel:
        files = wheel.namelist()
        package = {
            name.removeprefix("curvefit_stats/")
            for name in files
            if name.startswith("curvefit_stats/")
        }
        assert package == expected, package
        metadata_paths = [
            name for name in files if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_paths) == 1
        metadata = BytesParser().parsebytes(wheel.read(metadata_paths[0]))
        assert metadata["Name"] == "curvefit-stats"
        assert metadata["Version"] == "0.1.0"
        assert metadata["Requires-Python"] == ">=3.11"
        assert metadata["License-Expression"] == "MIT"
        assert any(name.endswith(".dist-info/licenses/LICENSE") for name in files)
        runtime = {
            re.split(r"[<>=!~\s\[]", dependency, maxsplit=1)[0].lower()
            for dependency in metadata.get_all("Requires-Dist", [])
            if ";" not in dependency
        }
        assert runtime == {"numpy", "scipy"}, runtime
        assert set(metadata.get_all("Provides-Extra", [])) == {"test", "dev"}
    with tarfile.open(sources[0]) as source:
        names = {member.name for member in source.getmembers() if member.isfile()}
        roots = {name.split("/", 1)[0] for name in names}
        assert len(roots) == 1
        root = roots.pop()
        relative = {name.removeprefix(root + "/") for name in names}
        required = {
            "pyproject.toml",
            "MANIFEST.in",
            "README.md",
            "LICENSE",
            "docs/user-guide.md",
            "docs/development.md",
            "docs/api.md",
            "examples/workflows.py",
            "tests/test_inference_reference.py",
            "requirements/dev-py312.lock",
            "tools/check_install.py",
            ".github/workflows/ci.yml",
            "zensical.toml",
            "docs/_tools/build.py",
            "docs/_tools/prepare.py",
            "docs/_tools/validate.py",
            "docs/assets/brand/logo.svg",
            "docs/assets/brand/favicon.svg",
            "docs/assets/stylesheets/extra.css",
            "docs/assets/javascripts/math.js",
            "docs/assets/vendor/katex/katex.min.css",
            "docs/assets/vendor/katex/katex.min.js",
            "docs/assets/vendor/katex/auto-render.min.js",
            "requirements/docs/pyproject.toml",
            "requirements/docs/uv.lock",
        }
        required |= {"src/curvefit_stats/" + name for name in expected}
        assert required <= relative, required - relative
        assert not any(".venv" in Path(name).parts for name in relative)
    print(f"Validated {wheels[0].name} and {sources[0].name}")


if __name__ == "__main__":
    main()
