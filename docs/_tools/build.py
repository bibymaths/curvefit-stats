"""Build or preview the documentation with the supported Zensical CLI."""

from __future__ import annotations

import argparse
import subprocess

from prepare import ROOT, prepare


def main() -> None:
    """Refresh the offline snapshot and execute a clean, strict site build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="Preview locally instead.")
    args = parser.parse_args()
    prepare()
    command = ["zensical", "serve" if args.serve else "build", "--strict"]
    if not args.serve:
        command.append("--clean")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
