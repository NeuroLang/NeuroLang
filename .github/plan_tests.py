"""
Plan the test jobs of a pull request from the packages it changes.

Each package depends on those before it: a change to a package runs its
tests on every Python version, and those of the packages after it on the
oldest one. A push, or a change to what all the packages share, runs all of
them on every version. ``CHANGES`` is the JSON list of the changed filters
of ``.github/workflows/tests.yml``; the matrix goes to ``GITHUB_OUTPUT``.
"""

import json
import os

PYTHON_VERSIONS = ("3.12", "3.13", "3.14")

# Package directory: (distribution, extras its tests need).
PACKAGES = {
    "nl_expr": ("nl-expr", "--extra dev"),
    "nl_core": ("nl-core", "--extra dev --extra ras-polars"),
    "neurolang": ("neurolang", "--extra dev"),
    "neurolang_apps": (
        "neurolang-apps", "--extra dev --extra tui --extra llm"
    ),
}


def plan(changed, event):
    """The jobs to run, as the ``include`` list of a matrix."""
    order = list(PACKAGES)
    changed = set(changed)
    if event == "push" or "shared" in changed:
        changed |= set(order)
    first = min(
        (order.index(p) for p in changed if p in PACKAGES),
        default=len(order),
    )
    return [
        {
            "package": package,
            "dist": PACKAGES[package][0],
            "extras": PACKAGES[package][1],
            "python": version,
        }
        for package in order[first:]
        for version in (
            PYTHON_VERSIONS if package in changed else PYTHON_VERSIONS[:1]
        )
    ]


def main():
    include = plan(
        json.loads(os.environ.get("CHANGES") or "[]"),
        os.environ.get("EVENT", "pull_request"),
    )
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"matrix={json.dumps({'include': include})}\n")
        output.write(f"any={'true' if include else 'false'}\n")
    print(json.dumps(include, indent=2))


if __name__ == "__main__":
    main()
