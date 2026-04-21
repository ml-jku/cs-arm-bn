from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


def _read_long_description() -> str:
    repo_root = Path(__file__).resolve().parent
    readme = repo_root / "plt" / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return ""


setup(
    name="plate_effects",
    version="0.0.0",
    description="Plate effects codebase",
    long_description=_read_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["plt", "plt.*"]),
    python_requires=">=3.9",
)

