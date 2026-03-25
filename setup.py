"""
setup.py — legacy fallback for older pip/setuptools versions.

Prefer: pip install -e .   (uses pyproject.toml)
Also works: python setup.py develop

This puts the project root on sys.path permanently via a .egg-link,
preventing 'ImportError: cannot import name MiniGPT from src.model (unknown location)'.
"""
from setuptools import setup, find_packages

setup(
    name="minigpt",
    version="1.0.0",
    packages=find_packages(
        where=".",
        include=["src", "src.*"],
        exclude=["tests*", "scripts*", "docs*", "web*", "mobile*"],
    ),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.26.0",
    ],
)
