from setuptools import setup, find_packages

setup(
    name="minigpt",
    version="1.0.0",
    packages=find_packages(
        where=".",
        include=["minigpt_core", "minigpt_core.*"],
    ),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.26.0",
    ],
)
