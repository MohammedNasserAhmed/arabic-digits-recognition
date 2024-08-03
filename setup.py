"""
Setup script for Arabic Digits Recognition package.
"""
from pathlib import Path
from setuptools import setup, find_packages

# Constants
VERSION = "0.0.1"
REPO_NAME = "arabic-digits-recognition"
AUTHOR = "MohammedNasserAhmed"
PACKAGE_NAME = "adr"
AUTHOR_EMAIL = "abunasserip@gmail.com"
DESCRIPTION = "A deep learning model for recognizing Arabic digits (0-9)"
LONG_DESCRIPTION = Path("README.md").read_text(encoding="utf-8")
GITHUB_URL = f"https://github.com/{AUTHOR}/{REPO_NAME}"
LICENSE = "Apache License 2.0"
LICENSE_TROVE = "License :: OSI Approved :: Apache Software License"

# Core dependencies
REQUIREMENTS = [
    "pandas",
    "numpy",
    "matplotlib",
    "python-box",
    "pyYAML",
    "types-pyYAML",
    "joblib",
    "mlflow",
    "librosa",
    "tqdm",
    "pydantic",
]


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=GITHUB_URL,
    project_urls={
        "Bug Tracker": f"{GITHUB_URL}/issues",
        "Documentation": f"{GITHUB_URL}/blob/main/README.md",
        "Source Code": GITHUB_URL,
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        LICENSE_TROVE,
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Topic :: Computing/DeepLearning :: Digit Recognition",
    ],
    python_requires=">=3.8",
    install_requires=REQUIREMENTS,
    license=LICENSE,
)