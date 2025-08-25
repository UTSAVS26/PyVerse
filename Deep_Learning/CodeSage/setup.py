from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

from setuptools import setup, find_packages
from pathlib import Path

base_dir = Path(__file__).parent.resolve()
with (base_dir / "README.md").open("r", encoding="utf-8") as fh:
    long_description = fh.read()

with (base_dir / "requirements.txt").open("r", encoding="utf-8") as fh:
    raw_reqs = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]
    # Exclude dev/test tools from runtime requirements
    _dev_prefixes = ("pytest", "pytest-cov", "black", "flake8")
    requirements = [r for r in raw_reqs if not r.startswith(_dev_prefixes)]

setup(
    # ...
    install_requires=requirements,
    packages=find_packages(),
    # ...
)

setup(
    name="codesage",
    version="0.1.0",
    author="CodeSage Team",
    author_email="team@codesage.dev",
    description="AI-Based Code Complexity Estimator using AST analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/codesage",
    packages=find_packages(exclude=("tests", "tests.*", "examples", "examples.*")),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
        "viz": [
            # Optional visualization backends
            "graphviz>=0.20.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "codesage=codesage.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
