"""
Setup script for PyRecon.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pyrecon",
    version="1.0.0",
    author="Shivansh Katiyar",
    author_email="shivansh.katiyar@example.com",
    description="High-Speed Port Scanner & Service Fingerprinter",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/<correct-owner>/pyrecon",

    project_urls={
        "Bug Tracker": "https://github.com/<correct-owner>/pyrecon/issues",
        "Source Code": "https://github.com/<correct-owner>/pyrecon",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: System :: Networking :: Monitoring",
        "Topic :: Security",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.10.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pyrecon=pyrecon.cli.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="port-scanner network-security reconnaissance fingerprinting",
    project_urls={
        "Bug Reports": "https://github.com/shivanshkatiyar/pyrecon/issues",
        "Source": "https://github.com/shivanshkatiyar/pyrecon",
        "Documentation": "https://github.com/shivanshkatiyar/pyrecon#readme",
    },
) 