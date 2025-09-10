#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    with open('requirements.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                requirements.append(line)
    return requirements

setup(
    name="treegpt",
    version="0.1.0",
    author="Zixi Li",
    author_email="lizx93@mail2.sysu.edu.cn",
    description="A Novel Hybrid Architecture for Abstract Syntax Tree Processing with Global Parent-Child Aggregation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TreeGPT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pre-commit>=2.17.0",
        ],
        "viz": [
            "torch-geometric>=2.0.0",
            "tensorboard>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "treegpt-train=src.arc_treegpt:train_arc_model",
        ],
    },
    keywords="deep-learning, transformer, tree-neural-networks, program-synthesis, arc-challenge",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/TreeGPT/issues",
        "Source": "https://github.com/yourusername/TreeGPT",
        "Documentation": "https://github.com/yourusername/TreeGPT/blob/main/README.md",
    },
)