from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mezo-bca-framework",
    version="0.1.0",
    author="Skripsi Research",
    author_email="example@example.com",
    description="MeZO + Block Circulant Analysis Framework for CPU-efficient LLM Fine-tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mezo-bca-framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "cpu": [
            "intel-openmp",
            "mkl",
        ],
        "visualization": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mezo-bca-train=src.trainer:main",
            "mezo-bca-demo=demo:full_demo",
        ],
    },
)