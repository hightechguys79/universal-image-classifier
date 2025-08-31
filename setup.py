"""
Setup script for Universal Image Classifier
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="universal-image-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A flexible, PyTorch-based image classification framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/2hightechguys/universal-image-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0", 
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "wandb": ["wandb>=0.12.0"],
        "huggingface": ["huggingface-hub>=0.10.0", "transformers>=4.20.0"],
    },
    entry_points={
        "console_scripts": [
            "uic-train=examples.train_example:main",
            "uic-infer=examples.inference_example:main",
        ],
    },
    keywords="image-classification pytorch machine-learning deep-learning computer-vision",
    project_urls={
        "Bug Reports": "https://github.com/2hightechguys/universal-image-classifier/issues",
        "Source": "https://github.com/2hightechguys/universal-image-classifier",
        "Documentation": "https://github.com/2hightechguys/universal-image-classifier#readme",
    },
)
