# Contributing to Universal Image Classifier

We love your input! We want to make contributing to Universal Image Classifier as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project.

## Report bugs using GitHub's [issue tracker](https://github.com/2hightechguys/universal-image-classifier/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/2hightechguys/universal-image-classifier/issues/new).

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Development Setup

```bash
# Clone your fork
git clone https://github.com/2hightechguys/universal-image-classifier.git
cd universal-image-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run these before submitting:

```bash
# Format code
black .
isort .

# Check linting
flake8 .

# Type checking
mypy .
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Add tests for new features in the `tests/` directory.

## Documentation
 
- Update docstrings for new functions/classes
- Update README.md if you change functionality
- Add examples for new features

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
