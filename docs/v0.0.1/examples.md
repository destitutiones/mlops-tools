# Examples

## Installation/Usage:

As the package has not been published on PyPi yet, it CANNOT be install using
pip.

To get started with `mlops_tools`, you’ll need to use Poetry, a powerful tool
for dependency management and packaging in Python. If you haven’t already
installed Poetry, you can follow the installation instructions provided in the
[official Poetry documentation](https://python-poetry.org/docs/#installation).

Once you have Poetry installed, you can easily set up `mlops_tools` and run its
examples.

### First Project Example

1. Navigate to your project directory.
2. Run the following commands:

```bash
poetry install
poetry run python examples/hw1_usage.py
```

This will install the necessary dependencies and execute the `hw1_usage.py`
script, allowing you to explore the first project example.

### Second Project Example

1. Navigate to your project directory (if not already there).
2. Run the following commands:

```bash
poetry install
pre-commit install
pre-commit run -a
python mlops_tools/train.py
python mlops_tools/infer.py
```

These commands will install the required dependencies, set up pre-commit hooks
for code formatting and linting, and then execute the train.py and infer.py
scripts, demonstrating the second project example.

By following these installation steps, you’ll have `mlops_tools` up and running,
ready for you to explore and utilize its capabilities for your MLOps projects.
