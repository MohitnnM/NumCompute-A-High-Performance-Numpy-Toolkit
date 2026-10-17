# NumCompute-A-High-Performance-Numpy-Toolkit
A Numpy based scientific computing toolkit that features preprocessing, statistics, metrics, ranking, numerical gradients, pipelines and performance benchmarking.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage its Python environment and dependencies. Python 3.12 or newer is required.

1. Install uv (if not already installed):

   ```bash
   # macOS (Homebrew)
   brew install uv

   # macOS / Linux (standalone installer)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create the virtual environment and install dependencies:

   ```bash
   uv sync
   ```

   This creates a `.venv/` folder in the project root, installs NumPy, and installs `numcompute` in editable mode. Exact versions are pinned in `uv.lock`.

3. (Optional) Activate the environment:

   ```bash
   source .venv/bin/activate
   ```

   Alternatively, prefix commands with `uv run` to use the environment without activating it.

## Running the tests

```bash
uv run python -m unittest discover -s tests -p "*_test.py"
```

## Formatting and linting

This project uses [Ruff](https://docs.astral.sh/ruff/) as both the code formatter and the linter. It is installed automatically as a development dependency by `uv sync`, and its settings live under `[tool.ruff]` in `pyproject.toml`.

```bash
uv run ruff format .        # format all code
uv run ruff check .         # report lint issues
uv run ruff check . --fix   # automatically fix the issues that can be fixed safely
```

To preview changes without modifying any files:

```bash
uv run ruff format --check --diff .
```

Run both `ruff format` and `ruff check` before committing.

## Adding dependencies

```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # development-only dependency
```

Both commands update `pyproject.toml` and `uv.lock`, so commit both files.
