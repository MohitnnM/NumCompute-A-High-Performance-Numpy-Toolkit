# NumCompute-A-High-Performance-Numpy-Toolkit
A Numpy based scientific computing toolkit that features preprocessing, statistics, metrics, ranking, numerical gradients, pipelines and performance benchmarking.

## Modules

| Module | What it provides | Status |
|---|---|---|
| `io` | `load_csv`: load a CSV into a NumPy array, with custom delimiters and NaN for missing values | Done |
| `preprocessing` | `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `SimpleImputer` | Done |
| `optim` | `grad` and `jacobian` by finite differences (forward or central) | Done |
| `sort_search` | Stable and multi-key sort, `topk`, quickselect, `binary_search` | Done |
| `rank` | `rank` with tie handling, `percentile` | Done |
| `stats` | Descriptive statistics, histogram, quantiles | Done |
| `metrics` | Classification metrics, confusion matrix, `mse` | Done |
| `pipeline` | `Pipeline` for chaining transformers and models | Done |
| `benchmarking` | Performance comparisons | In progress |
| `utils`, `exceptions` | Shared helpers (`check_array`, `check_is_fitted`) and `NotFittedError` | Done |

Module owners: update your row when you push.

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

## Quickstart

```python
from numcompute.io import load_csv
from numcompute.preprocessing import SimpleImputer, StandardScaler

X = load_csv("tests/test_csv_files/missing_data_multiple.csv")
X = SimpleImputer(fill_value=0.0).fit_transform(X)
X_scaled = StandardScaler().fit_transform(X)
```

For a full walkthrough of every module, open [`demo/quickstart.ipynb`](demo/quickstart.ipynb) and choose the `.venv` Python as the kernel. If VS Code asks to install `ipykernel`, accept.

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

## Conventions

- **Tests:** one file per module, named `tests/<module>_test.py`, written with `unittest`.
- **Input checks:** use `check_array` from `numcompute.utils` to validate array input. It rejects 1D input; reshape with `X.reshape(-1, 1)`.
- **Fitted attributes:** anything learned in `fit` ends with `_` (e.g. `self.mean_`) and is created only inside `fit`. Call `check_is_fitted` at the start of `transform` and `predict`.
- **Private helpers:** functions used only inside one module start with `_` (e.g. `_check_inputs`).
- **Docstrings:** include Parameters, Returns, Raises, and Time and Space Complexity (see `preprocessing.py`).
- **Quickstart notebook:** add a short section for your module to `demo/quickstart.ipynb`. Clear all outputs before committing to avoid merge conflicts.

## Adding dependencies

```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # development-only dependency
```

Both commands update `pyproject.toml` and `uv.lock`, so commit both files.
