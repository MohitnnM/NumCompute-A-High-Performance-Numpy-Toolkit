# NumCompute-A-High-Performance-Numpy-Toolkit
A Numpy based scientific computing toolkit that features preprocessing, statistics, metrics, ranking, numerical gradients, pipelines, decision trees, visualisation and performance benchmarking.

This toolkit supports both traditional batch processing and incremental streaming workflows using partial_fit() and per-chunk updates.

## Modules

| Module | What it provides | Status |
|---|---|---|
| `io` | `load_csv`: load a CSV into a NumPy array, with custom delimiters and NaN for missing values | Done |
| `preprocessing` | `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `SimpleImputer`, including incremental partial_fit() support | Done |
| `optim` | `grad` and `jacobian` by finite differences (forward or central) | Done |
| `sort_search` | Stable and multi-key sort, `topk`, quickselect, `binary_search` | Done |
| `rank` | `rank` with tie handling, `percentile` | Done |
| `stats` | Descriptive statistics, histogram, quantiles and `Streaming Stats` for incremental statistics | Done |
| `metrics` | Streaming `Accuracy`, `Precision`, `Recall`, `F1`, `ConfusionMatrix`, `MSE`, `AUC`, and `RollingWindowMetric` | Done |
| `tree` | `DecisionTreeClassifier` with Gini/entropy splitting and buffered incremental retraining | Done |
| `pipeline` | `Pipeline` for chaining transformers and estimators with batch and incremental training | Done |
| `stream` | `StreamTraniner` for traning and evaluating models chunk-by-chunl with metric and memory logging | Done |
| `visualis` | Reusable Matplotlib functions for metrics, model comparisions and predictions | Done |
| `benchmarking` | Performance comparisons for loop/vectorised operations and decision-tree buffer sizes | Done |
| `utils`, `exceptions` | Shared helpers (`check_array`, `check_is_fitted`) and `NotFittedError` | Done |

## Streaming Support
The NumCompute package support incremental processing of datasets, arriving in chunks instead of the entire dataset being processed at once.

The functionality includes:
   - Incremental preprocessing using `partial_fit()`
   - Running statistics using `StreamingStats`
   - Metrics that acumulate results across multiple chunks of data
   - Rolling-window metrics
   - Incremental buffered training for the `DecisionTreeClassifier`
   - Incremental pipelines
   - Chunk-based model training and evaluation with StreamTrainer
   - Streaming performance benchmarks

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

### Batch preprocessing
```python
from numcompute.io import load_csv
from numcompute.preprocessing import SimpleImputer, StandardScaler

X = load_csv("tests/test_csv_files/missing_data_multiple.csv")
X = SimpleImputer(fill_value=0.0).fit_transform(X)
X_scaled = StandardScaler().fit_transform(X)
```
### Incremental preprocessing

```python
import numpy as np
from numcompute.preprocessing import StandardScaler

X1 = np.array([[1.0, 10.0], [2.0, 20.0]])
X2 = np.array([[3.0, 30.0], [4.0, 40.0]])

scaler = StandardScaler()

scaler.partial_fit(X1)
scaler.partial_fit(X2)

X_scaled = scaler.transform(X2)
```

### Streaming metrics
```python
import numpy as np
from numcompute.metrics import Accuracy

metric = Accuracy()

metric.update(np.array([1, 0]), np.array([1, 1]))
metric.update(np.array([1, 0]), np.array([1, 0]))

print(metric.result())
```

### Streaming decision tree
```python
import numpy as np
from numcompute.tree import DecisionTreeClassifier

X = np.array([[1.0], [2.0], [3.0], [4.0]])

y = np.array([0, 0, 1, 1])

tree = DecisionTreeClassifier(max_depth=3, buffer_size=4)

tree.partial_fit(X[:2], y[:2])
tree.partial_fit(X[2:], y[2:])

pred = tree.predict(X)
```

### Pipeline
```python
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler
from numcompute.tree import DecisionTreeClassifier

pipeline = Pipeline(
    [
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=3, buffer_size=4)),
    ]
)
```

For a full walkthrough of every module, open [`demo/quickstart.ipynb`](demo/quickstart.ipynb) and choose the `.venv` Python as the kernel. If VS Code asks to install `ipykernel`, accept.

## Running the tests

```bash
uv run python -m unittest discover -s tests -p "*_test.py"
```

Currently there are a total of 185 tests that cover batch operations, streaming functionality, pipelines, decision trees, metrics, statistics, visualisation and benchmarking.

## Benchmarking

The NumCompute package also includes benchmarking where different implementations and streaming strategies are compared. 

### Streaming StandardScaler

This benchmark compares:
   - Python loop-based incremental statistics
   - Vectorised incremental statistics

### Decision Tree Buffer Sizes

This benchmark evaluates the differen `buffer_size` values by measuring:
   - Training time
   - The final prediction accuracy

The benchmark timings are machine-dependatn so they should be interpreted comparitively, instead of using them as fixed performance guarantees. 

## Visualisation
This module provides resuable Matplotlib function:

```python
plot_metric_over_time(...)
compare_model(...)
plot_predictions_vs_ground_truth(...)
```

The plots can be saved to a file or displayed interactively.

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
