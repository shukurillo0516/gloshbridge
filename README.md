# GLOSHBridge

**GLOSHBridge** is a Python package that provides a unified interface for calculating **GLOSH** (Global-Local Outlier Score from Hierarchies) outlier scores using different backend implementations: Python, Rust, and Java.

This allows for benchmarking, cross-validating results, and leveraging the most efficient implementation of GLOSH for your dataset.

## Features
- **Python Integration**: Built-in GLOSH calculation utilizing Python's `hdbscan` to extract linkage trees and directly derive GLOSH scores.
- **Rust Bridge**: Interacts with a fast Rust-based HDBSCAN binary executable for highly performant outlier detection.
- **Java Bridge**: Wraps around the robust Java-based ELKI environment to perform GLOSH clustering and outlier ranking.

## Prerequisites

To use all the bridges provided by this package, you must have the following system dependencies installed:
- **Python** `~> 3.10`
- **Cargo / Rust** (To modify or compile the Rust binary locally)
- **Java** (JRE/JDK, to run the ELKI jar)

## Installation

You can clone the repository and install the package locally:

```bash
git clone https://github.com/shukurillo0516/gloshbridge.git
cd gloshbridge
pip install -e .
```

*Key Python dependencies include `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `hdbscan==0.8.40`.*

## Usage

**GLOSHBridge** implements the Strategy Design Pattern. You can pass your data as a physical `.csv` file, a `pandas.DataFrame`, or a `numpy.ndarray` directly. You can inject whichever compute backend you prefer:

```python
import pandas as pd
from gloshbridge import GLOSHBridge, PythonBackend, RustBackend, JavaBackend

# 1. Load or define your data
# Can be a file path, Numpy array, or Pandas DataFrame
file_path = "datasets/your_data.csv"
data = pd.read_csv(file_path)

min_pts = 5
min_clsize = 4

# 2. Calculate using the Python Backend (Default)
bridge_py = GLOSHBridge(backend=PythonBackend())
py_scores = bridge_py.fit_predict(data, min_pts=min_pts, min_clsize=min_clsize)
print("Python GLOSH Scores:", py_scores)

# 3. Calculate using the fast Rust Backend
bridge_rust = GLOSHBridge(backend=RustBackend())
rust_scores = bridge_rust.fit_predict(data, min_pts=min_pts, min_clsize=min_clsize)
print("Rust GLOSH Scores:", rust_scores)

# 4. Calculate using the Java Backend (ELKI)
bridge_java = GLOSHBridge(backend=JavaBackend())
java_scores = bridge_java.fit_predict(data, min_pts=min_pts, min_clsize=min_clsize)
print("Java GLOSH Scores:", java_scores)
```

## How It Works

- **Python Backend**: Leverages `hdbscan` to build a single linkage tree. It traverses the tree to extract the minimal cluster sizes and distance variations to calculate the $1 - \frac{\epsilon_{min}}{\epsilon}$ scores. *(See `glosh_calc_cor.py`)*
- **Rust Backend**: Uses Python's secure `tempfile` library to stream your data into a temporary `.csv`, then calls the highly performant `rust_hdbscan` binary natively in Rust. The temporary formats are automatically cleaned up afterward, keeping your runtime fast and footprint secure.
- **Java Backend**: Functions similarly to the Rust bridge, spawning a safe thread subprocess to execute the official Java ELKI library (`elki-bundle-0.8.0.jar`) with `outlier.clustering.GLOSH`. Output parsing automatically handles results cleanly through secure temp buffers.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.