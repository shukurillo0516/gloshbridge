import numpy as np
import pandas as pd
from typing import Union
from .backends import OutlierComputeBackend, PythonBackend

class GLOSHBridge:
    """
    GLOSH Bridge delegates calculation to backend strategies.
    By default it uses the Python backend.
    """

    def __init__(self, backend: OutlierComputeBackend = None):
        self.backend = backend if backend is not None else PythonBackend()

    def fit_predict(self, data: Union[np.ndarray, pd.DataFrame, str], min_pts: int, min_clsize: int | None = None) -> np.ndarray:
        if isinstance(data, str):
            # Treat as file path
            data = pd.read_csv(data)
            data.drop(columns=[col for col in ["outlier", "outliers"] if col in data.columns], inplace=True)
            
        if min_clsize is None:
            min_clsize = min_pts - 1
            
        return self.backend.calculate(data, min_pts, min_clsize)
