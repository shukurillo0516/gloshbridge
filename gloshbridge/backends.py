import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd

from .glosh_calc_cor import GLOSH
from .utils import extract_java_outliers_data_from_txt

BASE_DIR = Path(__file__).resolve().parent

class OutlierComputeBackend(ABC):
    """Abstract Strategy interface for calculating GLOSH outlier scores."""
    
    @abstractmethod
    def calculate(self, data: Union[np.ndarray, pd.DataFrame], min_pts: int, min_clsize: int) -> np.ndarray:
        pass


class PythonBackend(OutlierComputeBackend):
    def calculate(self, data: Union[np.ndarray, pd.DataFrame], min_pts: int, min_clsize: int) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            data_to_use = data.select_dtypes(include=[np.number]).to_numpy()
        else:
            data_to_use = np.asarray(data)

        calc = GLOSH(data=data_to_use, min_pts=min_pts, min_clsize=min_clsize)
        return calc.calc_glosh_scores()


class RustBackend(OutlierComputeBackend):
    def calculate(self, data: Union[np.ndarray, pd.DataFrame], min_pts: int, min_clsize: int) -> np.ndarray:
        rust_binary = os.path.join(BASE_DIR, "binaries/rust_hdbscan")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as input_file, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as output_file:
            
            input_path = input_file.name
            out_path = output_file.name
            
            if isinstance(data, pd.DataFrame):
                data.to_csv(input_path, index=False)
            else:
                pd.DataFrame(data).to_csv(input_path, index=False, header=[f"col_{i}" for i in range(data.shape[1])])

        try:
            subprocess.run(
                [
                    rust_binary,
                    "--",
                    f"--file_path={input_path}",
                    f"--out_path={out_path}",
                    f"--min_pts={min_pts}",
                    f"--min_clsize={min_clsize}",
                ],
                check=True,
            )
            outlier_scores = np.loadtxt(out_path, delimiter=",")
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(out_path):
                os.remove(out_path)

        return outlier_scores


class JavaBackend(OutlierComputeBackend):
    def calculate(self, data: Union[np.ndarray, pd.DataFrame], min_pts: int, min_clsize: int) -> np.ndarray:
        java_jar = os.path.join(BASE_DIR, "binaries/elki-bundle-0.8.0.jar")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as input_file, \
             tempfile.TemporaryDirectory() as out_dir:
            
            input_path = input_file.name
            
            if isinstance(data, pd.DataFrame):
                original_df = data.copy()
                data.to_csv(input_path, index=False)
            else:
                original_df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(data.shape[1])])
                if data.shape[1] == 2:
                    original_df.columns = ["x", "y"]
                original_df.to_csv(input_path, index=False)

        try:
            subprocess.run(
                [
                    "java",
                    "-cp",
                    java_jar,
                    "elki.application.KDDCLIApplication",
                    "-dbc.in",
                    input_path,
                    "-dbc.parser",
                    "NumberVectorLabelParser",
                    "-algorithm",
                    "outlier.clustering.GLOSH",
                    "-hdbscan.minPts",
                    str(min_pts),
                    "-hdbscan.minclsize",
                    str(min_clsize),
                    "-out",
                    out_dir,
                ],
                check=True,
            )

            scores_file = os.path.join(out_dir, "GLOSH score Order.txt")
            score_df = extract_java_outliers_data_from_txt(scores_file)

            if "x" in original_df.columns and "y" in original_df.columns:
                merged_df = pd.merge(
                    original_df, score_df, on=["x", "y"], how="left"
                )
                return merged_df["outlier_score"].to_numpy()
            else:
                return score_df["outlier_score"].to_numpy()

        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
