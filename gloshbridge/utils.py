import pandas as pd
import numpy as np


def z_score_normalize(df: pd.DataFrame):
    for col in df.columns:
        df[col] = (df[col].astype(float) - df[col].astype(float).mean()) / df[col].astype(float).std()
    return df


def get_txt_file_in_df(path: str):
    rows = [["x", "y", "cluster"]]
    with open(path, "r") as file:
        for i, line in enumerate(file):
            if i < 7:
                continue
            row = line.strip().split("\t")
            if len(row) == 3:
                rows.append(row)
            else:
                print(row)
    df = pd.DataFrame(columns=rows[0], data=rows[1:])
    df = z_score_normalize(df)

    return df


def add_new_noise_points(df, num_points=100, noise_std=0.1):
    # Sample from existing x and y columns
    x_orig = df["x"].astype(float).to_numpy()
    y_orig = df["y"].astype(float).to_numpy()

    # Randomly sample indices
    indices = np.random.choice(len(df), size=num_points)

    # Create noisy new points
    new_x = x_orig[indices] + np.random.normal(0, noise_std, num_points)
    new_y = y_orig[indices] + np.random.normal(0, noise_std, num_points)

    # Optional: assign a special cluster label, e.g., -1 for noise
    new_cluster = [-1] * num_points

    # Create new DataFrame
    new_points = pd.DataFrame(
        {
            "x": new_x,
            "y": new_y,
            "cluster": new_cluster,
        }
    )

    # Combine with original
    return pd.concat([df, new_points], ignore_index=True)


def extract_java_outliers_data_from_txt(path: str = "out/GLOSH score Order.txt") -> pd.DataFrame:
    cols = ["x", "y", "outlier_score"]
    data = []
    with open(path, "r") as file:
        for line in file:
            vals = line.split(" ")
            x, y = map(float, vals[1:3])
            gl_score = float(vals[-1].split("=")[-1])
            data.append([x, y, gl_score])

    return pd.DataFrame(data, columns=cols)
