# -*- coding: utf-8 -*-
"""Hybrid CCM color prediction example.

This script demonstrates a simple hybrid color correction matrix (CCM) model
that can handle fluorescent and non-fluorescent color patches.

The dataset is expected to contain the following columns:
    sensor_r, sensor_g, sensor_b  - raw sensor RGB values
    target_L, target_a, target_b  - reference Lab values
    fluorescent                   - 1 if the sample is fluorescent else 0

If no dataset path is provided, values from the ``ColorChecker N Ohta`` dataset
distributed with ``colour-science`` will be used.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

import colour


@dataclass
class CCM:
    """Simple 3x3 color correction matrix."""

    matrix: np.ndarray

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        return rgb @ self.matrix


def fit_ccm(sensors: np.ndarray, targets: np.ndarray) -> CCM:
    """Fit a CCM using linear least squares."""
    matrix, _, _, _ = np.linalg.lstsq(sensors, targets, rcond=None)
    return CCM(matrix)


def generate_colorchecker_dataset() -> pd.DataFrame:
    """Create a dataset from the ColorChecker "N Ohta" spectral data."""
    sd_data = colour.characterisation.datasets.SDS_COLOURCHECKERS[
        "ColorChecker N Ohta"
    ]
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    illuminant = colour.SDS_ILLUMINANTS["D65"]

    records = []
    for name, sd in sd_data.items():
        XYZ = colour.sd_to_XYZ(sd, cmfs, illuminant) / 100
        rgb = np.clip(colour.XYZ_to_sRGB(XYZ), 0.0, 1.0)
        lab = colour.XYZ_to_Lab(XYZ)
        fluorescent = 1.0 if lab[0] > 80 else 0.0
        records.append(np.hstack([rgb, lab, fluorescent]))

    df = pd.DataFrame(
        records,
        columns=[
            "sensor_r",
            "sensor_g",
            "sensor_b",
            "target_L",
            "target_a",
            "target_b",
            "fluorescent",
        ],
    )
    return df


def load_dataset(path: Path | None) -> pd.DataFrame:
    if path is None:
        return generate_colorchecker_dataset()
    return pd.read_csv(path)


def train_models(df: pd.DataFrame) -> Tuple[CCM, CCM, LogisticRegression, np.ndarray, np.ndarray]:
    sensors = df[["sensor_r", "sensor_g", "sensor_b"]].to_numpy()
    targets = df[["target_L", "target_a", "target_b"]].to_numpy()
    fluo = df["fluorescent"].to_numpy()

    X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(
        sensors, targets, fluo, test_size=0.2, random_state=1
    )

    log_reg = LogisticRegression(max_iter=200)
    log_reg.fit(X_train, f_train)

    base_ccm = fit_ccm(X_train[f_train == 0], y_train[f_train == 0])
    fluo_ccm = fit_ccm(X_train[f_train == 1], y_train[f_train == 1])

    return base_ccm, fluo_ccm, log_reg, X_test, y_test


def evaluate(base_ccm: CCM, fluo_ccm: CCM, log_reg: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> float:
    probs = log_reg.predict_proba(X_test)[:, 1]
    pred_base = base_ccm.predict(X_test)
    pred_fluo = fluo_ccm.predict(X_test)
    preds = (1 - probs[:, None]) * pred_base + probs[:, None] * pred_fluo
    mse = mean_squared_error(y_test, preds)
    return mse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid CCM example")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to CSV dataset")
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    base_ccm, fluo_ccm, log_reg, X_test, y_test = train_models(df)
    mse = evaluate(base_ccm, fluo_ccm, log_reg, X_test, y_test)

    print("Base CCM:\n", base_ccm.matrix)
    print("Fluorescent CCM:\n", fluo_ccm.matrix)
    print(f"Test MSE: {mse:.6f}")


if __name__ == "__main__":
    main()
