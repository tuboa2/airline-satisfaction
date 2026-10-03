"""
Utility Functions: Memory Optimization, Hardware Detection, and Logging
"""

import logging
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd


def seed_everything(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def get_logger(name: str = "AirlineML") -> logging.Logger:
    """Configures a clean console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Iterates through all numerical columns of a dataframe and downcasts
    the data types to prevent Out-Of-Memory (OOM) errors on Kaggle.
    """
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and not pd.api.types.is_categorical_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        logging.info(
            f"Memory reduction: {start_mem:.2f} MB -> {end_mem:.2f} MB (-{reduction:.1f}%)"
        )

    return df


def detect_hardware() -> tuple[str, int]:
    """
    Detects whether an NVIDIA GPU is available and determines optimal thread count.
    Returns: (device_type, num_threads)
    """
    num_cpus = os.cpu_count() or 4
    gpu_available = False

    try:
        import torch

        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            logging.info(f"NVIDIA GPU Detected: {gpu_name}")
    except ImportError:
        pass

    device = "cuda" if gpu_available else "cpu"
    logging.info(
        f"Hardware Configuration: Device={device.upper()} | CPU Threads={num_cpus}"
    )
    return device, num_cpus


@contextmanager
def timer(name: str):
    """Context manager to measure and log execution time."""
    t0 = time.time()
    yield
    logging.info(f"[{name}] completed in {time.time() - t0:.2f} seconds.")


def resolve_binary_target(series: pd.Series | np.ndarray) -> np.ndarray:
    """
    Robustly resolves binary satisfaction targets across all dtypes:
    boolean (True/False), numeric (1/0 or 1.0/0.0), and text
    ('satisfied'/'neutral or dissatisfied', 'true'/'false', '1'/'0').
    Returns an int32 numpy array with values in {0, 1}.
    """
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    elif not isinstance(series, pd.Series):
        series = pd.Series(list(series))

    if pd.api.types.is_bool_dtype(series):
        return series.astype(np.int32).values
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) == 1.0).astype(np.int32).values

    val_str = series.astype(str).str.lower().str.strip()
    is_pos = val_str.isin(["satisfied", "true", "1", "t", "yes", "y"])
    return is_pos.astype(np.int32).values
