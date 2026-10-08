"""ETTh1 loading, train-only scaling, and Informer window datasets."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class InformerDataConfig:
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 24
    hours_per_month: int = 30 * 24

    @property
    def train_end(self) -> int:
        return 12 * self.hours_per_month

    @property
    def val_end(self) -> int:
        return self.train_end + 4 * self.hours_per_month

    @property
    def test_end(self) -> int:
        return self.val_end + 4 * self.hours_per_month


def build_time_marks(dates: pd.Series) -> np.ndarray:
    """Return [month, day, day_of_week, hour] for every timestamp."""
    dates = pd.to_datetime(dates)
    return np.stack(
        [
            dates.dt.month.to_numpy(),
            dates.dt.day.to_numpy(),
            dates.dt.dayofweek.to_numpy(),
            dates.dt.hour.to_numpy(),
        ],
        axis=1,
    ).astype(np.int64)


def load_etth1(
    data_path: Path,
    config: InformerDataConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Read ETTh1 and standardize every value column using train statistics only."""
    df = pd.read_csv(data_path)
    expected_columns = [
        "date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT",
    ]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    if df.isna().any().any():
        raise ValueError("ETTh1 contains missing values; clean them before training.")
    if df["date"].duplicated().any() or not df["date"].is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and sorted ascending.")
    if len(df) < config.test_end:
        raise ValueError(f"Need at least {config.test_end} rows, found {len(df)}.")

    feature_columns = [column for column in df.columns if column != "date"]
    values = df[feature_columns].to_numpy(dtype=np.float32)
    train_values = values[:config.train_end]
    mean = train_values.mean(axis=0, keepdims=True)
    std = train_values.std(axis=0, keepdims=True)
    if np.any(std == 0):
        raise ValueError("A training feature has zero standard deviation.")

    scaled_values = (values - mean) / std
    time_marks = build_time_marks(df["date"])
    return df, scaled_values, time_marks, feature_columns, mean, std


class InformerWindowDataset(Dataset):
    """Samples whose *prediction targets* belong to one time split.

    Each sample returns:
      x       : past seq_len values
      y       : last label_len known values + next pred_len true labels
      x_mark  : calendar markers for x
      y_mark  : calendar markers for y
    """

    def __init__(
        self,
        values: np.ndarray,
        time_marks: np.ndarray,
        config: InformerDataConfig,
        target_start: int,
        target_end: int,
    ):
        if target_start < config.seq_len:
            raise ValueError("target_start needs at least seq_len historical points.")
        if target_end > len(values):
            raise ValueError("target_end exceeds the available data.")

        self.values = values
        self.time_marks = time_marks
        self.config = config
        # target_time is the first of the pred_len future labels.
        self.target_times = range(target_start, target_end - config.pred_len + 1)

    def __len__(self) -> int:
        return len(self.target_times)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_time = self.target_times[index]
        s_begin = target_time - self.config.seq_len
        s_end = target_time
        r_begin = s_end - self.config.label_len
        r_end = target_time + self.config.pred_len

        return (
            torch.from_numpy(self.values[s_begin:s_end]).float(),
            torch.from_numpy(self.values[r_begin:r_end]).float(),
            torch.from_numpy(self.time_marks[s_begin:s_end]).long(),
            torch.from_numpy(self.time_marks[r_begin:r_end]).long(),
        )
