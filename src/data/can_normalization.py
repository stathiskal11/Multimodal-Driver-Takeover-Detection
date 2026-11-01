from __future__ import annotations

import csv
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from .hdbd_paper_dataset import CAN_BUS_SIGNAL_COLUMNS, _parse_float


def fit_can_zscore(
    csv_archive_path: Path,
    train_index: pd.DataFrame,
    train_participants: list[str],
    *,
    min_std: float = 1e-6,
) -> dict[str, dict[str, float]]:
    """Fit on unique timesteps covered by training windows, never val/test rows."""
    allowed = set(train_participants)
    if not allowed or train_index.empty:
        raise ValueError("Cannot fit CAN statistics on empty training data")
    if not set(train_index["participant_id"].astype(str)).issubset(allowed):
        raise ValueError("Training index contains a participant outside the train split")
    counts = {name: 0 for name in CAN_BUS_SIGNAL_COLUMNS}
    sums = {name: 0.0 for name in CAN_BUS_SIGNAL_COLUMNS}
    sums_sq = {name: 0.0 for name in CAN_BUS_SIGNAL_COLUMNS}
    with tarfile.open(csv_archive_path, "r:gz") as archive:
        for member_name, group in train_index.groupby("csv_member", sort=False):
            member_participant = str(member_name).replace("\\", "/").lstrip("./").split("/")[1]
            if set(group["participant_id"].astype(str)) != {member_participant}:
                raise ValueError(f"Participant mismatch for {member_name}")
            member = archive.extractfile(str(member_name))
            if member is None:
                raise FileNotFoundError(str(member_name))
            rows = list(csv.DictReader(member.read().decode("utf-8", errors="replace").splitlines()))
            selected = np.zeros(len(rows), dtype=bool)
            for row in group.itertuples(index=False):
                start, end = int(row.window_start_idx), int(row.window_end_idx)
                if start < 0 or end >= len(rows) or start > end:
                    raise ValueError(f"Invalid training window in {member_name}: {start}:{end}")
                selected[start : end + 1] = True
            for row in (rows[i] for i in np.flatnonzero(selected)):
                for name in CAN_BUS_SIGNAL_COLUMNS:
                    value = _parse_float(row.get(name))
                    if value is None or not np.isfinite(value):
                        continue
                    counts[name] += 1
                    sums[name] += value
                    sums_sq[name] += value * value
    result = {}
    for name in CAN_BUS_SIGNAL_COLUMNS:
        count = counts[name]
        if count == 0:
            result[name] = {"mean": 0.0, "std": 1.0, "count": 0}
            continue
        mean = sums[name] / count
        variance = max(sums_sq[name] / count - mean * mean, 0.0)
        std = variance**0.5
        result[name] = {"mean": mean, "std": std if std >= min_std else 1.0, "count": count}
    return result
