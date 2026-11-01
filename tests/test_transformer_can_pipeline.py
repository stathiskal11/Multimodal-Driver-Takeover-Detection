from __future__ import annotations

import csv
import io
import json
import tarfile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.train_paper_baseline import run_epoch
from src.data.can_normalization import fit_can_zscore
from src.data.hdbd_paper_dataset import _build_normalized_signal_column
from src.models import TransformerTakeoverModel
from src.training.experiment_logging import ExperimentRecorder


def test_can_fit_uses_train_windows_only(tmp_path):
    archive_path = tmp_path / "sessions.tar.gz"
    rows = [
        {"Throttle": "1", "RPM": "10", "Steering": "", "Speed": "2"},
        {"Throttle": "3", "RPM": "10", "Steering": "nan", "Speed": "4"},
        {"Throttle": "999", "RPM": "999", "Steering": "999", "Speed": "999"},
    ]
    data = io.StringIO()
    writer = csv.DictWriter(data, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    payload = data.getvalue().encode()
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("data/0001/session.csv")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    index = pd.DataFrame([{
        "participant_id": "0001", "csv_member": "data/0001/session.csv",
        "window_start_idx": 0, "window_end_idx": 1,
    }])
    stats = fit_can_zscore(archive_path, index, ["0001"])
    assert stats["Throttle"]["mean"] == 2
    assert stats["Throttle"]["std"] == 1
    assert stats["RPM"]["std"] == 1
    assert stats["Steering"]["count"] == 0
    transformed = _build_normalized_signal_column(
        rows[:2], participant_id="0001", column="Throttle", signal_stats={},
        can_bus_normalization="zscore", can_bus_stats=stats,
    )
    np.testing.assert_array_equal(transformed, [-1, 1])
    missing = _build_normalized_signal_column(
        rows[:2], participant_id="0001", column="Steering", signal_stats={},
        can_bus_normalization="zscore", can_bus_stats=stats,
    )
    np.testing.assert_array_equal(missing, [0, 0])


def test_transformer_shapes_checkpoint_and_training_smoke(tmp_path):
    torch.manual_seed(3)
    model = TransformerTakeoverModel()
    samples = [
        {"scene_gaze": torch.rand(2, 30, 32, 32), "signals": torch.rand(30, 6),
         "hmi": torch.eye(9)[i], "label": torch.tensor(float(i))}
        for i in range(2)
    ]
    batch = next(iter(DataLoader(samples, batch_size=2)))
    assert model(batch["scene_gaze"], batch["signals"], batch["hmi"]).shape == (2,)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics = run_epoch(model, DataLoader(samples, batch_size=2), nn.BCEWithLogitsLoss(), optimizer, 1, torch.device("cpu"))
    assert metrics.batch_count == 1 and metrics.roc_auc is not None
    recorder = ExperimentRecorder(experiment_root=tmp_path, run_name="smoke", args={}, report_only=False)
    stats = {name: {"mean": 0.0, "std": 1.0, "count": 2} for name in ("Throttle", "RPM", "Steering", "Speed")}
    path = recorder.save_checkpoint(
        split_seed=0, epoch=1, checkpoint_tag="best", checkpoint_metric_name="val_roc_auc",
        checkpoint_metric_value=0.5, model=model, optimizer=optimizer, train_metrics=metrics,
        val_metrics=metrics, preprocessing={"can_normalization": "zscore", "can_stats": stats},
        architecture="transformer",
    )
    saved = torch.load(path, weights_only=False)
    assert saved["preprocessing"]["can_stats"] == stats
    restored = TransformerTakeoverModel()
    restored.load_state_dict(saved["model_state_dict"])
    restored.eval()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            model(batch["scene_gaze"], batch["signals"], batch["hmi"]),
            restored(batch["scene_gaze"], batch["signals"], batch["hmi"]),
        )


def test_split_result_is_saved_before_all_splits_finish(tmp_path):
    recorder = ExperimentRecorder(experiment_root=tmp_path, run_name="progress", args={}, report_only=False)
    recorder.record_split_setup(
        split_seed=0, participant_splits={"train": ["a"], "val": ["b"], "test": ["c"]},
        full_summaries={}, loaded_summaries={},
    )
    assert recorder.summary_path.exists()
    recorder.record_split_result(
        split_seed=0, train_metrics=None, val_metrics=None,
        test_metrics={"roc_auc": 0.75}, best_checkpoint={}, test_evaluation_checkpoint={},
    )
    saved = json.loads(recorder.summary_path.read_text(encoding="utf-8"))
    assert saved["splits"][0]["final_metrics"]["test"]["roc_auc"] == 0.75
