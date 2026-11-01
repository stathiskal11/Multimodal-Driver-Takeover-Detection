from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_paper_baseline import make_dataset, make_loader, run_epoch
from src.models import PaperTakeoverBaselineModel, TransformerTakeoverModel
from src.training import make_participant_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one saved takeover checkpoint on its held-out participants")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=ROOT / "data/interim/paper_window_index.csv")
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--heatmap-variant", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    split_seed = int(checkpoint["split_seed"])
    participants = make_participant_split(args.index, seed=split_seed)["test"]
    preprocessing = checkpoint.get("preprocessing", {"can_normalization": "legacy", "can_stats": None})
    architecture = checkpoint.get("architecture", "baseline")
    saved_heatmap = checkpoint.get("args", {}).get("heatmap_variant", "sigma64")
    if args.heatmap_variant is not None and args.heatmap_variant != saved_heatmap:
        raise ValueError(f"Checkpoint used heatmap variant {saved_heatmap}")
    model = (PaperTakeoverBaselineModel() if architecture == "baseline" else TransformerTakeoverModel()).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = make_dataset(
        index_path=args.index, bundle_path=args.bundle, heatmap_variant=saved_heatmap,
        participant_ids=participants, sample_ids=None, limit_samples=None, cache_dir=args.cache_dir,
        can_bus_normalization=preprocessing["can_normalization"], can_bus_stats=preprocessing.get("can_stats"),
    )
    loader = make_loader(dataset, args.batch_size, False, args.num_workers, device.type == "cuda")
    metrics = run_epoch(model, loader, nn.BCEWithLogitsLoss(), None, None, device)
    print(f"split_seed={split_seed} architecture={architecture} can_normalization={preprocessing['can_normalization']}")
    print(f"test_roc_auc={metrics.roc_auc} test_loss={metrics.loss} samples={metrics.example_count}")


if __name__ == "__main__":
    main()
