from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common import ensure_dir
from src.detection import convert_labelme_to_yolo, create_detection_examples, train_yolo_detector
from src.retrieval import evaluate_retrieval
from src.validate_dataset import validate_dataset
from src.visualize import create_retrieval_detection_cases


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SJTU visual impression experiment 2 pipeline.")
    parser.add_argument("--data-root", required=True, help="Path to 交大视觉印象数据集2026")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--skip-train", action="store_true", help="Convert detection data but skip YOLO training.")
    parser.add_argument("--fast", action="store_true", help="Reduce detector epochs for a quick verification run.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)

    print("[1/5] validating dataset")
    validate_dataset(args.data_root, output_dir)

    print("[2/5] running image retrieval")
    evaluate_retrieval(args.data_root, output_dir, cfg.get("retrieval", {}))

    print("[3/5] converting detection labels")
    det_cfg = cfg.get("detection", {})
    data_yaml = convert_labelme_to_yolo(
        args.data_root,
        output_dir,
        float(det_cfg.get("train_ratio", 0.8)),
        float(det_cfg.get("val_ratio", 0.1)),
        int(det_cfg.get("seed", 2026)),
    )

    weights = None
    if not args.skip_train:
        print("[4/5] training YOLO detector")
        epochs = 2 if args.fast else int(det_cfg.get("epochs", 30))
        weights = train_yolo_detector(
            data_yaml,
            output_dir,
            str(det_cfg.get("model", "yolov8s.pt")),
            int(det_cfg.get("imgsz", 640)),
            epochs,
            int(det_cfg.get("batch", 8)),
        )
    else:
        print("[4/5] YOLO training skipped by flag")

    print("[5/5] creating visualizations")
    vis_cfg = cfg.get("visualization", {})
    create_detection_examples(args.data_root, output_dir, weights, limit=24)
    cases = create_retrieval_detection_cases(
        args.data_root,
        output_dir,
        weights,
        int(vis_cfg.get("cases_per_class", 2)),
        int(vis_cfg.get("retrieved_per_case", 3)),
    )

    print("Done.")


if __name__ == "__main__":
    main()

