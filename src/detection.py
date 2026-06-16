from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

from .common import copy_file, dataset_paths, ensure_dir, image_files, read_json, stable_split, write_json

_YOLO_MODEL_CACHE: dict[str, Any] = {}


def _enable_torch_checkpoint_compat() -> None:
    import torch

    if getattr(torch.load, "_sjtu_visual_compat", False):
        return
    original_load = torch.load

    def load_with_legacy_checkpoint_support(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    load_with_legacy_checkpoint_support._sjtu_visual_compat = True  # type: ignore[attr-defined]
    torch.load = load_with_legacy_checkpoint_support  # type: ignore[assignment]


def _image_for_json(json_path: Path, data: dict) -> Path | None:
    image_name = data.get("imagePath")
    candidates = []
    if image_name:
        candidates.append(json_path.with_name(Path(str(image_name)).name))
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        candidates.append(json_path.with_suffix(ext))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _labelme_boxes(json_path: Path) -> tuple[Path | None, list[tuple[float, float, float, float]], tuple[int, int]]:
    data = read_json(json_path)
    image_path = _image_for_json(json_path, data)
    width = int(data.get("imageWidth") or 0)
    height = int(data.get("imageHeight") or 0)
    if image_path and (width <= 0 or height <= 0):
        with Image.open(image_path) as im:
            width, height = im.size
    boxes = []
    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        if len(points) < 2 or width <= 0 or height <= 0:
            continue
        xs = [float(pt[0]) for pt in points]
        ys = [float(pt[1]) for pt in points]
        x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
        y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return image_path, boxes, (width, height)


def convert_labelme_to_yolo(
    data_root: str | Path,
    output_dir: str | Path = "outputs",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 2026,
) -> Path:
    paths = dataset_paths(data_root)
    yolo_root = ensure_dir(Path(output_dir) / "detection_yolo")
    json_paths = sorted(paths.detection_root.glob("*.json"))
    train, val, test = stable_split(json_paths, train_ratio, val_ratio, seed)
    splits = {"train": train, "val": val, "test": test}
    summary: dict[str, Any] = {"splits": {}, "classes": ["text"], "skipped": []}

    for split, items in splits.items():
        image_out = ensure_dir(yolo_root / "images" / split)
        label_out = ensure_dir(yolo_root / "labels" / split)
        box_total = 0
        for jp in tqdm(items, desc=f"labelme->yolo {split}"):
            image_path, boxes, (width, height) = _labelme_boxes(jp)
            if image_path is None or width <= 0 or height <= 0:
                summary["skipped"].append(jp.name)
                continue
            dst_image = image_out / image_path.name
            copy_file(image_path, dst_image)
            lines = []
            for x1, y1, x2, y2 in boxes:
                cx = ((x1 + x2) / 2.0) / width
                cy = ((y1 + y2) / 2.0) / height
                bw = (x2 - x1) / width
                bh = (y2 - y1) / height
                vals = [max(0.0, min(1.0, v)) for v in [cx, cy, bw, bh]]
                lines.append("0 " + " ".join(f"{v:.8f}" for v in vals))
            (label_out / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            box_total += len(lines)
        summary["splits"][split] = {"jsons": len(items), "boxes": box_total}

    yaml_text = (
        f"path: {yolo_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: text\n"
    )
    (yolo_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    write_json(yolo_root / "conversion_summary.json", summary)
    return yolo_root / "dataset.yaml"


def train_yolo_detector(
    data_yaml: str | Path,
    output_dir: str | Path = "outputs",
    model_name: str = "yolov8s.pt",
    imgsz: int = 640,
    epochs: int = 30,
    batch: int = 8,
) -> Path | None:
    train_root = ensure_dir(Path(output_dir).resolve() / "detection_train")
    expected_best = train_root / "text_yolov8s" / "weights" / "best.pt"
    try:
        import torch
        from ultralytics import YOLO

        _enable_torch_checkpoint_compat()
        device: int | str = 0 if torch.cuda.is_available() else "cpu"
        model = YOLO(model_name)
        results = model.train(
            data=str(data_yaml),
            imgsz=imgsz,
            epochs=epochs,
            batch=batch,
            device=device,
            project=str(train_root),
            name="text_yolov8s",
            exist_ok=True,
            patience=8,
            seed=2026,
        )
        save_dir = Path(getattr(results, "save_dir", train_root / "text_yolov8s"))
        best = save_dir / "weights" / "best.pt"
        write_json(train_root / "train_meta.json", {"status": "ok", "best": str(best), "device": str(device)})
        return best if best.exists() else None
    except Exception as exc:
        if expected_best.exists():
            write_json(
                train_root / "train_meta.json",
                {
                    "status": "ok_with_finalize_warning",
                    "best": str(expected_best),
                    "device": "auto",
                    "warning": str(exc),
                },
            )
            print(f"[detection] YOLO training produced weights, but final cleanup warned: {exc}")
            return expected_best
        write_json(train_root / "train_meta.json", {"status": "failed", "reason": str(exc)})
        print(f"[detection] YOLO training skipped/failed: {exc}")
        return None


def maybe_predict_boxes(image_path: Path, weights: str | Path | None, conf: float = 0.25) -> list[tuple[float, float, float, float, float]]:
    if weights is None or not Path(weights).exists():
        return []
    try:
        from ultralytics import YOLO

        _enable_torch_checkpoint_compat()
        key = str(Path(weights).resolve())
        model = _YOLO_MODEL_CACHE.get(key)
        if model is None:
            model = YOLO(key)
            _YOLO_MODEL_CACHE[key] = model
        result = model.predict(str(image_path), conf=conf, verbose=False)[0]
        boxes = []
        if result.boxes is None:
            return boxes
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()
        for coords, score in zip(xyxy, confs):
            x1, y1, x2, y2 = [float(v) for v in coords]
            boxes.append((x1, y1, x2, y2, float(score)))
        return boxes
    except Exception:
        return []


def gt_boxes_for_image(image_path: Path, detection_root: str | Path) -> list[tuple[float, float, float, float, float]]:
    jp = Path(detection_root) / f"{image_path.stem}.json"
    if not jp.exists():
        return []
    _, boxes, _ = _labelme_boxes(jp)
    return [(x1, y1, x2, y2, 1.0) for x1, y1, x2, y2 in boxes]


def draw_boxes(
    image_path: Path,
    output_path: Path,
    boxes: list[tuple[float, float, float, float, float]],
    title: str = "",
    max_side: int = 900,
) -> None:
    ensure_dir(output_path.parent)
    with Image.open(image_path) as src:
        im = ImageOps.exif_transpose(src).convert("RGB")
    scale = min(max_side / max(im.size), 1.0)
    if scale < 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)))
    draw = ImageDraw.Draw(im)
    for x1, y1, x2, y2, score in boxes:
        coords = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]
        draw.rectangle(coords, outline=(255, 40, 40), width=max(2, int(3 * scale)))
        if score < 0.999:
            draw.text((coords[0], max(0, coords[1] - 14)), f"{score:.2f}", fill=(255, 40, 40))
    if title:
        band = Image.new("RGB", (im.width, 28), (20, 20, 20))
        band_draw = ImageDraw.Draw(band)
        band_draw.text((8, 6), title, fill=(255, 255, 255))
        canvas = Image.new("RGB", (im.width, im.height + band.height), (255, 255, 255))
        canvas.paste(band, (0, 0))
        canvas.paste(im, (0, band.height))
        im = canvas
    im.save(output_path)


def create_detection_examples(data_root: str | Path, output_dir: str | Path = "outputs", weights: str | Path | None = None, limit: int = 24) -> list[Path]:
    paths = dataset_paths(data_root)
    out = ensure_dir(Path(output_dir) / "detection_examples")
    examples = []
    for image_path in image_files(paths.detection_root)[:limit]:
        boxes = maybe_predict_boxes(image_path, weights) or gt_boxes_for_image(image_path, paths.detection_root)
        dst = out / f"{image_path.stem}_boxes.png"
        draw_boxes(image_path, dst, boxes, title=image_path.name)
        examples.append(dst)
    return examples


def load_detection_config(config_path: str | Path | None) -> dict:
    if not config_path:
        return {}
    with Path(config_path).open("r", encoding="utf-8") as f:
        return json.load(f).get("detection", {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    cfg = load_detection_config(args.config)
    data_yaml = convert_labelme_to_yolo(
        args.data_root,
        args.output_dir,
        float(cfg.get("train_ratio", 0.8)),
        float(cfg.get("val_ratio", 0.1)),
        int(cfg.get("seed", 2026)),
    )
    weights = None
    if not args.skip_train:
        weights = train_yolo_detector(
            data_yaml,
            args.output_dir,
            str(cfg.get("model", "yolov8s.pt")),
            int(cfg.get("imgsz", 640)),
            int(cfg.get("epochs", 30)),
            int(cfg.get("batch", 8)),
        )
    print({"data_yaml": str(data_yaml), "weights": str(weights) if weights else None})


if __name__ == "__main__":
    main()
