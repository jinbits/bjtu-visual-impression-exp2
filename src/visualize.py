from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .common import LANDMARKS, dataset_paths, ensure_dir, image_files, prefix_from_path, write_json
from .detection import gt_boxes_for_image, maybe_predict_boxes


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _thumb(path: Path, label: str, boxes: list[tuple[float, float, float, float, float]] | None, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as src:
        im = ImageOps.exif_transpose(src).convert("RGB")
    original_w, original_h = im.size
    im.thumbnail((size[0], size[1] - 34), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (246, 248, 250))
    x = (size[0] - im.width) // 2
    y = 34 + (size[1] - 34 - im.height) // 2
    canvas.paste(im, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(210, 214, 220), width=1)
    draw.rectangle([0, 0, size[0], 32], fill=(30, 42, 56))
    draw.text((8, 9), label[:42], fill=(255, 255, 255))
    if boxes:
        sx = im.width / original_w
        sy = im.height / original_h
        for x1, y1, x2, y2, score in boxes:
            coords = [x + x1 * sx, y + y1 * sy, x + x2 * sx, y + y2 * sy]
            draw.rectangle(coords, outline=(232, 47, 47), width=3)
            if score < 0.999:
                draw.text((coords[0], max(34, coords[1] - 14)), f"{score:.2f}", fill=(232, 47, 47))
    return canvas


def create_retrieval_detection_cases(
    data_root: str | Path,
    output_dir: str | Path = "outputs",
    weights: str | Path | None = None,
    cases_per_class: int = 2,
    retrieved_per_case: int = 3,
) -> list[Path]:
    paths = dataset_paths(data_root)
    output_dir = Path(output_dir)
    top_rows = _read_csv(output_dir / "retrieval" / "top60_results.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in top_rows:
        if int(row.get("rank", "0")) <= retrieved_per_case:
            grouped[row["query"]].append(row)
    query_by_name = {p.name: p for p in image_files(paths.query_root)}
    queries_by_class: dict[str, list[Path]] = defaultdict(list)
    for p in image_files(paths.query_root):
        queries_by_class[prefix_from_path(p)].append(p)

    case_dir = ensure_dir(output_dir / "visualizations" / "cases")
    outputs = []
    for lm in LANDMARKS:
        for qpath in queries_by_class.get(lm, [])[:cases_per_class]:
            rows = sorted(grouped.get(qpath.name, []), key=lambda r: int(r["rank"]))
            if not rows:
                continue
            tile_w, tile_h = 320, 260
            tiles = [_thumb(qpath, f"Query {qpath.name}", None, (tile_w, tile_h))]
            for row in rows[:retrieved_per_case]:
                bpath = Path(row["base_path"])
                boxes = maybe_predict_boxes(bpath, weights) or gt_boxes_for_image(bpath, paths.detection_root)
                label = f"Top {row['rank']} {row['base']} [{row['base_class']}]"
                tiles.append(_thumb(bpath, label, boxes, (tile_w, tile_h)))
            canvas = Image.new("RGB", (tile_w * len(tiles), tile_h + 46), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            title = f"{lm}: image retrieval + text detection"
            draw.rectangle([0, 0, canvas.width, 44], fill=(14, 74, 113))
            draw.text((14, 14), title, fill=(255, 255, 255))
            for i, tile in enumerate(tiles):
                canvas.paste(tile, (i * tile_w, 46))
            out = case_dir / f"{lm}_{qpath.stem}_case.png"
            canvas.save(out)
            outputs.append(out)
    write_json(output_dir / "visualizations" / "case_manifest.json", [str(p) for p in outputs])
    return outputs


def load_visual_config(config_path: str | Path | None) -> dict:
    if not config_path:
        return {}
    with Path(config_path).open("r", encoding="utf-8") as f:
        return json.load(f).get("visualization", {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()
    cfg = load_visual_config(args.config)
    cases = create_retrieval_detection_cases(
        args.data_root,
        args.output_dir,
        args.weights,
        int(cfg.get("cases_per_class", 2)),
        int(cfg.get("retrieved_per_case", 3)),
    )
    print({"cases": len(cases)})

if __name__ == "__main__":
    main()

