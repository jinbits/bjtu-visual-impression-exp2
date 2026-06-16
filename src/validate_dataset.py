from __future__ import annotations

import argparse
import collections
from pathlib import Path

from PIL import Image

from .common import LANDMARKS, dataset_paths, image_files, prefix_from_path, read_json, write_json


def validate_dataset(data_root: str | Path, output_dir: str | Path = "outputs") -> dict:
    paths = dataset_paths(data_root)
    output_dir = Path(output_dir)

    base_bjtu = image_files(paths.base_bjtu)
    base_util = image_files(paths.base_util)
    query = image_files(paths.query_root)
    det_images = image_files(paths.detection_root)
    det_jsons = sorted(paths.detection_root.glob("*.json"))

    base_counts = collections.Counter(prefix_from_path(p) for p in base_bjtu)
    query_counts = collections.Counter(prefix_from_path(p) for p in query)

    paired = 0
    box_count = 0
    invalid_boxes = []
    labels = collections.Counter()
    for jp in det_jsons:
        data = read_json(jp)
        image_name = Path(str(data.get("imagePath") or (jp.stem + ".jpg"))).name
        image_path = jp.with_name(image_name)
        if not image_path.exists():
            matches = [p for p in det_images if p.stem == jp.stem]
            image_path = matches[0] if matches else image_path
        if image_path.exists():
            paired += 1
        width = int(data.get("imageWidth") or 0)
        height = int(data.get("imageHeight") or 0)
        if width <= 0 or height <= 0:
            try:
                with Image.open(image_path) as im:
                    width, height = im.size
            except Exception:
                width, height = 0, 0
        for shape in data.get("shapes", []):
            labels[str(shape.get("label", ""))] += 1
            points = shape.get("points", [])
            if len(points) < 2:
                invalid_boxes.append({"json": jp.name, "reason": "fewer than 2 points"})
                continue
            xs = [float(pt[0]) for pt in points]
            ys = [float(pt[1]) for pt in points]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            box_count += 1
            if width and height and not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                invalid_boxes.append({"json": jp.name, "box": [x1, y1, x2, y2], "size": [width, height]})

    summary = {
        "base_bjtu_images": len(base_bjtu),
        "base_util_images": len(base_util),
        "base_total_images": len(base_bjtu) + len(base_util),
        "query_images": len(query),
        "detection_images": len(det_images),
        "detection_jsons": len(det_jsons),
        "detection_paired_jsons": paired,
        "detection_boxes": box_count,
        "base_counts": {k: base_counts.get(k, 0) for k in LANDMARKS},
        "query_counts": {k: query_counts.get(k, 0) for k in LANDMARKS},
        "detection_label_top20": labels.most_common(20),
        "invalid_box_count": len(invalid_boxes),
        "invalid_box_examples": invalid_boxes[:20],
    }
    write_json(output_dir / "dataset_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    summary = validate_dataset(args.data_root, args.output_dir)
    print(summary)


if __name__ == "__main__":
    main()
