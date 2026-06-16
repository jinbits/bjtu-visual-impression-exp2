from __future__ import annotations

import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
LANDMARKS = ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    retrieval_root: Path
    base_root: Path
    base_bjtu: Path
    base_util: Path
    query_root: Path
    detection_root: Path


def dataset_paths(data_root: str | Path) -> DatasetPaths:
    root = Path(data_root).expanduser().resolve()
    retrieval_root = root / "image_retrieval"
    return DatasetPaths(
        root=root,
        retrieval_root=retrieval_root,
        base_root=retrieval_root / "base",
        base_bjtu=retrieval_root / "base" / "BJTU",
        base_util=retrieval_root / "base" / "util_pic",
        query_root=retrieval_root / "query",
        detection_root=root / "object_detection" / "data",
    )


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def image_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def prefix_from_path(path: str | Path) -> str:
    return Path(path).stem.split("-")[0].lower()


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: object) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        try:
            if src.stat().st_size == dst.stat().st_size:
                return
        except OSError:
            pass
    for attempt in range(4):
        try:
            shutil.copy2(src, dst)
            return
        except OSError:
            if attempt == 3:
                raise
            time.sleep(0.5 * (attempt + 1))


def stable_split(items: Iterable[Path], train_ratio: float, val_ratio: float, seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    values = list(items)
    rng = random.Random(seed)
    rng.shuffle(values)
    n = len(values)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return values[:n_train], values[n_train : n_train + n_val], values[n_train + n_val :]


def path_to_posix(path: Path) -> str:
    return path.resolve().as_posix()


def relative_or_name(path: str | Path, start: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(start))
    except ValueError:
        return path.name
