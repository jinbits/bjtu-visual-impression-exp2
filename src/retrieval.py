from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from .common import LANDMARKS, dataset_paths, ensure_dir, image_files, prefix_from_path, write_json


def _safe_import_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torchvision.transforms as T

        return torch, T
    except Exception:
        return None, None


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def _filter_readable(paths: list[Path]) -> tuple[list[Path], list[dict[str, str]]]:
    readable: list[Path] = []
    skipped: list[dict[str, str]] = []
    for path in tqdm(paths, desc="checking images"):
        try:
            with Image.open(path) as im:
                im.verify()
            readable.append(path)
        except Exception as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
    return readable, skipped


def _fallback_feature(path: Path, size: int = 224) -> np.ndarray:
    """Compact deterministic image descriptor used when deep dependencies are unavailable."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB").resize((size, size))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    # Color histograms at two granularities plus low-res luminance texture.
    feats: list[np.ndarray] = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=32, range=(0, 1), density=True)
        feats.append(hist.astype(np.float32))
    small = np.asarray(Image.fromarray((arr.mean(axis=2) * 255).astype(np.uint8)).resize((28, 28)), dtype=np.float32)
    small = (small - small.mean()) / (small.std() + 1e-6)
    feats.append(small.reshape(-1))
    # Simple edge magnitude histogram.
    gy, gx = np.gradient(arr.mean(axis=2))
    mag = np.sqrt(gx * gx + gy * gy)
    hist, _ = np.histogram(mag, bins=64, range=(0, 0.5), density=True)
    feats.append(hist.astype(np.float32))
    return np.concatenate(feats).astype(np.float32)


class DeepFeatureExtractor:
    def __init__(self, use_dino: bool = True, use_openclip: bool = True, batch_size: int = 24) -> None:
        self.torch, self.T = _safe_import_torch()
        self.batch_size = batch_size
        self.device = None
        self.dino = None
        self.dino_transform = None
        self.clip = None
        self.clip_preprocess = None
        if self.torch is None:
            return
        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        if use_dino:
            try:
                self.dino = self.torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
                self.dino.eval().to(self.device)
                self.dino_transform = self.T.Compose(
                    [
                        self.T.Resize(256, interpolation=self.T.InterpolationMode.BICUBIC),
                        self.T.CenterCrop(224),
                        self.T.ToTensor(),
                        self.T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ]
                )
            except Exception as exc:
                print(f"[retrieval] DINOv2 unavailable, continuing without it: {exc}")
                self.dino = None
        if use_openclip:
            try:
                import open_clip

                self.clip, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.device
                )
                self.clip.eval()
            except Exception as exc:
                print(f"[retrieval] OpenCLIP unavailable, continuing without it: {exc}")
                self.clip = None

    @property
    def available(self) -> bool:
        return self.torch is not None and (self.dino is not None or self.clip is not None)

    def encode(self, paths: list[Path]) -> np.ndarray:
        if not self.available:
            raise RuntimeError("deep feature extractor is not available")
        all_features: list[np.ndarray] = []
        with self.torch.no_grad():
            for start in tqdm(range(0, len(paths), self.batch_size), desc="deep features"):
                batch_paths = paths[start : start + self.batch_size]
                dino_features = None
                clip_features = None
                images = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in batch_paths]
                if self.dino is not None and self.dino_transform is not None:
                    x = self.torch.stack([self.dino_transform(im) for im in images]).to(self.device)
                    y = self.dino(x)
                    dino_features = y.detach().float().cpu().numpy()
                if self.clip is not None and self.clip_preprocess is not None:
                    x = self.torch.stack([self.clip_preprocess(im) for im in images]).to(self.device)
                    y = self.clip.encode_image(x)
                    clip_features = y.detach().float().cpu().numpy()
                for im in images:
                    im.close()
                parts = [p for p in [dino_features, clip_features] if p is not None]
                feat = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
                all_features.append(feat.astype(np.float32))
        return np.concatenate(all_features, axis=0)


def extract_features(paths: list[Path], output_path: Path, config: dict) -> np.ndarray:
    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as data:
            return data["features"]
    ensure_dir(output_path.parent)
    extractor = DeepFeatureExtractor(
        use_dino=bool(config.get("use_dino", True)),
        use_openclip=bool(config.get("use_openclip", True)),
        batch_size=int(config.get("batch_size", 24)),
    )
    if extractor.available:
        features = extractor.encode(paths)
        mode = "dinov2_openclip"
    else:
        size = int(config.get("fallback_size", 224))
        features = np.stack([_fallback_feature(p, size=size) for p in tqdm(paths, desc="fallback features")], axis=0)
        mode = "fallback_color_texture"
    features = _l2_normalize(features.astype(np.float32))
    np.savez_compressed(output_path, features=features, paths=np.array([str(p) for p in paths]), mode=np.array(mode))
    return features


def fit_pca_whitening(base_features: np.ndarray, query_features: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray, dict]:
    try:
        from sklearn.decomposition import PCA

        n = min(dim, base_features.shape[0] - 1, base_features.shape[1])
        pca = PCA(n_components=n, whiten=True, random_state=2026)
        base_w = pca.fit_transform(base_features)
        query_w = pca.transform(query_features)
        meta = {"method": "sklearn_pca_whiten", "dim": int(n)}
    except Exception:
        mean = base_features.mean(axis=0, keepdims=True)
        centered = base_features - mean
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        n = min(dim, vt.shape[0])
        comps = vt[:n].T
        scale = np.sqrt(max(base_features.shape[0] - 1, 1)) / np.maximum(s[:n], 1e-6)
        base_w = centered @ comps * scale
        query_w = (query_features - mean) @ comps * scale
        meta = {"method": "numpy_svd_whiten", "dim": int(n)}
    return _l2_normalize(base_w.astype(np.float32)), _l2_normalize(query_w.astype(np.float32)), meta


def evaluate_retrieval(data_root: str | Path, output_dir: str | Path = "outputs", config: dict | None = None) -> dict:
    config = config or {}
    paths = dataset_paths(data_root)
    output_dir = Path(output_dir)
    feature_dir = ensure_dir(output_dir / "features")
    result_dir = ensure_dir(output_dir / "retrieval")

    base_paths = image_files(paths.base_root)
    query_paths = image_files(paths.query_root)
    base_paths, skipped_base = _filter_readable(base_paths)
    query_paths, skipped_query = _filter_readable(query_paths)
    write_json(result_dir / "skipped_images.json", {"base": skipped_base, "query": skipped_query})
    base_features = extract_features(base_paths, feature_dir / "base_features.npz", config)
    query_features = extract_features(query_paths, feature_dir / "query_features.npz", config)

    base_w, query_w, pca_meta = fit_pca_whitening(base_features, query_features, int(config.get("pca_dim", 256)))
    scores = query_w @ base_w.T
    top_k = int(config.get("top_k", 60))
    order = np.argsort(-scores, axis=1)[:, :top_k]

    ks = [20, 40, 60]
    rows: list[dict[str, Any]] = []
    class_stats = {lm: {str(k): [] for k in ks} for lm in LANDMARKS}
    top_rows: list[dict[str, Any]] = []
    for qi, qpath in enumerate(query_paths):
        qprefix = prefix_from_path(qpath)
        retrieved = [base_paths[idx] for idx in order[qi]]
        retrieved_prefixes = [prefix_from_path(p) for p in retrieved]
        row = {"query": str(qpath), "query_name": qpath.name, "class": qprefix}
        for k in ks:
            value = sum(1 for rp in retrieved_prefixes[:k] if rp == qprefix) / k
            row[f"P@{k}"] = value
            if qprefix in class_stats:
                class_stats[qprefix][str(k)].append(value)
        rows.append(row)
        for rank, idx in enumerate(order[qi], start=1):
            top_rows.append(
                {
                    "query": qpath.name,
                    "class": qprefix,
                    "rank": rank,
                    "base": base_paths[idx].name,
                    "base_path": str(base_paths[idx]),
                    "base_class": prefix_from_path(base_paths[idx]),
                    "score": float(scores[qi, idx]),
                    "relevant": int(prefix_from_path(base_paths[idx]) == qprefix),
                }
            )

    class_summary = []
    for lm in LANDMARKS:
        item = {"class": lm}
        for k in ks:
            values = class_stats[lm][str(k)]
            item[f"P@{k}"] = float(np.mean(values)) if values else math.nan
        class_summary.append(item)
    overall = {"class": "overall"}
    for k in ks:
        overall[f"P@{k}"] = float(np.mean([row[f"P@{k}"] for row in rows])) if rows else math.nan
    class_summary.append(overall)

    _write_csv(result_dir / "query_precision.csv", rows)
    _write_csv(result_dir / "top60_results.csv", top_rows)
    _write_csv(result_dir / "class_precision.csv", class_summary)
    meta = {
        "base_images": len(base_paths),
        "query_images": len(query_paths),
        "top_k": top_k,
        "pca": pca_meta,
        "overall": overall,
    }
    write_json(result_dir / "retrieval_meta.json", meta)
    make_retrieval_plots(result_dir / "class_precision.csv", result_dir / "plots")
    return meta


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def make_retrieval_plots(class_precision_csv: str | Path, output_dir: str | Path) -> None:
    rows = _read_csv(Path(class_precision_csv))
    output_dir = ensure_dir(output_dir)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ks = [20, 40, 60]
        for row in rows:
            if row["class"] == "overall":
                continue
            vals = [float(row[f"P@{k}"]) for k in ks]
            fig, ax = plt.subplots(figsize=(5, 3.2), dpi=160)
            ax.plot(ks, vals, marker="o", linewidth=2, color="#1f77b4")
            ax.set_ylim(0, 1.02)
            ax.set_xticks(ks)
            ax.set_title(f"{row['class']} Precision@K")
            ax.set_xlabel("K")
            ax.set_ylabel("Precision")
            ax.grid(True, alpha=0.25)
            for x, y in zip(ks, vals):
                ax.text(x, min(y + 0.035, 1.0), f"{y:.2f}", ha="center", fontsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / f"{row['class']}_p_at_k.png")
            plt.close(fig)
        labels = [r["class"] for r in rows if r["class"] != "overall"]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
        for i, k in enumerate(ks):
            vals = [float(r[f"P@{k}"]) for r in rows if r["class"] != "overall"]
            ax.bar(x + (i - 1) * 0.25, vals, width=0.24, label=f"P@{k}")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Precision")
        ax.set_title("Image Retrieval Precision by Landmark")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "summary_p_at_k.png")
        plt.close(fig)
    except Exception as exc:
        print(f"[retrieval] matplotlib unavailable, writing JSON-only plots skipped: {exc}")


def load_config(path: str | Path | None) -> dict:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f).get("retrieval", {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    print(evaluate_retrieval(args.data_root, args.output_dir, load_config(args.config)))


if __name__ == "__main__":
    main()
