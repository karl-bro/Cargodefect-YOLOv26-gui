# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convert MVTec LOCO AD (multi-category) to unified YOLO layout for CargoDefect."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ultralytics.utils import LOGGER

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _bbox_from_mask(mask_path: Path) -> tuple[float, float, float, float] | None:
    """Extract normalized xywh bbox from binary mask."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    h, w = mask.shape[:2]
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def _write_yolo_label(path: Path, cls: int, xywh: tuple[float, float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{cls} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n", encoding="utf-8")


def _write_quality(path: Path, label: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{int(label)}\n", encoding="utf-8")


def iter_mvtec_categories(root: Path) -> list[Path]:
    """Return category roots like juice_bottle/juice_bottle."""
    cats = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        nested = p / p.name
        if nested.is_dir() and (nested / "train").exists():
            cats.append(nested)
    return cats


def convert_mvtec_loco(
    src: str | Path,
    dst: str | Path,
    anomaly_cls: int = 11,
    pseudo_bbox: tuple[float, float, float, float] = (0.5, 0.5, 0.4, 0.4),
) -> dict[str, int]:
    """Convert MVTec LOCO AD to YOLO images/labels/quality layout.

    Source layout (per category)::

        {category}/{category}/
          train/good/
          validation/good/
          test/good/
          test/logical_anomalies/
          test/structural_anomalies/
          ground_truth/logical_anomalies/{id}/{id}.png
          ground_truth/structural_anomalies/{id}/{id}.png

    Returns:
        (dict): Counts per split.
    """
    src, dst = Path(src), Path(dst)
    stats = {"train": 0, "val": 0}
    categories = iter_mvtec_categories(src)
    if not categories:
        LOGGER.warning(f"No MVTec LOCO categories found under {src}")
        return stats

    split_map = {
        ("train", "good"): ("train", 0, False, None),
        ("validation", "good"): ("val", 0, False, None),
        ("test", "good"): ("val", 0, False, None),
        ("test", "logical_anomalies"): ("val", 1, True, "logical_anomalies"),
        ("test", "structural_anomalies"): ("val", 1, True, "structural_anomalies"),
    }

    for cat_root in categories:
        cat = cat_root.parent.name
        for (split_name, sub_name), (out_split, q, is_anomaly, gt_type) in split_map.items():
            src_dir = cat_root / split_name / sub_name
            if not src_dir.exists():
                continue
            for im_path in sorted(src_dir.iterdir()):
                if im_path.suffix.lower() not in IMG_EXTS:
                    continue
                im = cv2.imread(str(im_path))
                if im is None:
                    continue
                out_name = f"{cat}_{split_name}_{sub_name}_{im_path.stem}{im_path.suffix.lower()}"
                out_img = dst / "images" / out_split / out_name
                out_lbl = dst / "labels" / out_split / f"{Path(out_name).stem}.txt"
                out_q = dst / "quality" / out_split / f"{Path(out_name).stem}.txt"
                out_img.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_img), im)

                if is_anomaly:
                    bbox = None
                    if gt_type:
                        gt_path = cat_root / "ground_truth" / gt_type / im_path.stem / f"{im_path.stem}.png"
                        bbox = _bbox_from_mask(gt_path)
                    _write_yolo_label(out_lbl, anomaly_cls, bbox or pseudo_bbox)
                else:
                    out_lbl.parent.mkdir(parents=True, exist_ok=True)
                    out_lbl.write_text("", encoding="utf-8")
                _write_quality(out_q, q)
                stats[out_split] += 1

    LOGGER.info(f"Converted MVTec LOCO AD -> {dst} | train={stats['train']} val={stats['val']} categories={len(categories)}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MVTec LOCO AD to YOLO format")
    parser.add_argument("--src", type=str, required=True, help="MVTec LOCO AD root")
    parser.add_argument("--dst", type=str, required=True, help="Output YOLO dataset root")
    parser.add_argument("--anomaly-cls", type=int, default=11, help="Unified anomaly class id")
    args = parser.parse_args()
    convert_mvtec_loco(args.src, args.dst, anomaly_cls=args.anomaly_cls)


if __name__ == "__main__":
    main()
