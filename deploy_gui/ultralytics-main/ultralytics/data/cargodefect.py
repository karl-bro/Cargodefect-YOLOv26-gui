"""CargoDefect multi-source dataset: packaging detection + defect cls + quality NG."""

from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.utils import DEFAULT_CFG, LOGGER, IterableSimpleNamespace
from ultralytics.utils.ops import segments2boxes

from .augment import Compose, Format, LetterBox, v8_transforms
from .convert_mvtec_loco import iter_mvtec_categories
from .dataset import YOLODataset
from .utils import img2label_paths

# PackDet legacy cargo index -> unified 4-class detection space
CARGO_CLASS_REMAP = {0: 1, 1: 1, 2: 2, 3: 3, 4: 0, 5: 3}
# Legacy unified defect indices (6-11) -> mapped to binary defect (=1)
LEGACY_DEFECT_RANGE = set(range(6, 12))  # indices 6-11 are all defect
DEFECT_NONE_IDX = 0  # binary: 0=none, any defect→1
DEFECT_BINARY_ANY = 1


def load_quality_label(path: Path, default: int = 0) -> int:
    """Load quality label from txt where 0=OK, 1=NG."""
    if not path.exists():
        return default
    try:
        return 1 if int(path.read_text(encoding="utf-8").strip().split()[0]) > 0 else 0
    except Exception:
        return default


def _read_yolo_boxes(label_file: str | Path) -> np.ndarray:
    """Read YOLO txt labels; supports bbox (5 cols) and polygon (class + xy pairs)."""
    p = Path(label_file)
    if not p.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = [r.strip().split() for r in p.read_text(encoding="utf-8").splitlines() if r.strip()]
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    out = []
    for row in rows:
        vals = [float(x) for x in row]
        if len(vals) == 5:
            out.append(vals)
        elif len(vals) > 5 and len(vals[1:]) % 2 == 0:
            seg = np.array(vals[1:], dtype=np.float32).reshape(-1, 2)
            box = segments2boxes([seg])[0]
            out.append([vals[0], *box.tolist()])
    if not out:
        return np.zeros((0, 5), dtype=np.float32)
    return np.array(out, dtype=np.float32)


def _is_mvtec_anomaly_path(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts.intersection({"logical_anomalies", "structural_anomalies", "anomaly"}))


def _pseudo_bbox() -> np.ndarray:
    """Deprecated: anomalies are image-level NG only, not detection bboxes."""
    return np.array([0.5, 0.5, 0.4, 0.4], dtype=np.float32)


def _mvtec_defect_label(im_path: Path) -> int:
    """Map MVTec anomaly path to defect classifier label (binary: 0=none, 1=defect)."""
    parts = set(im_path.parts)
    if "logical_anomalies" in parts or "structural_anomalies" in parts:
        return DEFECT_BINARY_ANY
    return DEFECT_NONE_IDX


def _split_name_from_img(path: Path) -> str:
    """Infer split name from image path."""
    for split in ("train", "val", "test"):
        if split in path.parts:
            return split
    return "train"


class CargoDefectFormat(Format):
    """Format transform preserving image-level quality_label and defect_label."""

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        labels = super().apply_instances(labels, params)
        ql = labels.pop("quality_label", 0)
        dl = labels.pop("defect_label", DEFECT_NONE_IDX)
        labels["quality_label"] = torch.tensor(int(ql), dtype=torch.long)
        labels["defect_label"] = torch.tensor(int(dl), dtype=torch.long)
        return labels


def cargodefect_transforms(dataset, imgsz: int, hyp: IterableSimpleNamespace, stretch: bool = False) -> Compose:
    """YOLOv8/26 transforms with defect-oriented blur, brightness, and copy-paste defaults."""
    if getattr(hyp, "cargodefect_copy_paste", None) is not None:
        hyp.copy_paste = hyp.cargodefect_copy_paste
    elif getattr(hyp, "copy_paste", 0.0) <= 0.0:
        hyp.copy_paste = 0.3

    if getattr(hyp, "augmentations", None) is None:
        try:
            import albumentations as A

            hyp.augmentations = [
                A.OneOf(
                    [
                        A.Blur(blur_limit=3, p=1.0),
                        A.GaussianBlur(blur_limit=3, p=1.0),
                        A.MotionBlur(blur_limit=3, p=1.0),
                    ],
                    p=0.2,
                ),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.15, p=0.35),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.1),
            ]
        except ImportError:
            LOGGER.warning("albumentations not installed; using default HSV augmentations only for CargoDefect.")

    return v8_transforms(dataset, imgsz=imgsz, hyp=hyp, stretch=stretch)


class CargoDefectDataset(YOLODataset):
    """Multi-source dataset: cargo detection + image-level defect & quality labels."""

    def __init__(self, *args, data: dict | None = None, task: str = "detect", **kwargs):
        self.data = data or {}
        self.mode = self.data.get("_mode", "train")
        self.nc_cargo = int(self.data.get("nc_cargo", 4))
        self.nc_defect = int(self.data.get("nc_defect", 5))
        self.quality_dir = self.data.get("quality_dir", "quality")
        self.data_sources = self.data.get("data_sources", {"packaging": 0.7, "mvtec_loco": 0.3})
        self.packaging_root = Path(self.data.get("packaging_path", self.data.get("path", "")) or "")
        self.mvtec_root = Path(self.data.get("mvtec_path", "")) if self.data.get("mvtec_path") else None
        self.defect_cp_prob = float(self.data.get("defect_copy_paste_prob", 0.25))
        self._sample_source: dict[str, str] = {}
        self._mvtec_anomaly_images: list[str] = []
        super().__init__(*args, data=data, task=task, **kwargs)

    def _packaging_split_dirs(self) -> list[Path]:
        """Resolve packaging split directories for train/val/test."""
        if self.mode == "train":
            names = ("train", "Train")
        elif self.mode == "val":
            names = ("val", "Val", "validation", "Validation")
        else:
            names = ("test", "Test", "val", "Validation")
        out = []
        for name in names:
            p = self.packaging_root / "images" / name
            if p.exists():
                out.append(p)
        return out

    def _packaging_images(self) -> list[str]:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = []
        for split_dir in self._packaging_split_dirs():
            files += [str(p) for p in split_dir.rglob("*") if p.suffix.lower() in exts]
        return files

    def _mvtec_images(self) -> list[str]:
        """Collect images from MVTec LOCO AD multi-category layout."""
        if not self.mvtec_root or not self.mvtec_root.exists():
            return []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        out: list[str] = []
        categories = iter_mvtec_categories(self.mvtec_root)
        if not categories:
            legacy = [
                self.mvtec_root / "train" / "good",
                self.mvtec_root / "test" / "good",
                self.mvtec_root / "test" / "anomaly",
            ]
            for d in legacy:
                if d.exists():
                    out += [str(p) for p in d.rglob("*") if p.suffix.lower() in exts]
            return out

        if self.mode == "train":
            folders = [("train", "good")]
            if self.data.get("mvtec_train_include_anomalies", True):
                folders.extend(
                    [
                        ("test", "logical_anomalies"),
                        ("test", "structural_anomalies"),
                    ]
                )
        else:
            folders = [
                ("validation", "good"),
                ("test", "good"),
                ("test", "logical_anomalies"),
                ("test", "structural_anomalies"),
            ]
        for cat_root in categories:
            for split_name, sub_name in folders:
                src_dir = cat_root / split_name / sub_name
                if src_dir.exists():
                    out += [str(p) for p in src_dir.iterdir() if p.suffix.lower() in exts]
        return out

    def _weighted_mix(self, packaging: list[str], mvtec: list[str]) -> list[str]:
        """Create a mixed image list with target source ratio for training."""
        if self.mode != "train":
            return packaging + mvtec

        anomalies = [f for f in mvtec if _is_mvtec_anomaly_path(Path(f))]
        mvtec_good = [f for f in mvtec if f not in set(anomalies)]
        wp = float(self.data_sources.get("packaging", 0.7))
        wm = float(self.data_sources.get("mvtec_loco", 0.3))

        if not packaging and not mvtec_good:
            return anomalies if anomalies else mvtec
        if not packaging:
            return mvtec_good + anomalies
        if not mvtec_good and not anomalies:
            return packaging

        # Keep every MVTec anomaly for NG supervision; subsample OK sources by ratio.
        base_total = max(len(packaging), len(mvtec_good) + len(anomalies), 1)
        n_pack = max(1, int(base_total * wp / max(wp + wm, 1e-6)))
        n_good = max(1, int(base_total * wm / max(wp + wm, 1e-6)))
        p_idx = np.random.choice(len(packaging), size=n_pack, replace=len(packaging) < n_pack)
        g_idx = (
            np.random.choice(len(mvtec_good), size=n_good, replace=len(mvtec_good) < n_good)
            if mvtec_good
            else np.array([], dtype=int)
        )
        mixed = [packaging[i] for i in p_idx.tolist()]
        mixed += [mvtec_good[i] for i in g_idx.tolist()]
        mixed += anomalies
        random.shuffle(mixed)
        return mixed

    def get_img_files(self, img_path) -> list[str]:
        """Collect and mix files from packaging + MVTec according to configured ratios."""
        packaging = self._packaging_images()
        mvtec = self._mvtec_images()
        mixed = self._weighted_mix(packaging, mvtec)
        for f in mixed:
            src = "mvtec_loco" if self.mvtec_root and str(self.mvtec_root) in str(Path(f)) else "packaging"
            self._sample_source[f] = src
        self._mvtec_anomaly_images = [f for f in mixed if _is_mvtec_anomaly_path(Path(f))]
        if not mixed:
            LOGGER.warning("CargoDefectDataset found no images in packaging/mvtec sources.")
        return mixed

    def _packaging_label(self, im_file: str) -> tuple[np.ndarray, np.ndarray, int]:
        """Cargo boxes only (4 classes) + image-level defect label from legacy defect rows."""
        label_file = Path(img2label_paths([im_file])[0])
        lb = _read_yolo_boxes(label_file)
        defect_label = DEFECT_NONE_IDX
        cargo_rows = []
        if len(lb):
            for row in lb:
                c = int(row[0])
                if c < 6:
                    cargo_rows.append([CARGO_CLASS_REMAP.get(c, 3), *row[1:].tolist()])
                elif c in LEGACY_DEFECT_RANGE:
                    defect_label = DEFECT_BINARY_ANY
        if cargo_rows:
            lb = np.array(cargo_rows, dtype=np.float32)
            lb[:, 0] = np.clip(lb[:, 0], 0, self.nc_cargo - 1)
        else:
            lb = np.zeros((0, 5), dtype=np.float32)
        cls = lb[:, 0:1] if len(lb) else np.zeros((0, 1), dtype=np.float32)
        boxes = lb[:, 1:5] if len(lb) else np.zeros((0, 4), dtype=np.float32)
        return cls, boxes, defect_label

    def _packaging_quality_path(self, im_path: Path) -> Path:
        split = _split_name_from_img(im_path)
        for split_name in (split, split.capitalize(), split.lower()):
            q = self.packaging_root / self.quality_dir / split_name / f"{im_path.stem}.txt"
            if q.exists():
                return q
        return self.packaging_root / self.quality_dir / split / f"{im_path.stem}.txt"

    def _mvtec_label(self, im_file: str) -> tuple[np.ndarray, np.ndarray, int, int]:
        """MVTec: no detection boxes; anomaly -> image-level NG + defect class only."""
        p = Path(im_file)
        is_anomaly = _is_mvtec_anomaly_path(p)
        quality_label = 1 if is_anomaly else 0
        defect_label = _mvtec_defect_label(p) if is_anomaly else DEFECT_NONE_IDX
        return (
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            quality_label,
            defect_label,
        )

    def get_labels(self) -> list[dict]:
        labels = []
        for im_file in self.im_files:
            p = Path(im_file)
            src = self._sample_source.get(im_file, "packaging")
            if src == "packaging":
                cls, bboxes, defect_label = self._packaging_label(im_file)
                quality_label = load_quality_label(self._packaging_quality_path(p), default=0)
                if quality_label and defect_label == DEFECT_NONE_IDX:
                    defect_label = DEFECT_BINARY_ANY
            else:
                cls, bboxes, quality_label, defect_label = self._mvtec_label(im_file)
            labels.append(
                {
                    "im_file": im_file,
                    "shape": (0, 0),
                    "cls": cls,
                    "bboxes": bboxes,
                    "segments": [],
                    "keypoints": None,
                    "normalized": True,
                    "bbox_format": "xywh",
                    "quality_label": quality_label,
                    "defect_label": defect_label,
                }
            )
        return labels

    def _copy_paste_defect_patch(self, label: dict[str, Any]) -> dict[str, Any]:
        """Paste anomaly texture for augmentation; sets defect/quality labels only (no bbox)."""
        if self.mode != "train" or random.random() > self.defect_cp_prob:
            return label
        if self._sample_source.get(label["im_file"], "packaging") != "packaging":
            return label
        if not self._mvtec_anomaly_images:
            return label
        src_file = random.choice(self._mvtec_anomaly_images)
        patch = cv2.imread(src_file)
        if patch is None or label["img"] is None:
            return label
        img = label["img"]
        h, w = img.shape[:2]
        ph, pw = patch.shape[:2]
        scale = random.uniform(0.15, 0.3) * min(h / max(ph, 1), w / max(pw, 1))
        nh, nw = max(8, int(ph * scale)), max(8, int(pw * scale))
        patch = cv2.resize(patch, (nw, nh))
        x1 = random.randint(0, max(w - nw, 1))
        y1 = random.randint(0, max(h - nh, 1))
        x2, y2 = min(w, x1 + nw), min(h, y1 + nh)
        img[y1:y2, x1:x2] = patch[: y2 - y1, : x2 - x1]
        label["img"] = img
        label["defect_label"] = _mvtec_defect_label(Path(src_file))
        label["quality_label"] = 1
        return label

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        label = deepcopy(self.labels[index])
        label.pop("shape", None)
        label["quality_label"] = int(label.get("quality_label", 0))
        label["defect_label"] = int(label.get("defect_label", DEFECT_NONE_IDX))
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        label = self._copy_paste_defect_patch(label)
        return self.update_labels_info(label)

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        quality_label = int(label.get("quality_label", 0))
        defect_label = int(label.get("defect_label", DEFECT_NONE_IDX))
        label = super().update_labels_info(label)
        label["quality_label"] = quality_label
        label["defect_label"] = defect_label
        return label

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        """Build transforms with defect-oriented augmentation and quality-aware formatting."""
        hyp = hyp or DEFAULT_CFG
        if self.augment:
            if not isinstance(hyp, IterableSimpleNamespace):
                hyp = IterableSimpleNamespace(**dict(vars(hyp) if hasattr(hyp, "__dict__") else hyp))
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = cargodefect_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            CargoDefectFormat(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=getattr(hyp, "mask_ratio", 4),
                mask_overlap=getattr(hyp, "overlap_mask", True),
                bgr=getattr(hyp, "bgr", 0.0) if self.augment else 0.0,
            )
        )
        return transforms

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        new_batch = YOLODataset.collate_fn(batch)
        if batch and "quality_label" in batch[0]:
            ql = [b["quality_label"] for b in batch]
            if isinstance(ql[0], torch.Tensor):
                new_batch["quality_label"] = torch.stack([q.reshape(()) for q in ql], 0).long()
            else:
                new_batch["quality_label"] = torch.tensor(ql, dtype=torch.long)
        if batch and "defect_label" in batch[0]:
            dl = [b["defect_label"] for b in batch]
            if isinstance(dl[0], torch.Tensor):
                new_batch["defect_label"] = torch.stack([d.reshape(()) for d in dl], 0).long()
            else:
                new_batch["defect_label"] = torch.tensor(dl, dtype=torch.long)
        return new_batch


def build_cargo_batch(batch: dict[str, torch.Tensor], nc_cargo: int) -> dict[str, torch.Tensor]:
    """Filter batch to cargo-only detection targets (cls < nc_cargo)."""
    cls = batch["cls"].view(-1)
    mask = cls < nc_cargo
    device = batch["batch_idx"].device
    if mask.sum() == 0:
        return {
            "img": batch["img"],
            "batch_idx": torch.zeros(0, device=device, dtype=batch["batch_idx"].dtype),
            "cls": torch.zeros(0, 1, device=batch["cls"].device, dtype=batch["cls"].dtype),
            "bboxes": torch.zeros(0, 4, device=batch["bboxes"].device, dtype=batch["bboxes"].dtype),
        }
    return {
        "img": batch["img"],
        "batch_idx": batch["batch_idx"].view(-1)[mask],
        "cls": cls[mask].view(-1, 1).to(batch["cls"].dtype),
        "bboxes": batch["bboxes"][mask],
    }


def build_defect_batch(batch: dict[str, torch.Tensor], nc_cargo: int) -> dict[str, torch.Tensor]:
    """Legacy helper kept for ablation configs using SmallDefectHead."""
    cls = batch["cls"].view(-1)
    mask = cls >= nc_cargo
    device = batch["batch_idx"].device
    if mask.sum() == 0:
        return {
            "img": batch["img"],
            "batch_idx": torch.zeros(0, device=device, dtype=batch["batch_idx"].dtype),
            "cls": torch.zeros(0, 1, device=batch["cls"].device, dtype=batch["cls"].dtype),
            "bboxes": torch.zeros(0, 4, device=batch["bboxes"].device, dtype=batch["bboxes"].dtype),
        }
    return {
        "img": batch["img"],
        "batch_idx": batch["batch_idx"].view(-1)[mask],
        "cls": (cls[mask] - nc_cargo).view(-1, 1).to(batch["cls"].dtype),
        "bboxes": batch["bboxes"][mask],
    }


def quality_targets_from_batch(batch: dict[str, torch.Tensor], nc_quality: int = 2) -> torch.Tensor:
    """Map binary quality_label (0=OK, 1=NG) to training targets."""
    if "quality_label" not in batch:
        from ultralytics.nn.modules.quality_head import QualityHead

        return QualityHead.derive_quality_labels(batch)

    ql = batch["quality_label"].long().view(-1)
    if nc_quality <= 2:
        return ql.clamp(0, 1)
    return torch.where(ql <= 0, torch.zeros_like(ql), torch.full_like(ql, min(2, nc_quality - 1)))


def resolve_cargodefect_data(data: dict) -> dict:
    """Resolve relative dataset paths from the fusion YAML location."""
    from pathlib import Path

    data = dict(data)
    yaml_dir = Path(data.get("yaml_file", "ultralytics/cfg/datasets")).parent
    if not yaml_dir.exists():
        yaml_dir = Path(__file__).resolve().parents[1] / "cfg" / "datasets"

    def _resolve(value: str) -> str:
        p = Path(value)
        if p.is_absolute():
            return str(p)
        for candidate in (p, yaml_dir / value, yaml_dir / value.lstrip("./")):
            resolved = candidate.expanduser().resolve()
            if resolved.exists():
                return str(resolved)
        return str((yaml_dir / value).expanduser().resolve())

    for key in ("path", "packaging_path", "mvtec_path"):
        if data.get(key):
            data[key] = _resolve(str(data[key]))
    if not data.get("packaging_path") and data.get("path"):
        data["packaging_path"] = data["path"]
    return data


def collect_label_statistics(data: dict, mode: str = "train") -> dict:
    """Summarize quality/defect/source label counts for a split."""
    from collections import Counter

    from ultralytics.data.augment import Compose
    from ultralytics.utils import DEFAULT_CFG

    data = resolve_cargodefect_data(data)
    data["_mode"] = mode

    class _StatsDS(CargoDefectDataset):
        def build_transforms(self, hyp=None):
            return Compose([])

    split_key = "val" if mode == "val" else "train"
    ds = _StatsDS(
        img_path=data[split_key],
        imgsz=640,
        batch_size=1,
        augment=False,
        hyp=DEFAULT_CFG,
        data=data,
        stride=32,
    )
    defect_names = data.get("defect_names", {0: "none", 1: "defect"})
    ql = Counter()
    dl = Counter()
    src = Counter()
    for lb in ds.labels:
        ql[int(lb["quality_label"])] += 1
        dl[int(lb["defect_label"])] += 1
        src[ds._sample_source.get(lb["im_file"], "?")] += 1
    return {
        "mode": mode,
        "total": len(ds.labels),
        "quality_ok": ql.get(0, 0),
        "quality_ng": ql.get(1, 0),
        "defect": {defect_names.get(k, str(k)): dl[k] for k in sorted(dl)},
        "source": dict(src),
    }
