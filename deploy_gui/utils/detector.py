#!/usr/bin/env python3
"""YOLO detector with background-diff fallback, spatial/geometric filtering."""

from __future__ import annotations

import time, traceback
from datetime import datetime
from pathlib import Path

import cv2, numpy as np
from ultralytics import YOLO

PACKAGE_ID, DEFECT_ID = 0, 1
CLASS_NAMES = {0: "package", 1: "defect"}


class CargoDefectDetector:

    def __init__(self, config: dict, base_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path(".")

        m = config["model"]
        w = Path(m["weights"])
        if not w.is_absolute():
            w = self.base_dir / w
        self.weights = w
        self.imgsz = int(m.get("imgsz", 640))
        self.device = m["device"]
        self.iou = float(m.get("iou", 0.45))
        self.raw_conf = float(m.get("raw_conf", 0.01))
        self.max_det = int(m.get("max_det", 20))

        t = config.get("threshold", {})
        self.package_conf = float(t.get("package_conf", 0.15))
        self.defect_show_conf = float(t.get("defect_show_conf", 0.12))
        self.defect_ng_conf = float(t.get("defect_ng_conf", 0.25))

        f = config.get("filter", {})
        self.enable_roi = bool(f.get("enable_roi", True))
        roi_raw = f.get("roi", [80, 40, 880, 520])
        self.roi = (int(roi_raw[0]), int(roi_raw[1]), int(roi_raw[2]), int(roi_raw[3]))
        self.require_inside = bool(f.get("require_defect_inside_package_or_roi", True))
        self.inside_ratio = float(f.get("defect_inside_ratio", 0.5))
        self.min_area_r = float(f.get("min_defect_area_ratio", 0.0008))
        self.max_area_r = float(f.get("max_defect_area_ratio", 0.15))
        self.max_aspect = float(f.get("max_aspect_ratio", 8.0))

        # Package detect — ROI crop mode
        pd = config.get("package_detect", {})
        self.pkg_use_roi_crop = bool(pd.get("use_roi_crop", False))
        pkg_roi_raw = pd.get("roi", [180, 70, 850, 500])
        self.pkg_roi = (int(pkg_roi_raw[0]), int(pkg_roi_raw[1]),
                        int(pkg_roi_raw[2]), int(pkg_roi_raw[3]))

        # Display
        disp = config.get("display", {})
        self.show_raw_package_boxes = bool(disp.get("show_raw_package_boxes", False))

        dec = config.get("decision", {})
        self.fixed_station_mode = bool(dec.get("fixed_station_mode", False))
        self.debug = bool(dec.get("debug", False))

        perf = config.get("performance", {})
        self.save_debug_frame = bool(perf.get("save_debug_frame", False))
        save_dir_raw = Path(config.get("save", {}).get("result_dir", "results"))
        if not save_dir_raw.is_absolute():
            save_dir_raw = self.base_dir / save_dir_raw
        save_dir = save_dir_raw
        self.debug_no_package_dir = save_dir / "debug_no_package"
        self.debug_no_package_dir.mkdir(parents=True, exist_ok=True)

        # Package fallback -- background_diff
        pf = config.get("package_fallback", {})
        self.fallback_enable = bool(pf.get("enable", False))
        self.fallback_method = str(pf.get("method", "background_diff"))
        fb_roi = pf.get("roi", [180, 70, 850, 500])
        self.fallback_roi = (int(fb_roi[0]), int(fb_roi[1]), int(fb_roi[2]), int(fb_roi[3]))
        self.min_diff_ratio = float(pf.get("min_diff_ratio", 0.04))
        self.diff_threshold = int(pf.get("diff_threshold", 20))

        # Background image for diff
        self.bg_frame: np.ndarray | None = None
        self.bg_loaded = False
        self.bg_path = save_dir / "background_empty.jpg"

        self.model: YOLO | None = None
        self.loaded = False

        self._fc = 0
        self._last_np_save = 0.0
        self._last_print = 0.0

    # -- setters --
    def set_package_conf(self, v: float): self.package_conf = float(v)
    def set_defect_show_conf(self, v: float): self.defect_show_conf = float(v)
    def set_defect_ng_conf(self, v: float): self.defect_ng_conf = float(v)
    def set_fixed_station_mode(self, v: bool): self.fixed_station_mode = bool(v)
    def set_debug(self, v: bool): self.debug = bool(v)

    def load(self) -> bool:
        try:
            self.model = YOLO(str(self.weights))
            self.model.model.eval()
            self.loaded = True
            return True
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Model load failed: {e}")

    # -- background capture/load --
    def capture_background(self, frame: np.ndarray) -> bool:
        try:
            cv2.imwrite(str(self.bg_path), frame)
            self.bg_frame = frame.copy()
            self.bg_loaded = True
            print(f"[detector] Background captured OK: {self.bg_path}")
            return True
        except Exception as e:
            traceback.print_exc()
            return False

    def load_background(self) -> bool:
        if self.bg_path.exists():
            self.bg_frame = cv2.imread(str(self.bg_path))
            self.bg_loaded = self.bg_frame is not None
            if self.bg_loaded:
                print(f"[detector] Background loaded from disk: {self.bg_path}")
            return self.bg_loaded
        return False

    # -- pure helpers --
    @staticmethod
    def _area(b: np.ndarray) -> float:
        w = max(b[2] - b[0], 0); h = max(b[3] - b[1], 0)
        return w * h

    @staticmethod
    def _inter(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _inside_pkg(self, d: np.ndarray, pkgs: np.ndarray) -> bool:
        da = self._area(d)
        if da <= 0: return False
        for i in range(pkgs.shape[0]):
            if self._inter(d, pkgs[i]) / da >= self.inside_ratio:
                return True
        return False

    def _in_roi(self, d: np.ndarray) -> bool:
        cx = (d[0] + d[2]) / 2; cy = (d[1] + d[3]) / 2
        return self.roi[0] <= cx <= self.roi[2] and self.roi[1] <= cy <= self.roi[3]

    def _valid_geom(self, b: np.ndarray, img_w: int, img_h: int) -> bool:
        bw = b[2] - b[0]; bh = b[3] - b[1]
        if bw <= 0 or bh <= 0: return False
        ar = (bw * bh) / max(img_w * img_h, 1)
        if ar < self.min_area_r or ar > self.max_area_r: return False
        asp = max(bw, bh) / max(min(bw, bh), 1)
        if asp > self.max_aspect: return False
        return True

    # ------------------------------------------------------------------
    # Background-diff package fallback
    # ------------------------------------------------------------------
    @staticmethod
    def detect_package_by_background_diff(frame: np.ndarray, bg_frame: np.ndarray,
                                          roi: tuple, min_diff_ratio: float = 0.04,
                                          diff_threshold: int = 20) -> tuple[bool, float]:
        """Returns (has_package, diff_ratio) by comparing ROI against background.
        Handles frame/bg size mismatch by resizing bg_frame to match frame."""
        # Ensure same dimensions
        if frame.shape[:2] != bg_frame.shape[:2]:
            bg_frame = cv2.resize(bg_frame, (frame.shape[1], frame.shape[0]))

        rx, ry, rx2, ry2 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        h_img, w_img = frame.shape[:2]
        rx = max(0, min(rx, w_img - 1))
        ry = max(0, min(ry, h_img - 1))
        rx2 = max(rx + 1, min(rx2, w_img))
        ry2 = max(ry + 1, min(ry2, h_img))
        if rx2 <= rx or ry2 <= ry:
            return False, 0.0

        cur_roi = frame[ry:ry2, rx:rx2]
        bg_roi = bg_frame[ry:ry2, rx:rx2]
        roi_area = (rx2 - rx) * (ry2 - ry)

        gray_cur = cv2.cvtColor(cur_roi, cv2.COLOR_BGR2GRAY)
        gray_bg = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)

        blur_cur = cv2.GaussianBlur(gray_cur, (5, 5), 0)
        blur_bg = cv2.GaussianBlur(gray_bg, (5, 5), 0)

        diff = cv2.absdiff(blur_cur, blur_bg)
        _, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        white_pixels = cv2.countNonZero(closed)
        diff_ratio = white_pixels / max(roi_area, 1)

        return diff_ratio >= min_diff_ratio, diff_ratio

    # -- main --
    def detect(self, image: np.ndarray) -> dict:
        if not self.loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        self._fc += 1
        t0 = datetime.now()
        h_img, w_img = image.shape[:2]

        # ============================================================
        # 1. Single inference — ROI crop (if enabled), both classes
        # ============================================================
        infer_frame = image
        offset_x, offset_y = 0, 0

        if self.pkg_use_roi_crop:
            px, py, px2, py2 = self.pkg_roi
            px = max(0, min(px, w_img - 1))
            py = max(0, min(py, h_img - 1))
            px2 = max(px + 1, min(px2, w_img))
            py2 = max(py + 1, min(py2, h_img))
            if px2 > px and py2 > py:
                infer_frame = image[py:py2, px:px2]
                offset_x, offset_y = px, py

        try:
            results = self.model(
                infer_frame, imgsz=640, conf=0.001, iou=self.iou,
                classes=None, max_det=100, verbose=False, device=self.device,
            )
            inf_ms = float(results[0].speed.get("inference", 0))
            raw = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else np.empty((0, 6))
        except Exception:
            traceback.print_exc()
            raw = np.empty((0, 6))
            inf_ms = 0.0

        n_raw = raw.shape[0]

        # ============================================================
        # 2. Split raw results into package & defect, map coords back
        # ============================================================
        raw_pkg_boxes: list[dict] = []
        final_pkg_boxes: list[dict] = []

        # All raw detections (with offset) for debug
        all_raw_boxes = []
        for i in range(n_raw):
            bx = float(raw[i, 0]) + offset_x
            by = float(raw[i, 1]) + offset_y
            bx2 = float(raw[i, 2]) + offset_x
            by2 = float(raw[i, 3]) + offset_y
            cf = float(raw[i, 4])
            cls_id = int(raw[i, 5])
            all_raw_boxes.append((bx, by, bx2, by2, cf, cls_id))

        # --- Package filtering ---
        cls_arr = raw[:, 5].astype(int) if n_raw > 0 else np.array([], dtype=int)
        cf_arr = raw[:, 4] if n_raw > 0 else np.array([])
        pkg_mask = cls_arr == 0

        for i in range(n_raw):
            if not pkg_mask[i]:
                continue
            bx, by, bx2, by2, cf, cls_id = all_raw_boxes[i]
            raw_pkg_boxes.append({
                "x1": bx, "y1": by, "x2": bx2, "y2": by2,
                "conf": cf, "cls": 0, "label": "raw_package",
            })
            if cf >= self.package_conf:
                final_pkg_boxes.append({
                    "x1": bx, "y1": by, "x2": bx2, "y2": by2,
                    "conf": cf, "cls": 0, "label": "package", "color": "blue",
                })

        has_pkg_yolo = len(final_pkg_boxes) > 0
        yolo_pkg_max_conf = max((b["conf"] for b in final_pkg_boxes), default=0.0)
        yolo_raw_pkg_count = len(raw_pkg_boxes)

        # Diagnostic (throttled — printing every frame freezes Windows console)
        now_diag = time.time()
        if self.debug or (now_diag - self._last_print) >= 1.0:
            if yolo_raw_pkg_count > 0:
                all_cf = sorted([b["conf"] for b in raw_pkg_boxes], reverse=True)
                top5_str = [f"{v:.3f}" for v in all_cf[:5]]
                print(f"  [YOLO pkg] raw={yolo_raw_pkg_count} | top5_cf={top5_str} | "
                      f"keep(>={self.package_conf:.3f})={len(final_pkg_boxes)}")
            else:
                print("  [YOLO pkg] raw=0 — NO package detections")

        # --- Defect filtering ---
        def_mask = cls_arr == 1
        if def_mask.any():
            ds_mask = def_mask & (cf_arr >= self.defect_show_conf)
            dng_mask = def_mask & (cf_arr >= self.defect_ng_conf)
        else:
            ds_mask = np.zeros(0, dtype=bool)
            dng_mask = np.zeros(0, dtype=bool)

        ds_indices = np.where(ds_mask)[0]
        dng_indices = np.where(dng_mask)[0]

        # Build defect arrays with mapped coords
        def_mapped = np.zeros((n_raw, 6)) if n_raw > 0 else np.zeros((0, 6))
        for i in range(n_raw):
            if not def_mask[i]:
                continue
            bx, by, bx2, by2, cf, cls_id = all_raw_boxes[i]
            def_mapped[i] = [bx, by, bx2, by2, cf, cls_id]

        ds = def_mapped[ds_indices] if len(ds_indices) > 0 else np.empty((0, 6))
        dng = def_mapped[dng_indices] if len(dng_indices) > 0 else np.empty((0, 6))

        # -- spatial filter NG defects --
        pkg_boxes_np = np.array([[b["x1"], b["y1"], b["x2"], b["y2"]]
                                 for b in final_pkg_boxes]) if final_pkg_boxes else np.empty((0, 4))
        valid_ng_list = []
        for i in range(dng.shape[0]):
            b = dng[i]
            if not self._valid_geom(b, w_img, h_img):
                continue
            ok = False
            if self.require_inside:
                if pkg_boxes_np.shape[0] > 0 and self._inside_pkg(b, pkg_boxes_np):
                    ok = True
                elif self.enable_roi and self._in_roi(b):
                    ok = True
            else:
                ok = True
            if ok:
                valid_ng_list.append(b)
        vng = np.array(valid_ng_list) if valid_ng_list else np.empty((0, 6))

        # -- suspect = show-level but NOT NG-level --
        sus_list = []
        ng_set = set()
        for i in range(vng.shape[0]):
            ng_set.add((vng[i, 0], vng[i, 1], vng[i, 2], vng[i, 3]))
        for i in range(ds.shape[0]):
            b = ds[i]
            key = (b[0], b[1], b[2], b[3])
            if key not in ng_set and self._valid_geom(b, w_img, h_img):
                sus_list.append(b)
        sus = np.array(sus_list) if sus_list else np.empty((0, 6))

        # -- build final output boxes --
        boxes: list[dict] = []

        # Always include final packages
        boxes.extend(final_pkg_boxes)

        # Raw package boxes (debug)
        if self.show_raw_package_boxes:
            for b in raw_pkg_boxes:
                already = any(
                    abs(bp["x1"] - b["x1"]) < 1 and abs(bp["y1"] - b["y1"]) < 1
                    for bp in final_pkg_boxes
                )
                if not already:
                    b["color"] = "gray"
                    boxes.append(b)

        for i in range(vng.shape[0]):
            boxes.append({
                "x1": float(vng[i, 0]), "y1": float(vng[i, 1]),
                "x2": float(vng[i, 2]), "y2": float(vng[i, 3]),
                "conf": float(vng[i, 4]), "cls": 1, "label": "defect", "color": "red",
            })
        for i in range(sus.shape[0]):
            boxes.append({
                "x1": float(sus[i, 0]), "y1": float(sus[i, 1]),
                "x2": float(sus[i, 2]), "y2": float(sus[i, 3]),
                "conf": float(sus[i, 4]), "cls": 1, "label": "suspect", "color": "yellow",
            })

        # ============================================================
        # 3. Background diff fallback
        # ============================================================
        has_pkg_bg = False
        diff_ratio = 0.0
        if self.fallback_enable and self.bg_loaded and self.bg_frame is not None:
            has_pkg_bg, diff_ratio = CargoDefectDetector.detect_package_by_background_diff(
                image, self.bg_frame, self.fallback_roi,
                self.min_diff_ratio, self.diff_threshold,
            )

        # ============================================================
        # 4. Package determination — YOLO > bg_diff > fixed_station
        # ============================================================
        has_pkg = False
        pkg_src = "none"

        if has_pkg_yolo:
            has_pkg = True
            pkg_src = "yolo"
        elif has_pkg_bg:
            has_pkg = True
            pkg_src = "background_diff"
        elif self.fixed_station_mode:
            has_pkg = True
            pkg_src = "fixed_station"

        # ============================================================
        # 5. Decision
        # ============================================================
        has_def = vng.shape[0] > 0
        now = time.time()

        if not has_pkg:
            quality, ds_status, ccls = "WAIT", "none", "none"
        elif has_def:
            quality, ds_status, ccls = "NG", "defect", "package"
        else:
            quality, ds_status, ccls = "OK", "normal", "package"

        max_conf = max(
            ([b["conf"] for b in final_pkg_boxes] +
             [b["conf"] for b in boxes if b.get("cls") == 1]),
            default=0.0,
        )

        if self.save_debug_frame and not has_pkg and (now - self._last_np_save) >= 2.0:
            self._last_np_save = now
            ts2 = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(str(self.debug_no_package_dir / f"np_{ts2}.jpg"), image)

        if self.debug or (now - self._last_print) >= 1.0:
            self._last_print = now
            elapsed = (datetime.now() - t0).total_seconds() * 1000
            print(f"[det #{self._fc}] {quality}  n_raw={n_raw}  pkg_yolo={len(final_pkg_boxes)}  "
                  f"pkg_src={pkg_src}  def_ng={vng.shape[0]}  def_sus={sus.shape[0]}  "
                  f"diff_r={diff_ratio:.4f}  inf={inf_ms:.0f}ms  tot={elapsed:.0f}ms")

        return {
            "has_package": has_pkg, "has_defect": has_def, "cargo_class": ccls,
            "defect_status": ds_status, "quality": quality, "boxes": boxes,
            "max_conf": max_conf, "inference_ms": inf_ms,
            "package_source": pkg_src,
            "diff_ratio": diff_ratio,
            "yolo_pkg_max_conf": yolo_pkg_max_conf,
            "n_raw_pkg": yolo_raw_pkg_count, "n_pkg": len(final_pkg_boxes),
            "n_def_ng": vng.shape[0], "n_def_suspect": sus.shape[0],
        }
