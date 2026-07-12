# MVTec LOCO AD Dataset Analysis Report

## 1. Dataset Structure

- Root: `/home/swot2486/0701/MVTec LOCO AD`
- Categories: 4  (breakfast_box, juice_bottle, pushpins, screw_bag)
- Anomaly types: **logical_anomalies** and **structural_anomalies** only

### Important: No scratch/crack/dent/stain labels exist
- Old heuristics mapping logical→stain, structural→scratch were **incorrect**
- Correct mapping: ALL anomalies → `defect` (class_id=4)

## 2. Sample Counts

| Category | train/good | val/good | test/good | logical_ano | structural_ano | logical_mask | structural_mask | logical_bbox | structural_bbox |
|----------|-----------|---------|-----------|-------------|----------------|-------------|----------------|-------------|----------------|
| breakfast_box | 351 | 62 | 102 | 83 | 90 | 83 | 90 | 100 | 108 |
| juice_bottle | 335 | 54 | 94 | 142 | 94 | 142 | 94 | 174 | 119 |
| pushpins | 372 | 69 | 138 | 91 | 81 | 91 | 81 | 429 | 83 |
| screw_bag | 360 | 60 | 122 | 137 | 82 | 137 | 82 | 213 | 89 |
| **Total** | 1418 | 245 | 456 | 453 | 347 | -- | -- | 1315 | -- |

## 3. Mask Coverage

- Total anomaly images: **800**
- Images with ground-truth masks: **800**
- Total mask files: **1053**
- Total bounding boxes generated: **1315**

| Anomaly Type | Samples | Has Mask | Missing Mask | Masks | BBoxes |
|--------------|---------|----------|-------------|-------|--------|
| logical_anomalies | 453 | 453 | 0 | 453 | 916 |
| structural_anomalies | 347 | 347 | 0 | 347 | 399 |

## 4. Quality Classification Summary (OK/NG)

- OK samples (all good): 2119
- NG samples (all anomalies): 800
- Total: 2919

## 5. Generated Labels

- YOLO labels: `/home/swot2486/0701/ultralytics-main/datasets/cargodefect_mvtec/labels`
  - Format: class x_center y_center width height (all normalized)
  - class=4 = defect (unified)

## 6. Recommendations

1. **Detection**: Use class 0-3 for cargo + class 4 for unified defect
2. **Quality head**: binary OK (0) / NG (1) — no fine-grained anomaly types
3. **Defect classifier**: binary none (0) / defect (1) — no scratch/crack/dent/stain
4. **Training strategy**: pseudo-bboxes from mask conversion give weak supervision for defect localization

## 7. Artifacts
- Detailed CSV: `/home/swot2486/0701/ultralytics-main/results/mvtec_analysis/anomaly_samples.csv`
- JSON stats: `/home/swot2486/0701/ultralytics-main/results/mvtec_analysis/mvtec_stats.json`
- Full report: `/home/swot2486/0701/ultralytics-main/results/mvtec_analysis/mvtec_analysis_report.md`