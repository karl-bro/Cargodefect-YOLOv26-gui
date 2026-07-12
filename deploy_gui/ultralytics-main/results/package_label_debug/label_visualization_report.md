# Package Label Debug Report

## Class Distribution
| class_id | class | train | val | total |
|---|---|---|---|---|
| 0 | package | 6303 | 670 | 6973 |
| 1 | defect | 21160 | 805 | 21965 |

## Label Files
- Train: 6297 files
- Val: 669 files
- Class 0 (package): 6973 bboxes
- Class 1 (defect): 21965 bboxes
- Other IDs: 0

## Kaggle Verification
- Objects found: {'hole': 34, 'dent': 76}
- Image sizes: W=1882 × H=2425
- Mapping: dent/dirt/hole → class 1 (defect)

## Roboflow Verification
- Cardboard Box Defect: no defect(0)→package, torn(1)→defect, wrinkle(2)→defect
- Corrugated Box Defect: No Defect(0)→package, Torn(1)→defect, Wrinkle(2)→defect

## Recommendations
1. Full-image pseudo bbox for package dominates → defect model ignores defect class
2. Need defect-only dataset variant (nc=1) for baseline verification
3. Need package+defect WITHOUT pseudo bbox (real boxes only)