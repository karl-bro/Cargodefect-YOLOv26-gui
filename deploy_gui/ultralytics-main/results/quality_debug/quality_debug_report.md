# Quality Debug Report

## Status

- **v3 (pos_weight=1.5)**: NG Recall achieved 1.0 (root cause fixed), but model heavily biased towards NG — 87.4% false NG rate on OK samples.
- **Next**: Testing pos_weight=1.0/1.2 and Focal Loss variants to balance OK Accuracy with NG Recall.

## Metrics (fusion_v3, threshold=0.5)
- Accuracy: **0.5584**
- OK Accuracy: 0.1261
- NG Recall: **1.0000**
- False OK Rate (NG missed): **0.0000** (0 / 800)
- False NG Rate (OK flagged): 0.8739 (714 / 817)

## Threshold Sweep

See `results/quality_debug_v3/best_threshold_report.md` — no threshold meets NG Recall >= 0.85 AND OK Accuracy >= 0.75 simultaneously. Model needs retraining with lower pos_weight.

## Root cause (fixed in dataset pipeline)
- Training previously used only MVTec `train/good` + packaging (all OK=0).
- `mvtec_train_include_anomalies: true` fixed the 0 NG label issue.
- `quality_pos_weight` tuning is ongoing.

## v4 Experiments (in progress)
1. pos_weight=1.0
2. pos_weight=1.2
3. pos_weight=1.5 (v3 baseline)
4. Focal Loss (gamma=2.0, alpha=0.25)
5. Focal Loss (gamma=2.0, alpha=0.35)