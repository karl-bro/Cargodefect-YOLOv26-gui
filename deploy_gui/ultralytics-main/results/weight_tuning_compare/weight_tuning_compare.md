# Weight Tuning Comparison

**Baseline**: `cargodefect_detect_finetune_from_baseline/weights/best.pt` (50 epoch finetune from baseline weights)

**Tuning experiments**: 30 epoch finetune with modified loss weights, `val=False` during training, then separate val.

## Val Metrics (conf=0.001 default)

| Model | package_mAP50 | package_R | defect_mAP50 | defect_R | all_mAP50 | 是否达标 |
|-------|--------------|-----------|--------------|----------|-----------|---------|
| **finetune baseline** | **0.937** | **91.2%** | **0.445** | **41.6%** | **0.691** | ✅ 当前最优 |
| defectw2 (cls×2) | 0.845 | 79.9% | 0.330 | 27.5% | 0.588 | ❌ 全面下降 |
| defectw2_cls075 | 0.826 | 78.4% | 0.322 | 25.4% | 0.574 | ❌ 全面下降 |
| pgme_alpha05 | 0.851 | 79.9% | 0.334 | 28.8% | 0.593 | ❌ 全面下降 |
| p2loss15 | 0.828 | 76.8% | 0.324 | **31.8%** | 0.576 | ❌ defect R 略高但仍远低于 baseline |

## 详细指标

| Model | Params | GFLOPs | FPS | package_P | package_R | package_mAP50 | package_mAP50-95 | defect_P | defect_R | defect_mAP50 | defect_mAP50-95 | all_mAP50 | all_mAP50-95 |
|-------|--------|--------|-----|-----------|-----------|---------------|------------------|----------|----------|--------------|-----------------|-----------|--------------|
| finetune baseline | 2.52M | 8.1 | ~333 | 83.8 | 91.2 | 0.937 | 0.829 | 66.3 | 41.6 | 0.445 | 0.222 | 0.691 | 0.526 |
| defectw2 | 2.52M | 8.1 | ~333 | 70.2 | 79.9 | 0.845 | 0.694 | 61.3 | 27.5 | 0.330 | 0.158 | 0.588 | 0.426 |
| defectw2_cls075 | 2.52M | 8.1 | ~333 | 70.1 | 78.4 | 0.826 | 0.639 | 61.3 | 25.4 | 0.322 | 0.139 | 0.574 | 0.389 |
| pgme_alpha05 | 2.52M | 8.1 | ~333 | 76.5 | 79.9 | 0.851 | 0.711 | 55.8 | 28.8 | 0.334 | 0.162 | 0.593 | 0.436 |
| p2loss15 | 2.52M | 8.1 | ~333 | 69.1 | 76.8 | 0.828 | 0.670 | 48.8 | 31.8 | 0.324 | 0.151 | 0.576 | 0.410 |

## 实验说明

| 实验 | 配置 | 训练结果 |
|------|------|---------|
| defectw2 | `class_weights=[1.0, 2.0]` | cls loss 加权后 defect 召回反而从 41.6% 降到 27.5% |
| defectw2_cls075 | + `box=7.5, cls=0.75, dfl=1.5` | 进一步恶化，defect R=25.4% |
| pgme_alpha05 | `alpha_pgme` 初始 0.5，训练中学习到 **1.05** | 模型自动增大 PGME 权重，说明削弱 PGME 不利于当前数据 |
| p2loss15 | `level_loss_weights=[1.5,1.0,1.0,0.75]` | defect R 在调权实验中最高 (31.8%)，但仍远低于 baseline |

## 选择结论

按选择标准（defect Recall 优先 → defect mAP50 ≥ baseline → package mAP50 ≥ 0.90）：

**不采用任何调权实验。** 继续以 **`cargodefect_detect_finetune_from_baseline`** 作为最终候选模型。

调权实验全面劣于 baseline 的可能原因：
1. 仅 30 epoch 微调 + `val=False`，无法保留最优 checkpoint
2. 在已收敛模型上大幅改变 loss 比例，破坏了 package/defect 平衡
3. defect 类权重×2 导致更多 FP，Precision 下降且 Recall 未提升

## 权重路径

```
runs/detect/runs/cargodefect/cargodefect_detect_finetune_from_baseline/weights/best.pt  ← 推荐
runs/detect/runs/cargodefect/cargodefect_detect_defectw2/weights/best.pt
runs/detect/runs/cargodefect/cargodefect_detect_defectw2_cls075/weights/best.pt
runs/detect/runs/cargodefect/cargodefect_detect_pgme_alpha05/weights/best.pt
runs/detect/runs/cargodefect/cargodefect_detect_p2loss15/weights/best.pt
```
