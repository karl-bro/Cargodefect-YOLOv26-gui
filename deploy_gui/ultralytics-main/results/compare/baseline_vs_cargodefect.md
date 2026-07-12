# Baseline vs CargoDefect-YOLOv26 对比

- Baseline 最佳 epoch: 83 | CargoDefect 最佳 epoch: 86
- 推理: Tesla V100, imgsz=640, batch=1

| Model | mAP50 | mAP50-95 | Precision | Recall | FPS | Params | GFLOPs |
|------|------|----------|-----------|--------|-----|--------|--------|
| YOLOv26 baseline | 0.8500 | 0.7016 | 0.9027 | 0.7187 | 98.0 | 2,375,616 | 5.2 |
| CargoDefect-YOLOv26 | 0.8541 | 0.6943 | 0.8871 | 0.7699 | 52.2 | 2,498,817 | 6.9 |

## 判定标准

- ✅ mAP50 提升
- ❌ mAP50-95 提升
- ✅ Recall 不下降或提升
- ❌ FPS 不严重下降 (>=70% baseline)
- ✅ cls_loss 正常 (<10)
- ✅ 无关键类别 AP=0

## 综合结论: **CargoDefect-YOLOv26 在检测指标上优于 baseline，但 FPS/mAP50-95 等未全部达标** (4/6 项通过)

### 关键差异
- mAP50: +0.0041
- mAP50-95: -0.0073
- Recall: +0.0512
- FPS: -45.8 (-46.7%)
- Params: +123,201

### 辅助头 (仅 CargoDefect)
- Defect classifier accuracy: 0.5053
- Quality head accuracy: 0.5053
- 正常→NG 误判 (FP): 0
- 缺陷→OK 漏判 (FN): 800
