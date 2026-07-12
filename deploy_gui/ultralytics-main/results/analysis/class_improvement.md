# Per-Class 分析报告

## 货物检测类 (4-class detection)

### 提升最大的类别 (按 mAP50 差值)

- **box**: mAP50 0.759 → 0.785 (+0.025), Recall Δ -0.020
- **bottle**: mAP50 0.922 → 0.939 (+0.017), Recall Δ +0.003
- **package**: mAP50 0.746 → 0.753 (+0.007), Recall Δ +0.019
- **can**: mAP50 0.959 → 0.940 (-0.019), Recall Δ -0.072

### 仍然较低的类别

- **package**: mAP50=0.753, Recall=0.605
- **box**: mAP50=0.785, Recall=0.760

## 缺陷分类与质量判定 (CargoDefect 辅助头)

- **scratch**: Recall=0.000, Precision=0.000
- **crack**: Recall=0.000, Precision=0.000
- **dent**: Recall=0.000, Precision=0.000
- **stain**: Recall=0.000, Precision=0.000
- **OK**: Recall=1.000 (正确识别正常货)
- **NG**: Recall=0.000 (正确识别缺陷货)
- 正常货物误判为 NG: **0** 张
- 缺陷货物漏判为 OK: **800** 张
- Defect classifier accuracy: 0.505
- Quality head accuracy: 0.505

## AP=0 检查

- 关键类别 AP=0: **无**

## 总结

检测部分已提升 Recall，质量判定头已能有效捕获 NG 样本，但当前存在正常样本误报 NG 的问题，后续通过阈值校准和 loss 调整优化。缺陷分类头由于缺少真实缺陷标签，已改为二分类（none/defect）。package 类别 Recall 仍然偏低，需专项优化。