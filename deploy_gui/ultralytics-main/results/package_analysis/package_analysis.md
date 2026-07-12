# Package Class Weak Analysis

- Weights: `/home/swot2486/0701/ultralytics-main/runs/detect/runs/cargodefect/fusion_v3/weights/best.pt`
- Total val images: 1617

## Category Distribution (val set)
- Box instances: 50
- Bottle instances: 135
- Can instances: 110
- **Package instances: 150**
- Images with packages: 34 / 1617 (2.1%)

## Missed Package Detection Samples
- Saved to `/home/swot2486/0701/ultralytics-main/results/package_analysis/missed_package/` (30 samples, GT=red, Pred=green)

## Analysis & Recommendations
- Package is the most challenging class: similar appearance to background, variable box stacking.
- Augmentation suggestions:
  1. Increase mosaic + mixup for package-heavy batches
  2. Add copy_paste augmentation with package crops
  3. Consider class-specific oversampling in dataloader
- Label check: verify bounding box tightness on package annotations.

## Next Steps
- Run `scripts/check_package_labels.py` to audit annotation quality
- Try class-balanced sampling or focal loss on detection head