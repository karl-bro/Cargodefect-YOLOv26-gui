# CargoDefect-YOLOv26-Detect — 货件缺陷检测系统

## 快速开始

1. **不要复制 `venv/`**（目标机需重新安装）
2. 双击 **`run.bat`**
3. 打开摄像头 → 采集空工位背景 → 放入纸箱检测

详细说明见 **[使用教程.md](使用教程.md)**

## 常见问题

- **卡顿**：已改为后台推理；仍卡则增大 `frame_skip`，或删掉 `venv` 后重装
- **无法采集**：自动采集需检测到包裹；可用「采集单张」强制抓拍

## 项目结构

```
app.py / config.yaml / run.bat
utils/          # 检测、摄像头、推理线程、日志、串口
ultralytics-main/
weights/
results/
```
