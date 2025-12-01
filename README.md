# YOLO4P - YOLOv5 风格的目标检测实现

一个基于 PyTorch 的简化版 YOLO 目标检测项目，用于学习和实验目标检测算法。

## 项目特点

- 🎯 **简化的 YOLOv5 架构**：包含 CSPDarknet backbone 和多尺度检测头
- 📊 **完整的训练流程**：支持 COCO128 数据集的训练和验证
- 🖼️ **可视化工具**：训练过程可视化，包括损失曲线、检测结果等
- 🔧 **模块化设计**：清晰的代码结构，易于理解和扩展

## 项目结构

```
yolo4p/
├── model.py                        # YOLO 模型定义 (Backbone + Head)
├── dataset.py                      # COCO128 数据集加载器
├── loss_function.py                # YOLO 损失函数实现
├── train.py                        # 训练脚本
├── interactive_dataset_visualizer.py  # 交互式数据集可视化工具
├── getdataset.py                   # 数据集下载脚本
└── coco128/                        # COCO128 数据集目录
    ├── images/
    └── labels/
```

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA (推荐用于 GPU 加速)

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/SeanComeOn/yolo4p.git
cd yolo4p
```

2. 安装依赖：
```bash
pip install torch torchvision numpy opencv-python matplotlib pillow
```

3. 下载 COCO128 数据集：
```bash
python getdataset.py
```

## 使用方法

### 训练模型

```bash
python train.py
```

训练结果会保存在 `runs/` 目录下，包括：
- 权重文件 (`.pt`)
- 训练可视化图表
- 检测结果样本

### 可视化数据集

```bash
python interactive_dataset_visualizer.py
```

交互式查看数据集中的图像和标注。

## 模型架构

- **Backbone**: CSPDarknet (简化版)
  - 4 个 stage，使用 C3 模块
  - 下采样倍数: /8, /16, /32
  
- **Detection Head**: 三个尺度的检测头
  - P3 (80x80): 小目标
  - P4 (40x40): 中目标
  - P5 (20x20): 大目标

- **损失函数**:
  - 边界框损失 (CIoU Loss)
  - 置信度损失 (BCE Loss)
  - 分类损失 (BCE Loss)

## 训练配置

默认训练参数：
- Batch size: 8
- Learning rate: 1e-3
- Optimizer: Adam
- Epochs: 100
- Input size: 640x640

## 输出示例

训练过程中会生成：
- 📈 Loss 曲线图
- 🎨 检测结果可视化
- 📝 训练日志 Markdown 文档

## 致谢

本项目参考了 Ultralytics YOLOv5 的设计思路，用于教学和实验目的。

## License

MIT License