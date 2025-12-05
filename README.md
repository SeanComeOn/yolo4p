# YOLO4P - YOLOv5 风格的目标检测实现

一个基于 PyTorch 的简化版 YOLO 目标检测项目，用于学习和实验目标检测算法。

## 项目特点

- 🎯 **简化的 YOLOv5 架构**：包含 CSPDarknet backbone 和多尺度检测头
- 📊 **完整的训练流程**：支持 COCO128/COCO2017 数据集的训练和验证
- 🔄 **断点续训**：支持从检查点继续训练，保存完整的训练状态
- 📈 **双向评估**：同时评估训练集和验证集，实时监控过拟合
- 🖼️ **可视化工具**：训练/验证过程可视化，包括损失曲线、检测结果等
- 🧬 **进化画廊**：自动生成样本在不同 epoch 的检测效果对比
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

#### 从头开始训练
```bash
python train.py
```

#### 从检查点继续训练
```bash
# 从指定 epoch 继续训练
python train.py --resume runs/2025-12-05_10-30-00/weights/epoch_100.pt

# 或使用最新的检查点
python train.py --resume runs/2025-12-05_10-30-00/weights/last.pt
```

训练结果会保存在 `runs/` 目录下，包括：
- 权重文件 (`.pt`) - 包含完整检查点（模型、优化器、epoch 信息）
- 训练/验证可视化图表（每 10 个 epoch）
- 训练集和验证集检测结果样本
- 样本进化史 Markdown 文档

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
- Batch size: 8 (训练集: 64, 验证集: 8)
- Learning rate: 1e-3
- Optimizer: Adam
- Epochs: 400
- Input size: 800x640 (宽x高)

## 数据集支持

项目支持训练集和验证集分离：
- 训练集：`coco2017/images/train2017` & `coco2017/labels/train2017`
- 验证集：`coco2017/images/val2017` & `coco2017/labels/val2017`

数据集类通过 `split` 参数自动选择对应目录。

## 输出示例

训练过程中会生成：

### 1. 终端输出
```
Epoch 1/400 | Train Loss: 2.3456 | Val Loss: 2.4567
🔍 正在生成 Epoch 1 的可视化结果...
```

### 2. 文件结构
```
runs/2025-12-05_10-30-00/
├── weights/
│   ├── epoch_1.pt          # 包含完整检查点
│   ├── epoch_10.pt
│   ├── epoch_20.pt
│   └── last.pt             # 最新检查点
├── visualizations/
│   ├── epoch_1/
│   │   ├── train_img_0.jpg  # 训练集样本
│   │   ├── train_img_1.jpg
│   │   ├── val_img_0.jpg    # 验证集样本
│   │   └── val_img_1.jpg
│   └── epoch_10/
│       └── ...
├── evolution_train_sample_0.md  # 训练集样本0的进化史
├── evolution_train_sample_1.md
├── evolution_val_sample_0.md    # 验证集样本0的进化史
└── evolution_val_sample_1.md
```

### 3. 功能说明

- **📈 实时损失监控**：每个 epoch 显示训练和验证损失
- **🎨 双向可视化**：同时保存训练集和验证集的检测结果
- **🧬 进化画廊**：Markdown 文档展示同一样本在不同 epoch 的检测演变
- **💾 完整检查点**：保存模型权重、优化器状态、epoch 编号和损失值

## 新功能亮点

### ✨ 验证集评估（Validation Loop）
- 每个 epoch 自动在验证集上评估
- 实时监控过拟合情况（对比 Train Loss vs Val Loss）
- 验证过程使用 `torch.no_grad()` 节省显存

### 🔄 断点续训（Resume Training）
- 支持从任意检查点继续训练
- 自动恢复训练进度、优化器状态
- 创建新的实验目录，保持历史记录完整

### 📊 双向可视化（Train & Val Visualization）
- 同时生成训练集和验证集的可视化结果
- 使用前缀 `train_` 和 `val_` 区分图片
- 每 10 个 epoch 自动保存

## 致谢

本项目参考了 Ultralytics YOLOv5 的设计思路，用于教学和实验目的。

## License

MIT License