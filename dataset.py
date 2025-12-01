import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class COCO128Dataset(Dataset):
    def __init__(self, root_dir='coco128', img_size=640):
        self.img_dir = os.path.join(root_dir, 'images/train2017')
        self.label_dir = os.path.join(root_dir, 'labels/train2017')
        
        # 支持元组 (width, height) 和整数输入
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)  # 正方形
        else:
            self.img_size = img_size  # 期望 (Width, Height)
        
        # 获取所有图片文件名
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))

        # 1. 读取图片 & 强制 Resize 到目标矩形尺寸
        img0 = cv2.imread(img_path)
        h0, w0 = img0.shape[:2]  # 原图尺寸
        
        # 获取目标尺寸: (width, height)
        dest_w, dest_h = self.img_size
        
        # 强制缩放到目标尺寸（不保留比例，不填充）
        # 注意: cv2.resize 参数顺序是 (width, height)
        img = cv2.resize(img0, (dest_w, dest_h))
        
        # BGR -> RGB -> Channel First -> Normalize
        # 最终 Tensor 形状: [3, dest_h, dest_w] (Channels, Height, Width)
        img = img[:, :, ::-1].transpose(2, 0, 1) 
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0

        # 2. 读取标签（标签已经是归一化坐标，可以直接使用）
        # YOLO 格式标签已经是相对于原图的归一化坐标 [0, 1]
        # 由于我们使用强制缩放，这些归一化坐标仍然有效
        labels = []
        if os.path.exists(label_path):
            # txt 每一行: class x_center y_center w h (归一化的)
            with open(label_path, 'r') as f:
                for line in f:
                    l = line.strip().split()
                    if len(l) == 5:
                        cls = int(l[0])
                        # 转换成 tensor 列表
                        labels.append([0, cls, float(l[1]), float(l[2]), float(l[3]), float(l[4])])
                        # 注意：第一个 0 是 batch_index，在 collate_fn 里面会填充
        
        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out = torch.tensor(labels)

        return torch.from_numpy(img), labels_out

    # 自定义 collate_fn，处理变长的 label
    @staticmethod
    def collate_fn(batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs, 0)
        
        # 给 label 加上 batch index (第几张图)
        # 格式变成: [image_index, class, x, y, w, h]
        new_labels = []
        for i, l in enumerate(labels):
            l[:, 0] = i 
            new_labels.append(l)
            
        return imgs, torch.cat(new_labels, 0)
    

# **`collate_fn`** (Collate Function) 是 PyTorch `DataLoader` 里的\*\*“打包员”\*\*。

# 简单来说，它的任务是：**把 `Dataset` 拿出来的零散数据，拼装成一个整齐的 Batch（批次），以便喂给 GPU。**

# 在 YOLO（目标检测）任务中，它之所以必不可少，是因为**默认的打包员处理不了“数量不定的框”**。

# 我们可以通过\*\*“普通打包”\*\* vs **“YOLO打包”** 的对比来理解：

# ### 1\. 默认打包员 (Default Collate)

# 适用于：图片分类（每张图只有 1 个标签）。
# 它假设每个样本的形状都是**完全一样**的，所以它直接把它们**堆叠 (Stack)** 起来。

#   * **样本 1**：图A (猫) $\rightarrow$ 标签 `0`
#   * **样本 2**：图B (狗) $\rightarrow$ 标签 `1`
#   * **打包结果**：`Tensor([0, 1])` $\rightarrow$ **成功！**

# ### 2\. YOLO 遇到的问题

# 适用于：目标检测（每张图的框数量不一样）。

#   * **样本 1**：图A (1只猫) $\rightarrow$ 标签是 **1行**数据 `[[class, x, y, w, h]]`
#   * **样本 2**：图B (10个人) $\rightarrow$ 标签是 **10行**数据 `[[class, ...], [class, ...], ...]`

# 如果你用默认打包员，它试图把“1行的数据”和“10行的数据”强行对齐堆叠，程序就会直接报错：

# > `RuntimeError: stack expects each tensor to be equal size...`

# ### 3\. 我们写的 `collate_fn` 做了什么？

# 为了解决这个问题，我们在代码里自定义了打包逻辑，做了两件事：

# 1.  **图片 (Images)**：因为都 Resize 到了 640x640，形状一样，**直接堆叠**。
# 2.  **标签 (Labels)**：形状不一样，不能堆叠。我们将它们**拼接 (Concat)** 成一个超长的列表，并**添加一列索引**。

# #### 数据变形演示：

# 假设 Batch Size = 2：

# **输入数据：**

#   * **第0张图**：有 1 个框 `[猫, x, y, w, h]`
#   * **第1张图**：有 2 个框 `[人, x, y, w, h]`, `[领带, x, y, w, h]`

# **经过 `collate_fn` 打包后的输出：**

# ```python
# # new_labels (Tensor)
# [
#   # index, class, x,   y,   w,   h
#   [ 0,     猫,    0.5, 0.5, 0.2, 0.2],  # index 0 表示这个框属于第 0 张图
#   [ 1,     人,    0.1, 0.1, 0.1, 0.1],  # index 1 表示这个框属于第 1 张图
#   [ 1,     领带,   0.2, 0.2, 0.05, 0.1]  # index 1 表示这个框也属于第 1 张图
# ]
# ```

# **总结：**
# `collate_fn` 在这里的作用就是**给每个框贴上“身份证号”（Batch Index）**，然后把所有图的框混在一起变成一个大列表。这样 GPU 在计算 Loss 时，就能通过这个 ID 知道哪个框属于哪张图片。