import torch
import torch.nn as nn
import numpy as np

# 1. 基础卷积模块：Conv2d + BatchNorm + SiLU (Swish)
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        # 自动计算 padding，保持特征图大小不变 (如果 s=1)
        if p is None: p = k // 2 
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# 2. Bottleneck 模块：ResNet 的残差结构
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# 3. C3 模块 (CSP Bottleneck with 3 convolutions)
# 这是 YOLOv5 的标志性结构，用于学习残差特征
class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        # CSP 结构：一部分经过残差层，一部分直接连接，最后 Concat
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    

class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.nc = num_classes
        # YOLOv5 通常有 3 个检测头 (对应大、中、小目标)
        # 这里为了演示简单，我们假设每个 grid 只预测 1 个框 (Anchor-free 风格，类似 YOLOv8)
        # 输出维度: 4 (xywh) + 1 (conf) + nc (classes)
        self.no = 5 + self.nc # one-hot 式的class
        
        # --- Backbone (简化版 CSPDarknet) ---
        # 假设输入是 640x640x3
        self.stage1 = nn.Sequential(
            Conv(3, 32, 6, 2, 2),    # /2
            Conv(32, 64, 3, 2),      # /4
            C3(64, 64, 1),
        )
        self.stage2 = nn.Sequential(
            Conv(64, 128, 3, 2),     # /8 (P3)
            C3(128, 128, 2),
        )
        self.stage3 = nn.Sequential(
            Conv(128, 256, 3, 2),    # /16 (P4)
            C3(256, 256, 2),
        )
        self.stage4 = nn.Sequential(
            Conv(256, 512, 3, 2),    # /32 (P5)
            C3(512, 512, 1),
        )
        
        # --- Detect Head ---
        # 我们只使用最后三层特征 P3, P4, P5
        self.detect_p3 = nn.Conv2d(128, self.no, 1)
        self.detect_p4 = nn.Conv2d(256, self.no, 1)
        self.detect_p5 = nn.Conv2d(512, self.no, 1)
        
        self.strides = [8, 16, 32]

    def forward(self, x):
        # Backbone Forward
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        
        # Head Forward
        out3 = self.detect_p3(p3) # [B, no, H/8, W/8]
        out4 = self.detect_p4(p4) # [B, no, H/16, W/16]
        out5 = self.detect_p5(p5) # [B, no, H/32, W/32]
        
        if self.training:
            # 训练模式：返回原始特征图用于计算 Loss
            return [out3, out4, out5]
        else:
            # 推理模式：需要将输出解码成 [x, y, w, h, conf, cls]
            return self.decode_inference([out3, out4, out5])

    def decode_inference(self, outs):
        # 这是推理的核心逻辑：将 Grid 坐标转换为图像绝对坐标
        predictions = []
        for i, pred in enumerate(outs):
            B, C, H, W = pred.shape
            stride = self.strides[i]
            
            # 1. 维度变换: [B, C, H, W] -> [B, H, W, C]
            pred = pred.permute(0, 2, 3, 1).contiguous()
            
            # 2. 生成网格 (Grid)
            yv, xv = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            grid = torch.stack((xv, yv), 2).to(pred.device).float()
            
            # 3. 解码
            # Sigmoid 将输出约束到 0-1
            # pred = pred.sigmoid() 
            
            # # xy: grid_center + offset * stride
            # # wh: exp(output) * stride (简化版逻辑，YOLOv5实际用的是 anchors)
            # box_xy = (grid + pred[..., 0:2] - 0.5) * stride
            # box_wh = (pred[..., 2:4] * 2) ** 2 * stride # YOLOv5 style encoding
            
            # conf = pred[..., 4:5]
            # cls_prob = pred[..., 5:]

            # 单独处理
            xy_prob = pred[..., 0:2].sigmoid()
            wh_raw  = pred[..., 2:4] # 保持原样，对应 log
            conf    = pred[..., 4:5].sigmoid()
            cls_prob = pred[..., 5:].sigmoid()
            
            box_xy = (grid + xy_prob - 0.5) * stride
            box_wh = torch.exp(wh_raw) * stride  # 还原 Log
            
            # 拼接: [xy, wh, conf, cls]
            out = torch.cat((box_xy, box_wh, conf, cls_prob), dim=-1)
            predictions.append(out.view(B, -1, self.no))
            
        return torch.cat(predictions, dim=1) # [B, all_anchors, no]


import torchvision

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """
    prediction: [B, num_anchors, 5 + num_classes]
    """
    output = []
    for xi, x in enumerate(prediction):  # 遍历 batch 中的每一张图
        # 1. 过滤掉置信度低的框
        x = x[x[:, 4] > conf_thres]
        
        if not x.shape[0]:
            output.append(None)
            continue

        # 2. 计算 class score = obj_conf * cls_conf
        x[:, 5:] *= x[:, 4:5]
        
        # 3. Box 转换 (Center x,y, w, h) -> (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        
        # 4. 获取最大类别概率
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        if not x.shape[0]:
            output.append(None)
            continue
            
        # 5. Torchvision 自带的 NMS
        # 为了防止不同类别的框互相抵消，我们给不同类别的框加上偏移量
        c = x[:, 5:6] * 7680  
        boxes, scores = x[:, :4] + c, x[:, 4]
        
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        output.append(x[i])
        
    return output

def xywh2xyxy(x):
    # 转换坐标格式
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


import os
import cv2
import time
import datetime
import numpy as np
import torch

# COCO 类别名称 (用于画图显示标签)
NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# 为每个类别生成一个随机颜色
COLORS = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(NAMES))]

def save_inference_sample(model, dataset, epoch, save_dir, prefix='train', num_samples=10):
    """
    推理前 num_samples 张图，画出预测框并保存到 save_dir
    prefix: 'train' or 'val', 用于区分训练集和验证集图片
    """
    model.eval() # 切换到评估模式 (关闭 BatchNorm 更新等)
    device = next(model.parameters()).device
    
    # 确保保存目录存在
    epoch_dir = os.path.join(save_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    # 我们直接从 dataset 里取前 N 张图
    for i in range(num_samples):
        # 1. 获取数据 (img_tensor: 3, 640, 640)
        img_tensor, _ = dataset[i]
        
        # 2. 预处理
        # 增加 batch 维度: [3, H, W] -> [1, 3, H, W]
        img_input = img_tensor.unsqueeze(0).to(device)
        
        # 3. 推理
        with torch.no_grad():
            pred = model(img_input)
            # 调用之前写的 NMS (conf_thres=0.25, iou_thres=0.45)
            # 这里的 non_max_suppression 需要是你之前定义的那个函数
            pred = non_max_suppression(pred, 0.25, 0.45) 
            
        # 4. 准备画图
        # Tensor -> Numpy: (C, H, W) -> (H, W, C) -> 反归一化
        img_vis = img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
        img_vis = np.ascontiguousarray(img_vis, dtype=np.uint8)
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR) # OpenCV 用 BGR
        
        # 5. 画框
        det = pred[0] # 取 batch 中第一张图的结果
        if det is not None and len(det):
            # det: [x1, y1, x2, y2, conf, cls]
            for *xyxy, conf, cls in det:
                c = int(cls)
                label = f'{NAMES[c]} {conf:.2f}'
                x1, y1, x2, y2 = map(int, xyxy)
                
                # 画矩形
                color = COLORS[c]
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                
                # 画标签背景和文字
                t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img_vis, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img_vis, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        
        # 6. 保存
        save_path = os.path.join(epoch_dir, f'{prefix}_img_{i}.jpg')
        cv2.imwrite(save_path, img_vis)
        
    print(f"Epoch {epoch} {prefix} 可视化结果已保存至: {epoch_dir}")
    model.train() # 切回训练模式