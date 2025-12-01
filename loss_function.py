import torch
import torch.nn as nn

# 一个简化版 Anchor-Free Loss（类似于 YOLOv1 + FCOS 的逻辑）。
# 原理： 如果一个物体的中心落在某个 Grid Cell 里，那么这个 Cell 就负责预测它。
# Box Loss: 预测框和真实框的均方误差 (MSE)。
# Obj Loss: 该 Cell 的置信度应当趋向 1，其他背景 Cell 趋向 0。
# Cls Loss: 该 Cell 的类别概率应当趋向真实类别。

def bbox_iou(box1, box2, eps=1e-7):
    """
    计算两个框的 IoU (Intersection over Union)
    box1: [N, 4] (x, y, w, h)
    box2: [N, 4] (x, y, w, h)
    返回: [N] iou
    """
    # 1. 转换成角点坐标 (x1, y1, x2, y2)
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # 2. 计算交集 (Intersection)
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # 3. 计算并集 (Union)
    w1, h1 = box1[:, 2], box1[:, 3]
    w2, h2 = box2[:, 2], box2[:, 3]
    union_area = w1 * h1 + w2 * h2 - inter_area + eps

    # 4. IoU
    iou = inter_area / union_area
    return iou

class SimpleComputeLoss:
    def __init__(self, strides=[8, 16, 32]):
        self.strides = strides
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 定义三个 Loss 组件
        # BCEWithLogitsLoss 内部集成了 Sigmoid，数值更稳定
        self.bce_conf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(self.device))
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(self.device))
        self.mse = nn.MSELoss()

    def __call__(self, preds, targets):
        # preds: list of [B, 85, H, W]
        # targets: [N, 6] -> (img_idx, cls, x, y, w, h)
        
        loss_box = 0
        loss_obj = 0
        loss_cls = 0
        
        # 遍历三个尺度的输出 (P3, P4, P5)
        for i, pred in enumerate(preds):
            B, C, H, W = pred.shape
            stride = self.strides[i]
            
            # 1. 创建目标网格 (Target Grids)
            # obj_mask: 标记哪个 grid 有物体 (1/0)
            # target_box: 对应 grid 应该预测的真实框坐标
            # target_cls: 对应 grid 应该预测的类别
            obj_mask = torch.zeros((B, H, W), device=self.device)
            target_box = torch.zeros((B, H, W, 4), device=self.device)
            target_cls = torch.zeros((B, H, W), dtype=torch.long, device=self.device)
            
            # 2. 将真实框映射到当前尺度的 Grid 上
            # targets shape: [num_objects, 6]
            if targets.numel() > 0:
                # 提取对应当前 batch 的 targets
                t = targets.clone()
                
                # 将归一化坐标 (0-1) 转为当前 Grid 坐标 (0-W, 0-H)
                # x_grid = x_norm * W
                gxy = t[:, 2:4] * torch.tensor([W, H], device=self.device)
                gwh = t[:, 4:6] * torch.tensor([W, H], device=self.device)
                
                # 取整，看中心点落在哪个 grid 坐标上
                gij = gxy.long()
                gi, gj = gij[:, 0], gij[:, 1]
                
                # 过滤掉越界的框 (安全措施)
                mask = (gi >= 0) & (gi < W) & (gj >= 0) & (gj < H)
                b, c = t[:, 0].long()[mask], t[:, 1].long()[mask]
                gi, gj = gi[mask], gj[mask]
                
                # 3. 设置正样本 (Positive Samples)
                if len(b) > 0:
                    # 在对应的 grid (b, gj, gi) 标记为有物体
                    obj_mask[b, gj, gi] = 1.0
                    
                    # 设置回归目标：我们让网络直接预测相对于 grid 的偏移量和宽高的 log 值(或直接预测宽高)
                    # 这里简化：直接预测相对于 grid 左上角的 x,y (范围 0-1) 和 归一化的 w,h
                    # pred_xy = Sigmoid(out) -> 0~1. target_xy = grid_x - floor(grid_x)
                    target_box[b, gj, gi, 0:2] = gxy[mask] - gij[mask] # offset
                    target_box[b, gj, gi, 2:4] = torch.log(gwh[mask] + 1e-16) # width/height 预测通常用 log 空间
                    target_cls[b, gj, gi] = c

            # 4. 计算 Loss
            # 重新排列 pred: [B, C, H, W] -> [B, H, W, C]
            pred = pred.permute(0, 2, 3, 1)
            
            # 提取各部分预测值
            pred_xy = pred[..., 0:2].sigmoid() # 限制在 0-1
            pred_wh = pred[..., 2:4]           # 无限制 (对应 log 空间)
            pred_conf = pred[..., 4]           # 还没做 sigmoid，给 BCELoss 用
            pred_cls = pred[..., 5:]           # 还没做 sigmoid
            
            # A. Objectness Loss (物体置信度)
            # 有物体的格子要趋向 1，没物体的趋向 0
            loss_obj += self.bce_conf(pred_conf, obj_mask)
            
            # 只计算有物体格子的 Box 和 Class Loss
            # 获取有物体的 mask 索引
            obj_mask_bool = obj_mask.bool()
            
            if obj_mask_bool.sum() > 0:
                # B. Box Loss (XY + WH)
                # 只计算正样本的 loss
                t_box = target_box[obj_mask_bool]
                p_xy = pred_xy[obj_mask_bool]
                p_wh = pred_wh[obj_mask_bool]
                
                loss_box += self.mse(p_xy, t_box[:, 0:2]) # xy 偏移 loss
                loss_box += self.mse(p_wh, t_box[:, 2:4]) # wh log loss
                
                # C. Class Loss
                # One-hot encoding for targets
                t_cls = torch.zeros_like(pred_cls[obj_mask_bool])
                t_cls[torch.arange(t_cls.size(0)), target_cls[obj_mask_bool]] = 1.0
                
                loss_cls += self.bce_cls(pred_cls[obj_mask_bool], t_cls)

        # 加权求和 (权重是经验值)
        return loss_box * 0.05 + loss_obj * 1.0 + loss_cls * 0.5