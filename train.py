import time
import datetime
import os
import numpy as np
import torch
from dataset import COCO128Dataset
from model import SimpleYOLO, save_inference_sample
from loss_function import SimpleComputeLoss


def train_professional(resume=None, data_root='coco2017'):
    """
    专业训练函数
    
    Args:
        resume (str, optional): 检查点路径，用于继续训练。
                                可以是 .pt 文件路径，如 'runs/2025-12-02_23-16-21/weights/epoch_100.pt'
                                如果为 None，则从头开始训练。
        data_root (str): 数据集根目录，默认为 'coco2017'。
                         目录下应包含 images/train2017, images/val2017 等子目录。
    """
    # --- 1. 实验环境设置 ---
    # 生成 runs/2023-10-27_10-30-00 这样的目录
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join('runs', start_time)
    weights_dir = os.path.join(save_dir, 'weights')
    vis_dir = os.path.join(save_dir, 'visualizations')
    
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    if resume:
        print(f"🔄 从检查点恢复训练: {resume}")
        print(f"📁 新日志目录: {save_dir}")
    else:
        print(f"🚀 从头开始训练！日志目录: {save_dir}")
    
    print(f"📂 数据集根目录: {data_root}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 2. 数据与模型 ---
    # 使用矩形输入: 800x640 (宽x高)
    # 注意: Tensor 形状将是 [Batch, 3, 640, 800] (Channels, Height, Width)
    train_dataset = COCO128Dataset(data_root, img_size=(800, 640), split='train')
    val_dataset = COCO128Dataset(data_root, img_size=(800, 640), split='val')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        collate_fn=COCO128Dataset.collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=False, 
        collate_fn=COCO128Dataset.collate_fn
    )
    
    model = SimpleYOLO(num_classes=80).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    compute_loss = SimpleComputeLoss() # 假设你已经定义了这个类
    
    # --- 3. 加载检查点 (如果提供) ---
    start_epoch = 0
    if resume and os.path.isfile(resume):
        print(f"📥 正在加载检查点: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        
        # 加载模型权重
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 完整检查点格式 (包含 optimizer, epoch 等)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            print(f"✅ 已加载模型、优化器状态，从 Epoch {start_epoch + 1} 继续训练")
        else:
            # 仅包含模型权重
            model.load_state_dict(checkpoint)
            print(f"✅ 已加载模型权重 (仅权重文件)")
            # 尝试从文件名推断 epoch
            if 'epoch_' in os.path.basename(resume):
                try:
                    start_epoch = int(os.path.basename(resume).split('epoch_')[1].split('.')[0])
                    print(f"📍 从文件名推断起始 Epoch: {start_epoch}")
                except:
                    pass
    elif resume:
        print(f"⚠️  警告: 找不到检查点文件 {resume}，从头开始训练")
    
    # --- 4. 训练循环 ---
    TOTAL_EPOCHS = 400 # 跑 400 个 epoch 效果比较明显
    
    model.train()
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        train_loss = 0
        
        # --- A. 训练阶段 ---
        for i, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # Forward
            preds = model(imgs)
            loss = compute_loss(preds, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- B. 验证阶段 (新增) ---
        # 注意: 保持 model.training=True 以获取原始特征图用于 Loss 计算
        # 但使用 torch.no_grad() 来禁用梯度计算
        val_loss = 0
        with torch.no_grad():
            for i, (imgs, targets) in enumerate(val_loader):
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                # Forward (模型仍在 training mode，返回原始特征图)
                preds = model(imgs)
                loss = compute_loss(preds, targets)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # --- C. 可视化阶段 (每 10 个 epoch) ---
        # 另外，epoch 0 也跑一下，看看初始的瞎猜是什么样的
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"🔍 正在生成 Epoch {epoch+1} 的可视化结果...")
            save_inference_sample(model, train_dataset, epoch+1, vis_dir, prefix='train', num_samples=10)
            save_inference_sample(model, val_dataset, epoch+1, vis_dir, prefix='val', num_samples=10)
            
            # --- D. 保存模型权重 ---
            # 保存完整检查点 (包含模型、优化器、epoch 信息)
            ckpt_path = os.path.join(weights_dir, f'epoch_{epoch+1}.pt')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }
            torch.save(checkpoint, ckpt_path)
            torch.save(checkpoint, os.path.join(weights_dir, 'last.pt'))

    print(f"✅ 训练结束。所有结果保存在: {save_dir}")
    # === 新增这一行 ===
    generate_evolution_gallery(save_dir, num_samples=10)

    return model


def generate_evolution_gallery(save_dir, num_samples=10):
    """
    生成 Markdown 文件，分别展示训练集和验证集样本的进化过程。
    每个文件专注于展示【同一张图片】在不同 Epoch 的变化过程。
    """
    vis_dir = os.path.join(save_dir, 'visualizations')
    if not os.path.exists(vis_dir):
        print("未找到可视化目录，跳过生成画廊。")
        return

    # 1. 获取所有 epoch 文件夹并排序
    # 比如: epoch_1, epoch_10, epoch_20...
    folders = [f for f in os.listdir(vis_dir) if f.startswith('epoch_') and os.path.isdir(os.path.join(vis_dir, f))]
    folders.sort(key=lambda x: int(x.split('_')[1]))

    if not folders:
        print("可视化目录为空。")
        return

    print(f"🎨 正在生成演化画廊 (共 {num_samples} 个样本，Train & Val 各一套)...")

    # 2. 为训练集和验证集分别生成画廊
    for prefix in ['train', 'val']:
        dataset_name = '训练集' if prefix == 'train' else '验证集'
        
        # 为每个样本索引 (0~9) 生成一个独立的 MD 文件
        for i in range(num_samples):
            md_filename = f'evolution_{prefix}_sample_{i}.md'
            md_path = os.path.join(save_dir, md_filename)
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# 🧬 {dataset_name}样本 {i} 的进化史\n\n")
                f.write(f"**观察对象**: {dataset_name}中的第 {i} 张图片\n\n")
                f.write(f"**说明**: 向下滚动查看该图片从 Epoch {folders[0].split('_')[1]} 到最后的训练变化。\n\n")
                f.write("---\n\n")

                # 遍历所有 epoch 文件夹
                for folder in folders:
                    epoch_num = folder.split('_')[1]
                    img_name = f"{prefix}_img_{i}.jpg"
                    
                    # 相对路径 (用于 Markdown 显示)
                    # 结构: visualizations/epoch_X/{prefix}_img_i.jpg
                    img_rel_path = f"visualizations/{folder}/{img_name}"
                    
                    # 绝对路径 (用于检查文件是否存在)
                    full_path = os.path.join(vis_dir, folder, img_name)
                    
                    if os.path.exists(full_path):
                        f.write(f"## Epoch {epoch_num}\n")
                        f.write(f"![Epoch {epoch_num}]({img_rel_path})\n\n")
                    else:
                        # 如果某个 epoch 没生成这张图 (极少见)
                        f.write(f"## Epoch {epoch_num}\n")
                        f.write(f"> *该 Epoch 缺失图片*\n\n")

    print(f"✅ 画廊生成完毕！请在 VS Code 中打开 '{save_dir}/evolution_train_sample_X.md' 和 'evolution_val_sample_X.md' 查看。")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO 训练脚本')
    parser.add_argument('--resume', type=str, default=None,
                        help='检查点路径，用于继续训练。例如: runs/2025-12-02_23-16-21/weights/epoch_100.pt')
    parser.add_argument('--data', type=str, default='coco2017',
                        help='数据集根目录。默认: coco2017')
    args = parser.parse_args()
    
    # 确保之前的 SimpleYOLO, COCO128Dataset, SimpleComputeLoss, non_max_suppression 都在上下文中
    train_professional(resume=args.resume, data_root=args.data)