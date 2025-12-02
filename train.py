import time
import datetime
import os
import numpy as np
import torch
from dataset import COCO128Dataset
from model import SimpleYOLO, save_inference_sample
from loss_function import SimpleComputeLoss


def train_professional():
    # --- 1. 实验环境设置 ---
    # 生成 runs/2023-10-27_10-30-00 这样的目录
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join('runs', start_time)
    weights_dir = os.path.join(save_dir, 'weights')
    vis_dir = os.path.join(save_dir, 'visualizations')
    
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"🚀 训练启动！日志目录: {save_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 2. 数据与模型 ---
    # 使用矩形输入: 800x640 (宽x高)
    # 注意: Tensor 形状将是 [Batch, 3, 640, 800] (Channels, Height, Width)
    dataset = COCO128Dataset('coco2017', img_size=(800, 640))
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=COCO128Dataset.collate_fn
    )
    
    model = SimpleYOLO(num_classes=80).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    compute_loss = SimpleComputeLoss() # 假设你已经定义了这个类
    
    # --- 3. 训练循环 ---
    TOTAL_EPOCHS = 100 # 跑 100 个 epoch 效果比较明显
    
    model.train()
    for epoch in range(TOTAL_EPOCHS):
        total_loss = 0
        
        # --- A. 训练阶段 ---
        for i, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # Forward
            preds = model(imgs)
            loss = compute_loss(preds, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} | Loss: {avg_loss:.4f}")
        
        # --- B. 可视化阶段 (每 10 个 epoch) ---
        # 另外，epoch 0 也跑一下，看看初始的瞎猜是什么样的
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"🔍 正在生成 Epoch {epoch+1} 的可视化结果...")
            save_inference_sample(model, dataset, epoch+1, vis_dir, num_samples=10)
            
            # --- C. 保存模型权重 ---
            # 保存 latest 和 当前 epoch 的权重
            ckpt_path = os.path.join(weights_dir, f'epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), os.path.join(weights_dir, 'last.pt'))

    print(f"✅ 训练结束。所有结果保存在: {save_dir}")
    # === 新增这一行 ===
    generate_evolution_gallery(save_dir, num_samples=10)

    return model


def generate_evolution_gallery(save_dir, num_samples=10):
    """
    生成 10 个 Markdown 文件。
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

    print(f"🎨 正在生成演化画廊 (共 {num_samples} 个样本)...")

    # 2. 为每个样本索引 (0~9) 生成一个独立的 MD 文件
    for i in range(num_samples):
        md_filename = f'evolution_sample_{i}.md'
        md_path = os.path.join(save_dir, md_filename)
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# 🧬 样本 {i} 的进化史\n\n")
            f.write(f"**观察对象**: 数据集中的第 {i} 张图片\n\n")
            f.write(f"**说明**: 向下滚动查看该图片从 Epoch {folders[0].split('_')[1]} 到最后的训练变化。\n\n")
            f.write("---\n\n")

            # 遍历所有 epoch 文件夹
            for folder in folders:
                epoch_num = folder.split('_')[1]
                img_name = f"val_img_{i}.jpg"
                
                # 相对路径 (用于 Markdown 显示)
                # 结构: visualizations/epoch_X/val_img_i.jpg
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

    print(f"✅ 画廊生成完毕！请在 VS Code 中打开 '{save_dir}/evolution_sample_X.md' 查看。")

if __name__ == "__main__":
    # 确保之前的 SimpleYOLO, COCO128Dataset, SimpleComputeLoss, non_max_suppression 都在上下文中
    train_professional()