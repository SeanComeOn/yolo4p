import torch
from model import SimpleYOLO
from thop import profile

# 1. 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleYOLO(num_classes=80).to(device)

# 2. 创建一个虚拟输入 (BatchSize=1, RGB, 640x640)
input_tensor = torch.randn(1, 3, 640, 640).to(device)

# 3. 使用 thop 进行分析
print("正在分析模型 (这可能需要几秒钟)...")
flops, params = profile(model, inputs=(input_tensor, ), verbose=False)

print("-" * 30)
print(f"模型名称: SimpleYOLO (YOLO4P)")
print(f"输入尺寸: 640x640")
print(f"参数量 (Params): {params / 1e6:.2f} M")
print(f"计算量 (GFLOPs): {flops / 1e9:.2f} G")
print("-" * 30)