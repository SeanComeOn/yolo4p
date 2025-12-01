import os
import json
from flask import Flask, render_template_string, jsonify, send_file

# ================= 配置区域 =================
# 确保这里的路径对应你之前下载的 coco128 目录结构
IMG_DIR = 'coco128/images/train2017'
LABEL_DIR = 'coco128/labels/train2017'

# COCO 80类名称 (对应索引 0-79)
COCO_CLASSES = [
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

app = Flask(__name__)

# ================= 前端 HTML/JS 代码 =================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <title>YOLO 数据集可视化工具</title>
    <style>
        body { margin: 0; display: flex; height: 100vh; font-family: sans-serif; background: #1e1e1e; color: #fff; }
        
        /* 左侧边栏 */
        #sidebar { width: 300px; overflow-y: auto; background: #252526; border-right: 1px solid #333; }
        .file-item { padding: 10px 15px; cursor: pointer; border-bottom: 1px solid #333; color: #ccc; font-size: 14px; }
        .file-item:hover { background: #37373d; color: #fff; }
        .file-item.active { background: #094771; color: #fff; }

        /* 右侧主视口 */
        #main { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; padding: 20px; }
        #canvas-container { position: relative; box-shadow: 0 0 20px rgba(0,0,0,0.5); border: 2px solid #444; }
        canvas { display: block; }
        
        /* 信息面板 */
        #info { margin-top: 10px; color: #aaa; }
    </style>
</head>
<body>

<div id="sidebar">
    <div style="padding:15px; font-weight:bold; background:#333;">文件列表 (<span id="count">0</span>)</div>
    <div id="file-list"></div>
</div>

<div id="main">
    <div id="canvas-container">
        <canvas id="canvas"></canvas>
    </div>
    <div id="info">请选择左侧图片</div>
</div>

<script>
    const colors = ['#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231', '#48F90A', '#92CC17', '#3DDB86', '#1A9334', '#00D4BB', '#2C99A8', '#00C2FF', '#344593', '#6473FF', '#0018EC', '#8438FF', '#520085', '#CB38FF', '#FF95C8', '#FF37C7'];
    
    let currentImage = null;

    // 1. 加载文件列表
    fetch('/api/list')
        .then(r => r.json())
        .then(files => {
            document.getElementById('count').innerText = files.length;
            const listDiv = document.getElementById('file-list');
            files.forEach(f => {
                const div = document.createElement('div');
                div.className = 'file-item';
                div.innerText = f;
                div.onclick = () => loadFile(f, div);
                listDiv.appendChild(div);
            });
        });

    function loadFile(filename, elem) {
        // 高亮选中
        document.querySelectorAll('.file-item').forEach(e => e.classList.remove('active'));
        elem.classList.add('active');
        document.getElementById('info').innerText = `正在加载: ${filename}`;

        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        // 加载图片
        img.src = `/api/image/${filename}`;
        img.onload = () => {
            // 设置画布大小
            canvas.width = img.width;
            canvas.height = img.height;
            
            // 绘制图片
            ctx.drawImage(img, 0, 0);
            
            // 加载并绘制标签
            fetch(`/api/label/${filename}`)
                .then(r => r.json())
                .then(labels => {
                    labels.forEach(label => drawBox(ctx, label, img.width, img.height));
                    document.getElementById('info').innerText = `${filename} | 分辨率: ${img.width}x${img.height} | 目标数: ${labels.length}`;
                });
        };
    }

    // 绘制 YOLO 格式框 (class x_center y_center w h)
    function drawBox(ctx, label, imgW, imgH) {
        const [clsName, xc, yc, w, h] = label;
        
        // YOLO 坐标 (归一化中心点+宽高) -> 像素左上角坐标
        const boxW = w * imgW;
        const boxH = h * imgH;
        const boxX = (xc * imgW) - (boxW / 2);
        const boxY = (yc * imgH) - (boxH / 2);

        // 随机颜色 (根据类名 Hash)
        const colorIdx = clsName.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % colors.length;
        const color = colors[colorIdx];

        // 画框
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(boxX, boxY, boxW, boxH);

        // 画标签背景
        ctx.fillStyle = color;
        ctx.font = "bold 16px Arial";
        const text = clsName;
        const textMetrics = ctx.measureText(text);
        const textHeight = 20; 
        
        ctx.fillRect(boxX, boxY - textHeight, textMetrics.width + 10, textHeight);
        
        // 画文字
        ctx.fillStyle = "black";
        ctx.fillText(text, boxX + 5, boxY - 5);
    }
</script>
</body>
</html>
"""

# ================= 后端路由逻辑 =================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/list')
def get_list():
    if not os.path.exists(IMG_DIR):
        return jsonify([])
    files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])
    return jsonify(files)

@app.route('/api/image/<filename>')
def get_image(filename):
    return send_file(os.path.join(IMG_DIR, filename))

@app.route('/api/label/<filename>')
def get_label(filename):
    # 找到对应的 txt 文件
    txt_name = os.path.splitext(filename)[0] + '.txt'
    txt_path = os.path.join(LABEL_DIR, txt_name)
    
    parsed_labels = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    # 获取类名，防止索引越界
                    cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
                    # 返回: [类名, x, y, w, h] (保持 float)
                    parsed_labels.append([
                        cls_name, 
                        float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    ])
    return jsonify(parsed_labels)

if __name__ == '__main__':
    print(f"数据目录检查: {IMG_DIR}")
    if not os.path.exists(IMG_DIR):
        print(f"错误: 找不到目录 {IMG_DIR}，请确保你已经下载了 coco128 并解压在当前目录下。")
    else:
        print("启动服务中... 请在浏览器访问 http://127.0.0.1:5000")
        app.run(debug=True, port=5000)