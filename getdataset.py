import torch

# 使用 torch.hub 下载 (这是最快的方法，会自动解压到当前目录或 torch cache)
# 如果下载慢，你可以手动去 Kaggle 或 Ultralytics Github 搜 coco128.zip 下载
torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip', 'coco128.zip')
import zipfile
with zipfile.ZipFile('coco128.zip', 'r') as zip_ref:
    zip_ref.extractall('.')  # 解压到当前文件夹