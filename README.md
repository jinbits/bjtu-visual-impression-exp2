# 交大视觉印象实验2：图像检索与文字检测

本项目用于完成交大视觉印象-阶段2实验：  
1. 使用 query 图像在 base 图像库中做以图搜图，并计算每类 landmark 的 P@20 / P@40 / P@60。  
2. 将 LabelMe 文字框标注转换为 YOLO 数据集，训练单类 `text` 检测器。  
3. 生成检索曲线、24 组“检索-检测”可视化


- 学号：`保密`
- 姓名：保密
- GitHub链接：`https://github.com/ts-ts-ts-1/bjtu-visual-impression-exp2.git`

## 环境安装

建议使用 Python 3.10+。GPU 可用时会自动使用 CUDA；显存不足时可调小 `configs/default.json` 中的 batch 或图片尺寸。

```powershell
python -m pip install -r requirements.txt
```

如果 PyTorch 没有自动装到合适的 CUDA 版本，可先按本机环境安装 PyTorch，再安装其余依赖：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## 一键运行

```powershell
python run_all.py --data-root "D:\学业\26first\微专业\计算机视觉基础\作业\交大视觉印象-阶段2\交大视觉印象数据集2026"
```

快速验证流程可跳过完整训练或缩短训练：

```powershell
python run_all.py --data-root "D:\学业\26first\微专业\计算机视觉基础\作业\交大视觉印象-阶段2\交大视觉印象数据集2026" --skip-train
python run_all.py --data-root "D:\学业\26first\微专业\计算机视觉基础\作业\交大视觉印象-阶段2\交大视觉印象数据集2026" --fast
```

## 单步运行

```powershell
python -m src.validate_dataset --data-root "D:\...\交大视觉印象数据集2026"
python -m src.retrieval --data-root "D:\...\交大视觉印象数据集2026"
python -m src.detection --data-root "D:\...\交大视觉印象数据集2026"
python -m src.visualize --data-root "D:\...\交大视觉印象数据集2026"
```

## 主要输出

- `outputs/dataset_summary.json`：数据读取和标注一致性校验。
- `outputs/retrieval/class_precision.csv`：12 类和 overall 的 P@K 结果。
- `outputs/retrieval/top60_results.csv`：每张 query 的 Top60 检索结果。
- `outputs/retrieval/plots/`：12 张每类 P@K 曲线和总览图。
- `outputs/detection_yolo/`：YOLO 格式数据集和 `dataset.yaml`。
- `outputs/detection_train/text_yolov8s/weights/best.pt`：训练完成后的最佳检测权重。
- `outputs/visualizations/cases/`：24 组“query -> Top-K 检索 -> 文字检测”可视化图。

## 方法说明

图像检索默认优先加载 DINOv2 与 OpenCLIP 预训练模型，拼接图像特征后在 base 集上拟合 PCA-whitening，再以余弦相似度排序。训练和白化过程中不使用类别标签，文件名前缀只用于最终 P@K 评测。

文字检测将所有 LabelMe 框统一为单类 `text`，按固定随机种子划分 train / val / test，默认训练 YOLOv8s。若深度学习依赖或权重下载失败，代码会记录失败原因，并退回到标注框绘制。

