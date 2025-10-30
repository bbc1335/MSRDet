# A Lightweight Detection Network Integrating Multi-Scale Semantic Refinement for Steel Strip Defects

## 📖 Overview

MSRDet is an advanced object detection built upon the Ultralytics YOLO architecture, specifically designed for steel surface defect detection. 

<!-- ## 🚀 Key Features

- **Multi-Scale Feature Fusion**: Advanced feature pyramid networks with residual connections
- **Lightweight Architecture**: Optimized for real-time inference on edge devices
- **High Precision Detection**: State-of-the-art performance on steel defect datasets
- **YOLOv8 Integration**: Built upon the robust Ultralytics YOLOv8 framework
- **Custom Modules**: Includes specialized components like C2fk, GatedSPPF, FEM, HTEM, and Fusion_2in_mod

## 📊 Model Architecture

The MSRDet model incorporates several innovative components:

- **Backbone**: Enhanced YOLOv8 backbone with C2f and C2fk modules
- **Neck**: Multi-scale feature fusion with upsampling and concatenation
- **Head**: Custom detection head with Feature Enhancement Module (FEM) and Hierarchical Texture Enhancement Module (HTEM)
- **Detection**: Multi-scale detection at P3, P4, and P5 levels -->

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (for GPU acceleration)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/bbc1335/MSRDet.git
cd MSRDet

# install in development mode for code modifications
pip install -e .

```

## 📈 Usage

### Training
```python
from ultralytics import YOLO

# Load MSRDet model
model = YOLO("ultralytics/cfg/models/v8/MSRDet.yaml")

# Train on your dataset
model.train(data="your_dataset.yaml", epochs=300, imgsz=512, batch=32)
```

### Inference
```python
from ultralytics import YOLO

# Load trained model
model = YOLO("path/to/best.pt")

# Run inference
results = model.predict("path/to/image.jpg")
```

### Validation
```python
from ultralytics import YOLO

# Validate model performance
model = YOLO("path/to/best.pt")
validation_results = model.val(data="your_dataset.yaml")
```

## 📁 Project Structure

```plaintext
project/
├── ultralytics/             # 核心代码库
│   ├── cfg/                 # 配置文件
│   │   ├── datasets/        # 数据集配置
│   │   ├── models/          # 模型配置
│   │   └── trackers/        # 跟踪器配置
│   ├── data/                # 数据处理
│   │   ├── augment.py       # 数据增强
│   │   ├── dataset.py       # 数据集处理
│   │   ├── loaders.py       # 数据加载器
│   │   └── utils.py         # 数据工具函数
│   ├── engine/              # 引擎模块
│   │   ├── exporter.py      # 模型导出
│   │   ├── model.py         # 模型管理
│   │   ├── predictor.py     # 预测器
│   │   ├── trainer.py       # 训练器
│   │   └── validator.py     # 验证器
│   ├── hub/                 # 模型中心集成
│   ├── models/              # 模型定义
│   │   ├── fastsam/         # FastSAM模型
│   │   ├── nas/             # 神经架构搜索模型
│   │   ├── rtdetr/          # RT-DETR模型
│   │   ├── sam/             # SAM模型
│   │   └── yolo/            # YOLO系列模型
│   ├── nn/                  # 神经网络模块（核心改进）
│   │   ├── modules/         # 自定义模块
│   │   │   ├── block.py     # 核心改进模块（包含GatedSPPF、FEM、HTEM等）
│   │   │   ├── conv.py      # 卷积模块
│   │   │   ├── head.py      # 检测头模块
│   │   │   ├── transformer.py # Transformer模块
│   │   │   └── utils.py     # 模块工具函数
│   │   ├── autobackend.py   # 自动后端支持
│   │   └── tasks.py         # 任务定义
│   ├── solutions/        # Task-specific files
│   ├── utils/            # Utility function files
└── README.md             # Project documentation
```

# Acknowledgments
This repo is built upon a very early version of [Ultralytics](https://github.com/ultralytics/ultralytics). Sincere thanks to their excellent work!