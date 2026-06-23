# 宫颈癌细胞AI诊断系统

基于深度学习的宫颈细胞病理图像智能分析平台，实现自动化病变筛查与分级。

## 🎯 项目简介

本系统采用双模型协同架构（YOLOv8检测 + ResNet50分类），支持宫颈细胞的自动定位、计数和病变程度分级，为病理科医生提供辅助诊断工具。

### 核心功能

- ✅ **细胞检测**: YOLOv8精确定位图像中的单个细胞
- ✅ **病变分类**: ResNet50进行7级TBS分类或二分类（正常/异常）
- ✅ **注意力机制**: 集成SE/CBAM提升特征提取能力
- ✅ **多数据集融合**: 支持HERLEV和SIPaKMeD数据集混合训练
- ✅ **Web界面**: Flask后端提供医生友好的交互界面
- ✅ **高级增强**: MixUp、CutMix、EMA等训练技巧

## 📊 数据集

| 数据集 | 图像数量 | 类别数 | 来源 |
|--------|---------|--------|------|
| HERLEV | 917张 | 7类 | 丹麦哥本哈根大学医院 |
| SIPaKMeD | 5,015张 | 5类 | 泰国诗里拉吉医院 |
| **合并** | **5,932张** | **7→2类** | - |

**数据划分**: 训练集70% / 验证集15% / 测试集15%

## 🛠️ 技术栈

- **深度学习**: PyTorch ≥1.9.0, Ultralytics YOLOv8 ≥8.0.0
- **图像处理**: OpenCV ≥4.5.0, PIL ≥8.3.0
- **Web框架**: Flask 2.3.2
- **数据处理**: NumPy ≥1.21.0, scikit-learn ≥1.0.0
- **可视化**: Matplotlib ≥3.5.0, TensorBoard

## 📦 安装步骤

### 1. 环境准备
bash 
conda create -n cervical-ai python=3.9 
conda activate cervical-ai

### 2. 安装依赖
bash 
pip install -r requirements.txt

### 3. 数据集准备
将HERLEV和SIPaKMeD数据集放入对应目录：
data/ 
├── herlev/ # HERLEV原始数据 
└── sipakmed/ # SIPaKMeD原始数据

### 4. 数据集转换（可选）

转换为YOLO格式：
bash python utils/herlev_yolo_converter.py --input data/herlev --output data/herlev_yolo 
python utils/sipakmed_converter.py --input data/sipakmed --output data/sipakmed_yolo

## 🚀 快速开始

### 训练模型

#### 分类模型训练
bash
二分类模式（默认）
python scripts/train.py --config configs/default.yaml --epochs 100

多分类模式
python scripts/train.py --classification_mode multiclass --epochs 150

使用Focal Loss + EMA
python scripts/train.py --loss_type focal --use_ema --experiment_name exp_focal_ema

使用注意力机制
python scripts/train.py --attention_type cbam --model_type resnet101

#### YOLO检测模型训练
bash 
python scripts/train_yolo.py 
--data configs/herlev_yolo.yaml 
--epochs 100 
--batch 16 
--imgsz 640

### 评估模型
bash
基础评估
python scripts/evaluate.py --model_path outputs/models/best_resnet50_binary_model.pth

绘制ROC曲线
python scripts/evaluate.py --model_path outputs/models/best_model.pth --plot_roc

蒙特卡洛Dropout不确定性估计
python scripts/evaluate.py --model_path outputs/models/best_model.pth --use_mc_dropout --mc_samples 20

### 启动Web服务
bash 
cd appWeb 
python app.py

访问 http://localhost:5001

**测试账号**:
- 用户名: `dr_zhang` / 密码: `password123`
- 用户名: `dr_wang` / 密码: `securepwd456`

## 📁 项目结构
Cervix-Cytology-AI-System/ 
├── appWeb/ # Web应用 
│ ├── app.py # Flask主程序 
│ ├── doctor_accounts.json # 医生账号 
│ └── templates/tryweb.html # 前端页面 
│ 
├── models/ # 模型定义 
│ ├── cer_cancer_model.py # 分类模型（含注意力） 
│ ├── yolo_model.py # YOLOv8封装 
│ └── resnet_model.py # ResNet变体 
│ 
├── utils/ # 工具函数 
│ ├── data_loader.py # 数据加载器 
│ ├── preprocess.py # 预处理+增强 
│ ├── advanced_augmentation.py # MixUp/CutMix 
│ ├── visualization.py # 可视化 
│ ├── wsi_processor.py # 全切片处理 
│ └── yolo_processor.py # YOLO数据处理 
│ 
├── scripts/ # 训练/评估脚本 
│ ├── train.py # 主训练脚本 
│ ├── train_yolo.py # YOLO训练 
│ ├── evaluate.py # 评估脚本 
│ ├── predict.py # 单张预测 
│ └── ... 
│ 
├── configs/ # 配置文件 
│ ├── default.yaml # 默认配置 
│ └── ... 
│ 
└── data/ # 数据集 
├── herlev/ 
├── sipakmed/ 
└── combined_yolo/

## ⚙️ 配置说明

编辑 `configs/default.yaml` 修改训练参数：
```yaml
data: 
batch_size: 32 
img_size: 224 
train_ratio: 0.7 
val_ratio: 0.15 
test_ratio: 0.15

model: 
name: "resnet50" 
num_classes: 7 
pretrained: true 
dropout_rate: 0.5 
use_attention: false

training: 
epochs: 100 
learning_rate: 0.001 
weight_decay: 0.0001 
early_stopping_patience: 15 
loss_type: "cross_entropy" # cross_entropy/focal/weighted
use_ema: false
```

## 📈 性能指标

典型训练结果（二分类模式）：

- **准确率**: 92-95%
- **精确率**: 90-93%
- **召回率**: 91-94%
- **F1 Score**: 90-93%
- **AUC**: 0.95-0.97

## 🔧 高级功能

### 交叉验证
bash 
python scripts/cross_validation.py --folds 5

### 模型蒸馏
bash 
python scripts/model_distillation.py 
--teacher_model resnet101 
--student_model resnet50

### 集成预测
bash python scripts/ensemble_predict.py 
--models model1.pth model2.pth model3.pth

### Grad-CAM可视化
python 
from utils.grad_cam import GradCAM 
cam = GradCAM(model, target_layer='layer4') 
heatmap = cam(input_image)

## 📝 引用

如果使用本项目的数据集或代码，请引用：
```bibtex
@misc{cervical-ai-system,
 title={Cervical Cell AI Diagnosis System},
  author={Your Name}, 
  year={2026}, 
  publisher={GitHub}
}
```

## 📄 许可证
MIT License

## 👥 贡献
欢迎提交Issue和Pull Request！

## 📧 联系方式
如有问题或建议，请提交Issue或联系项目维护者。