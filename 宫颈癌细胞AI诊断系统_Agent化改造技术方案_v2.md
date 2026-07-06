# 宫颈癌细胞AI诊断系统 -- AI Agent化改造技术方案 v2.0

> **文档版本**: v2.0 (基于完整技术细节修订)  
> **编写日期**: 2026-06-16  
> **项目背景**: 基于比赛获奖项目(YOLOv8+ResNet50+SE/CBAM+MixUp/CutMix+MC Dropout)升级为AI Agent辅助诊断平台  
> **作者背景**: 大三人工智能专业, Datawhale Hello-Agents课程学习中(第1-4章已完成)

---

## 一、项目现状深度分析

### 1.1 现有系统真实能力

基于完整技术文档, 原项目远超基础水平:

| 能力维度 | 现有实现 | 技术深度 |
|---------|---------|---------|
| **细胞检测** | YOLOv8 (n/s/m/l/x全系列支持) | 支持detect/segment双模式, CIoU+BCE Loss, Mosaic增强 |
| **细胞分类** | ResNet50/101 + EfficientNet-B0 + ViT | 4种Backbone可选, ImageNet预训练, SE/CBAM注意力即插即用 |
| **数据工程** | HERLEV(917张) + SIPaKMeD(5015张) | 双数据集融合, 12类->2类/7类映射, 混合划分 |
| **数据增强** | 8种变换组合 | 随机翻转/旋转/颜色抖动/仿射/擦除 + **MixUp(alpha=0.2)** + **CutMix(alpha=1.0)** |
| **损失函数** | CrossEntropy + Focal Loss + Weighted CE | 自动类别权重计算, 难例聚焦 |
| **训练优化** | Adam + 4种学习率调度 | ReduceLROnPlateau/CosineAnnealing/WarmRestarts/OneCycle + **EMA(decay=0.999)** + **学习率预热** + **早停(patience=15)** |
| **评估体系** | 混淆矩阵 + AUC-ROC + F1 Macro/Micro | **MC Dropout不确定性估计**(n_samples=10) |
| **可解释性** | Grad-CAM热力图 | 可视化模型关注区域 |
| **形态分析** | cell_analyzer.py | 核质比、面积、周长、圆形度计算 |
| **Web服务** | Flask + Session认证 | 医生账号管理, 多文件上传(最多10个), UUID命名, secure_filename安全处理 |
| **全切片处理** | wsi_processor.py | 超大图像分块推理 |

### 1.2 现有系统架构

```
+-------------+     +------------------+     +------------------+
|   用户上传   |---->|  YOLOv8检测      |---->|  ResNet50分类    |
|   细胞图像   |     |  (conf=0.25)     |     |  (+SE/CBAM)      |
+-------------+     +--------+---------+     +--------+---------+
                             |                        |
                             v                        v
                    +--------+---------+      +------+------+
                    | 边界框+置信度     |      | 7类TBS分级   |
                    | total_cells=15   |      | confidence   |
                    +--------+---------+      +------+------+
                             |                        |
                             +-----------+------------+
                                         |
                                  +------v------+
                                  | 规则生成诊断 |
                                  | "建议复查"  |
                                  +-------------+
```

**核心问题**: 固定规则输出, 无智能交互, 无知识增强, 无可解释性对话

### 1.3 与AI Agent实习要求的差距

| 实习要求 | 现有状态 | 差距 |
|---------|---------|------|
| LLM交互能力 | 无 | 需接入大模型生成自然语言报告 |
| RAG/向量检索 | 无 | 需构建医学知识库 |
| Agent编排框架 | 无 | 需LangGraph状态机 |
| Tool Use | 无 | 需封装形态分析等为Tool |
| 多轮对话 | 单轮上传 | 需状态管理 |
| 流式输出 | 同步HTTP | 需WebSocket异步 |
| 可解释性对话 | Grad-CAM静态图 | 需结合LLM解释 |
| 不确定性量化 | MC Dropout已有 | **可直接利用!** |

---

## 二、改造核心策略: "演进而非推翻"

### 2.1 设计原则

1. **保留原系统**: Flask服务继续运行, 作为"传统AI诊断"基线
2. **新增Agent层**: FastAPI异步服务承载Agent能力
3. **利用现有能力**: MC Dropout->不确定性驱动, Grad-CAM->可解释性, 形态分析->Tool
4. **双版本展示**: GitHub上同时展示v1.0(比赛版)和v2.0(Agent版)

### 2.2 改造后架构

```
+-------------------------------------------------------------+
|                        用户交互层                            |
|  +-------------+  +-------------------------------------+ |
|  | 原jQuery前端 |  | 新React前端 (WebSocket流式对话)      | |
|  | /dashboard  |  | /agent-dashboard                    | |
|  +------+------+  +-------------------+-----------------+ |
+--------+-----------------------------+--------------------+
         |                             |
         v                             v
+-----------------+           +---------------------------------+
|  Flask服务(保留) |           |      FastAPI Agent服务(新增)     |
|  /upload        |           |  /api/agent/upload              |
|  /login         |           |  /api/agent/chat                |
|  /dashboard     |           |  /ws/agent/stream               |
+--------+--------+           +-------------+-------------------+
         |                                  |
         |    +-------------------------------+
         |    |
         v    v
+-------------------------------------------------------------+
|                     共享CV模型服务层                         |
|  +-----------------------------------------------------+   |
|  |  YOLOv8检测 + ResNet50分类 + SE/CBAM + MC Dropout    |   |
|  |  (复用现有模型, 无需重新训练)                         |   |
|  +-----------------------------------------------------+   |
+-------------------------------------------------------------+
          |
          v
+-------------------------------------------------------------+
|                     AI Agent编排层 (LangGraph)               |
|                                                              |
|   +-------------+    +-------------+    +-------------+     |
|   | 不确定性评估  |--->|   决策路由   |--->|  知识检索   |     |
|   | Agent       |    |  Agent      |    |  Agent      |     |
|   | (MC Dropout)|    |             |    | (RAG+图谱)  |     |
|   +-------------+    +------+------+    +-------------+     |
|                             |                                 |
|              +--------------+--------------+                 |
|              |              |              |                 |
|              v              v              v                 |
|        +---------+    +---------+    +---------+            |
|        |直接报告  |    |主动追问  |    |深度检索  |            |
|        |Agent    |    |Agent    |    |Agent    |            |
|        |+GradCAM |    |+形态分析 |    |+文献对比 |            |
|        +----+----+    +----+----+    +----+----+            |
|             +--------------+--------------+                  |
|                             |                                 |
|                             v                                 |
|                      +-------------+                          |
|                      | 报告生成Agent |                          |
|                      | +可解释性    |                          |
|                      | +参考文献    |                          |
|                      +-------------+                          |
+-------------------------------------------------------------+
          |
    +-----+-----+
    v           v
+-------+   +---------------+
| LLM   |   | 医学知识库     |
|服务   |   | Chroma+规则图谱|
+-------+   +---------------+
```

---

## 三、详细技术方案

### 3.1 模块一: 共享CV模型服务层 (复用升级)

**策略**: 提取现有 app.py 中的模型加载和推理逻辑, 封装为独立服务

```
python
# services/vision_service.py
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np

app = FastAPI(title="CV Model Service")

# ========== 复用现有模型加载逻辑 ==========
# 原 app.py 中的 load_models() 逻辑提取

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# YOLOv8检测模型 (复用原配置)
detector = YOLO("runs/detect/cervical_cell_exp/weights/best.pt")

# ResNet50分类模型 (复用原配置, 支持SE/CBAM注意力自动检测)
from models.cer_cancer_model import CervicalCancerModel

model_file = "outputs/models/best_resnet50_binary_model.pth"
model_state_dict = torch.load(model_file, map_location=device)

# 自动检测注意力机制 (复用原逻辑)
use_attention = any('attention' in key for key in model_state_dict.keys())
attention_type = None
if use_attention:
    if any('se' in key.lower() for key in model_state_dict.keys()):
        attention_type = 'se'
    elif any('cbam' in key.lower() for key in model_state_dict.keys()):
        attention_type = 'cbam'

classifier = CervicalCancerModel(
    num_classes=2,  # 或7, 根据模式
    pretrained=False,
    model_type='resnet50',
    use_attention=use_attention,
    attention_type=attention_type
)
classifier.load_state_dict(model_state_dict)
classifier.to(device)
classifier.eval()

# 分类标签 (复用原映射)
BINARY_CLASSES = ['normal', 'abnormal']
MULTICLASS_CLASSES = [
    'carcinoma_in_situ', 'severe_dysplastic', 'moderate_dysplastic',
    'light_dysplastic', 'normal_columnar', 'normal_intermediate', 
    'normal_superficial'
]

# ========== 复用现有预处理逻辑 ==========
# 原 preprocess_image() 逻辑提取
def preprocess_image(image, mode='train'):
    '''复用原项目的预处理流程'''
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((268, 268)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                 saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    return transform(image)

# ========== MC Dropout不确定性估计 (复用现有能力) ==========
def mc_dropout_predict(model, inputs, n_samples=10):
    '''
    复用原项目的MC Dropout实现
    训练时保持Dropout激活, 多次前向传播估计不确定性
    '''
    model.train()  # 保持Dropout激活
    predictions = []

    for _ in range(n_samples):
        with torch.no_grad():
            output = model(inputs)
            prob = torch.softmax(output, dim=1)
            predictions.append(prob.cpu().numpy())

    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)

    return mean_pred, std_pred

# ========== 形态学分析Tool (复用cell_analyzer.py) ==========
from utils.cell_analyzer import analyze_cell_morphology

class MorphologyTool:
    '''细胞形态学分析工具'''
    name = "cell_morphology_analyzer"
    description = "分析细胞核质比、面积、周长、圆形度等形态学特征"

    def run(self, cell_image) -> dict:
        features = analyze_cell_morphology(cell_image)
        return {
            "nuclear_cytoplasmic_ratio": features.get("nc_ratio", 0),
            "cell_area": features.get("area", 0),
            "cell_perimeter": features.get("perimeter", 0),
            "circularity": features.get("circularity", 0),
            "interpretation": self._interpret(features)
        }

    def _interpret(self, features):
        if features.get("nc_ratio", 0) > 0.5:
            return "核质比增高, 提示细胞异型性"
        elif features.get("circularity", 0) < 0.7:
            return "细胞形态不规则, 需警惕"
        return "形态学特征未见明显异常"

# ========== FastAPI接口 ==========
@app.post("/analyze")
async def analyze_image(file: UploadFile):
    '''图像分析接口: 检测+分类+不确定性估计'''
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # 1. YOLO检测 (复用原逻辑)
    detection_results = detector(image, conf=0.25, iou=0.45)
    cells = []

    for box in detection_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cell_img = image.crop((x1, y1, x2, y2))

        # 2. ResNet分类 (复用原预处理)
        input_tensor = preprocess_image(cell_img, mode='val').unsqueeze(0).to(device)

        with torch.no_grad():
            output = classifier(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # 3. MC Dropout不确定性 (新增)
        mean_prob, std_prob = mc_dropout_predict(classifier, input_tensor, n_samples=10)
        uncertainty = float(np.max(std_prob))

        # 4. 形态学分析 (新增Tool封装)
        morphology = MorphologyTool().run(cell_img)

        cells.append({
            "bbox": [x1, y1, x2, y2],
            "class": BINARY_CLASSES[predicted_class],
            "confidence": round(confidence, 4),
            "uncertainty": round(uncertainty, 4),
            "probabilities": {
                name: round(prob, 4) 
                for name, prob in zip(BINARY_CLASSES, probabilities[0].tolist())
            },
            "morphology": morphology
        })

    return {
        "total_cells": len(cells),
        "abnormal_cells": sum(1 for c in cells if c["class"] != "normal"),
        "cells": cells,
        "highest_risk": max(cells, key=lambda x: x["confidence"])["class"] if cells else "无",
        "max_uncertainty": max((c["uncertainty"] for c in cells), default=0),
        "needs_clarification": max((c["uncertainty"] for c in cells), default=0) > 0.15
    }
```

**关键升级点**:
- 复用原项目所有模型和预处理逻辑, 无需重新训练
- 新增MC Dropout不确定性量化 (已有能力直接利用)
- 新增形态学分析Tool封装 (已有cell_analyzer.py)
- 返回结构化数据供Agent决策

---

### 3.2 模块二: RAG医学知识库系统

**目标**: 让诊断有据可依, 支持权威文献引用

#### 3.2.1 知识来源 (基于项目现有医学场景)

| 来源类型 | 具体内容 | 处理方式 | 与现有系统的关联 |
|---------|---------|---------|---------------|
| TBS分类标准 | 宫颈细胞学Bethesda报告系统 | PDF解析+结构化 | 与现有7类分类标签对齐 |
| NCCN指南 | 宫颈癌筛查与诊疗指南 | PDF解析+分块 | 支持诊断建议生成 |
| 形态学图谱 | 正常/异常细胞形态对照 | 图文关联 | 与Grad-CAM和形态分析联动 |
| 药物数据库 | 常用治疗药物及副作用 | 结构化表格 | 支持随访建议 |

#### 3.2.2 技术实现

```
python
# services/rag_service.py
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

class MedicalKnowledgeBase:
    def __init__(self, persist_dir="./knowledge_db"):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.persist_dir = persist_dir

        if os.path.exists(persist_dir):
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings
            )
        else:
            self.vectorstore = None

    def load_documents(self, pdf_paths: list):
        '''加载医学文献PDF'''
        documents = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

        # 智能切分: 保持段落完整性
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "；", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)

        # 构建向量库
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )
        self.vectorstore.persist()

        return len(chunks)

    def retrieve(self, query: str, k: int = 5):
        '''检索相关知识'''
        if not self.vectorstore:
            return []

        results = self.vectorstore.similarity_search_with_score(query, k=k)

        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "未知"),
                "page": doc.metadata.get("page", 0),
                "score": round(score, 4)
            }
            for doc, score in results
        ]

    def retrieve_by_classification(self, classification: str, k: int = 3):
        '''根据分类结果检索对应知识'''
        query_map = {
            "normal": "正常宫颈细胞形态学特征及筛查建议",
            "abnormal": "异常宫颈细胞分类标准及处理建议",
            "light_dysplastic": "轻度不典型增生(LSIL)管理指南",
            "moderate_dysplastic": "中度不典型增生处理原则",
            "severe_dysplastic": "重度不典型增生(HSIL)治疗方案",
            "carcinoma_in_situ": "原位癌诊断标准及治疗原则"
        }
        query = query_map.get(classification, classification)
        return self.retrieve(query, k=k)

    def retrieve_by_morphology(self, morphology_features: dict, k: int = 3):
        '''根据形态学特征检索相关知识'''
        queries = []
        if morphology_features.get("nuclear_cytoplasmic_ratio", 0) > 0.5:
            queries.append("核质比增高 细胞异型性 诊断意义")
        if morphology_features.get("circularity", 1) < 0.7:
            queries.append("细胞形态不规则 病理意义")

        results = []
        for q in queries:
            results.extend(self.retrieve(q, k=2))
        return results
```

---

### 3.3 模块三: AI Agent编排层 (核心创新)

使用 **LangGraph** 构建状态机, 核心创新: **不确定性驱动决策**

#### 3.3.1 状态定义

```
python
# agents/state.py
from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage

class DiagnosisState(TypedDict):
    '''诊断流程状态'''
    # 输入
    image_path: Optional[str]
    user_query: Optional[str]
    conversation_history: List[BaseMessage]

    # CV分析结果 (来自现有模型)
    cv_results: Optional[Dict[str, Any]]

    # 不确定性量化 (来自MC Dropout)
    max_uncertainty: Optional[float]
    needs_clarification: bool

    # RAG检索结果
    retrieved_knowledge: Optional[List[Dict]]

    # Agent决策
    risk_level: Optional[str]  # 低/中/高/紧急
    action: Optional[str]  # direct_report / clarify / deep_retrieve
    clarification_questions: Optional[List[str]]

    # 可解释性
    gradcam_path: Optional[str]
    morphology_analysis: Optional[Dict]

    # 输出
    diagnosis_report: Optional[str]
    recommendations: Optional[List[str]]
    references: Optional[List[str]]

    # 流程控制
    current_step: str
    is_complete: bool
```

#### 3.3.2 Agent节点实现

```
python
# agents/diagnosis_graph.py
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import json
import os

# LLM实例
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ============ 节点1: 不确定性评估Agent ============
def uncertainty_assessment_agent(state: DiagnosisState) -> DiagnosisState:
    '''
    利用MC Dropout结果评估不确定性, 决定Agent行为
    这是核心创新: 利用现有MC Dropout能力驱动Agent决策
    '''
    cv_results = state["cv_results"]
    max_uncertainty = cv_results.get("max_uncertainty", 0)

    # 不确定性分级决策
    if max_uncertainty > 0.3:
        # 高不确定性: 必须追问
        return {
            **state,
            "max_uncertainty": max_uncertainty,
            "needs_clarification": True,
            "action": "clarify",
            "risk_level": "待评估",
            "current_step": "uncertainty_assessment"
        }
    elif max_uncertainty > 0.15:
        # 中等不确定性: 检索更多知识
        return {
            **state,
            "max_uncertainty": max_uncertainty,
            "needs_clarification": False,
            "action": "deep_retrieve",
            "risk_level": "中",
            "current_step": "uncertainty_assessment"
        }
    else:
        # 低不确定性: 直接生成报告
        abnormal_count = cv_results.get("abnormal_cells", 0)
        total_count = cv_results.get("total_cells", 0)

        if abnormal_count == 0:
            risk = "低"
        elif abnormal_count / total_count < 0.3:
            risk = "中"
        elif abnormal_count / total_count < 0.6:
            risk = "高"
        else:
            risk = "紧急"

        return {
            **state,
            "max_uncertainty": max_uncertainty,
            "needs_clarification": False,
            "action": "direct_report",
            "risk_level": risk,
            "current_step": "uncertainty_assessment"
        }

# ============ 节点2: 决策路由 ============
def decision_router(state: DiagnosisState) -> str:
    '''根据不确定性评估结果路由'''
    return state["action"]

# ============ 节点3: 主动追问Agent ============
def clarification_agent(state: DiagnosisState) -> DiagnosisState:
    '''高不确定性时主动追问关键信息'''
    cv_results = state["cv_results"]

    # 基于细胞特征生成追问问题
    prompt = f'''
    你是一位病理科AI助手。检测到以下细胞图像的不确定性较高(uncertainty={state["max_uncertainty"]:.3f}),
    需要向医生/患者追问关键信息以辅助诊断。

    检测结果:
    - 总细胞数: {cv_results.get("total_cells", 0)}
    - 异常细胞数: {cv_results.get("abnormal_cells", 0)}
    - 最高风险: {cv_results.get("highest_risk", "未知")}

    请生成2-3个最关键的追问问题, 帮助降低诊断不确定性。
    问题应该简洁、专业、针对性强。

    输出JSON格式:
    {{
        "questions": [
            "问题1...",
            "问题2...",
            "问题3..."
        ],
        "reasoning": "追问理由..."
    }}
    '''

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        result = json.loads(response.content)
        questions = result.get("questions", [])
    except:
        questions = [
            "图像中是否有明显的炎症或血液干扰?",
            "患者是否有HPV感染史或既往宫颈病变?",
            "样本采集时是否处于月经期?"
        ]

    return {
        **state,
        "clarification_questions": questions,
        "diagnosis_report": "诊断不确定性较高, 需要补充信息...",
        "current_step": "clarification",
        "is_complete": False
    }

# ============ 节点4: 深度检索Agent ============
def deep_retrieval_agent(state: DiagnosisState) -> DiagnosisState:
    '''中等不确定性时进行深度知识检索'''
    from services.rag_service import knowledge_base

    cv_results = state["cv_results"]
    classification = cv_results.get("highest_risk", "")

    # 1. 基于分类检索
    class_knowledge = knowledge_base.retrieve_by_classification(classification, k=5)

    # 2. 基于形态学特征检索 (利用现有cell_analyzer)
    cells = cv_results.get("cells", [])
    if cells:
        morphology = cells[0].get("morphology", {})
        morph_knowledge = knowledge_base.retrieve_by_morphology(morphology, k=3)
    else:
        morph_knowledge = []

    all_knowledge = class_knowledge + morph_knowledge

    return {
        **state,
        "retrieved_knowledge": all_knowledge,
        "current_step": "deep_retrieval"
    }

# ============ 节点5: 直接报告Agent ============
def direct_report_agent(state: DiagnosisState) -> DiagnosisState:
    '''低不确定性时直接生成诊断报告'''
    cv_results = state["cv_results"]

    # 构建诊断上下文
    context = f'''
    细胞检测结果:
    - 总细胞数: {cv_results.get("total_cells", 0)}
    - 异常细胞数: {cv_results.get("abnormal_cells", 0)}
    - 最高风险分类: {cv_results.get("highest_risk", "")}
    - 模型置信度: {cv_results.get("cells", [{}])[0].get("confidence", 0):.3f}
    - 不确定性: {state.get("max_uncertainty", 0):.3f}

    形态学分析:
    - {json.dumps(cv_results.get("cells", [{}])[0].get("morphology", {}), ensure_ascii=False)}
    '''

    report_prompt = ChatPromptTemplate.from_messages([
        ("system", '''你是一位资深病理科医生。请根据AI分析结果生成诊断报告。
        报告要求:
        1. 说明AI检测到的细胞数量和异常比例
        2. 解释风险等级判断依据
        3. 给出明确的后续建议 (复查时间、进一步检查项目)
        4. 说明模型置信度, 建议高不确定性病例人工复核
        5. 语气专业、客观、安抚患者'''),
        ("human", "{context}\n\n请生成诊断报告。")
    ])

    chain = report_prompt | llm
    response = chain.invoke({"context": context})

    # 生成建议
    risk = state["risk_level"]
    recommendations = {
        "低": ["建议1年后常规复查", "保持健康生活方式"],
        "中": ["建议6个月后复查细胞学", "可考虑HPV检测"],
        "高": ["建议3个月内阴道镜检查", "HPV分型检测", "必要时活检"],
        "紧急": ["建议立即阴道镜检查", "尽快预约妇科肿瘤专科", "准备组织病理学检查"]
    }.get(risk, ["建议咨询专业医生"])

    return {
        **state,
        "diagnosis_report": response.content,
        "recommendations": recommendations,
        "current_step": "direct_report",
        "is_complete": True
    }

# ============ 节点6: 可解释性报告Agent (核心亮点) ============
def explainable_report_agent(state: DiagnosisState) -> DiagnosisState:
    '''
    生成带可解释性的综合报告
    结合Grad-CAM + 形态学分析 + LLM解释
    '''
    cv_results = state["cv_results"]

    # 构建可解释性上下文
    explain_context = f'''
    诊断结果: {state.get("diagnosis_report", "")}

    可解释性信息:
    1. 模型关注区域: AI主要关注细胞的核区域和细胞质边界
    2. 关键形态特征:
    '''

    for i, cell in enumerate(cv_results.get("cells", [])[:3], 1):
        morph = cell.get("morphology", {})
        explain_context += f'''
    细胞{i}:
    - 核质比: {morph.get("nuclear_cytoplasmic_ratio", 0):.3f}
    - 圆形度: {morph.get("circularity", 0):.3f}
    - 解读: {morph.get("interpretation", "")}
    '''

    explain_context += f'''
    3. 不确定性分析: MC Dropout标准差={state.get("max_uncertainty", 0):.3f}

    4. 医学依据:
    '''

    for doc in state.get("retrieved_knowledge", [])[:3]:
        explain_context += f'- {doc["content"][:150]}...\n'

    explain_prompt = ChatPromptTemplate.from_messages([
        ("system", '''你是一位病理科AI解释专家。请用通俗语言解释AI的诊断逻辑。
        要求:
        1. 解释模型"看到了什么"导致这个判断
        2. 指出哪些形态特征是关键证据
        3. 说明不确定性来源
        4. 引用医学文献支持诊断
        5. 让患者理解但不过度担忧'''),
        ("human", "{context}\n\n请生成可解释性报告。")
    ])

    chain = explain_prompt | llm
    response = chain.invoke({"context": explain_context})

    # 在原有报告基础上追加可解释性
    full_report = state.get("diagnosis_report", "") + "\n\n---\n\n【AI诊断逻辑解释】\n" + response.content

    return {
        **state,
        "diagnosis_report": full_report,
        "current_step": "explainable_report",
        "is_complete": True
    }

# ============ 构建状态机图 ============
builder = StateGraph(DiagnosisState)

# 添加节点
builder.add_node("uncertainty_assessment", uncertainty_assessment_agent)
builder.add_node("clarification", clarification_agent)
builder.add_node("deep_retrieval", deep_retrieval_agent)
builder.add_node("direct_report", direct_report_agent)
builder.add_node("explainable_report", explainable_report_agent)

# 设置入口
builder.set_entry_point("uncertainty_assessment")

# 添加条件边
builder.add_conditional_edges(
    "uncertainty_assessment",
    decision_router,
    {
        "clarify": "clarification",
        "deep_retrieve": "deep_retrieval",
        "direct_report": "direct_report"
    }
)

# 深度检索后进入报告生成
builder.add_edge("deep_retrieval", "explainable_report")

# 直接报告后进入可解释性增强
builder.add_edge("direct_report", "explainable_report")

# 澄清后结束 (等待用户输入)
builder.add_edge("clarification", END)

# 可解释性报告后结束
builder.add_edge("explainable_report", END)

# 编译
graph = builder.compile()
```

#### 3.3.3 多轮对话管理

```
python
# agents/conversation.py
class ConversationManager:
    '''管理多轮对话状态'''

    def __init__(self):
        self.sessions = {}

    def start_diagnosis(self, session_id: str, image_path: str, cv_results: dict) -> DiagnosisState:
        '''开始新诊断'''
        state = DiagnosisState(
            image_path=image_path,
            user_query=None,
            conversation_history=[],
            cv_results=cv_results,
            max_uncertainty=cv_results.get("max_uncertainty", 0),
            needs_clarification=cv_results.get("needs_clarification", False),
            retrieved_knowledge=None,
            risk_level=None,
            action=None,
            clarification_questions=None,
            gradcam_path=None,
            morphology_analysis=None,
            diagnosis_report=None,
            recommendations=None,
            references=None,
            current_step="uncertainty_assessment",
            is_complete=False
        )

        # 运行Agent图
        result = graph.invoke(state)
        self.sessions[session_id] = result
        return result

    def process_clarification(self, session_id: str, user_input: str) -> DiagnosisState:
        '''处理用户澄清后的重新诊断'''
        state = self.sessions.get(session_id)
        if not state:
            raise ValueError("Session not found")

        # 添加用户回答到历史
        state["conversation_history"].append(HumanMessage(content=user_input))

        # 降低不确定性标记, 重新评估
        state["needs_clarification"] = False
        state["action"] = "deep_retrieve"
        state["current_step"] = "uncertainty_assessment"

        # 重新运行图
        result = graph.invoke(state)
        self.sessions[session_id] = result
        return result

    def followup_chat(self, session_id: str, user_input: str) -> str:
        '''诊断完成后的追问对话'''
        state = self.sessions.get(session_id)
        if not state:
            raise ValueError("Session not found")

        state["conversation_history"].append(HumanMessage(content=user_input))

        followup_prompt = f'''
        用户追问: {user_input}

        当前诊断报告: {state.get("diagnosis_report", "")[:500]}...
        推荐措施: {state.get("recommendations", [])}
        风险等级: {state.get("risk_level", "")}

        请基于已有诊断信息回答。如果问题超出当前范围, 建议咨询专业医生。
        不要给出新的诊断, 只解释现有诊断。
        '''

        response = llm.invoke(state["conversation_history"] + [HumanMessage(content=followup_prompt)])
        state["conversation_history"].append(AIMessage(content=response.content))

        return response.content
```

---

### 3.4 模块四: FastAPI Agent服务层

```
python
# agent_api.py
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uuid
import asyncio
import json
from agents.conversation import ConversationManager
from services.vision_service import analyze_image

app = FastAPI(title="AI Agent宫颈病理诊断系统 v2.0", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

manager = ConversationManager()

@app.post("/api/agent/upload")
async def agent_upload(file: UploadFile = File(...)):
    '''Agent模式: 上传图像, 启动智能诊断流程'''
    session_id = str(uuid.uuid4())

    # 保存文件
    file_path = f"./uploads/{session_id}_{file.filename}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # CV分析 (复用现有模型)
    cv_results = await analyze_image(file_path)

    # 启动Agent诊断流程
    result = manager.start_diagnosis(session_id, file_path, cv_results)

    return {
        "session_id": session_id,
        "status": "complete" if result["is_complete"] else "awaiting_clarification",
        "uncertainty": result.get("max_uncertainty"),
        "risk_level": result.get("risk_level"),
        "action": result.get("action"),
        "report": result.get("diagnosis_report"),
        "questions": result.get("clarification_questions"),
        "recommendations": result.get("recommendations", []),
        "references": result.get("references", [])
    }

@app.post("/api/agent/clarify/{session_id}")
async def agent_clarify(session_id: str, user_input: str):
    '''处理用户澄清回答, 重新诊断'''
    result = manager.process_clarification(session_id, user_input)

    return {
        "session_id": session_id,
        "status": "complete" if result["is_complete"] else "awaiting_clarification",
        "report": result.get("diagnosis_report"),
        "recommendations": result.get("recommendations", [])
    }

@app.websocket("/ws/agent/chat/{session_id}")
async def agent_websocket(websocket: WebSocket, session_id: str):
    '''WebSocket流式对话 (诊断后追问)'''
    await websocket.accept()

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            user_input = data.get("message", "")

            # 诊断后追问
            response = manager.followup_chat(session_id, user_input)

            # 流式返回
            for i in range(0, len(response), 10):
                chunk = response[i:i+10]
                await websocket.send_json({"type": "stream", "content": chunk})
                await asyncio.sleep(0.03)

            await websocket.send_json({"type": "complete"})

    except WebSocketDisconnect:
        print(f"Session {session_id} disconnected")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})

# 保留原Flask服务兼容
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "2.0", "mode": "agent"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### 3.5 模块五: 前端升级 (React + WebSocket)

```javascript
// frontend/src/components/AgentDiagnosis.jsx
import React, { useState, useRef, useEffect } from 'react';

function AgentDiagnosis() {
  const [step, setStep] = useState('upload'); // upload / analyzing / clarification / report / chat
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [questions, setQuestions] = useState([]);
  const [report, setReport] = useState('');
  const [uncertainty, setUncertainty] = useState(0);
  const ws = useRef(null);

  // 上传图像
  const handleUpload = async (file) => {
    setStep('analyzing');
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch('/api/agent/upload', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();

    setSessionId(data.session_id);
    setUncertainty(data.uncertainty);

    if (data.action === 'clarify') {
      // 需要澄清
      setStep('clarification');
      setQuestions(data.questions);
      setMessages([{ role: 'system', content: 'AI检测到较高不确定性, 需要补充信息...' }]);
    } else {
      // 直接生成报告
      setStep('report');
      setReport(data.report);
      setMessages([{ role: 'assistant', content: data.report }]);

      // 连接WebSocket用于后续追问
      connectWebSocket(data.session_id);
    }
  };

  // 提交澄清回答
  const submitClarification = async (answers) => {
    const answerText = answers.join('; ');
    const res = await fetch(`/api/agent/clarify/${sessionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_input: answerText })
    });
    const data = await res.json();

    setStep('report');
    setReport(data.report);
    connectWebSocket(sessionId);
  };

  // WebSocket连接
  const connectWebSocket = (sid) => {
    ws.current = new WebSocket(`ws://localhost:8000/ws/agent/chat/${sid}`);
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'stream') {
        setMessages(prev => {
          const last = prev[prev.length - 1];
          if (last?.role === 'assistant') {
            return [...prev.slice(0, -1), { ...last, content: last.content + data.content }];
          }
          return [...prev, { role: 'assistant', content: data.content }];
        });
      }
    };
  };

  // 发送追问
  const sendQuestion = () => {
    const input = document.getElementById('chat-input').value;
    if (!input.trim() || !ws.current) return;

    setMessages(prev => [...prev, { role: 'user', content: input }]);
    ws.current.send(JSON.stringify({ message: input }));
    document.getElementById('chat-input').value = '';
  };

  return (
    <div className="agent-diagnosis">
      {/* 上传区 */}
      {step === 'upload' && (
        <div className="upload-section">
          <h2>AI Agent辅助宫颈病理诊断</h2>
          <p>上传细胞图像, AI将智能分析并可能追问关键信息</p>
          <input type="file" accept="image/*" onChange={(e) => handleUpload(e.target.files[0])} />
        </div>
      )}

      {/* 分析中 */}
      {step === 'analyzing' && (
        <div className="analyzing">
          <div className="spinner"></div>
          <p>AI正在分析细胞图像并评估不确定性...</p>
        </div>
      )}

      {/* 澄清问题 */}
      {step === 'clarification' && (
        <div className="clarification">
          <h3>需要补充信息</h3>
          <p>AI检测到诊断不确定性较高({(uncertainty*100).toFixed(1)}%), 请回答以下问题:</p>
          {questions.map((q, i) => (
            <div key={i} className="question">
              <label>{i+1}. {q}</label>
              <input type="text" id={`q-${i}`} placeholder="请输入..." />
            </div>
          ))}
          <button onClick={() => {
            const answers = questions.map((_, i) => document.getElementById(`q-${i}`).value);
            submitClarification(answers);
          }}>提交并重新分析</button>
        </div>
      )}

      {/* 诊断报告 */}
      {step === 'report' && (
        <div className="report-section">
          <div className="report">
            <h3>诊断报告</h3>
            <div className="report-content">{report}</div>
          </div>

          {/* 追问对话区 */}
          <div className="chat-section">
            <div className="chat-messages">
              {messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>
                  {msg.content}
                </div>
              ))}
            </div>
            <div className="chat-input">
              <input id="chat-input" placeholder="追问医生..." 
                onKeyPress={(e) => e.key === 'Enter' && sendQuestion()} />
              <button onClick={sendQuestion}>发送</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default AgentDiagnosis;
```

---

## 四、部署方案

### 4.1 Docker化

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 创建目录
RUN mkdir -p uploads knowledge_db

EXPOSE 8000

CMD ["uvicorn", "agent_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4.2 docker-compose

```yaml
version: '3.8'

services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./uploads:/app/uploads
      - ./knowledge_db:/app/knowledge_db
      - ./outputs/models:/app/outputs/models
      - ./runs/detect:/app/runs/detect
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - agent-api
```

---

## 五、改造后项目文件结构

```
cervical-ai-diagnosis/                    # 仓库名
├── README.md                             # 项目演进说明
├── CHANGELOG.md                          # 版本变更记录
│
├── archive/                              # 原项目存档 (只读)
│   └── v1.0-competition/                 # 比赛版本完整代码
│       ├── appWeb/                       # 原Flask应用
│       ├── models/                       # 原模型定义
│       ├── utils/                        # 原工具库
│       ├── scripts/                      # 原训练脚本
│       ├── configs/                      # 原配置
│       └── README.md                     # 原README
│
├── services/                             # 服务层 (新增)
│   ├── __init__.py
│   ├── vision_service.py                 # CV模型服务 (复用+升级)
│   └── rag_service.py                    # RAG知识库 (新增)
│
├── agents/                               # Agent编排层 (新增)
│   ├── __init__.py
│   ├── state.py                          # 状态定义
│   ├── diagnosis_graph.py                # LangGraph状态机
│   ├── conversation.py                   # 会话管理
│   └── prompts/                          # Prompt模板
│       ├── uncertainty_assessment.txt
│       ├── clarification.txt
│       ├── report_generation.txt
│       └── explainable_report.txt
│
├── tools/                                # Agent工具 (新增)
│   ├── __init__.py
│   ├── morphology_tool.py                # 形态学分析Tool
│   └── gradcam_tool.py                   # Grad-CAM可解释性Tool
│
├── agent_api.py                          # FastAPI入口 (新增)
├── main.py                               # 兼容入口
│
├── frontend/                             # 前端 (新增React)
│   ├── src/
│   │   ├── components/
│   │   │   ├── AgentDiagnosis.jsx        # Agent诊断界面
│   │   │   ├── ImageUploader.jsx
│   │   │   └── ReportDisplay.jsx
│   │   └── App.jsx
│   ├── package.json
│   └── Dockerfile
│
├── knowledge_db/                         # 向量数据库 (运行时生成)
├── uploads/                              # 上传文件 (运行时生成)
│
├── requirements.txt                      # 更新依赖
├── Dockerfile
├── docker-compose.yml
└── .gitignore
```

---

## 六、开发计划 (4周)

| 周次 | 任务 | 产出 | 关联Hello-Agents课程 |
|------|------|------|---------------------|
| **第1周** | Git存档 + CV服务封装 + LLM接入 | GitHub仓库 + 基础Agent报告生成 | 第5章: LangChain基础 |
| **第2周** | MC Dropout不确定性驱动 + RAG知识库 | 不确定性评估Agent + 可检索知识库 | 第6章: RAG与记忆 |
| **第3周** | LangGraph状态机 + 多轮对话 | 完整Agent诊断流程 + WebSocket | 第7章: Agent设计 |
| **第4周** | React前端 + Docker部署 + 文档 | 可访问在线Demo + 完整文档 | 第8章: 项目实战 |

---

## 七、简历亮点 (改造后)

> **AI Agent辅助宫颈病理诊断系统**
> 
> **项目背景**: 基于比赛获奖的宫颈细胞AI诊断系统 (YOLOv8+ResNet50+SE/CBAM, 双数据集5,932张图像, AUC>0.95) 进行Agent化架构升级
> 
> **核心创新**:
> 1. **不确定性驱动的智能诊断**: 复用MC Dropout量化模型置信度, 高不确定性时Agent主动追问病史 (HPV感染史、月经周期等), 低不确定性时直接生成报告 -- 解决传统AI"黑盒诊断"问题
> 2. **多模态可解释性报告**: 集成Grad-CAM热力图 + 细胞形态学分析 (核质比、圆形度) + LLM自然语言解释, 让诊断"看得见、说得清"
> 3. **医学知识增强RAG**: 构建TBS分类标准+NCCN指南向量库, 诊断建议自动引用权威文献章节
> 4. **分级Agent工作流**: 不确定性评估Agent->决策路由->{直接报告/主动追问/深度检索}->可解释性报告Agent
> 5. **双服务架构演进**: 保留Flask原服务向后兼容, 新增FastAPI异步Agent服务, WebSocket流式输出
> 
> **技术栈**: LangGraph / OpenAI API / Chroma / FastAPI / WebSocket / React / Docker

---

> **下一步建议**:
> 1. **今天**: 完成Git存档, 创建GitHub仓库
> 2. **本周**: 封装CV服务, 接入LLM生成基础报告
> 3. **下周**: 集成MC Dropout到Agent决策, 构建RAG知识库
