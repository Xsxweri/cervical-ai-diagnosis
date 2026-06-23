from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import uuid
import hashlib
import json
from datetime import timedelta
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

# 导入模型和相关库
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
# 导入YOLO检测模型
from models.yolo_model import YOLOv8CellDetector

# 初始化设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_model_file(model_name):
    """在多个可能的位置查找模型文件"""
    # 定义可能的模型文件路径
    possible_paths = [
        model_name,
        os.path.join('outputs', 'models', model_name),
        os.path.join('runs', 'detect', 'cervical_cell_exp3', 'weights', model_name),
        os.path.join(os.path.dirname(__file__), 'outputs', 'models', model_name),
        os.path.join(os.path.dirname(__file__), 'runs', 'detect', 'cervical_cell_exp3', 'weights', model_name),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def load_models():
    """加载训练好的模型"""
    global classification_model, detection_model, classification_mode, class_names

    # 设置分类模式 ('multiclass' 或 'binary')
    classification_mode = 'binary'  # 可以修改为 'binary' 来使用二分类模型

    # 加载分类模型
    try:
        if classification_mode == 'binary':
            # 二分类模型配置
            num_classes = 2
            model_file = find_model_file('best_resnet50_binary_model.pth')
            # 二分类的类别名称
            class_names = ['normal', 'abnormal']
        else:
            # 多分类模型配置
            num_classes = 7
            model_file = find_model_file('best_resnet50_multiclass_model.pth')
            # 多分类的类别名称
            class_names = [
                'carcinoma_in_situ', 'light_dysplastic', 'moderate_dysplastic',
                'normal_columnar', 'normal_intermediate', 'normal_superficial', 'severe_dysplastic'
            ]

        if model_file is None:
            raise FileNotFoundError("找不到分类模型文件")

        # 延迟导入模型类以避免循环导入
        from models.cer_cancer_model import CervicalCancerModel

        # 加载模型权重以检查是否包含注意力机制
        model_state_dict = torch.load(model_file, map_location=device)

        # 检查state_dict中是否包含注意力机制相关的键
        use_attention = any('attention' in key for key in model_state_dict.keys())
        attention_type = None
        if use_attention:
            # 根据键名判断注意力机制类型
            if any('se' in key.lower() for key in model_state_dict.keys() if 'attention' in key):
                attention_type = 'se'
            elif any('cbam' in key.lower() for key in model_state_dict.keys() if 'attention' in key):
                attention_type = 'cbam'
            else:
                # 默认使用SE注意力机制
                attention_type = 'se'

        # 初始化模型，根据权重文件是否包含注意力机制来决定是否启用
        classification_model = CervicalCancerModel(
            num_classes=num_classes,
            pretrained=False,
            model_type='resnet50',
            use_attention=use_attention,
            attention_type=attention_type if use_attention else None
        )

        # 加载模型权重
        classification_model.load_state_dict(model_state_dict)
        print(f"成功加载分类模型: {model_file} (模式: {classification_mode})")
        print(f"模型使用注意力机制: {use_attention}, 类型: {attention_type}")

        classification_model.to(device)
        classification_model.eval()

    except Exception as e:
        print(f"加载分类模型时出错: {e}")
        classification_model = None

    # 加载检测模型
    try:
        detection_model_file = find_model_file('best.pt')
        if detection_model_file is None:
            # 如果找不到训练好的模型，尝试使用默认的YOLO模型
            detection_model_file = 'yolov8n.pt'
            print("未找到训练好的检测模型，将使用默认的YOLOv8n模型")

        detection_model = YOLOv8CellDetector(
            model_path=detection_model_file
        )
        print(f"成功加载检测模型: {detection_model_file}")

    except Exception as e:
        print(f"加载检测模型时出错: {e}")
        detection_model = None


# 加载模型
load_models()


def preprocess_image(image_path):
    """预处理图像用于分类"""
    # 定义图像预处理流程
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 应用预处理
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)


def preprocess_image_for_detection(image_path):
    """预处理图像用于检测"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def classify_cells(image_path):
    """
    使用训练好的分类模型进行预测
    """
    if classification_model is None:
        return {"error": "分类模型未加载"}

    try:
        # 预处理图像
        image_tensor = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)

        # 模型预测
        with torch.no_grad():
            outputs = classification_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # 获取各类别的概率
        probs = probabilities.cpu().numpy()[0]
        results = {}
        for i, class_name in enumerate(class_names):
            results[class_name] = round(float(probs[i]), 4)

        # 诊断建议
        predicted_class = class_names[predicted.item()]
        confidence_value = confidence.item()

        # 根据预测结果生成诊断建议
        if len(class_names) == 2:  # 二分类模式
            if predicted_class == 'abnormal':
                diagnosis = "发现异常细胞"
                recommendation = "建议进一步检查"
            else:
                diagnosis = "未见异常细胞"
                recommendation = "建议定期复查"
        else:  # 多分类模式
            if predicted_class in ['severe_dysplastic', 'carcinoma_in_situ']:
                diagnosis = "高度鳞状上皮内病变（HSIL）"
                recommendation = "建议立即进一步检查"
            elif predicted_class in ['moderate_dysplastic', 'light_dysplastic']:
                diagnosis = "低度鳞状上皮内病变（LSIL）"
                recommendation = "建议定期复查并进一步检查"
            elif predicted_class in ['normal_columnar', 'normal_intermediate', 'normal_superficial']:
                diagnosis = "未见上皮内病变细胞或恶性细胞（NILM）"
                recommendation = "建议定期复查"
            else:
                diagnosis = "非典型鳞状细胞（ASC-US）"
                recommendation = "建议3-6个月后复查"

        return {
            "probabilities": results,
            "predicted_class": predicted_class,
            "confidence": round(confidence_value, 4),
            "diagnosis": diagnosis,
            "recommendation": recommendation
        }
    except Exception as e:
        raise Exception(f"分类模型预测失败: {str(e)}")


def detect_cells(image_path):
    """
    使用YOLO检测模型检测图像中的细胞
    """
    if detection_model is None:
        return {"error": "检测模型未加载"}

    try:
        # 预处理图像
        image_rgb = preprocess_image_for_detection(image_path)

        # 检测细胞
        results = detection_model.detect_cells(
            image=image_rgb,
            conf_threshold=0.25,
            iou_threshold=0.45
        )

        # 分析检测结果
        total_cells = len(results['class_ids'])

        # 统计正常和异常细胞（这里假设类别0是正常细胞，类别1是异常细胞）
        normal_cells = sum(1 for class_id in results['class_ids'] if int(class_id) == 0)
        abnormal_cells = total_cells - normal_cells

        return {
            "total_cells": total_cells,
            "normal_cells": normal_cells,
            "abnormal_cells": abnormal_cells,
            "detection_results": results
        }
    except Exception as e:
        raise Exception(f"检测模型预测失败: {str(e)}")


def model_predict(image_path):
    """
    使用训练好的检测和分类模型进行预测
    """
    try:
        # 使用检测模型检测细胞
        detection_results = detect_cells(image_path)

        # 使用分类模型分类细胞
        classification_results = classify_cells(image_path)

        return {
            "detection": detection_results,
            "classification": classification_results
        }
    except Exception as e:
        raise Exception(f"模型预测失败: {str(e)}")


# 初始化Flask应用
app = Flask(__name__)
app.secret_key = 'cervical-cell-ai-system-secret-key-2023'  # 请在生产环境中更改这个密钥
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)  # 会话有效期1小时
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 最大文件大小30MB
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'tif', 'tiff'}

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 模拟数据库中的医生数据（在实际应用中应该使用真实数据库）
doctors = [
    {
        "id": 1,
        "username": "dr_zhang",
        "password_hash": generate_password_hash("password123"),  # 实际密码: password123
        "name": "张医生",
        "department": "妇科",
        "hospital": "北京协和医院"
    },
    {
        "id": 2,
        "username": "dr_wang",
        "password_hash": generate_password_hash("securepwd456"),  # 实际密码: securepwd456
        "name": "王医生",
        "department": "病理科",
        "hospital": "上海瑞金医院"
    },
    {
        "id": 3,
        "username": "dr_li",
        "password_hash": generate_password_hash("secret789"),  # 实际密码: secret789
        "name": "李医生",
        "department": "检验科",
        "hospital": "广州中山医院"
    }
]


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    # 如果用户已登录，重定向到仪表板
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    # 否则显示登录页面
    return render_template('tryweb.html')


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # 查找用户
    doctor = next((d for d in doctors if d['username'] == username), None)

    if doctor and check_password_hash(doctor['password_hash'], password):
        # 登录成功
        session.permanent = True
        session['user_id'] = doctor['id']
        session['username'] = doctor['username']
        session['doctor_name'] = doctor['name']
        session['department'] = doctor['department']
        session['hospital'] = doctor['hospital']

        return jsonify({
            'success': True,
            'message': '登录成功',
            'doctor': {
                'name': doctor['name'],
                'department': doctor['department'],
                'hospital': doctor['hospital']
            }
        })
    else:
        # 登录失败
        return jsonify({
            'success': False,
            'message': '用户名或密码错误'
        }), 401


@app.route('/dashboard')
def dashboard():
    # 检查用户是否已登录
    if 'user_id' not in session:
        return redirect(url_for('index'))

    # 在实际应用中，这里可以获取更多用户数据
    return render_template('tryweb.html')


@app.route('/logout')
def logout():
    session.clear()
    return jsonify({'success': True, 'message': '已退出登录'})


@app.route('/upload', methods=['POST'])
def upload_file():
    # 检查用户是否已登录
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'}), 401

    # 检查是否有文件上传
    if 'files' not in request.files:
        return jsonify({'success': False, 'message': '没有选择文件'}), 400

    files = request.files.getlist('files')

    # 检查文件数量
    if len(files) > 10:
        return jsonify({'success': False, 'message': '最多只能上传10个文件'}), 400

    results = []

    for file in files:
        if file.filename == '':
            continue

        if file and allowed_file(file.filename):
            # 生成安全的文件名
            filename = secure_filename(file.filename)
            # 添加唯一标识符避免重名
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

            # 保存文件
            file.save(filepath)

            # 调用模型进行预测
            try:
                prediction = model_predict(filepath)
                results.append({
                    'filename': filename,
                    'success': True,
                    'result': prediction
                })
            except Exception as e:
                results.append({
                    'filename': filename,
                    'success': False,
                    'message': f'分析失败: {str(e)}'
                })
        else:
            results.append({
                'filename': file.filename,
                'success': False,
                'message': '不支持的文件格式'
            })

    return jsonify({
        'success': True,
        'message': f'成功处理 {len([r for r in results if r["success"]])} 个文件',
        'results': results
    })


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5001)