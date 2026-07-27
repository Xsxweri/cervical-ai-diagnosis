from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
# import uuid
import json
from datetime import timedelta
import sys
import requests

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)


# 初始化Flask应用
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())
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
        filename = secure_filename(file.filename)
        if file and allowed_file(file.filename):
            files_binary = {'file': (file.filename, file.read(), file.content_type)}
            resp = requests.post("http://localhost:8000/api/agent/upload", files=files_binary, timeout=120)
            prediction = resp.json()
            results.append({
                'filename': filename,
                'success': prediction.get('status') == 'complete',
                'result': prediction
            })
        else:
            results.append({
            'filename': file.filename,
            'success': False,
            'message': '不支持的文件格式'
        })
    return jsonify({
        'success': True,
        'message': f'成功处理 {len(results)} 个文件',
        'results': results
    })


if __name__ == '__main__':
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, host='127.0.0.1', port=5001)