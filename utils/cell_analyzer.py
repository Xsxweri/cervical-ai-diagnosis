import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class CellAnalyzer:
    def __init__(self):
        pass

    def analyze_nucleus_cytoplasm(self, image_path, nucleus_dat_path=None, cytoplasm_dat_path=None):
        """分析细胞核和细胞质特征"""
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取.dat文件（如果提供）
        nucleus_contours = self._parse_dat_file(nucleus_dat_path) if nucleus_dat_path else None
        cytoplasm_contours = self._parse_dat_file(cytoplasm_dat_path) if cytoplasm_dat_path else None

        # 如果没有提供.dat文件，尝试自动检测
        if nucleus_contours is None:
            nucleus_contours = self._detect_nucleus(image)
        if cytoplasm_contours is None:
            cytoplasm_contours = self._detect_cytoplasm(image, nucleus_contours)

        # 提取特征
        features = {}

        # 细胞核特征
        if nucleus_contours is not None:
            features['nuclear_area'] = self._calculate_area(nucleus_contours)
            features['nuclear_perimeter'] = self._calculate_perimeter(nucleus_contours)
            features['nuclear_circularity'] = self._calculate_circularity(nucleus_contours)
            features['nuclear_intensity'] = self._calculate_intensity(image, nucleus_contours)

        # 细胞质特征
        if cytoplasm_contours is not None:
            features['cytoplasmic_area'] = self._calculate_area(cytoplasm_contours)
            if 'nuclear_area' in features and features['nuclear_area'] > 0:
                features['nuclear_cytoplasmic_ratio'] = features['nuclear_area'] / features['cytoplasmic_area']

        return features, nucleus_contours, cytoplasm_contours

    def _parse_dat_file(self, dat_path):
        """解析.dat文件，提取轮廓坐标"""
        try:
            with open(dat_path, 'r') as f:
                points = []
                for line in f:
                    if ',' in line:
                        x, y = map(float, line.strip().split(','))
                        points.append([x, y])
                return np.array(points, dtype=np.int32)
        except:
            return None

    def _detect_nucleus(self, image):
        """自动检测细胞核"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 使用Otsu's二值化
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 选择最大的轮廓作为细胞核
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            return largest_contour
        return None

    def _detect_cytoplasm(self, image, nucleus_contour):
        """自动检测细胞质"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # 根据细胞核位置估计细胞质区域
        if nucleus_contour is not None:
            # 获取细胞核边界框
            x, y, w, h = cv2.boundingRect(nucleus_contour)

            # 扩大边界框以包含细胞质
            expansion = 2
            x = max(0, x - w // expansion)
            y = max(0, y - h // expansion)
            w = min(image.shape[1] - x, w + 2 * w // expansion)
            h = min(image.shape[0] - y, h + 2 * h // expansion)

            # 提取感兴趣区域
            roi = image[y:y + h, x:x + w]
            roi_hsv = hsv[y:y + h, x:x + w]

            # 使用颜色阈值检测细胞质
            lower_bound = np.array([0, 0, 100])
            upper_bound = np.array([180, 100, 255])
            mask = cv2.inRange(roi_hsv, lower_bound, upper_bound)

            # 形态学操作
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 选择最大的轮廓作为细胞质
                largest_contour = max(contours, key=cv2.contourArea)

                # 调整坐标到原始图像
                largest_contour += np.array([x, y])
                return largest_contour

        return None

    def _calculate_area(self, contour):
        """计算轮廓面积"""
        return cv2.contourArea(contour)

    def _calculate_perimeter(self, contour):
        """计算轮廓周长"""
        return cv2.arcLength(contour, True)

    def _calculate_circularity(self, contour):
        """计算圆形度 (4π*面积/周长²)"""
        area = self._calculate_area(contour)
        perimeter = self._calculate_perimeter(contour)
        if perimeter > 0:
            return (4 * np.pi * area) / (perimeter ** 2)
        return 0

    def _calculate_intensity(self, image, contour):
        """计算轮廓内像素的平均强度"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_intensity = cv2.mean(image, mask=mask)[:3]
        return np.mean(mean_intensity)

    def visualize_cell_analysis(self, image_path, nucleus_contours, cytoplasm_contours, features):
        """可视化细胞分析结果"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 绘制轮廓
        if nucleus_contours is not None:
            cv2.drawContours(image, [nucleus_contours], -1, (255, 0, 0), 2)
        if cytoplasm_contours is not None:
            cv2.drawContours(image, [cytoplasm_contours], -1, (0, 255, 0), 2)

        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 显示图像和轮廓
        ax1.imshow(image)
        ax1.set_title('细胞核（蓝色）和细胞质（绿色）')
        ax1.axis('off')

        # 显示特征
        feature_names = list(features.keys())
        feature_values = [features[name] for name in feature_names]

        ax2.barh(feature_names, feature_values)
        ax2.set_title('Cell Features')
        ax2.set_xlabel('Value')

        plt.tight_layout()
        return fig


def analyze_cell_morphology(image):
    """
    独立的细胞形态学分析函数（供 vision_service 调用）
    Args:
        image: OpenCV格式的图像 (numpy array)

    Returns:
        dict: 形态学特征字典
    """
    analyzer = CellAnalyzer()

    # 如果输入是PIL Image，转换为OpenCV格式
    if hasattr(image, 'convert'):
        from PIL import Image
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    try:
        # 自动检测细胞核
        nucleus_contour = analyzer._detect_nucleus(image)

        # 自动检测细胞质
        cytoplasm_contour = analyzer._detect_cytoplasm(image, nucleus_contour)

        # 提取特征
        features = {}

        if nucleus_contour is not None:
            area = analyzer._calculate_area(nucleus_contour)
            perimeter = analyzer._calculate_perimeter(nucleus_contour)
            circularity = analyzer._calculate_circularity(nucleus_contour)

            features['area'] = area
            features['perimeter'] = perimeter
            features['circularity'] = circularity

            features['nc_ratio'] = 0.0
            features['nc_ratio_valid'] = False

            # 计算核质比
            if cytoplasm_contour is not None:
                cytoplasmic_area = analyzer._calculate_area(cytoplasm_contour)
                if 0 < area <= cytoplasmic_area:
                    features['nc_ratio'] = area / cytoplasmic_area
                    features['nc_ratio_valid'] = True
        else:
            features['area'] = 0
            features['perimeter'] = 0
            features['circularity'] = 0
            features['nc_ratio'] = 0
            features['nc_ratio_valid'] = False

        # 整体可信度：核分割成功且核质比可计算时才算可靠
        features['measurement_reliable'] = bool(
            nucleus_contour is not None and features.get('nc_ratio_valid')
        )

        return features

    except Exception as e:
        # 如果分析失败，返回默认值
        return {
            'area': 0,
            'perimeter': 0,
            'circularity': 0,
            'nc_ratio': 0,
            'nc_ratio_valid': False,
            'measurement_reliable': False,
        }