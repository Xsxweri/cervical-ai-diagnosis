"""
细胞形态学分析 Tool - 供 LangGraph Agent 调用
"""

from typing import Dict, Any
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class MorphologyTool:
    """
    LangGraph Tool: 细胞形态学特征分析
    """
    name: str = "cell_morphology_analyzer"
    description: str = """
    分析单个宫颈细胞的形态学特征，包括：
    - 核质比 (nuclear_cytoplasmic_ratio)
    - 细胞面积 (area)
    - 细胞周长 (perimeter)  
    - 圆形度 (circularity)
    
    输入: 裁剪后的细胞图像 (PIL.Image)
    输出: 形态学特征字典 + 医学解读
    """
    
    def __init__(self):
        # 延迟导入，避免循环依赖
        from utils.cell_analyzer import analyze_cell_morphology
        self._analyzer = analyze_cell_morphology
    
    def run(self, cell_image: Image.Image) -> Dict[str, Any]:
        """
        执行形态学分析
        
        Args:
            cell_image: 裁剪后的细胞图像 (RGB)
            
        Returns:
            {
                "nc_ratio": float,      # 核质比
                "area": float,          # 面积
                "perimeter": float,     # 周长
                "circularity": float,   # 圆形度 (0-1)
                "interpretation": str,   # 医学解读
                "risk_indicators": list  # 风险指标列表
            }
        """
        try:
            features = self._analyzer(cell_image)
        except Exception as e:
            logger.warning(f"形态分析失败: {e}")
            features = {}
        
        # 医学解读
        interpretation, risk_indicators = self._medical_interpret(features)
        
        return {
            "nc_ratio": round(features.get("nc_ratio", 0), 4),
            "area": round(features.get("area", 0), 2),
            "perimeter": round(features.get("perimeter", 0), 2),
            "circularity": round(features.get("circularity", 0), 4),
            "interpretation": interpretation,
            "risk_indicators": risk_indicators,
            "raw_features": features
        }
    
    def _medical_interpret(self, features: Dict[str, Any]) -> tuple:
        """生成医学解读和风险指标"""
        interpretation_parts = []
        risk_indicators = []
        
        nc_ratio = features.get("nc_ratio", 0)
        circularity = features.get("circularity", 1)
        area = features.get("area", 0)
        
        # 核质比分析
        if nc_ratio > 0.7:
            interpretation_parts.append("核质比显著增高 (>0.7)，高度提示细胞异型性，见于HSIL或鳞癌")
            risk_indicators.append("high_nc_ratio")
        elif nc_ratio > 0.5:
            interpretation_parts.append("核质比增高 (0.5-0.7)，提示轻-中度异型性，见于LSIL或炎症修复")
            risk_indicators.append("elevated_nc_ratio")
        else:
            interpretation_parts.append("核质比正常 (<0.5)，细胞分化良好")
        
        # 圆形度分析
        if circularity < 0.6:
            interpretation_parts.append("细胞形态显著不规则，核膜皱褶或呈多形性，高度提示恶性")
            risk_indicators.append("irregular_shape")
        elif circularity < 0.75:
            interpretation_parts.append("细胞形态轻度不规则，可见于炎症或低度病变")
            risk_indicators.append("mild_irregularity")
        else:
            interpretation_parts.append("细胞形态规则，圆形度良好")
        
        # 面积分析
        if area > 5000:  # 阈值根据实际数据调整
            interpretation_parts.append("细胞体积增大，可见于化生细胞或癌细胞")
            risk_indicators.append("enlarged_cell")
        
        return "；".join(interpretation_parts), risk_indicators
    
    async def arun(self, cell_image: Image.Image) -> Dict[str, Any]:
        """异步版本"""
        return self.run(cell_image)