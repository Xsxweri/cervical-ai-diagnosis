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
            "nuclear_cytoplasmic_ratio": round(features.get("nc_ratio", 0), 4),
            "nc_ratio_valid": bool(features.get("nc_ratio_valid", False)),
            "measurement_reliable": bool(features.get("measurement_reliable", False)),
            "area": round(features.get("area", 0), 2),
            "perimeter": round(features.get("perimeter", 0), 2),
            "circularity": round(features.get("circularity", 0), 4),
            "interpretation": interpretation,
            "risk_indicators": risk_indicators,
            "raw_features": features
        }
    
    # 形态学自动测量的免责声明与可信度门槛
    DISCLAIMER = "（形态学为自动测量结果，仅供辅助参考，不作为诊断依据，请以人工镜检为准）"

    def _medical_interpret(self, features: Dict[str, Any]) -> tuple:
        """
        生成医学解读和风险指标

        设计原则：自动分割结果不可靠时宁可"不下结论"，也不输出带有
        恶性提示倾向的断言。历史上曾把圆形度偏低的正常细胞判为"高度提示恶性"。
        """
        interpretation_parts = []
        risk_indicators = []

        nc_ratio = features.get("nc_ratio", 0.0)
        nc_valid = bool(features.get("nc_ratio_valid", False))
        circularity = features.get("circularity", 0.0)
        area = features.get("area", 0)

        # 核质比分析：仅在测量可信时才给出倾向性描述
        if not nc_valid:
            interpretation_parts.append("核质比无法可靠测量（胞浆自动分割失败），本项不纳入判断")
            risk_indicators.append("nc_ratio_unreliable")
        elif nc_ratio > 0.7:
            interpretation_parts.append(f"核质比偏高（{nc_ratio:.2f}），建议人工复核")
            risk_indicators.append("high_nc_ratio")
        elif nc_ratio > 0.5:
            interpretation_parts.append(
                f"核质比轻度升高（{nc_ratio:.2f}），可见于炎症、修复或低度病变，需结合镜检")
            risk_indicators.append("elevated_nc_ratio")
        else:
            interpretation_parts.append(f"核质比未见升高（{nc_ratio:.2f}）")

        # 圆形度分析：阈值式判据在启发式分割下不可靠，只作提示，不下恶性结论
        if circularity <= 0:
            interpretation_parts.append("圆形度无法测量")
            risk_indicators.append("circularity_unreliable")
        elif circularity < 0.6:
            interpretation_parts.append(
                f"轮廓圆形度偏低（{circularity:.2f}），可能受分割质量影响，建议人工复核")
            risk_indicators.append("low_circularity")
        elif circularity < 0.75:
            interpretation_parts.append(f"轮廓轻度不规则（{circularity:.2f}）")
            risk_indicators.append("mild_irregularity")
        else:
            interpretation_parts.append(f"轮廓较规则（{circularity:.2f}）")

        # 面积分析
        if area > 5000:
            interpretation_parts.append("细胞面积偏大")
            risk_indicators.append("enlarged_cell")

        # 整体不可信时显式标注，供下游阻断
        if not features.get("measurement_reliable", False):
            risk_indicators.append("morphology_unreliable")

        interpretation_parts.append(self.DISCLAIMER)
        return "；".join(interpretation_parts), risk_indicators