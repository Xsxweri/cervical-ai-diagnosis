"""
路由汇总
"""

from api.routes.agent import router as agent_router
from api.routes.auth import router as auth_router
from api.routes.vision import router as vision_router

__all__ = ["agent_router", "auth_router", "vision_router"]