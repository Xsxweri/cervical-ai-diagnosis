"""
路由汇总
"""
from api.routes.accounts import router as accounts_router
from api.routes.agent import router as agent_router
from api.routes.auth import router as auth_router
from api.routes.vision import router as vision_router

__all__ = ["accounts_router", "agent_router", "auth_router", "vision_router"]