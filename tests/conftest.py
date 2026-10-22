import torch
import torch._dynamo
import os

# pytest 先加载 conftest，这些 setdefault 在 import agent_api 之前生效
os.environ.setdefault("JWT_SECRET_KEY", "test-only-" + "0" * 48)
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL_ID", "test-model")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:9999")
os.environ.setdefault("CHECKPOINT_BACKEND", "sqlite")