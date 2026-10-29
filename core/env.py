from pathlib import Path

from dotenv import load_dotenv

# 显式定位项目根目录的 .env，避免当前工作目录不同导致找不到文件
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")