import os
from string import Template
from functools import lru_cache

_PROMPT_DIR = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=None)
def _read(name: str) -> str:
    path = os.path.join(_PROMPT_DIR, f"{name}.md")
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def raw(name: str) -> str:
    """加载原始模板（不做变量替换），用于 ChatPromptTemplate 的 system 段"""
    return _read(name)


def render(name: str, **kwargs) -> str:
    """加载 prompts/<name>.md 并用 ${var} 占位符渲染，缺失变量原样保留"""
    return Template(_read(name)).safe_substitute(**kwargs)