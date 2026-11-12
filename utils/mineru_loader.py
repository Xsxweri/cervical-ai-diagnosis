"""MinerU JSON 解析：版面感知的文档加载 + 标题感知分块。

MinerU（opendatalab/MinerU）输出结构：
    {"pdf_info": [{"page_idx", "page_size", "para_blocks", "discarded_blocks"}]}

关键设计：
  * 用 para_blocks 而非 preproc_blocks —— 前者已做段落合并、已排序，
    页眉/页脚/页码已被分离到 discarded_blocks。
  * 递归处理嵌套的 blocks：image 块内含 image_body（只有 URL）与
    image_caption（形态学描述）。**图注必须保留** —— 细胞学图谱的
    形态学知识几乎全在图注里，如「核面积是正常中层鳞状细胞核面积的
    2～3 倍」，正文反而很少展开。
  * 按 title 层级维护 section_path，用于标题感知分块与溯源。
"""

from __future__ import annotations

import json
import os
import re

_MINERU_PREFIX = re.compile(r"^MinerU_")
_TS_SUFFIX = re.compile(r"__\d{14}$")

# 有文本内容的块类型。image_body 只有 image_path，无文本，跳过；
# page_number / header / footer 在 discarded_blocks 里，本就不读。
_TEXT_TYPES = {
    "title", "text", "list", "equation",
    "image_caption", "image_footnote",
    "table_caption", "table_footnote", "table",
}

_SEPARATORS = ("。", "\n", "；", "，", " ")

_STANDALONE_TYPES = {
    "image_caption", "image_footnote",
    "table_caption", "table_footnote", "table", "equation",
}

def source_name_from(path: str) -> str:
    """MinerU_<书名-章节>__20261110023552.json → <书名-章节>

    用于 citation 展示与 corpus_manifest 匹配。保留章节名信息，
    去掉工具前缀与时间戳噪声。
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    return _TS_SUFFIX.sub("", _MINERU_PREFIX.sub("", stem))


def _span_text(block: dict) -> str:
    """拼接 block 内所有 span 的文本内容（跳过 image / 无 content 的 span）。"""
    parts = []
    for line in block.get("lines") or []:
        for span in line.get("spans") or []:
            content = span.get("content")
            if content:
                parts.append(content)
    return "".join(parts).strip()


def _iter_blocks(block: dict):
    """递归展开嵌套块，只产出带文本的 (type, level, text)。"""
    children = block.get("blocks")
    if children:
        for child in children:
            yield from _iter_blocks(child)
        return
    btype = block.get("type") or ""
    if btype not in _TEXT_TYPES:
        return
    text = _span_text(block)
    if text:
        yield btype, block.get("level"), text


def _split_long(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """超长 section 的兜底切分，优先在句子边界断开。"""
    if len(text) <= chunk_size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            floor = start + chunk_size // 2
            for sep in _SEPARATORS:
                idx = text.rfind(sep, floor, end)
                if idx != -1:
                    end = idx + 1
                    break
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return out


def load_mineru_documents(path: str, chunk_size: int = 1000,
                          chunk_overlap: int = 200) -> list[dict]:
    """解析单个 MinerU JSON，返回 [{"text", "metadata"}]。

    metadata: source / page / section_path / block_type
      block_type = "section" | "image_caption" | "table_caption" | ...

    分块策略：
      * title 更新 section_path，并作为正文 chunk 的首行（对 BM25 有益）；
      * text/list 合并进当前 section，按 chunk_size 切分；
      * **图注/表注独立成块**，不并入正文；
      * 纯标题（标题后无正文）不单独成块 —— section_path 已在 metadata 里。
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    source = source_name_from(path)
    docs: list[dict] = []
    stack: dict[int, str] = {}
    cur: list[str] = []
    cur_page = 0
    cur_path = source
    cur_title_only = True

    def add(text: str, page: int, btype: str, section_path: str) -> None:
        body = (text or "").strip()
        if not body:
            return
        for piece in _split_long(body, chunk_size, chunk_overlap):
            docs.append({
                "text": piece,
                "metadata": {
                    "source": source,
                    "page": page,
                    "section_path": section_path,
                    "block_type": btype,
                },
            })

    def flush() -> None:
        nonlocal cur, cur_title_only
        if cur and not cur_title_only:
            add("\n".join(cur), cur_page, "section", cur_path)
        cur = []
        cur_title_only = True

    for page in data.get("pdf_info") or []:
        page_idx = int(page.get("page_idx") or 0)
        for block in page.get("para_blocks") or []:
            for btype, level, text in _iter_blocks(block):
                if btype == "title":
                    flush()
                    lvl = int(level or 1)
                    stack[lvl] = text
                    for deeper in [k for k in stack if k > lvl]:
                        del stack[deeper]
                    cur_path = " > ".join(stack[k] for k in sorted(stack))
                    cur_page = page_idx
                    cur = [text]
                    cur_title_only = True
                    continue

                if btype in _STANDALONE_TYPES:
                    flush()                                  # 先把正文段落收尾
                    add(text, page_idx, btype, cur_path)     # 图注独立成块
                    continue

                if not cur:
                    cur_page = page_idx
                cur.append(text)
                cur_title_only = False

    flush()
    return docs


def discover_mineru_json(root: str) -> list[str]:
    """递归发现 root 下的 MinerU_*.json。"""
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("MinerU_") and name.endswith(".json"):
                found.append(os.path.join(dirpath, name))
    return sorted(found)