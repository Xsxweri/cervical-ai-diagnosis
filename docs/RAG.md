# RAG 架构与决策记录

## 当前状态（v14）
| 指标 | 值 |
|---|---|
| 语料 | 2043 chunks / 14 PDF + 18 MinerU JSON |
| recall | 1.0（41 正样本 + 1 负样本）|
| mrr | 0.9065 |
| neg_pass | 1.0 |
| corpus_gaps | 0 |

## 关键决策
1. **混合检索**：向量 + BM25 + RRF，`_DOC_BONUS` 加性偏移（不用乘性——RRF 动态范围仅 1.57）
2. **RRF 主导排序**：多查询走 `fused_rank`，单查询走 `cosine`（`_apply_doc_bonus` 分流）
3. **MinerU 版面感知解析**：13+5 个 JSON，`image_caption` 独立成块（327 条图注是形态学知识来源）
4. **查询改写三层**：`normalize_query` → `rule_rewrite`（alias）+ `detect_intents`（意图）→ 多查询 RRF
5. **CEBM 证据分级**：`doc_type → I/II/III`，用于 `evidence_sufficiency` 加权
6. **中文别名**：`宫颈↔子宫颈`、`TCT→细胞学检查`、`HPV52→非HPV16/18亚型`

## 已知局限
- 单领域语料下绝对余弦不可分（N01 Python 0.577 > 多条医学查询）
- R05/R08/R09/LB10/H02 的 rr 在 0.25-0.5，需交叉编码器重排
- `expect_files_any` 易造成误判（已三次踩坑），clinical 类用例应留空

## 换语料流程

1. PDF / MinerU JSON 放进 `Cervical_cytology/`
2. **更新 `corpus_manifest.yaml`**（`file` 值必须与 `source_name_from()` 输出逐字一致：
   MinerU JSON 去掉 `MinerU_` 前缀与 `__时间戳` 后缀，**不含 `.pdf`**）
3. **`python scripts/check_manifest.py`** → 必须输出 ✅，否则会 doc_type 静默降级
4. 重建：`python -c "import asyncio; from Agents.knowledge import init_kb; asyncio.run(init_kb())"`
5. 确认日志 `[RAG] doc_type 已标注 N/N`（分母=分子）
6. `python scripts/_diag_after_rebuild.py` 检查图注数与页眉噪声
7. `python scripts/eval_retrieval.py --json outputs/rag_eval_baseline_vN.json`

>改切分/清洗逻辑时需手动 `CORPUS_VERSION += 1`，指纹不含代码逻辑。

## 踩坑记录

| 坑 | 症状 | 修法 |
|---|---|---|
| `retrieve_n1`（数字1）typo | `search_guidelines` 永远返回「知识库不可用」，被 except 吞掉 | 加测试断言 KB 方法名存在 |
| manifest 缺失但库非空 | 语料重复灌入，chunks 翻倍 | `rebuild_reason` 判 `actual > 0` 即重建 |
| 乘性 doc 权重 | RRF 动态范围仅 1.57，权重 2.0 后排序≈「先按文件类型」 | 改加性 `_DOC_BONUS` + 归一化 |
| RRF 被 cosine 覆盖 | 多查询融合白做，目标 chunk rrf 最高却排第 4 | `_apply_doc_bonus` 用 `fused_rank` |
| alias target 不在 canonical | 静默丢弃，`rule_rewrite` 少返回词 | 加载器 `logger.warning` + 检查 `len(_ALIAS)` |
| `expect_files_any` 过严 | 正确答案在 top-5 却被文件过滤排除 | clinical 类用例留空 |
| `_diag` 脚本打印截断内容 | 误判「同页分块混淆」 | 打印完整 chunk 或 `len(content)` |

## 未做（含重启条件）

| 项 | 为什么不做 | 重启条件 |
|---|---|---|
| Cross-Encoder 重排 | v14 已 recall=1.0，5 条 rr<1 的用例答案都在 top-5；成本 600MB + CPU 1-3s/次 | 真实报告中「[E1] 答非所问」比例 > 20% |
| 父子分块 | MinerU 已按 section + 图注独立切分，语义单元正确 | 出现「召回了片段但缺上下文」的反馈 |
| 索引侧术语归一 | 破坏溯源（引用文本与实际存储不一致） | 不做（除非接受引用失真） |
| GraphRAG | 任务全是单跳事实查询，无多跳需求 | 出现「A 药与 B 药相互作用」类查询 |
| 扩评测集到 50+ | 当前 43 条足够守回归 | 要做 reranker / 换嵌入模型 / 调分块策略前 |

## 配置与数据文件索引

| 文件 | 位置 | 作用 | 进 Git |
|---|---|---|---|
| `rag_terms.yaml` | `configs/` | 术语白名单、口语 alias、意图规则 | ✅ |
| `corpus_manifest.yaml` | `configs/` | 每个语料文件的 doc_type/role/lang/年份 | ✅ |
| `queries.yaml` | `tests/rag_eval/` | 43 条检索评测用例 | ✅ |
| `.env.example` | 根目录 | 运行时开关（`RAG_*`） | ✅ |
| `kb_manifest.json` | `knowledge_db/` | 语料指纹 + chunk 数（自动生成） | ❌ |
| `Cervical_cytology/*` | 语料目录 | PDF / MinerU JSON 原文 | ❌ |