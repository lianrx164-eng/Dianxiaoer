# Ch04 混合检索 + 重排 + 评估体系 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ch03 的 dense 单路语义检索升级为「混合检索(Milvus 原生 BM25 + dense,hybrid_search RRF)+ 百炼 qwen3-reranker 精排 + 生成质量控制(引用/拒答/自评落池)」,并建一套四策略分桶评估体系,前端聊天页引用可点。

**Architecture:** 沿用 Milvus Lite(milvus-lite 3.2.2 已支持原生 BM25 + hybrid_search,不引入 Docker);保持单一 agent tool-calling 入口,`query_faq` 内部升级为完整 RAG 检索管线(query 理解 → 元数据过滤 → 混合召回 → 重排 → 两道生成前证据闸 → 带编号证据回传);引用元数据经 SSE 事件下发前端。纯 async 编排,不引入 LangGraph/Langfuse。

**Tech Stack:** Milvus Lite 3.2.2 (pymilvus: `Function`/`FunctionType.BM25`/`AnnSearchRequest`/`RRFRanker`/jieba 分词)、阿里云百炼 rerank(`qwen3-rerank`)、阿里云百炼 `text-embedding-v4` 嵌入(沿用 ch03)、FastAPI/SQLAlchemy/项目聊天模型(沿用)。

## Global Constraints

- **固定选型(定死,不自换)**:Milvus 2.5+ 原生 BM25 全文检索 + `hybrid_search` RRF;重排百炼 `qwen3-reranker`(2026-11-14 用户拍板);嵌入百炼 `text-embedding-v4`(ch03 实际交付)。实现中发现矛盾/走不通 → **停下问用户,只在选型内换法**(见 Task 1 红线)。
- **地址与密钥只从配置来**:嵌入、重排、聊天各一组 `.env`,代码里不写死任何上游。
- **不引入 LangGraph / Langfuse**,沿用 ch03 纯 async 函数编排。
- **结构化输出用扁平字段 + `method="function_calling"`**(禁 `list[对象]` 与默认 json_schema):上游对嵌套对象数组会 502,DeepSeek 对 json_schema response_format 会 400(ch03 教训)。
- **Milvus `id` = MySQL `knowledge_chunks.id`**(auto_id=False),重跑 upsert 幂等。
- **BM25 `text` 字段入库内容 = `category + questions + answer` 拼接**(与 dense 嵌入源同,两路召回同批 chunk)。
- **中文分词** analyzer_params = `{"tokenizer": "jieba"}`(milvus-lite 不识别 `chinese` 别名;需 `uv add jieba`)。
- **单一 agent 入口不变**,事务型工具(order/product/logistics/ticket)零改动。
- **凭据只在 `.env`**,不写进代码/文档/留痕。
- **确定性代码走 TDD**(红→绿→提交);纯 Prompt/数据任务用标注样例/评估集验证代替 TDD;前端走 Vibe Coding(不套 TDD/review)。
- Milvus 单测用**真实 Lite 临时文件**(每个测试唯一 `*.db` + 唯一集合名 + teardown drop),`milvus_client` 各函数加 `collection` 参数以隔离;Lite 是进程级独占锁,测试文件互不共享。

---

## 文件布局

**改:**
- `pyproject.toml` — 加 `jieba`(Lite BM25 分词依赖)
- `app/config.py` — milvus_uri 改 http、加 rerank/召回参数
- `app/kb/milvus_client.py` — 新 schema(dense+text/analyzer+sparse/BM25 Function+标量)、hybrid/dense/bm25 search、collection 参数
- `app/kb/dualwrite.py` — vectorize 写入 text+section_path+content_type+category
- `data/kb/*.md` — 知识库内容(faq/policy/manual/spec 四类;按评估集补齐缺失条款;spec 含易混型号族,使四策略评估可区分)
- `app/core/retrieval.py` — 重写:strategy 分支 + 首尾组装
- `app/core/prompts.py` — RAG 生成/自评/改写/faithfulness prompt + 负面知识
- `app/tools/business.py` — query_faq 升级为 RAG 管线,返回带编号证据 + citations + sufficient
- `app/core/agent.py` — 消费 query_faq citations、sufficient=false 落池、citations 事件
- `app/api/chat.py` — citations SSE 事件
- `app/static/index.html` — 可点引用 + 每段回答满意度反馈 👍/👎(Vibe Coding)
- `tests/conftest.py` — 加 ch04-ddl、low_confidence_questions 表
- `Makefile` — eval-rag / smoke-rag / eval-check;kb-reset 沿用现有 Lite 清理

**新:**
- `app/core/rerank.py` — 百炼 rerank 客户端
- `app/core/query_understanding.py` — 改写 + 同义词扩展
- `app/db/repository.py`(加 `insert_low_confidence`)
- `scripts/smoke_rerank.py`、`scripts/smoke_milvus_bm25.py` — 冒烟
- `scripts/eval_ch04.py` — 四策略分桶评估(检索 / 证据覆盖度 / 生成)+ 自动生成报告(.txt + .html)
- `app/api/rageval.py` — RAG 评估页只读 API(端出 `data/ch04/reports/rag_eval.json`,不重算)
- `app/static/rageval.html` — RAG 评估页(四策略对照的报告在后台里看,也在后台里重跑)
- `tests/data/eval_ch04.jsonl` — 评估集(300 题五桶,带 expect_section / expect_sections_all + expect_points)
- `tests/kb/test_milvus_hybrid.py`、`tests/core/test_rerank.py`、`tests/core/test_retrieval_ch04.py`、`tests/tools/test_query_faq_rag.py`、`tests/core/test_agent_citations.py`、`tests/db/test_low_confidence.py`

---

## Task 1: Fail-fast 冒烟(红线,GO/NO-GO)

**目的:** 动手改代码前先用真实服务验证两条新链路——不通即停、只在选型内换法、把卡点写进 dev-notes 顶部。非 TDD。

> **2026-11-14 冒烟实测记录:** 百炼 rerank 已通。用 `.env` 的 `EMBED_API_KEY` 调
> `POST https://maas.qianwenaiapi.com/api/v1/services/rerank/text-rerank/text-rerank`,
> 模型 `qwen3-rerank`,原生请求体 `{model, input:{query,documents}, parameters:{top_n,return_documents}}`,
> 返回 `output.results[]`(`index` + 0~1 的 `relevance_score`);「退货运费」正确排第 1(0.833)。
> 失败过的路径:`{EMBED_BASE_URL}/reranks` 与 `/rerank` 均 404,不用 OpenAI 兼容形状。
> **Milvus 已实测:** milvus-lite 3.2.2 支持 BM25 Function + hybrid_search;用 `analyzer_params={"tokenizer":"jieba"}`(Lite 不识别 `chinese`),装 `jieba` 后纯 BM25 命中型号词、hybrid RRF 返回结果。因此**不走 Docker/Standalone**,直接用本地 Lite 文件。

**Files:**
- Modify: `pyproject.toml`(加 `jieba`)
- Modify: `Makefile`
- Create: `scripts/smoke_rerank.py`
- Create: `scripts/smoke_milvus_bm25.py`

**Interfaces:**
- Produces: 可用的本地 Milvus Lite(URI = `settings.milvus_uri`,临时文件冒烟)、百炼 rerank 可用。

- [ ] **Step 1: 加 jieba 依赖**

```bash
uv add jieba
```

`jieba` 是 milvus-lite 的 BM25 jieba 分词器运行时依赖,不加会在 `create_collection` 时报 `JiebaAnalyzer requires the 'jieba' package`.

- [ ] **Step 2: Makefile 加 smoke 目标**

```makefile
smoke-rag:
        PYTHONPATH=. uv run python scripts/smoke_milvus_bm25.py
        PYTHONPATH=. uv run python scripts/smoke_rerank.py
```

`.PHONY` 行追加 `smoke-rag eval-rag`。不引入 `milvus-up/down`(Lite 是本地文件,没有服务要起)。

- [ ] **Step 3: 加重排那一组配置**

百炼 rerank 是 DashScope 原生形状(请求带 `input.query` + `input.documents`,回
`output.results` 里的 `index` 与 `relevance_score`),不是 OpenAI 协议,所以单独一组配置、请求也手写。

Edit `.env.example`:

```bash
# 重排 百炼 qwen3-rerank。URL 有内置默认值(百炼原生 rerank 网关)
# RERANK_API_KEY=sk-xxx      # 可选;不填则复用 EMBED_API_KEY
# RERANK_URL=https://maas.qianwenaiapi.com/api/v1/services/rerank/text-rerank/text-rerank
```

Edit `app/config.py`:

```python
    rerank_url: str = "https://maas.qianwenaiapi.com/api/v1/services/rerank/text-rerank/text-rerank"
    rerank_api_key: str | None = None        # 缺省复用 embed_api_key(同一百炼账号)
    rerank_model: str = "qwen3-rerank"       # 百炼模型真名
```


- [ ] **Step 4: 写 rerank 冒烟脚本**

`scripts/smoke_rerank.py`:
```python
"""直连百炼原生 rerank 冒烟。不通即红线停。"""
import httpx

from app.config import settings


def main() -> None:
    payload = {
        "model": settings.rerank_model,
        "input": {
            "query": "退货运费谁承担",
            "documents": [
                "满99元包邮,不满收取10元运费。",
                "七天无理由退货,非质量问题退货运费由买家承担。",
                "智能猫砂盆 Pro 型号支持自动清理。",
            ],
        },
        "parameters": {"top_n": 3, "return_documents": True},
    }
    key = settings.rerank_api_key or settings.embed_api_key
    r = httpx.post(settings.rerank_url, json=payload,
                   headers={"Authorization": f"Bearer {key}"}, timeout=60)
    r.raise_for_status()
    data = r.json()
    print("原始返回:", data)
    results = data["output"]["results"]
    assert results, "rerank 返回空"
    top = max(results, key=lambda x: x["relevance_score"])
    print(f"最相关 index={top['index']} score={top['relevance_score']:.4f}")
    assert top["index"] == 1, "退货运费问题应命中第 2 条(买家承担)"
    print("GO: rerank 链路通")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 写 Milvus BM25 冒烟脚本**

`scripts/smoke_milvus_bm25.py`:
```python
"""Milvus Lite 原生 BM25 全文检索 + hybrid_search 冒烟。不通即红线停。"""
import tempfile
from pathlib import Path

from pymilvus import (
    AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker,
)

URI = str(Path(tempfile.mkdtemp()) / "smoke_bm25.db")
COLL = "smoke_bm25"


def main() -> None:
    client = MilvusClient(uri=URI)
    if client.has_collection(COLL):
        client.drop_collection(COLL)

    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=4)
    schema.add_field("text", DataType.VARCHAR, max_length=2048,
                     enable_analyzer=True, analyzer_params={"tokenizer": "jieba"})
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(
        name="text_bm25", input_field_names=["text"],
        output_field_names=["sparse"], function_type=FunctionType.BM25,
    ))
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client.create_collection(COLL, schema=schema, index_params=index_params)

    client.insert(COLL, [
        {"id": 1, "dense": [0.1, 0.2, 0.3, 0.4], "text": "满99元包邮,不满收取10元运费"},
        {"id": 2, "dense": [0.2, 0.1, 0.4, 0.3], "text": "智能猫砂盆 Pro 型号支持自动清理和除臭"},
    ])
    client.load_collection(COLL)

    # 纯 BM25:问型号应命中第 2 条
    res = client.search(COLL, data=["猫砂盆 Pro 型号"], anns_field="sparse",
                        limit=2, output_fields=["text"], search_params={"metric_type": "BM25"})
    print("BM25:", [(h["id"], round(float(h["distance"]), 3)) for h in res[0]])
    assert res[0][0]["id"] == 2, "型号词 BM25 应命中第 2 条"

    # hybrid RRF
    dense_req = AnnSearchRequest(data=[[0.1, 0.2, 0.3, 0.4]], anns_field="dense",
                                 param={"metric_type": "COSINE"}, limit=2)
    sparse_req = AnnSearchRequest(data=["运费"], anns_field="sparse",
                                  param={"metric_type": "BM25"}, limit=2)
    hres = client.hybrid_search(COLL, reqs=[dense_req, sparse_req], ranker=RRFRanker(),
                                limit=2, output_fields=["text"])
    print("hybrid:", [(h["id"], round(float(h["distance"]), 3)) for h in hres[0]])
    assert len(hres[0]) >= 1, "hybrid_search 应有结果"

    client.drop_collection(COLL)
    print("GO: Milvus Lite BM25 + hybrid_search 通")


if __name__ == "__main__":
    main()
```

> Milvus Lite 对数据目录是进程级独占锁:冒烟用的是临时目录,不会碰 `data/milvus_knowledge.db`,可在服务运行时跑。

- [ ] **Step 6: 跑两个冒烟(GO/NO-GO)**

```bash
PYTHONPATH=. uv run python scripts/smoke_milvus_bm25.py
PYTHONPATH=. uv run python scripts/smoke_rerank.py
```
Expected: 两个都打印 `GO: ...`。

**NO-GO 处理(红线):**
- rerank 报错:先确认用的是 DashScope 原生形状(`input`/`parameters`/`output.results`),不是 OpenAI 的 `/rerank`;模型名用 `qwen3-rerank`;仍不通 → **停,把请求/响应/错误写进 `dev-notes/ch04.md` 顶部「⚠️ 红线卡点」并等用户**,不自换重排模型或供应商。
- Milvus BM25 报错:确认已 `uv add jieba` 且 tokenizer 是 `jieba`;仍不通 → 停+记录,不改回标准分词/不弃 BM25。

- [ ] **Step 7: 提交**

```bash
git add pyproject.toml uv.lock Makefile scripts/smoke_rerank.py scripts/smoke_milvus_bm25.py
git commit -m "feat(ch04): Milvus Lite BM25 + 百炼 rerank 冒烟(fail-fast 红线通过)"
```

---

## Task 2: config 参数

**Files:**
- Modify: `app/config.py`

**Interfaces:**
- Produces: `settings.milvus_uri`(http)、`settings.rerank_model`、`settings.recall_top_k`、`settings.rerank_top_k`、`settings.rerank_min_score`。

- [ ] **Step 1: 改 config**

`app/config.py` 的 ch03 段替换/追加:
```python
    # ch03 知识库检索(嵌入沿用百炼 text-embedding-v4)
    embed_model: str = "text-embedding-v4"
    milvus_uri: str = "data/milvus_knowledge.db"   # 沿用 ch03 Lite
    retrieval_top_k: int = 3
    retrieval_min_score: float = 0.4
    # ch04 混合检索 + 重排(百炼 qwen3-rerank)
    rerank_url: str = "https://maas.qianwenaiapi.com/api/v1/services/rerank/text-rerank/text-rerank"
    rerank_api_key: str | None = None            # 缺省复用 embed_api_key
    rerank_model: str = "qwen3-rerank"
    recall_top_k: int = 50        # dense/BM25 各召回 Top-50
    rerank_top_k: int = 10        # 精排出 Top-10
    rerank_min_score: float = 0.3 # 重排最高分低于此 → 检索证据低,拒答
```

- [ ] **Step 2: 提交**
```bash
git add app/config.py
git commit -m "feat(ch04): config 加 rerank/召回参数"
```

---

## Task 3: milvus_client 重写(新 schema + 混合/单路检索)

**Files:**
- Modify: `app/kb/milvus_client.py`
- Test: `tests/kb/test_milvus_hybrid.py`

**Interfaces:**
- Consumes: pymilvus `MilvusClient/DataType/Function/FunctionType/AnnSearchRequest/RRFRanker`。
- Produces:
  - `get_client(uri=None) -> MilvusClient`(不变)
  - `ensure_collection(client, collection=COLLECTION) -> None`(新 schema)
  - `upsert_vectors(client, rows, collection=COLLECTION)`;row = `{id, dense, text, question, answer, section_path, content_type, category}`
  - `dense_search(client, vector, top_k, category=None, collection=COLLECTION) -> list[dict]`
  - `bm25_search(client, text, top_k, category=None, collection=COLLECTION) -> list[dict]`
  - `hybrid_search(client, vector, text, top_k, recall=50, category=None, collection=COLLECTION) -> list[dict]`
  - `count(client, collection=COLLECTION) -> int`;`drop(client, collection)`
  - 每个 hit dict = `{id, score, question, answer, section_path, content_type, category}`

- [ ] **Step 1: 写失败测试**

`tests/kb/test_milvus_hybrid.py`(真实 Lite 临时文件,唯一集合名):
```python
import tempfile
import uuid
from pathlib import Path

import pytest

from app.kb import milvus_client as mc


@pytest.fixture()
def coll():
    uri = str(Path(tempfile.mkdtemp()) / f"{uuid.uuid4().hex[:8]}.db")
    client = mc.get_client(uri=uri)
    name = f"test_kb_{uuid.uuid4().hex[:8]}"
    mc.ensure_collection(client, collection=name)
    rows = [
        {"id": 1, "dense": [0.1] * 1024, "text": "运费 满99元包邮 不满10元",
         "question": "运费怎么算", "answer": "满99包邮,否则10元",
         "section_path": "运费政策", "content_type": "faq", "category": "运费"},
        {"id": 2, "dense": [0.2] * 1024, "text": "智能猫砂盆 Pro 型号 自动清理",
         "question": "Pro 型号功能", "answer": "自动清理除臭",
         "section_path": "商品手册 / 猫砂盆", "content_type": "manual", "category": "商品手册"},
    ]
    mc.upsert_vectors(client, rows, collection=name)
    client.load_collection(name)
    yield client, name
    mc.drop(client, name)


def test_bm25_hits_model_number(coll):
    client, name = coll
    hits = mc.bm25_search(client, "猫砂盆 Pro 型号", top_k=2, collection=name)
    assert hits[0]["id"] == 2
    assert hits[0]["section_path"] == "商品手册 / 猫砂盆"


def test_hybrid_returns_scored_hits(coll):
    client, name = coll
    hits = mc.hybrid_search(client, [0.1] * 1024, "运费", top_k=2, recall=10, collection=name)
    assert hits
    assert {"id", "score", "question", "answer", "section_path", "content_type", "category"} <= hits[0].keys()


def test_category_filter(coll):
    client, name = coll
    hits = mc.bm25_search(client, "型号 运费", top_k=5, category="运费", collection=name)
    assert all(h["category"] == "运费" for h in hits)
    assert all(h["id"] == 1 for h in hits)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH=. uv run pytest tests/kb/test_milvus_hybrid.py -v`
Expected: FAIL(`ensure_collection` 签名不含 collection / 无 bm25_search 等)。

- [ ] **Step 3: 重写 milvus_client**

`app/kb/milvus_client.py` 全文替换:
```python
from pymilvus import (
    AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker,
)

from app.config import settings

COLLECTION = "knowledge"
DIM = 1024

_CLIENT: MilvusClient | None = None
_ensured: set[str] = set()

_OUTPUT = ["question", "answer", "section_path", "content_type", "category"]


def get_client(uri: str | None = None) -> MilvusClient:
    """默认返回进程内单例(连 Lite 文件);传 uri(测试)返回独立客户端。"""
    global _CLIENT
    if uri is not None:
        return MilvusClient(uri=uri)
    if _CLIENT is None:
        _CLIENT = MilvusClient(uri=settings.milvus_uri)
    return _CLIENT


def ensure_collection(client: MilvusClient, collection: str = COLLECTION) -> None:
    """幂等建集合:dense(COSINE) + text(jieba analyzer) + sparse(BM25 Function) + 标量字段。
    单例默认集合只实际执行一次;不同集合名(测试临时库)各自 ensure。"""
    if client is _CLIENT and collection in _ensured:
        return
    if client.has_collection(collection):
        client.load_collection(collection)
    else:
        schema = client.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("text", DataType.VARCHAR, max_length=8192,
                         enable_analyzer=True, analyzer_params={"tokenizer": "jieba"})
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("question", DataType.VARCHAR, max_length=2048)
        schema.add_field("answer", DataType.VARCHAR, max_length=8192)
        schema.add_field("section_path", DataType.VARCHAR, max_length=512)
        schema.add_field("content_type", DataType.VARCHAR, max_length=32)
        schema.add_field("category", DataType.VARCHAR, max_length=255)
        schema.add_function(Function(
            name="text_bm25", input_field_names=["text"],
            output_field_names=["sparse"], function_type=FunctionType.BM25,
        ))
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        client.create_collection(collection, schema=schema, index_params=index_params)
        client.load_collection(collection)
    if client is _CLIENT:
        _ensured.add(collection)


def upsert_vectors(client: MilvusClient, rows: list[dict], collection: str = COLLECTION) -> None:
    if rows:
        client.upsert(collection, rows)


def flush(client: MilvusClient, collection: str = COLLECTION) -> None:
    """刷盘:新 upsert 的数据须 flush 后才可被 BM25/检索命中(Task 1 实证)。"""
    client.flush(collection)


def _cat_expr(category: str | None) -> str:
    return f'category == "{category}"' if category else ""


def _hit(h: dict) -> dict:
    e = h["entity"]
    return {"id": h["id"], "score": float(h["distance"]),
            "question": e["question"], "answer": e["answer"],
            "section_path": e["section_path"], "content_type": e["content_type"],
            "category": e["category"]}


def dense_search(client, vector, top_k, category=None, collection=COLLECTION) -> list[dict]:
    res = client.search(collection, data=[vector], anns_field="dense", limit=top_k,
                        output_fields=_OUTPUT, search_params={"metric_type": "COSINE"},
                        filter=_cat_expr(category))
    # 实测 Lite COSINE 的 distance 即相似度(identical 向量=1.0),直接当分数用
    return [_hit(h) for h in res[0]]


def bm25_search(client, text, top_k, category=None, collection=COLLECTION) -> list[dict]:
    res = client.search(collection, data=[text], anns_field="sparse", limit=top_k,
                        output_fields=_OUTPUT, search_params={"metric_type": "BM25"},
                        filter=_cat_expr(category))
    return [_hit(h) for h in res[0]]


def hybrid_search(client, vector, text, top_k, recall=50, category=None, collection=COLLECTION) -> list[dict]:
    expr = _cat_expr(category)
    dense_req = AnnSearchRequest(data=[vector], anns_field="dense",
                                 param={"metric_type": "COSINE"}, limit=recall, expr=expr)
    sparse_req = AnnSearchRequest(data=[text], anns_field="sparse",
                                  param={"metric_type": "BM25"}, limit=recall, expr=expr)
    res = client.hybrid_search(collection, reqs=[dense_req, sparse_req], ranker=RRFRanker(),
                               limit=top_k, output_fields=_OUTPUT)
    return [_hit(h) for h in res[0]]


def count(client, collection: str = COLLECTION) -> int:
    return client.query(collection, filter="id >= 0", output_fields=["count(*)"])[0]["count(*)"]


def drop(client, collection: str) -> None:
    if client.has_collection(collection):
        client.drop_collection(collection)
```

> **COSINE 语义(已实测)**:milvus-lite 3.2.2 的 COSINE `distance` 返回**相似度本身**(identical 向量=1.0,正交=0.0),与 ch03 一致,`score=float(distance)` 不做 `1−`。若换 Lite 版本后实测相反,以实测为准调整并记录。

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH=. uv run pytest tests/kb/test_milvus_hybrid.py -v`
Expected: 3 passed。

- [ ] **Step 5: 提交**
```bash
git add app/kb/milvus_client.py tests/kb/test_milvus_hybrid.py
git commit -m "feat(ch04): milvus_client 新 schema + hybrid/dense/bm25 检索 + 品类过滤"
```

---

## Task 4: 建库写入新字段(dualwrite/vectorize)

**Files:**
- Modify: `app/kb/dualwrite.py`
- Test: `tests/kb/test_dualwrite_ch04.py`(新增,复用 ch03 真实 test MySQL + 真实 Milvus 临时集合)

**Interfaces:**
- Consumes: `milvus_client.upsert_vectors`(新 row 结构)、`repository.list_pending_chunks`(含 section_path/content_type/category)。
- Produces: `vectorize_pending(client, batch_size=64, collection=COLLECTION)` 写入含 `dense/text/section_path/content_type/category` 的 row。

- [ ] **Step 1: 写失败测试**

`tests/kb/test_dualwrite_ch04.py`:
```python
import tempfile
import uuid
from pathlib import Path

import pytest

from app.kb import dualwrite, milvus_client as mc
from app.kb.documents import Chunk


@pytest.mark.asyncio
async def test_vectorize_writes_bm25_and_metadata(db_session_factory, monkeypatch):
    # 嵌入 mock 成定长向量
    async def fake_embed(texts):
        return [[0.05] * 1024 for _ in texts]
    monkeypatch.setattr("app.core.embeddings.embed_texts", fake_embed)

    chunks = [Chunk(category="运费", questions="运费怎么算", answer="满99包邮 型号无关",
                    section_path="运费政策", content_type="faq", is_key_clause=1)]
    await dualwrite.write_pending(chunks)

    uri = str(Path(tempfile.mkdtemp()) / f"{uuid.uuid4().hex[:8]}.db")
    client = mc.get_client(uri=uri)
    name = f"test_kb_{uuid.uuid4().hex[:8]}"
    mc.ensure_collection(client, collection=name)
    done = await dualwrite.vectorize_pending(client, collection=name)
    client.load_collection(name)
    assert done == 1
    hits = mc.bm25_search(client, "运费", top_k=1, collection=name)
    assert hits and hits[0]["section_path"] == "运费政策" and hits[0]["category"] == "运费"
    mc.drop(client, name)
```

- [ ] **Step 2: 跑确认失败**

Run: `PYTHONPATH=. uv run pytest tests/kb/test_dualwrite_ch04.py -v`
Expected: FAIL(`vectorize_pending` 无 collection 参数 / row 缺字段)。

- [ ] **Step 3: 改 dualwrite.vectorize_pending**

替换 `vectorize_pending`:
```python
async def vectorize_pending(client, batch_size: int = 64, collection: str = "knowledge") -> int:
    """幂等可重跑:取 pending → 拼 category+questions+answer → 嵌入 + BM25 text → Milvus upsert(PK=id)
    → 回填 vector_id、status=done。"""
    pending = await repository.list_pending_chunks()
    done = 0
    for batch in _batches(pending, batch_size):
        texts = [f"{r.category}\n{r.questions}\n{r.answer}" for r in batch]
        vectors = await embeddings.embed_texts(texts)
        rows = [
            {"id": r.id, "dense": v, "text": t,
             "question": r.questions, "answer": r.answer,
             "section_path": r.section_path or "", "content_type": r.content_type or "",
             "category": r.category or ""}
            for r, v, t in zip(batch, vectors, texts)
        ]
        milvus_client.upsert_vectors(client, rows, collection=collection)
        for r in batch:
            await repository.mark_chunk_vectorized(r.id, str(r.id))
        done += len(batch)
    if done:
        milvus_client.flush(client, collection=collection)  # 刷盘,数据方可被 BM25 检索
    return done
```

- [ ] **Step 4: 跑确认通过**

Run: `PYTHONPATH=. uv run pytest tests/kb/test_dualwrite_ch04.py -v`
Expected: 1 passed。

- [ ] **Step 5: 提交**
```bash
git add app/kb/dualwrite.py tests/kb/test_dualwrite_ch04.py
git commit -m "feat(ch04): 向量化写入 BM25 text + section_path/content_type/category"
```

---

## Task 5: rerank 客户端

**Files:**
- Create: `app/core/rerank.py`
- Test: `tests/core/test_rerank.py`

**Interfaces:**
- Produces: `async rerank(query: str, docs: list[str], top_n: int|None=None) -> list[tuple[int, float]]`(返回 `(原始索引, 相关分)` 按分降序,截断 top_n)。空 docs 返回 `[]`。

- [ ] **Step 1: 写失败测试**

`tests/core/test_rerank.py`:
```python
import pytest

from app.core import rerank as rr


@pytest.mark.asyncio
async def test_rerank_orders_and_truncates(monkeypatch):
    async def fake_post(url, json, headers, timeout):
        class R:
            def raise_for_status(self): pass
            def json(self):
                return {"output": {"results": [
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 2, "relevance_score": 0.5},
                ]}}
        return R()
    monkeypatch.setattr("app.core.rerank._post", fake_post)
    out = await rr.rerank("q", ["a", "b", "c"], top_n=2)
    assert out == [(1, 0.9), (2, 0.5)]


@pytest.mark.asyncio
async def test_rerank_empty_docs():
    assert await rr.rerank("q", []) == []
```

- [ ] **Step 2: 跑确认失败**

Run: `PYTHONPATH=. uv run pytest tests/core/test_rerank.py -v`
Expected: FAIL(模块不存在)。

- [ ] **Step 3: 实现 rerank**

`app/core/rerank.py`:
```python
import httpx

from app.config import settings


async def _post(url, json, headers, timeout):
    async with httpx.AsyncClient() as c:
        return await c.post(url, json=json, headers=headers, timeout=timeout)


async def rerank(query: str, docs: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
    """调百炼原生 rerank(qwen3-rerank)。返回 [(原始索引, 相关分)] 按分降序,截断 top_n。

    百炼 rerank 不是 OpenAI 协议:请求体是 {model, input:{query,documents}, parameters},
    结果是 output.results[].index / relevance_score,故手写请求、不用 openai 客户端。"""
    if not docs:
        return []
    payload = {"model": settings.rerank_model,
               "input": {"query": query, "documents": docs},
               "parameters": {"top_n": top_n or len(docs), "return_documents": False}}
    key = settings.rerank_api_key or settings.embed_api_key
    resp = await _post(settings.rerank_url, payload,
                       {"Authorization": f"Bearer {key}"}, 60)
    resp.raise_for_status()
    results = resp.json()["output"]["results"]
    ranked = sorted(((r["index"], float(r["relevance_score"])) for r in results),
                    key=lambda x: x[1], reverse=True)
    return ranked[:top_n] if top_n else ranked
```

- [ ] **Step 4: 跑确认通过**

Run: `PYTHONPATH=. uv run pytest tests/core/test_rerank.py -v`
Expected: 2 passed。

- [ ] **Step 5: 提交**
```bash
git add app/core/rerank.py tests/core/test_rerank.py
git commit -m "feat(ch04): 百炼 qwen3-rerank 重排客户端"
```

---

## Task 6: Query 理解(改写 + 同义词扩展)

**纯 Prompt 任务:TDD 换成标注样例验证。** 只作用于检索侧,不改入库。

**Files:**
- Modify: `app/core/prompts.py`(加 `QUERY_REWRITE_SYSTEM` / prompt)
- Create: `app/core/query_understanding.py`
- Create: `tests/data/query_rewrite_samples.jsonl`
- Modify: `scripts/eval_ch04.py`(Task 13 建;本任务先建独立 mini 验证或并入)

**Interfaces:**
- Produces: `async understand(query: str) -> dict`,返回扁平字段 `{"standard": str, "expanded": list[str]}`(`standard`=归一标准问法,`expanded`=同义词/近义扩展词,仅供 BM25 检索文本拼接)。

- [ ] **Step 1: 写改写 prompt**

`app/core/prompts.py` 追加:
```python
QUERY_REWRITE_SYSTEM = """你是电商客服检索前的 Query 归一化器。把用户口语、模糊、带情绪的问法改写成简洁标准的问法,并给出同义词/近义扩展词(用于关键词召回)。
- standard:一句话标准问法,去口语和情绪,保留关键实体(型号、品类、政策词)。
- expanded:3-6 个与问题相关的同义词/近义词/别称(如「邮费↔运费」「多久到↔时效」),只列词,不含原词。
- 不臆造原问题没有的实体或型号。"""

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", QUERY_REWRITE_SYSTEM), ("human", "用户问法:{query}")]
)
```

- [ ] **Step 2: 实现 understand(扁平结构化输出)**

`app/core/query_understanding.py`:
```python
from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import QUERY_REWRITE_PROMPT


class _Rewrite(BaseModel):
    standard: str = Field(description="标准问法")
    expanded: list[str] = Field(default_factory=list, description="同义/近义扩展词")


async def understand(query: str) -> dict:
    """口语→标准问法 + 同义词扩展。扁平字段(list[str],非嵌套对象),避开 glm-5.2 502。"""
    model = get_chat_model().with_structured_output(_Rewrite, method="function_calling")
    r: _Rewrite = await (QUERY_REWRITE_PROMPT | model).ainvoke({"query": query})
    return {"standard": r.standard or query, "expanded": list(r.expanded or [])}
```

- [ ] **Step 3: 造标注样例**

`tests/data/query_rewrite_samples.jsonl`(每行一条,标注期望关键词应出现在 standard 或 expanded):
```json
{"query": "那个邮费到底要给多少钱啊急", "expect_any": ["运费", "邮费"]}
{"query": "东西啥时候能到我手上", "expect_any": ["时效", "多久", "配送"]}
{"query": "Pro款猫砂盆能自动铲屎不", "expect_any": ["Pro", "猫砂盆"]}
{"query": "不想要了能退不", "expect_any": ["退货", "退款", "无理由"]}
{"query": "坏了咋整", "expect_any": ["维修", "售后", "质量"]}
```

- [ ] **Step 4: 跑标注验证(真实 glm-5.2)**

写临时验证片段(或并入 `scripts/eval_ch04.py --check-rewrite`):对每条样例调 `understand`,断言 `expect_any` 至少一个词出现在 `standard+expanded` 文本里。运行:
```bash
PYTHONPATH=. uv run python -c "
import asyncio, json
from app.core.query_understanding import understand
async def main():
    ok=0; tot=0
    for ln in open('tests/data/query_rewrite_samples.jsonl'):
        s=json.loads(ln); tot+=1
        r=await understand(s['query']); blob=r['standard']+' '+' '.join(r['expanded'])
        hit=any(w in blob for w in s['expect_any'])
        print('OK' if hit else 'MISS', s['query'], '->', r)
        ok+=hit
    print(f'{ok}/{tot}')
asyncio.run(main())
"
```
Expected: ≥4/5 命中(口语归一有效)。低于则调 prompt。

- [ ] **Step 5: 提交**
```bash
git add app/core/prompts.py app/core/query_understanding.py tests/data/query_rewrite_samples.jsonl
git commit -m "feat(ch04): Query 理解(口语归一 + 同义词扩展,标注样例验证)"
```

---

## Task 7: retrieval 重写(strategy 分支 + 首尾组装)

**Files:**
- Modify: `app/core/retrieval.py`
- Test: `tests/core/test_retrieval_ch04.py`

**Interfaces:**
- Consumes: `embeddings.embed_query`、`milvus_client.{dense_search,bm25_search,hybrid_search}`、`rerank.rerank`。
- Produces:
  - `async search_knowledge(query, strategy="hybrid_rerank", top_k=None, category=None, client=None, collection="knowledge") -> list[dict]`;strategy ∈ {vector,bm25,hybrid,hybrid_rerank};返回按最终排序的 hit dict(hybrid_rerank 时含 `rerank_score`)。
  - `arrange_head_tail(items: list) -> list`:最相关放首、次相关放尾,其余居中。

- [ ] **Step 1: 写失败测试**

`tests/core/test_retrieval_ch04.py`:
```python
import pytest

from app.core import retrieval


def test_arrange_head_tail():
    # 输入按相关度降序 [a,b,c,d,e] → a 首、b 尾、剩 c,d,e 居中
    out = retrieval.arrange_head_tail(["a", "b", "c", "d", "e"])
    assert out[0] == "a" and out[-1] == "b"
    assert set(out[1:-1]) == {"c", "d", "e"}


@pytest.mark.asyncio
async def test_strategy_vector(monkeypatch):
    async def fake_embed(q): return [0.1] * 1024
    monkeypatch.setattr("app.core.embeddings.embed_query", fake_embed)
    called = {}
    def fake_dense(client, vec, top_k, category=None, collection="knowledge"):
        called["dense"] = True
        return [{"id": 1, "score": 0.9, "question": "q", "answer": "a",
                 "section_path": "p", "content_type": "faq", "category": "运费"}]
    monkeypatch.setattr("app.kb.milvus_client.dense_search", fake_dense)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    out = await retrieval.search_knowledge("运费", strategy="vector", client=object())
    assert called.get("dense") and out[0]["id"] == 1


@pytest.mark.asyncio
async def test_strategy_hybrid_rerank(monkeypatch):
    async def fake_embed(q): return [0.1] * 1024
    monkeypatch.setattr("app.core.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    def fake_hybrid(client, vec, text, top_k, recall=50, category=None, collection="knowledge"):
        return [
            {"id": 1, "score": 0.5, "question": "运费", "answer": "满99包邮",
             "section_path": "p1", "content_type": "faq", "category": "运费"},
            {"id": 2, "score": 0.4, "question": "型号", "answer": "Pro自动清理",
             "section_path": "p2", "content_type": "manual", "category": "商品"},
        ]
    monkeypatch.setattr("app.kb.milvus_client.hybrid_search", fake_hybrid)
    async def fake_rerank(q, docs, top_n=None):
        return [(1, 0.95), (0, 0.2)]  # 第 2 条(index1)最相关
    monkeypatch.setattr("app.core.rerank.rerank", fake_rerank)
    out = await retrieval.search_knowledge("Pro 型号", strategy="hybrid_rerank", client=object())
    assert out[0]["id"] == 2 and out[0]["rerank_score"] == 0.95
```

- [ ] **Step 2: 跑确认失败**

Run: `PYTHONPATH=. uv run pytest tests/core/test_retrieval_ch04.py -v`
Expected: FAIL。

- [ ] **Step 3: 重写 retrieval**

`app/core/retrieval.py` 全文替换:
```python
from app.config import settings
from app.core import embeddings, rerank
from app.kb import milvus_client


def arrange_head_tail(items: list) -> list:
    """最相关放首、次相关放尾,其余按序居中(缓解 lost-in-the-middle)。"""
    if len(items) <= 2:
        return items
    return [items[0], *items[2:], items[1]]


async def search_knowledge(
    query: str, strategy: str = "hybrid_rerank", top_k: int | None = None,
    category: str | None = None, client=None, collection: str = "knowledge",
) -> list[dict]:
    """strategy: vector | bm25 | hybrid | hybrid_rerank。
    返回最终排序的 hit(hybrid_rerank 含 rerank_score);不做首尾组装(调用侧按需)。"""
    top_k = top_k or settings.rerank_top_k
    client = client or milvus_client.get_client()
    milvus_client.ensure_collection(client, collection=collection)

    if strategy == "vector":
        vec = await embeddings.embed_query(query)
        return milvus_client.dense_search(client, vec, top_k, category, collection)
    if strategy == "bm25":
        return milvus_client.bm25_search(client, query, top_k, category, collection)

    # hybrid / hybrid_rerank:先混合召回 recall_top_k
    vec = await embeddings.embed_query(query)
    hits = milvus_client.hybrid_search(
        client, vec, query, settings.recall_top_k, settings.recall_top_k, category, collection)
    if strategy == "hybrid":
        return hits[:top_k]

    # hybrid_rerank:对召回结果精排
    docs = [f"{h['question']} {h['answer']}" for h in hits]
    ranked = await rerank.rerank(query, docs, top_n=top_k)
    out = []
    for idx, score in ranked:
        h = dict(hits[idx]); h["rerank_score"] = score
        out.append(h)
    return out
```

- [ ] **Step 4: 跑确认通过**

Run: `PYTHONPATH=. uv run pytest tests/core/test_retrieval_ch04.py -v`
Expected: 3 passed。

- [ ] **Step 5: 提交**
```bash
git add app/core/retrieval.py tests/core/test_retrieval_ch04.py
git commit -m "feat(ch04): retrieval strategy 分支(四策略)+ 首尾组装"
```

---

## Task 8: 生成/自评/faithfulness prompt + 负面知识

**纯 Prompt 任务。** 本任务只落 prompt 文本 + 自评函数(自评函数走 TDD:mock 模型断言扁平输出解析)。

**Files:**
- Modify: `app/core/prompts.py`
- Create: `app/core/selfcheck.py`
- Test: `tests/core/test_selfcheck.py`

**Interfaces:**
- Produces:
  - `prompts.RAG_ANSWER_SYSTEM`(带引用规则 + 负面知识)、`RAG_ANSWER_PROMPT`
  - `prompts.SELF_CHECK_SYSTEM`、`SELF_CHECK_PROMPT`
  - `prompts.FAITHFULNESS_SYSTEM`、`FAITHFULNESS_PROMPT`(Task 13 用)
  - `selfcheck.check_sufficient(query, evidence_texts) -> dict`,返回扁平 `{"useful": bool, "reason": str}`

- [ ] **Step 1: 写 prompt**

`app/core/prompts.py` 追加:
```python
RAG_ANSWER_SYSTEM = """你是「喵喵优选」电商平台的智能客服「小喵」。下面提供了带编号的知识证据,请严格依据证据回答用户问题。

## 引用规则
- 答案里每个关键结论后标注来源编号,如「满99元包邮[1]」;编号对应下方证据的序号,可多个如[1][2]。
- 只使用提供的证据作答,不要编造证据之外的信息。

## 拒答规则
- 若证据不足以回答用户问题,明确告知「暂时没有查到相关信息」并引导用户联系人工客服,不要硬编答案。

## 禁止承诺(负面知识,必须遵守)
- 不承诺具体到账时间、到货/配送时间、维修时长等时效;统一表述「以平台实际处理为准」。
- 不承诺赔偿金额或赔付时效;退款政策统一「以平台售后规则为准」。
- 不臆造订单、物流、库存、价格;无权限转接/提交工单时引导用户走 App 人工客服入口。
- 语气亲切专业、简洁,中文作答。"""

RAG_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", RAG_ANSWER_SYSTEM), ("human", "用户问题:{query}\n\n知识证据:\n{evidence}")]
)

SELF_CHECK_SYSTEM = """你是检索质量评审员。给定用户问题和检索到的知识证据,判断这些证据是否足以准确回答该问题。
- useful=true:证据包含回答该问题所需的关键信息。
- useful=false:证据与问题无关、或缺少关键信息、或只能部分回答核心诉求。
- reason:一句话说明判断依据。
严格只看证据是否够答,不要脑补证据外的知识。"""

SELF_CHECK_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SELF_CHECK_SYSTEM), ("human", "用户问题:{query}\n\n检索证据:\n{evidence}")]
)

FAITHFULNESS_SYSTEM = """你是回答忠实度评审员。给定检索证据和客服回答,判断回答中的事实性主张是否都能被证据支撑。
- faithful=true:回答的关键事实都能在证据中找到依据(或为合理拒答)。
- faithful=false:回答包含证据未支撑的编造内容。
- reason:一句话说明。"""

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [("system", FAITHFULNESS_SYSTEM), ("human", "检索证据:\n{evidence}\n\n客服回答:\n{answer}")]
)
```

- [ ] **Step 2: 写失败测试**

`tests/core/test_selfcheck.py`:
```python
import pytest

from app.core import selfcheck


@pytest.mark.asyncio
async def test_check_sufficient_parses_flat(monkeypatch):
    class Fake:
        useful = False
        reason = "证据只讲运费,未涉及型号"
    async def fake_invoke(_): return Fake()
    class Chain:
        def __or__(self, o): return self
        async def ainvoke(self, _): return Fake()
    monkeypatch.setattr("app.core.selfcheck._chain", lambda: Chain())
    out = await selfcheck.check_sufficient("Pro型号功能", ["满99包邮"])
    assert out == {"useful": False, "reason": "证据只讲运费,未涉及型号"}
```

- [ ] **Step 3: 实现 selfcheck(扁平输出)**

`app/core/selfcheck.py`:
```python
from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import SELF_CHECK_PROMPT


class _Check(BaseModel):
    useful: bool = Field(description="证据是否足以回答")
    reason: str = Field(default="", description="判断依据")


def _chain():
    return SELF_CHECK_PROMPT | get_chat_model().with_structured_output(
        _Check, method="function_calling")


async def check_sufficient(query: str, evidence_texts: list[str]) -> dict:
    """生成前证据充分性自评。扁平字段 useful/reason(避开 glm-5.2 嵌套数组 502)。"""
    evidence = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(evidence_texts)) or "(无证据)"
    r: _Check = await _chain().ainvoke({"query": query, "evidence": evidence})
    return {"useful": bool(r.useful), "reason": r.reason or ""}
```

- [ ] **Step 4: 跑确认通过**

Run: `PYTHONPATH=. uv run pytest tests/core/test_selfcheck.py -v`
Expected: 1 passed。

- [ ] **Step 5: 提交**
```bash
git add app/core/prompts.py app/core/selfcheck.py tests/core/test_selfcheck.py
git commit -m "feat(ch04): RAG 生成/自评/faithfulness prompt + 负面知识 + 生成前自评"
```

---

## Task 9: low_confidence_questions 写入 + conftest 扩表

**Files:**
- Modify: `app/db/repository.py`
- Modify: `app/db/models.py`(加 `LowConfidenceQuestion` 模型)
- Modify: `tests/conftest.py`(加 ch04-ddl + 表名)
- Test: `tests/db/test_low_confidence.py`

**Interfaces:**
- Produces: `async repository.insert_low_confidence(conversation_id: int|None, raw_question: str, source: str, reason: str|None) -> int`;source ∈ {retrieval_low_conf, self_check, user_feedback}。

- [ ] **Step 1: 加 ORM 模型**

`app/db/models.py` 追加:
```python
class LowConfidenceQuestion(Base):
    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("conversations.id"), nullable=True)
    raw_question: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Enum("retrieval_low_conf", "self_check", "user_feedback"))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

- [ ] **Step 2: conftest 扩表**

`tests/conftest.py`:`_DDL_FILES` 追加 ch04-ddl;`_TABLES` 头部加 `low_confidence_questions`(子表,先删):
```python
_DDL_FILES = [
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch02-ddl.sql",
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch03-ddl.sql",
    pathlib.Path(__file__).resolve().parent.parent / "sql" / "ch04-ddl.sql",
]
_TABLES = ["low_confidence_questions", "messages", "tickets", "conversations", "faq",
           "qa_extraction_staging", "knowledge_chunks"]
```

- [ ] **Step 3: 写失败测试**

`tests/db/test_low_confidence.py`:
```python
import pytest

from app.db import repository


@pytest.mark.asyncio
async def test_insert_low_confidence(db_session_factory):
    cid = await repository.create_conversation("u1")
    lid = await repository.insert_low_confidence(cid, "邮费到底多少啊", "self_check", "证据不足")
    assert lid > 0
    factory = db_session_factory
    from sqlalchemy import text
    async with factory() as s:
        row = (await s.execute(text("SELECT raw_question, source, reason FROM low_confidence_questions WHERE id=:i"), {"i": lid})).first()
    assert row.source == "self_check" and row.raw_question == "邮费到底多少啊"


@pytest.mark.asyncio
async def test_insert_low_confidence_null_conv(db_session_factory):
    lid = await repository.insert_low_confidence(None, "无会话问题", "retrieval_low_conf", None)
    assert lid > 0
```

- [ ] **Step 4: 跑确认失败**

Run: `PYTHONPATH=. uv run pytest tests/db/test_low_confidence.py -v`
Expected: FAIL(`insert_low_confidence` 不存在)。

- [ ] **Step 5: 实现 repository 方法**

`app/db/repository.py` 追加(顶部 import 加 `LowConfidenceQuestion`):
```python
async def insert_low_confidence(
    conversation_id: int | None, raw_question: str, source: str, reason: str | None
) -> int:
    async with async_session() as s:
        row = LowConfidenceQuestion(
            conversation_id=conversation_id, raw_question=raw_question,
            source=source, reason=reason,
        )
        s.add(row)
        await s.commit()
        await s.refresh(row)
        return row.id
```

- [ ] **Step 6: 跑确认通过 + 全量回归**

Run: `PYTHONPATH=. uv run pytest tests/db/test_low_confidence.py -v`
Expected: 2 passed。

- [ ] **Step 7: 提交**
```bash
git add app/db/models.py app/db/repository.py tests/conftest.py tests/db/test_low_confidence.py
git commit -m "feat(ch04): low_confidence_questions ORM + 写入 + conftest 扩表"
```

---

## Task 10: query_faq 升级为 RAG 管线

**Files:**
- Modify: `app/tools/business.py`
- Test: `tests/tools/test_query_faq_rag.py`

**Interfaces:**
- Consumes: `retrieval.search_knowledge`、`selfcheck.check_sufficient`、`query_understanding.understand`、`settings.rerank_min_score`。
- Produces: `query_faq(keyword, category=None)` 返回:
  - 足:`{"sufficient": True, "evidence": "[1] ...\n[2] ...", "citations": [{"n":1,"id":..,"section_path":..,"question":..,"answer":..}, ...]}`
  - 不足:`{"sufficient": False, "source": "retrieval_low_conf"|"self_check", "reason": str, "citations": []}`
  - citations 已按首尾组装顺序编号;evidence 文本与 citations 编号一致。

- [ ] **Step 1: 写失败测试**

`tests/tools/test_query_faq_rag.py`:
```python
import pytest

from app.tools import business


@pytest.mark.asyncio
async def test_query_faq_sufficient(monkeypatch):
    async def fake_understand(q): return {"standard": q, "expanded": ["运费"]}
    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)
    async def fake_search(query, strategy="hybrid_rerank", top_k=None, category=None, client=None, collection="knowledge"):
        return [
            {"id": 1, "rerank_score": 0.9, "question": "运费", "answer": "满99包邮",
             "section_path": "运费政策", "content_type": "faq", "category": "运费"},
            {"id": 2, "rerank_score": 0.6, "question": "退货", "answer": "七天无理由",
             "section_path": "退货政策", "content_type": "policy", "category": "退货"},
        ]
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    async def fake_check(q, texts): return {"useful": True, "reason": "够"}
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)

    out = await business.query_faq.ainvoke({"keyword": "邮费多少"})
    assert out["sufficient"] is True
    assert out["citations"][0]["n"] == 1 and out["citations"][0]["section_path"]
    assert "[1]" in out["evidence"]


@pytest.mark.asyncio
async def test_query_faq_retrieval_low_conf(monkeypatch):
    async def fake_understand(q): return {"standard": q, "expanded": []}
    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)
    async def fake_search(**kw): return []  # 无召回
    monkeypatch.setattr("app.core.retrieval.search_knowledge", lambda *a, **k: fake_search(**k))
    out = await business.query_faq.ainvoke({"keyword": "火星车怎么买"})
    assert out["sufficient"] is False and out["source"] == "retrieval_low_conf"


@pytest.mark.asyncio
async def test_query_faq_self_check_fail(monkeypatch):
    async def fake_understand(q): return {"standard": q, "expanded": []}
    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)
    async def fake_search(query, **kw):
        return [{"id": 1, "rerank_score": 0.8, "question": "运费", "answer": "满99包邮",
                 "section_path": "p", "content_type": "faq", "category": "运费"}]
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    async def fake_check(q, texts): return {"useful": False, "reason": "问型号但证据只讲运费"}
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    out = await business.query_faq.ainvoke({"keyword": "Pro型号能自动铲屎吗"})
    assert out["sufficient"] is False and out["source"] == "self_check"
    assert "型号" in out["reason"]
```

- [ ] **Step 2: 跑确认失败**

Run: `PYTHONPATH=. uv run pytest tests/tools/test_query_faq_rag.py -v`
Expected: FAIL。

- [ ] **Step 3: 重写 query_faq**

`app/tools/business.py`:替换 `FaqInput` + `query_faq`(顶部加 import `from app.config import settings`、`from app.core import retrieval, selfcheck, query_understanding`):
```python
class FaqInput(BaseModel):
    keyword: str = Field(description="用户咨询的政策/规则/操作类问题(可用原话)")
    category: str | None = Field(default=None, description="可选:按品类过滤,如『运费』『退货』『商品手册』")


@tool(args_schema=FaqInput)
async def query_faq(keyword: str, category: str | None = None) -> dict:
    """查询常见问题/政策知识库(混合检索+重排)。用于政策、规则、时效、费用、商品手册等通用问题。
    返回带编号证据供作答引用;证据不足时返回 sufficient=False,请据此向用户拒答。"""
    u = await query_understanding.understand(keyword)
    query = u["standard"]
    # 同义词扩展只拼进检索文本(检索侧),不改标准问法语义
    search_query = query + (" " + " ".join(u["expanded"]) if u["expanded"] else "")

    hits = await retrieval.search_knowledge(
        search_query, strategy="hybrid_rerank", category=category)

    # 机械闸:无召回 or 最高分低于阈值
    top = hits[0]["rerank_score"] if hits else 0.0
    if not hits or top < settings.rerank_min_score:
        return {"sufficient": False, "source": "retrieval_low_conf",
                "reason": f"检索证据不足(top={top:.3f})", "citations": []}

    # 语义闸:生成前自评证据够不够
    ev_texts = [f"{h['question']} {h['answer']}" for h in hits]
    chk = await selfcheck.check_sufficient(query, ev_texts)
    if not chk["useful"]:
        return {"sufficient": False, "source": "self_check",
                "reason": chk["reason"], "citations": []}

    # 首尾组装 + 编号
    arranged = retrieval.arrange_head_tail(hits)
    citations = [
        {"n": i + 1, "id": h["id"], "section_path": h["section_path"],
         "question": h["question"], "answer": h["answer"], "content_type": h["content_type"]}
        for i, h in enumerate(arranged)
    ]
    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    return {"sufficient": True, "evidence": evidence, "citations": citations}
```

- [ ] **Step 4: 跑确认通过**

Run: `PYTHONPATH=. uv run pytest tests/tools/test_query_faq_rag.py -v`
Expected: 3 passed。

- [ ] **Step 5: 提交**
```bash
git add app/tools/business.py tests/tools/test_query_faq_rag.py
git commit -m "feat(ch04): query_faq 升级为 RAG 管线(理解→混合→重排→两道证据闸→编号证据)"
```

---

## Task 11: agent/chat 接线(citations 事件 + 拒答落池)

**Files:**
- Modify: `app/core/agent.py`
- Modify: `app/api/chat.py`
- Test: `tests/core/test_agent_citations.py`

**Interfaces:**
- Consumes: query_faq ToolRun 结果里的 `sufficient/citations/source/reason`;`repository.insert_low_confidence`。
- Produces:
  - agent 流式事件新增 `{"type": "citations", "items": [...]}`(足时下发)。
  - sufficient=false → 落 `low_confidence_questions`(raw_question=用户原话)+ 生成走拒答(证据置空,system prompt 引导拒答)。
  - `/api/chat` SSE 转发 `citations` 事件。

- [ ] **Step 1: 设计接线点(读 infra.ToolRun 结构)**

先读 `app/tools/infra.py` 确认 `ToolRun` 暴露 `name` 与结果 JSON(`tool_message.content`)。agent `_prepare_turn` 后,遍历 runs 找 `name=="query_faq"` 的结果、`json.loads(content)`,得 `faq`。

- [ ] **Step 2: 写失败测试**

`tests/core/test_agent_citations.py`:
```python
import json

import pytest

from app.core import agent


class _FakeToolRun:
    def __init__(self, name, payload, cid="tc1"):
        self.name = name
        self.tool_call_id = cid
        class _M:
            content = json.dumps(payload, ensure_ascii=False)
        self.tool_message = _M()


def test_extract_faq_citations():
    runs = [_FakeToolRun("query_faq", {"sufficient": True, "evidence": "[1] x",
            "citations": [{"n": 1, "id": 5, "section_path": "运费政策", "question": "运费", "answer": "满99包邮", "content_type": "faq"}]})]
    faq = agent._faq_result(runs)
    assert faq["sufficient"] is True and faq["citations"][0]["id"] == 5


def test_extract_faq_none_when_absent():
    runs = [_FakeToolRun("query_order", {"x": 1})]
    assert agent._faq_result(runs) is None
```

- [ ] **Step 3: 加 `_faq_result` 辅助 + 落池 + citations 事件**

`app/core/agent.py`:顶部 import 加 `import json`、`from app.db import repository`(已在)。加辅助:
```python
def _faq_result(runs) -> dict | None:
    for r in runs:
        if getattr(r, "name", None) == "query_faq":
            try:
                return json.loads(r.tool_message.content)
            except (ValueError, TypeError):
                return None
    return None
```

在 `stream_agent_turn` 有工具分支里,推工具轨迹帧后、收敛生成前插入:
```python
    faq = _faq_result(runs)
    if faq is not None:
        if faq.get("sufficient"):
            yield {"type": "citations", "items": faq.get("citations", [])}
        else:
            await repository.insert_low_confidence(
                conversation_id, message, faq.get("source", "self_check"), faq.get("reason"))
```
(非流式 `run_agent_turn` 同样在收敛前落池;citations 放进 `AgentResult`,`AgentResult` 加 `citations: list` 字段默认 `[]`。)

- [ ] **Step 4: chat.py 转发 citations 事件**

`app/api/chat.py` 的事件循环加分支:
```python
                elif ev["type"] == "citations":
                    yield _sse({"event": "citations", "items": ev["items"]})
```

- [ ] **Step 5: 跑确认通过 + 全量回归**

Run: `PYTHONPATH=. uv run pytest tests/core/test_agent_citations.py -v && PYTHONPATH=. uv run pytest -v`
Expected: 新测试 passed;全量绿(含真实 Lite 临时集合测试;注意 Lite 独占锁,跑测试前停掉占库的服务进程)。

- [ ] **Step 6: 提交**
```bash
git add app/core/agent.py app/api/chat.py tests/core/test_agent_citations.py
git commit -m "feat(ch04): agent/chat 接线 citations 事件 + 证据不足拒答落池"
```

---

## Task 12: 评估集(300 题五桶,直接从 CH_04_text.csv 转换)

**数据任务:** 无 TDD,产出评估集 + 结构校验脚本。用户已提供 300 题的 CSV(`CH_04_text.md`),A/B/C/E 四个可作答桶各 60、D 桶 60 库外题;本任务把它转成 jsonl,标注 `expect_section` / `expect_sections_all` 与 `expect_points`。建库后 Task 14 用 section_path 匹配算 Recall/MRR、用 expect_points 算证据/答案覆盖度(避免硬绑 id)。

**Files:**
- Create: `tests/data/eval_ch04.jsonl`

**Interfaces:**
- Produces: 每行 `{"id","bucket","query","expect_section":[...],"expect_sections_all":[[别名...],...],"expect_points":[...],"should_refuse":bool}`。bucket ∈ {A_policy,B_model,C_colloquial,D_absent,E_multi};D 桶 `expect_section`/`expect_points` 为空、`should_refuse=true`。
- `expect_section` 是全量别名列表(任一小节名命中即算);`expect_sections_all` 只给 E 桶,每组是别名列表,组内任一命中即可,凑齐所有组才算完整召回。

- [ ] **Step 1: 转换评估集**

用一次性脚本读 `CH_04_text.md`(UTF-8-BOM CSV),按列名 `id / 桶(bucket) / 问题(query) / 期望章节(expect_section) / 标准要点(expect_points) / 应拒答(should_refuse)` 转 jsonl:
- `expect_section` 列按 `+` 分组、组内按 `|` 拆别名;单组题也存成 `expect_sections_all=[[...]]`,方便统一算分。
- `expect_points` 按 `|` 拆,去首尾空白。
- `should_refuse` 为「是」→ true,D 桶 `expect_section=[]`、`expect_points=[]`。
- 保留原始 id(A1…A60/B…/C…/D…/E…)。

- [ ] **Step 2: 结构校验**
```bash
PYTHONPATH=. uv run python -c "
import json
from collections import Counter
c=Counter()
for ln in open('tests/data/eval_ch04.jsonl', encoding='utf-8'):
    r=json.loads(ln); c[r['bucket']]+=1
    assert r['query'] and 'should_refuse' in r
    if r['bucket']!='D_absent': assert r['expect_points'], r['id']
print(dict(c)); assert len(c)==5 and sum(c.values())==300 and all(v==60 for v in c.values())
print('OK')
"
```

- [ ] **Step 3: 提交**
```bash
git add tests/data/eval_ch04.jsonl
git commit -m "feat(ch04): 评估集 300 题五桶(expect_section + expect_points + should_refuse)"
```

---

## Task 13: 四策略分桶评估脚本(检索 + 证据覆盖度 + 生成)

**Files:**
- Create: `scripts/eval_ch04.py`
- Modify: `Makefile`(`eval-rag`)

**Interfaces:**
- Consumes: `retrieval.search_knowledge(strategy=...)`、`prompts.RAG_ANSWER_PROMPT`/`FAITHFULNESS_PROMPT`、`business.query_faq`、`milvus_client`、评估集。
- Produces: 三段指标打印 + 落 `data/ch04/reports/rag_eval.{txt,json}`(`.txt` 是运行日志给人读,`.json` 给评估页读):
  - **段1 检索**(确定性):四策略 × 桶 Recall@K/MRR(按 `expect_section` 命中 `section_path`)。
  - **段2 证据覆盖度**(确定性 · 跨策略):四策略召回 Top-K 证据机械匹配盖住 `expect_points` 的比例。
  - **段3 生成**(glm,带超时/跳过/低并发):四策略答案覆盖度(证据 → 生成 → glm 判盖住几个要点)+ hybrid_rerank 的 Faithfulness + D 桶拒答率。

- [ ] **Step 1: 写脚本(三段 + 健壮性 + 报告注入)**

`scripts/eval_ch04.py`(结构;完整实现见源文件):
```python
STRATEGIES = ["vector", "bm25", "hybrid", "hybrid_rerank"]
GRADED_BUCKETS = ["A_policy", "B_model", "C_colloquial"]
K = 10
RETR_CONCURRENCY, GEN_CONCURRENCY, CALL_TIMEOUT = 8, 3, 45.0   # 生成压低并发 + 每调用超时

def _norm(s): return "".join((s or "").split())               # 去空白,兼容"每 2 周"↔"每2周"
def _coverage_mech(points, hits):                             # 证据覆盖度:要点子串是否在召回证据里
    ev = _norm("\n".join(h["answer"] for h in hits))
    return sum(1 for p in points if _norm(p) in ev) / len(points) if points else None

async def _try(coro, label):                                  # 带超时跑 glm;超时/异常记账返回 None
    try: return await asyncio.wait_for(coro, CALL_TIMEOUT)
    except Exception as e: _errs.append(f"{label}: {type(e).__name__}"); return None

async def _deterministic(samples):    # 段1+2:每(策略,题)检索一次,复用算 Recall/MRR + 证据覆盖度,返回 HITS
    ...
async def _generation(samples, HITS): # 段3:四策略 gather(_gen_one)(生成+覆盖度judge+faith)+ D refusal;return_exceptions
    ...
def _write_report(retrieval, coverage, generation, meta):    # 运行日志给人读,json 给页面读
    report = {"meta": meta, "retrieval": retrieval, "evidence_coverage": coverage, "generation": generation}
    _OUT_TXT.write_text("\n".join(_LINES))
    _OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=1))   # generation 为 None 也照样落

async def main():
    samples = _load()
    retrieval_d, coverage_d, HITS = await _deterministic(samples)   # 确定性,始终产出
    generation_d = None
    try: generation_d = await _generation(samples, HITS)            # glm,失败不拖垮整轮
    except Exception as e: _log(f"[生成段未完成] {e}")               # 页面标注未完成,上游恢复重跑补全
    _write_report(retrieval_d, coverage_d, generation_d, meta)
```
关键点:证据覆盖度机械匹配,确定性可复现。生成段 `_try` 每调用 45s 超时,配 `gather(return_exceptions=True)` 做单点故障隔离,不死等。产物与呈现分离,页面读的是这份 json,数字与本轮一致。

- [ ] **Step 2: Makefile 加 eval-rag**
```makefile
eval-rag:
        PYTHONPATH=. uv run python scripts/eval_ch04.py
```

- [ ] **Step 3: 提交(先不跑,建库后 Task 14 跑)**
```bash
git add scripts/eval_ch04.py Makefile
git commit -m "feat(ch04): 四策略评估脚本——检索、证据覆盖度、生成三段一次跑齐"
```

---

## Task 14: 知识库补齐 + 真实建库 + 跑评估(验收1、2)

**Files:**
- Modify: `data/kb/*.md`(从桌面 CH_04 收录,修标题粘连,补评估集缺失条款)
- Modify: `scripts/build_kb.py` / `vectorize_kb.py`(传 collection=默认即可,一般无需改)
- Modify: `tests/data/eval_ch04.jsonl`(建库后按真实 section 回填/校准 expect_section)

- [ ] **Step 1: 收录知识文档并修标题**

把桌面 `CH_04/{after-sales-manual,product-faq,product-specs,returns-policy}.md` 复制到 `data/kb/`。
三个文件的首行把多个标题粘在一起(`# 退货退款政策## 无理由退货### 适用范围`),用一次性脚本在 `#` 前补换行,使每个 `#`/`##`/`###` 各占一行;正文一个字不改。`product-specs.md` 标题本来就正常。

- [ ] **Step 2: 按评估集补齐缺失条款**

用 `scripts/validate_eval_ch04.py` 或一次性脚本跑「评估集要点 → 文档子串」核对,把缺失的 46 个要点补进对应文档(用户 2026-11-14 拍板):
- `product-faq.md`:会员等级门槛(银卡 1000 元起 / 金卡 5000 元起、生日礼猫罐头礼盒、95 折/9 折)、积分规则(1 元 1 分、100 分抵 1 元、单笔最多抵 30 元、退货同步扣回、双倍积分不计等级)、开票类型与时效(电子专票、纳税人识别号、仅企业账户、30 天内、快递费到付)。
- `returns-policy.md` 或 `after-sales-manual.md`:猫窝机洗(拆垫芯、轻柔冷水、不可机洗局部擦洗、洗衣袋)、偏远地区不参与包邮。
- 新条款挂在已有对应小节下(如「会员等级与积分规则」「发票如何开具」),不新造与评估集无关的内容。

- [ ] **Step 3: 重置并建库(Lite 独占锁:先停占库的服务进程)**

```bash
PYTHONPATH=. uv run python scripts/reset_kb.py
PYTHONPATH=. uv run python scripts/build_kb.py --all
PYTHONPATH=. uv run python scripts/vectorize_kb.py
PYTHONPATH=. uv run python -c "from app.kb import milvus_client as m; print('Milvus count:', m.count(m.get_client()))"
```
Expected: count 与 MySQL `knowledge_chunks` done 数一致。

> 本机 MySQL 是本地 8.0(root/123456),不是 docker 容器;`make kb-reset` 现有实现走脚本/API,不要用 `docker exec -i dianxiaoer-mysql`。Lite 对数据目录进程级独占:CLI 建库前先停掉 8001/8002 的服务进程,或改走作业 API。

- [ ] **Step 4: 校准评估集 expect_section / expect_points**

跑 `PYTHONPATH=. uv run python scripts/validate_eval_ch04.py`,确保每条 A/B/C/E 的 `expect_section` 能匹配真实 chunk 的 `section_path`、`expect_points` 能在正确 chunk 原文里去空白子串命中;B 桶型号词用真实存在的型号(易混族里各自的正确规格)。改后重新提交评估集。

- [ ] **Step 5: 跑四策略报告(验收1)**
```bash
make eval-rag
```
Expected: 打印三段(检索 Recall@5/MRR + 证据覆盖度 + 生成四策略答案覆盖度/Faithfulness/拒答率),并落 `data/ch04/reports/rag_eval.{txt,json}`。**验收1 达成**:数字能跑出且四策略拉开差距——hybrid_rerank 总体 MRR 最高;BM25 在 B 桶型号题优于纯 vector、在 C 口语桶显著偏弱(验收2 数据侧佐证);证据/答案覆盖度随检索质量同向变化。

- [ ] **Step 6: BM25 命中型号实证(验收2)**
```bash
PYTHONPATH=. uv run python -c "
import asyncio; from app.core import retrieval
async def m():
    hits=await retrieval.search_knowledge('智能猫砂盆Pro 型号', strategy='bm25', top_k=5)
    for h in hits: print(h['id'], h['section_path'], h['question'])
asyncio.run(m())
"
```
Expected: 纯 BM25 路命中含该型号的 chunk(验收2 CLI 侧)。

- [ ] **Step 7: 提交**
```bash
git add data/kb Makefile tests/data/eval_ch04.jsonl scripts/validate_eval_ch04.py
git commit -m "feat(ch04): KB 补齐评估要点 + 真实建库 + 四策略报告跑通(验收1/2)"
```

---

## Task 15: 前端可点引用 + 每段回答满意度反馈(Vibe Coding,不套 TDD/review)

**Files:**
- Modify: `app/static/index.html`

**目标效果(用户醒来可微调):** 助手回答里的 `[n]` 渲染成可点角标;点击弹出浮层/侧栏,显示该引用的 `section_path`(章节路径)+ 来源 chunk 原文(question/answer)。数据来自 SSE `citations` 事件。

- [ ] **Step 1: 读现有 index.html 的 SSE 消费与消息渲染结构**,定位:①SSE `event` 分支处理;②assistant 消息 DOM 渲染函数。

- [ ] **Step 2: 接收 citations 事件**:在 SSE 解析里,遇 `{"event":"citations","items":[...]}` 把 items 存到当前 assistant 消息的 `citations` 数据上。

- [ ] **Step 3: 渲染可点角标**:assistant 文本渲染时,用正则把 `[n]` 替换成 `<sup class="cite" data-n="n">[n]</sup>`;点击时按 data-n 找 citations 项,弹浮层显示 `section_path` + question + answer。

- [ ] **Step 4: 样式**:角标可点(蓝色/hover);浮层简洁(标题=section_path,正文=原文,点外部关闭)。

- [ ] **Step 5: 每段回答满意度反馈(👍/👎)**:有效回答流式结束后,在气泡左下角挂一组反馈按钮(`.feedback` 容器 + 两个 `.fb-btn`);点击即点亮所选(`.active`)、另一个淡出(`.dim`)、显示「已反馈」并锁定两键(一次性)。纯前端交互、不落库(反馈入问题池 `source=user_feedback` 归后续数据飞轮章);样式沿用聊天页像素方块风(border + box-shadow + 按下位移)。挂载点:
```js
// index.html send():有效回答末尾挂反馈条
if (textStarted) addFeedbackBar(bubble);
```

- [ ] **Step 6: 手动浏览器自测**(chrome-devtools):①问「邮费是多少」→ 答案带 `[1]` → 点击 → 浮层显示运费政策 section + 原文;②同一条回答左下角有 👍/👎 → 点击 👍 → 点亮 + 显示「已反馈」+ 两键锁定。截图留 `dev-notes/ch04-citation-click.png`、`dev-notes/ch04-feedback.png`。

- [ ] **Step 7: 提交**
```bash
git add app/static/index.html
git commit -m "feat(ch04): 聊天页可点引用角标 + 来源浮层 + 每段回答满意度反馈(Vibe Coding 首版)"
```

---

## Task 16: RAG 评估页 + 挂进后台外壳

**Files:**
- Create: `app/api/rageval.py`、`app/static/rageval.html`
- Modify: `app/main.py`(include_router + `/rag-eval` 页面路由)、`app/core/jobs.py`(注册表加 `eval-rag`)、`app/static/admin.js`(导航加「RAG 评估」)、`app/api/admin.py`(首页多一张卡)
- Test: `tests/test_rageval_api.py`

**流程例外:** API 走 TDD,页面走 Vibe Coding——先出第一版,起服务截图给用户看,用户描述效果直接改。

**Interfaces:**
- Consumes: `data/ch04/reports/rag_eval.json`(Task 13 的产物)+ `app/core/jobs.status("eval-rag")`
- Produces: `GET /api/rag-eval/overview` → `{present, meta, retrieval, evidence_coverage, generation, generation_done, best, job}`

- [ ] **Step 1: 写失败测试**

`tests/test_rageval_api.py` 四条口径:产物缺失回 200 + `present=false` + 该按哪个作业(不是 500、不是白屏);产物在时 `retrieval` / `evidence_coverage` 与文件里一字不差(API 不重算);`best` 取总体 MRR 最高那一路(把 vector 的分改高,结论跟着换);`generation` 为 null 时检索段照样端出去、只标生成段未完成;产物写坏成半个 json 按「没跑过」处理。

- [ ] **Step 2: 写 `app/api/rageval.py`**

只读那份 json,一个指标都不在 web 层重算——页面上的数与终端 `make eval-rag` 跑出来的必须是同一份。`best` 这类结论性派生量算一次就够,页面的 KPI 与读图句都指着它。

- [ ] **Step 3: 注册作业 + 挂进外壳**

`JobSpec("eval-rag", "RAG 评估(四策略对照)", ("make", "eval-rag"), "需 Milvus + 已建库 + 上游可用,分钟级", heavy=True)`。分钟级重活标 heavy,页面二次确认才发起。导航注册表加一行,首页加一张卡(最佳 MRR、评估集题数、库外拒答率)。

- [ ] **Step 4: 写 `app/static/rageval.html`(Vibe Coding 首版)**

配色与硬阴影沿用后台那套(`acceptance.css`),取数、提示、重跑按钮与日志窗口用 `acceptance.js` 的公用件,导航用 `admin.js` 的 `mountAdminNav('/rag-eval')`。页面构成见 spec §4.1 的六块表。柱状图、忠实度横条、拒答率圆环都是纯 HTML/CSS/SVG,不引图表库。

- [ ] **Step 5: 在页面上按一次「重跑 RAG 评估」**

Expected: 日志窗口逐行回显 `make eval-rag` 的输出,跑完页面自动重新取数,报告头的时间换成本轮。

- [ ] **Step 6: 【检查点·Vibe 循环】按用户描述改到满意为止**

- [ ] **Step 7: Commit**
```bash
git add app/api/rageval.py app/static/rageval.html app/main.py app/core/jobs.py app/static/admin.js app/api/admin.py tests/test_rageval_api.py
git commit -m "feat(ch04): RAG 评估页——四策略对照的报告在后台看,也在后台重跑"
```

---

## Task 17: 端到端验收(浏览器 + CLI)+ 全量回归

- [ ] **Step 1: 全量单测**(先停占库的服务进程,释放 Lite 锁)
```bash
PYTHONPATH=. uv run pytest -v
```
Expected: 全绿(ch01-03 旧测试 + ch04 新测试)。ch03 的 milvus 旧测试若因新 schema/COSINE 语义失败,按新语义修正(记录)。

- [ ] **Step 2: 起服务浏览器验收**
```bash
make dev   # 8002 应用(先停掉占用 Lite 锁的旧进程)
```
- **验收3**(引用可点):问「邮费是多少」→ 答案带引用角标 → 点击看到来源 section_path + 原文。截图 `dev-notes/ch04-acceptance3.png`。
- **验收4**(拒答+落池):问「你们卖火星车吗」→ 明确拒答 → 查库:
```bash
docker exec -i dianxiaoer-mysql mysql -uroot -proot dianxiaoer -e "SELECT id,source,LEFT(raw_question,20),LEFT(reason,30) FROM low_confidence_questions ORDER BY id DESC LIMIT 5;"
```
Expected: 有新记录,source ∈ {retrieval_low_conf, self_check}。截图 `dev-notes/ch04-acceptance4.png`。
- **验收5**(满意度反馈):任一回答左下角有 👍/👎 → 点击其一 → 点亮所选 + 显示「已反馈」+ 两键锁定。截图 `dev-notes/ch04-acceptance5.png`。

- [ ] **Step 3: 验收1/2 复跑留档**:验收1 直接在 `/rag-eval` 页面上按「重跑 RAG 评估」,四策略报告跑完就在页面上;验收2 的 BM25 型号命中走 CLI。报告文本贴进 dev-notes。

- [ ] **Step 4: dev-notes 追记阶段(任务执行 + 验收)**,提交。

---

## Task 18: 评估集校准 + Recall@5 口径 + 跨文档桶

**数据任务:** Task 12 已直接生成 300 题五桶(含 E_multi);本任务只做建库后的 ground-truth 校准与口径收口。

- [ ] **Step 1: 校准 ground-truth**:建库后核对每条 `expect_section` 在真实 `section_path` 里至少命中一个别名;`expect_points` 必须逐字(去空白后)出现在目标小节正文里;不符的按真实文档修正评估集(或补 KB,见 Task 14)。
- [ ] **Step 2: 口径**:`scripts/eval_ch04.py` 里 `RECALL_K = 5`(检索深度 `K = 10` 不变,两者分开配);跨文档题 Recall 按组给部分分(凑齐才 1.00),MRR 按每块证据各自名次的倒数取平均、漏的记 0——**不用「最后一块到位」的名次算**,那样两块证据的题上限只有 0.5,和单证据桶放同一张图会被读成检索变差。
- [ ] **Step 3: 自检**:`scripts/validate_eval_ch04.py`(`make eval-check`)守四件事:id/query 不重复、五桶题数一致;每组 `expect_section` 在库里至少命中一个小节;`expect_points` 必须逐字(去空白后)出现在目标小节正文里;D 桶不带 ground truth。
- [ ] **Step 4: 页面跟着改口径**:`app/static/rageval.html` 的指标切换 `Recall@5`、桶列表加 `E_multi`。(本仓库没有 `app/api/observability.py`,Spec 里那句趋势指标兼容不适用,跳过。)

Expected: `make eval-check` 0 错;`make eval-rag` 报告与页面上都是五桶 + Recall@5。

---

## Task 19: 编造个案台账(跨轮累计 + 处置状态 + 证据快照)

- [ ] **Step 1: 建表**:`faith_cases` 并入 `sql/ch04-ddl.sql`(一章一份 DDL);`app/db/models.py` 加 `FaithCase`,`tests/conftest.py` 的清表清单加 `faith_cases`。
- [ ] **Step 2: 仓储**:`repository.upsert_faith_case`(按 `eval_id` 一题一行:已存在则更新正文/理由/角标快照并 `seen_count+1`;**已处置过的再判出来退回「未解决」并返回复发标记**)、`list_faith_cases(status, page, size)`(未解决排前,带三状态计数)、`set_faith_case_status`。
- [ ] **Step 3: API**:`GET /api/rag-eval/faith-cases`(分页 + 状态筛选)、`POST /api/rag-eval/faith-cases/{id}/status`(Literal 校验状态,挡脏值)。
- [ ] **Step 4: 评估脚本落台账**:个案里带上 `citations`——这一轮喂给模型的 Top-K 证据全集快照(问法/正文/`section_path`/chunk id),顺序与 evidence 拼接一致,答案里的 `[n]` 即其序号;写库失败只打一行日志,**不影响本轮报告**(台账是附加的管理视图)。
- [ ] **Step 5: 页面**:`/rag-eval` 加「编造个案台账」块——卡片沿用「编造个案」那一套 `.fcase` 结构(同一种东西长一个样),补状态徽标、「已解决 / 无需解决 / 退回未解决」按钮、证据原文折叠块(标出答案引用了哪几条)、状态页签与分页。
- [ ] **Step 6: 处置说明**:`resolution` 列 + API 校验(标已解决/无需解决时必填,空白也算没填 → 400;退回未解决时连说明与处置时间一起清空);页面点「已解决 / 无需解决」先在卡片里展开一行输入,填了才提交,卡片上把说明显示成「怎么解决的 / 为什么不用改」。
- [ ] **Step 7: 测试**:`tests/test_faith_cases_api.py` 覆盖一题一行、复发退回、分页与状态筛选、证据快照如实带出、处置必须写说明、退回清说明、非法状态 422。
- [ ] **Step 8: 幻觉率只算本轮**:分子取报告 `generation.faithfulness_cases` 的题号,处置状态用 `repository.faith_case_status_map()` 回查;台账累计走 `hallucination.ledger` 单独报。**别拿台账总条数当分子**——第一轮台账恰好等于那一轮的个案时看不出问题,改掉几条之后旧账会摊到新一轮头上(实测虚高成 2.33%,本轮实际 0.33%)。补一条测试:台账 3 条、本轮只判出 1 条时,判出率是 1/300。

Expected: 全量单测绿;页面上点按钮状态即时流转,重跑评估后台账累加而不是被覆盖。

---

## Task 20: 裁判整改(尺子去抖 + 口径写清 + 台账反过来当回归集)

- [ ] **Step 1: 回归工具**:`scripts/judge_check.py` + `make judge-check`——读台账里已处置且有 `citations` 快照的个案,用快照原样重放(不重检索、不重生成,只换裁判),人工处置翻译成标准答案(已解决→该判编造,无需解决→该判忠实,未解决跳过),输出逐条对照与一致率;不一致时退出码 1。上游偶发回空 completion(结构化解析成 `None`)时重试一次,仍失败记「调用失败」且不计入一致。
- [ ] **Step 2: 尺子去抖**:`get_chat_model` / `llm.structured` 加 `temperature` 参数(默认仍 0.3);评估里两个裁判与 `judge_check` 显式传 `temperature=0`。依据:同一份输入在 0.3 下重放四次得到 5/6、5/6、6/6、6/6,且挂的不是同一条。
- [ ] **Step 3: 口径写清**:`FAITHFULNESS_SYSTEM` 的豁免从三类补到七类(+跨证据合并 / 不确定语气的提醒 / 无害安全提示 / 通用常识注解),并反向点名两类真编造(凭空数值、条件张冠李戴),给出判数字的两问次序。
- [ ] **Step 4: 边界样例**:把六条真实判例写成 few-shot 附在提示词末尾(只加规则不加样例的那一版会把 A50 判反)。
- [ ] **Step 5: 测试**:`tests/test_judge_check.py`(证据拼回格式逐字一致、老快照缺 `n` 时按序补、处置→标准答案映射、能考个案的筛选、调用失败不算一致)+ `tests/test_llm_structured.py` 补 temperature 透传与默认值。
- [ ] **Step 6: 验证**:`make judge-check` 连跑四次 6/6;再做留一验证(逐条抽掉它自己那行样例)得 5/6,记录 E17 闭卷翻车这一事实。

Expected: 全量单测绿;`make judge-check` 6/6;**下一轮 `make eval-rag` 的判出率应朝确认幻觉率靠拢,且 C2/E15 这类仍被判出**——那才是这条整改的真验收。

---

## Self-Review(写完计划的自查)

**Spec 覆盖:**
- §1.1 沿用 Lite(milvus-lite 3.2.2 支持 BM25) → Task 1(jieba + 冒烟)
- §1.2 新 schema(BM25 Function/analyzer/双向量/标量)→ Task 3
- §1.3 在线管线 ①理解②过滤③混合④重排⑤⑥两闸⑦组装⑧回传 → Task 6/7/10/11
- §2 组件表(milvus/rerank/retrieval/query理解)→ Task 3/5/7/6
- §3 引用生成 / 生成前自评 / 检索证据低 / 负面知识 / 落池 → Task 8/9/10/11
- §4 评估(四策略:检索 Recall@5/MRR + 证据覆盖度 + 生成答案覆盖度/Faithfulness/拒答率 + 分桶 + 报告产物)→ Task 12/13/14,口径与扩集见 Task 18;§4.1 RAG 评估页(只读产物 + 就地重跑 + 后台外壳)→ Task 16,编造个案台账 → Task 19,裁判口径与回归 → Task 20
- §5 前端可点引用 → Task 15
- §6 冒烟 / TDD / 标注验证 / 验收映射 → Task 1(冒烟)、各 TDD 任务、Task 6/8/12(样例)、Task 14/17(验收)
- §7 数据模型 low_confidence_questions → Task 9;faith_cases → Task 19

**占位符扫描:** 无 TBD/TODO;每步含真实代码或真实命令。评估集内容 Task 12 给出模板 + Task 14 用真实文档校准(非占位,是数据任务的正常两阶段)。

**类型一致性:** `search_knowledge(query, strategy, top_k, category, client, collection)` 跨 Task 7/10/13 一致;hit dict 字段(id/score/question/answer/section_path/content_type/category,+rerank_score)跨 Task 3/7/10 一致;query_faq 返回(sufficient/evidence/citations/source/reason)跨 Task 10/11/13 一致;`insert_low_confidence(conversation_id, raw_question, source, reason)` 跨 Task 9/11 一致;citations item(n/id/section_path/question/answer/content_type)跨 Task 10/11/15 一致。

**已知风险(执行时留意):**
- Milvus Lite COSINE `distance` 语义已在 Task 1 实测:返回相似度本身(identical=1.0),保持 `score=distance`;换版本后重验。
- rerank provider 适配(jina_ai vs cohere)→ Task 1 冒烟拍死。
- ch03 旧 milvus 测试(Lite/COSINE)可能需随 Task 3/16 修正。
- 评估集 expect_section 必须对齐真实文档 → Task 14 Step 3 强制校准。