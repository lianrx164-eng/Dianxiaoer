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
