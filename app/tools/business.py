"""mock 数据源(纯函数),工具实现见 app/tools/builtin/。"""
import random


def order_snapshot(order_id: str) -> dict:
    """订单快照(纯函数,随机种子固定 → 同 order_id 稳定)。
    query_order 工具与退款子流程的 fetch_order 节点同源,避免两处数据对不上。"""
    rng = random.Random(f"order:{order_id}")
    return {
        "order_id": order_id,
        "status": rng.choice(["待付款", "已付款", "已发货", "已签收"]),
        "amount": rng.randint(50, 2000),
        "created_at": f"2026-07-{rng.randint(1, 12):02d} 10:00",
        "product": rng.choice(["智能猫砂盆", "猫粮 5kg", "猫爬架", "自动饮水机"]),
        "tracking_no": f"SF{rng.randint(10**9, 10**10 - 1)}",
    }


def list_user_orders(user_id: str) -> list[dict]:
    """按 user_id 稳定列出该用户的 2-4 笔订单(mock,不落库)。每笔与 order_snapshot 同源,
    前端选中后回填 order_id 即可 query_order。"""
    rng = random.Random(f"user_orders:{user_id}")
    n = rng.randint(2, 4)
    ids: list[str] = []
    while len(ids) < n:
        oid = str(rng.randint(1000, 9999))
        if oid not in ids:
            ids.append(oid)
    out = []
    for oid in ids:
        s = order_snapshot(oid)
        out.append({"order_id": oid, "product": s["product"],
                    "status": s["status"], "amount": s["amount"]})
    return out
