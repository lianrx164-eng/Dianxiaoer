from app.tools.business import query_logistics, query_order, query_product


async def test_query_order_deterministic_and_shaped():
    r1 = await query_order.ainvoke({"order_id": "1001"})
    r2 = await query_order.ainvoke({"order_id": "1001"})
    assert r1 == r2                                  # 同种子可复现
    assert r1["order_id"] == "1001"
    assert r1["status"] in {"待付款", "已付款", "已发货", "已签收"}
    assert r1["tracking_no"].startswith("SF")        # 物流单号只能从订单里拿到


async def test_query_logistics_takes_tracking_no_only():
    props = query_logistics.tool_call_schema.model_json_schema()["properties"]
    assert "tracking_no" in props and "order_id" not in props


async def test_query_product_and_logistics_shapes():
    assert query_product.name == "query_product"
    assert query_logistics.name == "query_logistics"
    p = await query_product.ainvoke({"product_name": "猫粮"})
    assert p["product_name"] == "猫粮" and "price" in p
    lg = await query_logistics.ainvoke({"tracking_no": "SF123456789"})
    assert lg["tracking_no"] == "SF123456789"
    assert "status" in lg and "timeline" in lg
