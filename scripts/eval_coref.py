"""指代消解标注评估:多轮带指代补全 + 已完整透传。需上游可用。
用法:PYTHONUTF8=1 uv run python -m scripts.eval_coref
"""
import asyncio

from app.core.coref import resolve

# (history, query, 期望类型, 补全后必须出现的实体;透传类 keyword 为 None)
CASES = [
    ("用户:蓝牙耳机什么时候到货\n客服:预计明天送达", "这个能退吗", "rewrite", "蓝牙耳机"),
    ("用户:订单1001买的猫粮\n客服:已发货", "它到哪了", "rewrite", "1001"),
    ("", "蓝牙耳机的保修期是多久", "passthrough", None),
    ("", "退货运费谁承担", "passthrough", None),
    ("", "我要退款", "passthrough", None),
    ("", "我要退货", "passthrough", None),
]


async def main():
    passed = 0
    for hist, q, kind, keyword in CASES:
        got = await resolve(q, hist)
        changed = got != q
        if kind == "rewrite":
            ok = changed and (keyword is None or keyword in got)
        else:
            ok = not changed
        passed += ok
        print(f"{'✅' if ok else '❌'} {q!r} -> {got!r} 期望={kind}"
              f"{'' if keyword is None else f'(含 {keyword})'}")

    print(f"\n通过 {passed}/{len(CASES)}(模型非确定性;透传类必须不改,补全类应含上文实体)")


if __name__ == "__main__":
    asyncio.run(main())
