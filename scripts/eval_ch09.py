"""ch09 验收 2/3/4 端到端:缺知识问题 → 兜底落池 → 飞轮入队 → 审核通过写回 → 再问答对;
以及聊天 👎 落池后进入待审队列。
运行:make eval-ch09(需服务在跑、mysql + 上游可用;知识库已建)
"""
import argparse
import asyncio
import subprocess
import sys

import httpx

from app.db import repository

MISSING_Q = "有没有以旧换新的活动"
RATED_Q = "退货政策是什么样的"
APPROVED_ANSWER = "目前平台暂不支持以旧换新,后续活动请关注商品页公告。"
EXPECT_KEYWORD = "以旧换新"


def ok(name: str, cond: bool, detail: str = "") -> bool:
    print(("✓ " if cond else "✗ ") + name + (f"  {detail}" if detail else ""))
    return cond


async def ask(client: httpx.AsyncClient, q: str, cid: int | None = None) -> dict:
    r = await client.post("/api/agent", json={
        "user_id": "ch09-eval", "message": q, "conversation_id": cid})
    r.raise_for_status()
    return r.json()


def run_flywheel() -> None:
    subprocess.run([sys.executable, "-m", "scripts.flywheel_pipeline"], check=True)


async def find_review_for(question: str):
    for item in await repository.list_review_queue("待审"):
        detail = await repository.get_review_detail(item.id)
        if detail and any(r.raw_question == question for r in detail[1]):
            return detail
    return None


async def main(base_url: str, missing_q: str, approved_answer: str,
               expect_keyword: str) -> int:
    checks = []
    async with httpx.AsyncClient(base_url=base_url, timeout=180) as c:
        # --- 验收 2:缺知识 → 兜底 + 落池 + 快照 ---
        first = await ask(c, missing_q)
        state = first.get("trace", {})
        fallback = state.get("gate") in ("retrieval_low_conf", "self_check")
        checks.append(ok("验收2a 缺知识问题走兜底", fallback,
                         f"intent={state.get('intent')} gate={state.get('gate')}"))
        run_flywheel()
        detail = await find_review_for(missing_q)
        checks.append(ok("验收2b 飞轮把问题并入待审队列", detail is not None))
        if detail:
            item, raws = detail
            raw = next(r for r in raws if r.raw_question == missing_q)
            checks.append(ok("验收2c 详情带用户原话与召回快照",
                             bool(raw.retrieved_chunks),
                             f"source={raw.source} chunks={len(raw.retrieved_chunks or [])}"))

            # --- 验收 3:审核通过写回 → 再问答对 ---
            r = await c.post(f"/api/review/{item.id}/approve",
                             json={"approved_answer": approved_answer})
            checks.append(ok("验收3a 审核通过并写回知识库", r.status_code == 200,
                             r.text[:80]))
            second = await ask(c, missing_q)
            answered = expect_keyword in (second.get("answer") or "")
            checks.append(ok("验收3b 同一问题再问能答对", answered,
                             (second.get("answer") or "")[:60]))

        # --- 验收 4:👎 落池 → 飞轮入待审 ---
        rated = await ask(c, RATED_Q)
        fb = await c.post("/api/feedback", json={
            "conversation_id": rated["conversation_id"], "rating": "down",
            "question": RATED_Q})
        checks.append(ok("验收4a 👎 落池", fb.status_code == 200 and fb.json()["pooled"]))
        run_flywheel()
        feedback_detail = await find_review_for(RATED_Q)
        checks.append(ok("验收4b 👎 经飞轮进入待审队列", feedback_detail is not None))
        if feedback_detail:
            _, raws = feedback_detail
            raw = next(r for r in raws if r.raw_question == RATED_Q)
            checks.append(ok("验收4c 入口标 user_feedback", raw.source == "user_feedback",
                             f"chunks={len(raw.retrieved_chunks or [])}"))

    passed = sum(checks)
    print(f"\nch09 验收 2/3/4:{passed}/{len(checks)} 通过")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8002")
    # 一轮跑完会把答案写进知识库,想重复演示就换一条库里没有的新问题
    ap.add_argument("--missing-question", default=MISSING_Q)
    ap.add_argument("--approved-answer", default=APPROVED_ANSWER)
    ap.add_argument("--expect-keyword", default=EXPECT_KEYWORD)
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args.base_url, args.missing_question,
                              args.approved_answer, args.expect_keyword)))
