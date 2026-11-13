from app.core.judges import (
    expected_faithful, judge_model, rebuild_evidence, summarize_check,
)


def test_rebuild_evidence_matches_eval_format():
    cits = [{"n": 1, "question": "运费", "answer": "满99包邮"},
            {"n": 2, "question": "退款", "answer": "原路退回"}]
    assert rebuild_evidence(cits) == "[1] 运费: 满99包邮\n[2] 退款: 原路退回"


def test_rebuild_evidence_fills_missing_n():
    cits = [{"question": "a", "answer": "A"}, {"question": "b", "answer": "B"}]
    assert rebuild_evidence(cits) == "[1] a: A\n[2] b: B"


def test_disposition_maps_to_expected_verdict():
    assert expected_faithful("已解决") is False
    assert expected_faithful("无需解决") is True
    assert expected_faithful("未解决") is None


def test_failed_calls_are_not_counted():
    rows = [
        {"id": "A1", "expected": False, "verdict": False, "error": None},
        {"id": "A2", "expected": True, "verdict": True, "error": None},
        {"id": "A3", "expected": True, "verdict": None, "error": "timeout"},
    ]
    s = summarize_check(rows)
    assert s["judged"] == 2 and s["agree"] == 2 and s["failed"] == 1
    assert s["agreement"] == 1.0


def test_judge_model_runs_at_temperature_zero():
    assert judge_model().temperature == 0
