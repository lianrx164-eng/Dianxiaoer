from dataclasses import dataclass

from app.kb import dedup


@dataclass
class Item:
    question: str


def test_normalize_strips_punct_keeps_chinese():
    assert dedup.normalize_question(" 邮费,是多少? ") == "邮费是多少"


def test_dedupe_within_batch_and_against_existing():
    items = [Item("邮费是多少"), Item("邮费是多少?"), Item("怎么退货"), Item("运费怎么算")]
    kept, discarded = dedup.dedupe(items, existing_questions=["运费怎么算"])
    kept_q = [i.question for i in kept]
    assert "邮费是多少" in kept_q          # 首次出现保留
    assert "怎么退货" in kept_q
    assert "运费怎么算" not in kept_q       # 与库内已有重复 → 丢弃
    assert len(kept) == 2 and len(discarded) == 2


@dataclass
class _Chunk:
    questions: str
    answer: str


def test_filter_new_chunks_skips_existing_and_batch_duplicates():
    items = [_Chunk("q", "a"), _Chunk("q", "a2"), _Chunk("q", "a2")]
    fresh = dedup.filter_new_chunks(items, [("q", "a")])
    assert fresh == [items[1]]  # 库内已存在的跳过;批内第二个 a2 也跳过
    assert dedup.filter_new_chunks([_Chunk("x", "y")], []) == [_Chunk("x", "y")]
