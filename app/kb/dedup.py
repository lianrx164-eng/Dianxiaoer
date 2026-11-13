import re

_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)  # \W 不含中文(中文属 \w),故只去空白/标点


def normalize_question(q: str) -> str:
    return _STRIP_RE.sub("", q.strip().lower())


def fingerprint(questions: str, answer: str) -> str:
    """查重指纹 = 归一化问法 + 归一化正文;同节多块问法相同、正文不同,不会误杀。"""
    return normalize_question(questions) + "|" + normalize_question(answer)


def filter_new_chunks(chunks: list, existing_pairs: list[tuple[str, str]]) -> list:
    """返回指纹未出现过的块;库内已有与批内重复都跳过。"""
    seen = {fingerprint(q, a) for q, a in existing_pairs}
    fresh = []
    for c in chunks:
        fp = fingerprint(c.questions, c.answer)
        if fp in seen:
            continue
        seen.add(fp)
        fresh.append(c)
    return fresh


def dedupe(items: list, existing_questions: list[str]) -> tuple[list, list]:
    seen = {normalize_question(q) for q in existing_questions}
    kept, discarded = [], []
    for item in items:
        key = normalize_question(item.question)
        if not key or key in seen:
            discarded.append(item)
        else:
            seen.add(key)
            kept.append(item)
    return kept, discarded
