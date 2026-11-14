"""SQL 脚本切分:给 DDL/seed 灌库用。

顺序不能反:先去掉 `--` 行注释,再按 `;` 切分——注释里含半角分号时
(如「…数据源;ch03 起…」),先切分会把注释劈成两半,第二半失去 `--` 前缀。
字符串字面量里的 `;`(如 COMMENT 文案)不能被当成分隔符,故切分时跟踪引号状态。
"""

_QUOTES = {"'", '"', "`"}


def split_statements(text: str) -> list[str]:
    body = "\n".join(
        ln for ln in text.splitlines() if not ln.strip().startswith("--")
    )
    chunks: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    escaped = False
    for ch in body:
        if quote is not None:
            buf.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in _QUOTES:
            quote = ch
            buf.append(ch)
        elif ch == ";":
            s = "".join(buf).strip()
            if s:
                chunks.append(s)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        chunks.append(tail)
    return chunks
