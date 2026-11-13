"""SQL 脚本切分:给 DDL/seed 灌库用。

顺序不能反:先去掉 `--` 行注释,再按 `;` 切分——注释里含半角分号时
(如「…数据源;ch03 起…」),先切分会把注释劈成两半,第二半失去 `--` 前缀。
"""


def split_statements(text: str) -> list[str]:
    body = "\n".join(
        ln for ln in text.splitlines() if not ln.strip().startswith("--")
    )
    return [s.strip() for s in body.split(";") if s.strip()]
