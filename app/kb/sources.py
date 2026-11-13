import pathlib

KB_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "kb"

# 建库材料清单:文件 → content_type。CLI 与录入页同读这一份。
SOURCE_TYPES: dict[str, str] = {
    "product-faq.md": "faq",
    "returns-policy.md": "policy",
    "after-sales-manual.md": "manual",
    "product-specs.md": "spec",
}

CONTENT_TYPES = ("faq", "policy", "manual", "spec", "mined")
