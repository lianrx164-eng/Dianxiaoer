"""测试不依赖仓库根的 .env:给缺省的 dummy 值,真实值只在运行时需要。"""

import os

os.environ.setdefault("CHAT_MODEL", "test-model")
os.environ.setdefault("CHAT_BASE_URL", "http://localhost:9/v1")
os.environ.setdefault("CHAT_API_KEY", "test-key")
