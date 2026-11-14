"""ch08 MCP Server 集成测试:真起子进程 → adapters client 列工具 + 真调。
连不上视为环境问题直接 fail(两台 Server 是本章交付物,不 skip)。"""
import os
import pathlib
import socket
import subprocess
import sys
import time

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient


def _wait_port(port: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"port {port} not ready")


@pytest.fixture(scope="module")
def mcp_procs():
    env = {**os.environ, "PYTHONUTF8": "1"}
    p1 = subprocess.Popen([sys.executable, "mcp_servers/logistics_server.py"],
                          env={**env, "PORT": "18101"})
    p2 = subprocess.Popen([sys.executable, "mcp_servers/aftersales_server.py"],
                          env={**env, "PORT": "18102"})
    try:
        _wait_port(18101)
        _wait_port(18102)
        yield
    finally:
        for p in (p1, p2):
            p.terminate()
            p.wait(timeout=5)


def _client() -> MultiServerMCPClient:
    return MultiServerMCPClient({
        "logistics": {"transport": "streamable_http", "url": "http://127.0.0.1:18101/mcp"},
        "aftersales": {"transport": "streamable_http", "url": "http://127.0.0.1:18102/mcp"},
    }, handle_tool_errors=False)


async def test_list_tools_three_essentials(mcp_procs):
    tools = await _client().get_tools()
    by = {t.name: t for t in tools}
    assert set(by) == {"query_logistics", "query_warranty", "query_return_status"}
    for t in by.values():
        assert t.description
        schema = t.args_schema if isinstance(t.args_schema, dict) else t.args_schema.model_json_schema()
        assert schema.get("properties")


async def test_invoke_logistics_stable_mock(mcp_procs):
    tools = {t.name: t for t in await _client().get_tools(server_name="logistics")}
    r1 = await tools["query_logistics"].ainvoke({"tracking_no": "SF123"})
    r2 = await tools["query_logistics"].ainvoke({"tracking_no": "SF123"})

    def text(result):
        # adapters 0.3.x 返回 content block 列表,每块带一次性的 LangChain id;只比文本
        if isinstance(result, list):
            return "".join(b.get("text", "") for b in result if isinstance(b, dict))
        return str(result)

    assert text(r1) == text(r2) and "status_code" in text(r1)


def _start_server(path, port: int):
    env = {**os.environ, "PYTHONUTF8": "1", "PORT": str(port)}
    return subprocess.Popen([sys.executable, str(path)], env=env)


async def test_tool_added_after_server_restart_becomes_visible(tmp_path):
    """验收 3:Server 侧加工具、只重启 Server,同一个 client 下一次 get_tools 就能看到。"""
    src = pathlib.Path("mcp_servers/logistics_server.py").read_text(encoding="utf-8")
    marker = 'if __name__ == "__main__":'
    probe_src = src.replace(marker, (
        "\n\n@mcp.tool()\n"
        "async def ping_probe() -> dict:\n"
        '    """测试探针工具。"""\n'
        '    return {"ok": True}\n\n') + marker)
    probe = tmp_path / "logistics_with_probe.py"
    probe.write_text(probe_src, encoding="utf-8")

    port = 18301
    client = MultiServerMCPClient({
        "logistics": {"transport": "streamable_http",
                      "url": f"http://127.0.0.1:{port}/mcp"},
    }, handle_tool_errors=False)

    p1 = _start_server("mcp_servers/logistics_server.py", port)
    try:
        _wait_port(port)
        names = {t.name for t in await client.get_tools(server_name="logistics")}
        assert "ping_probe" not in names
    finally:
        p1.terminate()
        p1.wait(timeout=5)

    p2 = _start_server(probe, port)
    try:
        _wait_port(port)
        names = {t.name for t in await client.get_tools(server_name="logistics")}
        assert "ping_probe" in names
    finally:
        p2.terminate()
        p2.wait(timeout=5)
