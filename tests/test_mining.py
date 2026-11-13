from app.kb import mining


async def _value(v):
    return v


async def test_mine_uses_per_batch_number_and_real_source(monkeypatch):
    sources = [("conv:1", "u1"), ("conv:2", "u2"), ("conv:3", "u3")]
    monkeypatch.setattr(mining, "_load_conversation_texts", lambda: _value(sources))

    async def fake_extract(batch, model=None):
        return [
            mining.QaPair(question=f"q-{ref}", answer="a", source_ref=ref)
            for ref, _ in batch
        ]

    monkeypatch.setattr(mining, "extract_qa_with_sources", fake_extract)

    inserted = []

    async def fake_insert(batch_no, source_ref, question, answer):
        inserted.append((batch_no, source_ref, question))
        return len(inserted)

    monkeypatch.setattr(mining.repository, "insert_staging", fake_insert)
    monkeypatch.setattr(mining.repository, "list_staging_by_status", lambda status: _value([]))
    monkeypatch.setattr(mining.repository, "list_all_questions", lambda: _value([]))
    monkeypatch.setattr(mining.repository, "set_staging_status", lambda ids, status: _value(None))

    stats = await mining.mine(batch_size=2)

    assert stats["sources"] == 3
    nos = [row[0] for row in inserted]
    refs = [row[1] for row in inserted]
    assert nos[0] == nos[1] and nos[2] != nos[0]      # 两个批次号不同
    assert refs == ["conv:1", "conv:2", "conv:3"]      # 每条问答对记自己的来源


async def test_mine_drops_unknown_source_ref(monkeypatch):
    sources = [("conv:1", "u1")]
    monkeypatch.setattr(mining, "_load_conversation_texts", lambda: _value(sources))

    async def fake_extract(batch, model=None):
        return [mining.QaPair(question="q", answer="a", source_ref="bogus")]

    monkeypatch.setattr(mining, "extract_qa_with_sources", fake_extract)

    inserted = []

    async def fake_insert(batch_no, source_ref, question, answer):
        inserted.append((batch_no, source_ref))
        return 1

    monkeypatch.setattr(mining.repository, "insert_staging", fake_insert)
    monkeypatch.setattr(mining.repository, "list_staging_by_status", lambda status: _value([]))
    monkeypatch.setattr(mining.repository, "list_all_questions", lambda: _value([]))
    monkeypatch.setattr(mining.repository, "set_staging_status", lambda ids, status: _value(None))

    await mining.mine(batch_size=2)
    assert inserted[0][1] is None  # 模型编造的来源不落库
