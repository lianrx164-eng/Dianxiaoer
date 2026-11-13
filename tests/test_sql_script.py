from app.db.sql_script import split_statements


def test_splits_and_strips_comment_lines():
    # 注释里的半角分号不能把语句劈开,注释行本身也不进结果
    text = "-- 表注释:数据源;ch03 起升级\nCREATE TABLE a (id INT);\n-- b\nCREATE TABLE b (id INT);\n"
    assert split_statements(text) == ["CREATE TABLE a (id INT)", "CREATE TABLE b (id INT)"]


def test_ignores_empty_fragments_and_comment_only_scripts():
    assert split_statements(";;\n-- only a comment\n;") == []
