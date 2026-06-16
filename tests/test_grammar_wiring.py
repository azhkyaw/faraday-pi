from faraday.config import Settings
from faraday.grammar import build_citation_grammar
from faraday.server import make_engine


def test_settings_reads_use_grammar_env(monkeypatch):
    monkeypatch.setenv("FARADAY_USE_GRAMMAR", "1")
    assert Settings.from_env().use_grammar is True
    monkeypatch.delenv("FARADAY_USE_GRAMMAR")
    assert Settings.from_env().use_grammar is False


def test_make_engine_wires_grammar_builder(tmp_path):
    on = Settings(db_path=str(tmp_path / "a.sqlite"), use_grammar=True)
    engine, store = make_engine(on)
    assert engine.grammar_builder is build_citation_grammar
    store.close()
    off = Settings(db_path=str(tmp_path / "b.sqlite"))
    engine, store = make_engine(off)
    assert engine.grammar_builder is None
    store.close()
