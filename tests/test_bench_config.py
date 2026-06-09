from faraday.bench.config import CSV_COLUMNS, Cell, cells


def test_cells_is_the_full_18_cell_matrix():
    all_cells = cells()
    assert len(all_cells) == 18                      # 3 sizes x 6 quants
    assert all_cells[0] == Cell("0.5B", "Q8_0")      # smallest model first
    assert len({c.key for c in all_cells}) == 18     # all distinct


def test_cell_repo_and_filename_follow_bartowski_naming():
    c = Cell("1.5B", "Q4_K_M")
    assert c.repo == "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
    assert c.filename == "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"


def test_csv_schema_is_stable():
    assert CSV_COLUMNS == (
        "size", "quant", "status", "disk_bytes", "peak_rss_bytes",
        "prefill_tps", "decode_tps", "perplexity", "notes",
    )
