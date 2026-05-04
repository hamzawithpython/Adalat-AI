import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.chunker import clean_text, _sliding_window, _structural_split


def test_clean_text():
    raw = "Hello   World\n\n\n\nTest"
    result = clean_text(raw)
    assert "   " not in result


def test_sliding_window():
    text = "A" * 2000
    chunks = _sliding_window(text, size=800, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)


def test_structural_split_pk():
    text = "Article 9 Right to life\nNo person shall be deprived.\nArticle 10 Rights of arrested"
    patterns = [r"(Article\s+\d+[\w\-]*\.?)"]
    parts = _structural_split(text, patterns)
    assert len(parts) >= 1


def test_structural_split_empty():
    result = _structural_split("", [])
    assert result == [""]