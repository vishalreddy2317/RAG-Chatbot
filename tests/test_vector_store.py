import numpy as np
import pytest

from rag.chunking import Chunk
from rag.vector_store import FaissVectorStore


def _chunk(i: int, doc: str = "a.pdf", page: int = 1) -> Chunk:
    return Chunk(f"{doc}::p{page}::c{i}", doc, page, f"chunk {i}", i, 0, 7)


def _unit(dim: int, hot: int) -> np.ndarray:
    v = np.zeros(dim, dtype="float32")
    v[hot] = 1.0
    return v


def test_add_and_search_returns_nearest_first():
    store = FaissVectorStore(4)
    chunks = [_chunk(0), _chunk(1), _chunk(2)]
    store.add(chunks, np.stack([_unit(4, 0), _unit(4, 1), _unit(4, 2)]))
    assert len(store) == 3
    results = store.search(_unit(4, 1), top_k=2)
    assert results[0].chunk.chunk_id == chunks[1].chunk_id
    assert results[0].score == pytest.approx(1.0)
    assert results[0].rank == 1 and results[1].rank == 2


def test_search_on_empty_store_returns_nothing():
    store = FaissVectorStore(4)
    assert store.search(_unit(4, 0), top_k=3) == []


def test_dimension_mismatch_raises():
    store = FaissVectorStore(4)
    with pytest.raises(ValueError):
        store.add([_chunk(0)], np.zeros((1, 5), dtype="float32"))
    with pytest.raises(ValueError):
        store.add([_chunk(0), _chunk(1)], np.zeros((1, 4), dtype="float32"))


def test_document_filter_and_stats():
    store = FaissVectorStore(4)
    chunks = [_chunk(0, "a.pdf", 1), _chunk(1, "a.pdf", 2), _chunk(2, "b.pdf", 1)]
    store.add(chunks, np.stack([_unit(4, 0), _unit(4, 1), _unit(4, 0)]))
    only_b = store.search(_unit(4, 0), top_k=3, doc_names=["b.pdf"])
    assert [r.chunk.doc_name for r in only_b] == ["b.pdf"]
    assert store.document_names() == ["a.pdf", "b.pdf"]
    assert store.document_stats() == {"a.pdf": {"chunks": 2, "pages": 2}, "b.pdf": {"chunks": 1, "pages": 1}}


def test_remove_document_rebuilds_index():
    store = FaissVectorStore(4)
    chunks = [_chunk(0, "a.pdf"), _chunk(1, "b.pdf"), _chunk(2, "a.pdf")]
    store.add(chunks, np.stack([_unit(4, 0), _unit(4, 1), _unit(4, 2)]))
    assert store.remove_document("a.pdf") == 2
    assert len(store) == 1
    assert store.document_names() == ["b.pdf"]
    assert store.search(_unit(4, 0), top_k=3)[0].chunk.doc_name == "b.pdf"
    assert store.remove_document("missing.pdf") == 0
    store.clear()
    assert store.is_empty()


def test_save_and_load_round_trip(tmp_path):
    store = FaissVectorStore(4)
    store.add([_chunk(0), _chunk(1)], np.stack([_unit(4, 0), _unit(4, 3)]))
    store.save(tmp_path / "index")
    loaded = FaissVectorStore.load(tmp_path / "index")
    assert len(loaded) == 2
    assert loaded.search(_unit(4, 3), top_k=1)[0].chunk.chunk_id == "a.pdf::p1::c1"
