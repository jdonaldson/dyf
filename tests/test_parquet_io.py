"""Round-trip and guard tests for dyf.parquet_io."""

from __future__ import annotations

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from dyf import LazyIndex, build_dyf_tree, write_lazy_index  # noqa: E402
from dyf.parquet_io import from_parquet, to_parquet  # noqa: E402


def _make_index(path, n=400, d=16, quantization="float32", seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, d)).astype(np.float32)
    tree = build_dyf_tree(emb, max_depth=4, num_bits=3, min_leaf_size=4, seed=42)
    write_lazy_index(
        tree,
        emb,
        str(path),
        quantization=quantization,
        stored_fields={
            "sku": [f"SKU-{i}" for i in range(n)],
            "spend": rng.normal(size=n).astype(np.float64),
            "cat": np.arange(n, dtype=np.int32),
        },
    )
    return emb


def test_export_preserves_vectors_and_leaf_order(tmp_path):
    emb = _make_index(tmp_path / "src.dyf")
    out = tmp_path / "out.parquet"
    report = to_parquet(str(tmp_path / "src.dyf"), str(out))

    assert report["rows"] == len(emb)
    assert report["embedding_dim"] == emb.shape[1]
    assert report["reconstructed"] is False

    table = pq.read_table(out)
    assert table.schema.field("embedding").type == pa.list_(pa.float32(), emb.shape[1])

    leaf = table.column("leaf_id").to_numpy()
    assert np.all(np.diff(leaf) >= 0), "rows must be written in leaf order for row-group pruning"

    # every row present exactly once, and vectors match the source at their own item_index
    item = table.column("item_index").to_numpy()
    assert sorted(item) == list(range(len(emb)))
    got = np.stack(table.column("embedding").to_pylist()).astype(np.float32)
    assert np.allclose(emb[item], got, atol=1e-6)


def test_round_trip_rebuilds_an_index_with_fields(tmp_path):
    emb = _make_index(tmp_path / "src.dyf")
    parquet = tmp_path / "out.parquet"
    to_parquet(str(tmp_path / "src.dyf"), str(parquet))

    report = from_parquet(str(parquet), str(tmp_path / "back.dyf"), max_depth=4, num_bits=3, quantization="float32")
    assert report["rows"] == len(emb)
    assert report["embedding_dim"] == emb.shape[1]
    assert report["skipped"] == []
    assert report["leaves"] >= 1

    with LazyIndex(str(tmp_path / "back.dyf")) as idx:
        # leaves reported by from_parquet must match the file, not the tree dict
        assert report["leaves"] == idx.num_leaves
        data = idx.extract_all_fields()

    assert set(data["fields"]) == {"sku", "spend", "cat", "source_item_index"}
    # source_item_index lets a row be tied back to the exporting index
    src_idx = np.asarray(data["fields"]["source_item_index"], dtype=np.int64)
    assert sorted(src_idx) == list(range(len(emb)))
    skus = list(data["fields"]["sku"])
    assert all(skus[i] == f"SKU-{src_idx[i]}" for i in range(len(emb)))


def test_item_index_is_renamed_not_dropped(tmp_path):
    """`item_index` is reserved by write_lazy_index, so it must survive under a new name."""
    _make_index(tmp_path / "src.dyf")
    parquet = tmp_path / "out.parquet"
    to_parquet(str(tmp_path / "src.dyf"), str(parquet))

    report = from_parquet(str(parquet), str(tmp_path / "a.dyf"), max_depth=3)
    assert "source_item_index" in report["fields"]
    assert "item_index" not in report["fields"]

    report = from_parquet(str(parquet), str(tmp_path / "b.dyf"), max_depth=3, preserve_item_index=False)
    assert "source_item_index" not in report["fields"]
    assert any(s["column"] == "item_index" for s in report["skipped"])


def test_export_only_columns_are_not_reimported(tmp_path):
    """A leaf_id from the exporting tree is meaningless against a freshly fitted one."""
    _make_index(tmp_path / "src.dyf")
    parquet = tmp_path / "out.parquet"
    to_parquet(str(tmp_path / "src.dyf"), str(parquet))
    report = from_parquet(str(parquet), str(tmp_path / "back.dyf"), max_depth=3)
    assert "leaf_id" not in report["fields"]
    assert "node_id" not in report["fields"]


def test_provenance_metadata_round_trips(tmp_path):
    _make_index(tmp_path / "src.dyf")
    parquet = tmp_path / "out.parquet"
    to_parquet(str(tmp_path / "src.dyf"), str(parquet))

    meta = {k.decode(): v.decode() for k, v in pq.read_table(parquet).schema.metadata.items()}
    assert meta["dyf_quantization"] == "float32"
    assert meta["dyf_is_pq"] == "false"
    assert len(meta["dyf_sha256"]) == 64
    assert "leaf_id ascending" in meta["dyf_row_order"]

    report = from_parquet(str(parquet), str(tmp_path / "back.dyf"), max_depth=3)
    assert report["source_metadata"]["dyf_sha256"] == meta["dyf_sha256"]


def test_fingerprint_can_be_skipped(tmp_path):
    _make_index(tmp_path / "src.dyf")
    parquet = tmp_path / "out.parquet"
    to_parquet(str(tmp_path / "src.dyf"), str(parquet), fingerprint=False)
    meta = {k.decode(): v.decode() for k, v in pq.read_table(parquet).schema.metadata.items()}
    assert "dyf_sha256" not in meta


def test_include_and_exclude(tmp_path):
    _make_index(tmp_path / "src.dyf")
    r = to_parquet(str(tmp_path / "src.dyf"), str(tmp_path / "a.parquet"), include=["sku"])
    assert r["fields"] == ["sku"]
    assert set(r["skipped"]) == {"spend", "cat"}

    r = to_parquet(str(tmp_path / "src.dyf"), str(tmp_path / "b.parquet"), exclude=["spend"])
    assert "spend" not in r["fields"]


def test_hive_partitioning_refused_on_thin_leaves(tmp_path):
    """gudid_viz.dyf has 5,938 leaves averaging 8 rows; partitioning it is a small-files trap."""
    _make_index(tmp_path / "src.dyf")
    with pytest.raises(ValueError, match="refusing to Hive-partition"):
        to_parquet(str(tmp_path / "src.dyf"), str(tmp_path / "parts"), partition_by_leaf=True)


def test_pq_index_requires_allow_lossy(tmp_path):
    pytest.importorskip("faiss", reason="PQ quantization is trained with faiss")
    _make_index(tmp_path / "pq.dyf", n=400, d=16, quantization="pq-4")
    with pytest.raises(ValueError, match="allow_lossy"):
        to_parquet(str(tmp_path / "pq.dyf"), str(tmp_path / "out.parquet"))

    report = to_parquet(str(tmp_path / "pq.dyf"), str(tmp_path / "out.parquet"), allow_lossy=True)
    assert report["reconstructed"] is True
    meta = {k.decode(): v.decode() for k, v in pq.read_table(tmp_path / "out.parquet").schema.metadata.items()}
    assert meta["dyf_embeddings_reconstructed"] == "true"


def test_variable_length_embeddings_rejected(tmp_path):
    table = pa.table(
        {
            "embedding": pa.array([[1.0, 2.0], [3.0, 4.0, 5.0]], type=pa.list_(pa.float32())),
            "sku": ["a", "b"],
        }
    )
    path = tmp_path / "ragged.parquet"
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="variable-length list"):
        from_parquet(str(path), str(tmp_path / "out.dyf"))


def test_missing_embedding_column_is_named(tmp_path):
    pq.write_table(pa.table({"sku": ["a", "b"]}), tmp_path / "no_emb.parquet")
    with pytest.raises(ValueError, match="embedding column 'embedding' not in"):
        from_parquet(str(tmp_path / "no_emb.parquet"), str(tmp_path / "out.dyf"))


def test_build_params_are_inherited_so_the_round_trip_keeps_its_shape(tmp_path):
    """Without this, a round trip silently rebuilds a differently shaped tree.

    Measured on a 50k/384d product index before the fix: 842 leaves became 14,385 and the
    file went 48 MB -> 136 MB, with nothing raised.
    """
    rng = np.random.default_rng(7)
    n, d = 600, 12
    emb = rng.normal(size=(n, d)).astype(np.float32)
    tree = build_dyf_tree(emb, max_depth=5, num_bits=2, min_leaf_size=8, seed=11)
    src = tmp_path / "src.dyf"
    write_lazy_index(
        tree,
        emb,
        str(src),
        quantization="float32",
        build_params={"max_depth": 5, "num_bits": 2, "min_leaf_size": 8, "seed": 11},
    )
    with LazyIndex(str(src)) as idx:
        src_leaves = idx.num_leaves

    parquet = tmp_path / "out.parquet"
    to_parquet(str(src), str(parquet))
    meta = {k.decode(): v.decode() for k, v in pq.read_table(parquet).schema.metadata.items()}
    assert meta["dyf_build_max_depth"] == "5"
    assert meta["dyf_build_num_bits"] == "2"
    assert meta["dyf_build_min_leaf_size"] == "8"
    assert meta["dyf_build_seed"] == "11"

    report = from_parquet(str(parquet), str(tmp_path / "back.dyf"), quantization="float32")
    assert report["build_params"] == {"max_depth": 5, "num_bits": 2, "min_leaf_size": 8, "seed": 11}
    assert set(report["build_params_inherited"]) == {"max_depth", "num_bits", "min_leaf_size", "seed"}
    assert report["leaves"] == src_leaves, "inherited params must reproduce the source tree shape"


def test_explicit_build_params_override_inheritance(tmp_path):
    _make_index(tmp_path / "src.dyf")
    parquet = tmp_path / "out.parquet"
    to_parquet(str(tmp_path / "src.dyf"), str(parquet))
    report = from_parquet(str(parquet), str(tmp_path / "back.dyf"), max_depth=2, quantization="float32")
    assert report["build_params"]["max_depth"] == 2
    assert "max_depth" not in report["build_params_inherited"]


def test_unsupported_column_is_skipped_loudly(tmp_path):
    """A struct column has no stored-field representation; it must be reported, not coerced."""
    n = 40
    rng = np.random.default_rng(3)
    table = pa.table(
        {
            "embedding": pa.FixedSizeListArray.from_arrays(pa.array(rng.normal(size=n * 8).astype(np.float32)), 8),
            "nested": pa.array([{"a": i} for i in range(n)]),
            "flag": pa.array([i % 2 == 0 for i in range(n)]),
            "label": pa.array([None if i == 0 else f"l{i}" for i in range(n)]),
        }
    )
    path = tmp_path / "mixed.parquet"
    pq.write_table(table, path)

    report = from_parquet(str(path), str(tmp_path / "out.dyf"), max_depth=2, min_leaf_size=2)
    assert "nested" not in report["fields"]
    assert any(s["column"] == "nested" and "unsupported arrow type" in s["reason"] for s in report["skipped"])
    # bool coerces to int32, null string becomes "" rather than failing the write
    assert "flag" in report["fields"]
    with LazyIndex(str(tmp_path / "out.dyf")) as idx:
        fields = idx.extract_all_fields()["fields"]
    assert set(np.unique(np.asarray(fields["flag"]))) <= {0, 1}
    assert "" in list(fields["label"])
