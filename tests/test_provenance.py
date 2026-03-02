"""Tests for dyf.provenance — artifact provenance tracking."""

import json

from dyf.provenance import (
    Provenance,
    file_hash,
    params_hash,
    create_provenance,
    check_compatible,
    provenance_to_dict,
    provenance_from_dict,
)


# ---------------------------------------------------------------------------
# file_hash
# ---------------------------------------------------------------------------

class TestFileHash:
    def test_deterministic(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"hello world")
        assert file_hash(p) == file_hash(p)

    def test_different_content(self, tmp_path):
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"hello")
        b.write_bytes(b"world")
        assert file_hash(a) != file_hash(b)

    def test_length_12(self, tmp_path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"test")
        assert len(file_hash(p)) == 12

    def test_large_file_uses_prefix(self, tmp_path):
        """Files larger than 64 KB still produce a hash (from first 64 KB)."""
        p = tmp_path / "big.bin"
        p.write_bytes(b"x" * 100_000)
        h = file_hash(p)
        assert len(h) == 12


# ---------------------------------------------------------------------------
# params_hash
# ---------------------------------------------------------------------------

class TestParamsHash:
    def test_deterministic(self):
        d = {"sample": 0, "bridge_level": 100}
        assert params_hash(d) == params_hash(d)

    def test_order_independent(self):
        a = {"b": 2, "a": 1}
        b = {"a": 1, "b": 2}
        assert params_hash(a) == params_hash(b)

    def test_different_values(self):
        a = {"sample": 0}
        b = {"sample": 5000}
        assert params_hash(a) != params_hash(b)

    def test_length_12(self):
        assert len(params_hash({"x": 1})) == 12


# ---------------------------------------------------------------------------
# create_provenance
# ---------------------------------------------------------------------------

class TestCreateProvenance:
    def test_basic(self, tmp_path):
        src = tmp_path / "source.parquet"
        src.write_bytes(b"fake parquet data")

        prov = create_provenance(
            artifact_type="rog_cache",
            n_items=1000,
            source_paths=[src],
            params={"sample": 0, "bridge_level": 100},
            sample_seed=42,
            sample_n=0,
        )

        assert prov.artifact_type == "rog_cache"
        assert prov.n_items == 1000
        assert prov.sample_seed == 42
        assert prov.sample_n == 0
        assert len(prov.source_hash) == 12
        assert len(prov.params_hash) == 12
        assert prov.created_at  # non-empty ISO timestamp
        assert prov.params == {"sample": 0, "bridge_level": 100}

    def test_no_sampling(self, tmp_path):
        src = tmp_path / "source.parquet"
        src.write_bytes(b"data")

        prov = create_provenance(
            artifact_type="dyf",
            n_items=500,
            source_paths=[src],
            params={},
        )

        assert prov.sample_seed is None
        assert prov.sample_n is None

    def test_nonexistent_source_path(self):
        """Stage names (not files) are hashed by name."""
        prov = create_provenance(
            artifact_type="viz",
            n_items=100,
            source_paths=["embed_stage", "cluster_stage"],
            params={},
        )
        assert len(prov.source_hash) == 12


# ---------------------------------------------------------------------------
# check_compatible
# ---------------------------------------------------------------------------

class TestCheckCompatible:
    def _make_prov(self, n_items=1000, sample_n=0, sample_seed=42):
        return Provenance(
            artifact_type="rog_cache",
            n_items=n_items,
            source_hash="aabbccddee00",
            params_hash="112233445566",
            created_at="2026-01-01T00:00:00+00:00",
            params={"sample": sample_n},
            sample_seed=sample_seed,
            sample_n=sample_n,
        )

    def test_compatible(self):
        prov = self._make_prov(n_items=1000, sample_n=0)
        ok, warnings = check_compatible(prov, downstream_n_items=1000, downstream_sample_n=0)
        assert ok
        assert warnings == []

    def test_n_items_mismatch(self):
        prov = self._make_prov(n_items=1000)
        ok, warnings = check_compatible(prov, downstream_n_items=5000)
        assert not ok
        assert any("n_items" in w for w in warnings)

    def test_sample_n_mismatch(self):
        prov = self._make_prov(sample_n=0)
        ok, warnings = check_compatible(prov, downstream_sample_n=5000)
        assert not ok
        assert any("sample_n" in w for w in warnings)

    def test_sample_seed_mismatch(self):
        prov = self._make_prov(sample_seed=42)
        ok, warnings = check_compatible(prov, downstream_sample_seed=99)
        assert not ok
        assert any("sample_seed" in w for w in warnings)

    def test_no_downstream_checks(self):
        """When no downstream values provided, everything is compatible."""
        prov = self._make_prov()
        ok, warnings = check_compatible(prov)
        assert ok
        assert warnings == []


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_roundtrip(self):
        prov = Provenance(
            artifact_type="rog_cache",
            n_items=34228,
            source_hash="abcdef012345",
            params_hash="fedcba543210",
            created_at="2026-02-26T10:00:00+00:00",
            params={"sample": 0, "bridge_level": 100},
            sample_seed=42,
            sample_n=0,
        )

        d = provenance_to_dict(prov)
        assert isinstance(d, dict)

        # Must be JSON-serializable
        json_str = json.dumps(d)
        d2 = json.loads(json_str)

        prov2 = provenance_from_dict(d2)
        assert prov2.artifact_type == prov.artifact_type
        assert prov2.n_items == prov.n_items
        assert prov2.source_hash == prov.source_hash
        assert prov2.params_hash == prov.params_hash
        assert prov2.created_at == prov.created_at
        assert prov2.params == prov.params
        assert prov2.sample_seed == prov.sample_seed
        assert prov2.sample_n == prov.sample_n

    def test_roundtrip_no_sampling(self):
        prov = Provenance(
            artifact_type="dyf",
            n_items=500,
            source_hash="000000000000",
            params_hash="111111111111",
            created_at="2026-01-01T00:00:00+00:00",
        )

        d = provenance_to_dict(prov)
        prov2 = provenance_from_dict(d)
        assert prov2.sample_seed is None
        assert prov2.sample_n is None
        assert prov2.params == {}
