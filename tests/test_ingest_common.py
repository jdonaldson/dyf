"""The shared ingest tail — `finalize_index` and `add_common_index_args`.

Tested directly rather than through the three commands, because two of those need torch
and a vision model to run at all and their test files skip entirely here. The extracted
tail is exactly where a refactor could break something silently, so it gets its own
coverage that does not depend on any optional stack.

The point of extracting it was not tidiness. `--dedup` existed only for `index_source`
because the tail was copy-pasted three times, so a capability could not spread. These
tests assert it now reaches all three.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

pytest.importorskip("flatbuffers")
pytest.importorskip("dyf_rs")

from dyf._ingest_common import (  # noqa: E402
    DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_LEAF_SIZE,
    DEFAULT_NUM_BITS,
    DEFAULT_SEED,
    add_common_index_args,
    finalize_index,
)
from dyf.lazy_index import LazyIndex  # noqa: E402


def _corpus(n=64, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal((n, dim)).astype(np.float32))


def _duplicated_corpus(n_unique=20, copies=4, dim=16, seed=7):
    """Each unique vector repeated `copies` times with tiny jitter — dedup should collapse it."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_unique, dim)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    X = np.repeat(base, copies, axis=0)
    X = X + 0.0005 * rng.standard_normal(X.shape).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return np.ascontiguousarray(X.astype(np.float32))


class TestFinalizeIndex:
    def test_writes_a_readable_index(self, tmp_path):
        X = _corpus()
        out = tmp_path / "a.dyf"
        finalize_index(
            X,
            out,
            stored_fields={"title": [f"t{i}" for i in range(len(X))]},
            metadata={"domain": "test"},
        )
        assert out.exists()
        with LazyIndex(str(out)) as idx:
            assert idx.total_items == len(X)
            assert idx.stored_field_names == ["title"]

    def test_normalizes_embeddings(self, tmp_path):
        """Every caller normalized before writing; the shared tail must still do it."""
        X = _corpus() * 17.0  # deliberately not unit-norm
        out = tmp_path / "b.dyf"
        finalize_index(X, out, stored_fields={}, metadata={"domain": "test"})

        with LazyIndex(str(out)) as idx:
            vecs = idx.extract_all_fields()["embeddings"]
        norms = np.linalg.norm(vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-2), f"not unit-norm: {norms[:5]}"

    def test_metadata_and_build_params_are_recorded(self, tmp_path):
        out = tmp_path / "c.dyf"
        finalize_index(
            _corpus(),
            out,
            stored_fields={},
            metadata={"domain": "images", "embedding_model": "m"},
            max_depth=3,
            num_bits=3,
            min_leaf_size=2,
            seed=99,
        )
        with LazyIndex(str(out)) as idx:
            summary = idx.tree_summary
            assert summary["build_params"]["max_depth"] == 3
            assert summary["build_params"]["seed"] == 99
            assert idx._get_metadata()["domain"] == "images"

    def test_defaults_match_what_the_three_commands_used(self, tmp_path):
        """The extracted constants must equal the values that were hard-coded in each copy."""
        assert (DEFAULT_MAX_DEPTH, DEFAULT_NUM_BITS, DEFAULT_MIN_LEAF_SIZE, DEFAULT_SEED) == (4, 4, 5, 42)

        out = tmp_path / "d.dyf"
        finalize_index(_corpus(), out, stored_fields={}, metadata={"domain": "test"})
        with LazyIndex(str(out)) as idx:
            bp = idx.tree_summary["build_params"]
        assert bp["max_depth"] == 4
        assert bp["num_bits"] == 4
        assert bp["min_leaf_size"] == 5
        assert bp["seed"] == 42


class TestDedupReachesEveryCommand:
    """`--dedup` existed only for source before the tail was shared."""

    def test_dedup_collapses_and_keeps_fields_aligned(self, tmp_path):
        X = _duplicated_corpus()
        titles = [f"item{i}" for i in range(len(X))]
        out = tmp_path / "dd.dyf"

        finalize_index(
            X,
            out,
            stored_fields={"title": titles},
            metadata={"domain": "test"},
            dedup=0.99,
        )

        with LazyIndex(str(out)) as idx:
            assert idx.total_items < len(X), "dedup did not collapse a duplicated corpus"
            fields = set(idx.stored_field_names)
        # The bookkeeping that lets a caller map back to pre-dedup rows.
        assert {"title", "orig_index", "dup_members"} <= fields

    def test_without_dedup_every_item_is_kept(self, tmp_path):
        X = _duplicated_corpus()
        out = tmp_path / "nodd.dyf"
        finalize_index(X, out, stored_fields={"title": [f"i{i}" for i in range(len(X))]}, metadata={"domain": "t"})
        with LazyIndex(str(out)) as idx:
            assert idx.total_items == len(X)
            assert "orig_index" not in idx.stored_field_names


class TestCommonArgs:
    def _parse(self, argv):
        parser = argparse.ArgumentParser()
        add_common_index_args(parser, default_model="test-model")
        return parser.parse_args(argv)

    def test_every_command_gets_the_same_flags(self):
        args = self._parse([])
        for flag in (
            "output",
            "model",
            "max_depth",
            "num_bits",
            "min_leaf_size",
            "seed",
            "dedup",
            "dry_run",
            "as_json",
        ):
            assert hasattr(args, flag), f"shared flag missing: {flag}"

    def test_dedup_defaults_to_off_and_to_099_when_bare(self):
        assert self._parse([]).dedup is None
        assert self._parse(["--dedup"]).dedup == 0.99
        assert self._parse(["--dedup", "0.95"]).dedup == 0.95

    def test_tree_defaults_come_from_the_shared_constants(self):
        args = self._parse([])
        assert args.max_depth == DEFAULT_MAX_DEPTH
        assert args.num_bits == DEFAULT_NUM_BITS
        assert args.min_leaf_size == DEFAULT_MIN_LEAF_SIZE
        assert args.seed == DEFAULT_SEED


@pytest.mark.parametrize("module_name", ["index_source", "index_images", "index_video"])
def test_all_three_commands_accept_dedup(module_name):
    """The capability that could not spread while the tail was copy-pasted."""
    import importlib
    import inspect

    module = importlib.import_module(f"dyf.{module_name}")
    entry = getattr(module, module_name)
    assert "dedup" in inspect.signature(entry).parameters, f"{module_name} still cannot dedup"


class TestProvenanceStamp:
    """`finalize_index` writes `_provenance_level_0`; `Pipeline` and `dyf info` read it."""

    @staticmethod
    def _record(path):
        import json

        with LazyIndex(str(path)) as idx:
            meta = dict(idx._get_metadata())
        assert "_provenance_level_0" in meta, sorted(meta)
        return json.loads(meta["_provenance_level_0"])

    def test_stamps_level_0_with_counts_and_params(self, tmp_path):
        X = _corpus()
        out = tmp_path / "p.dyf"
        finalize_index(X, out, stored_fields={}, metadata={"domain": "test", "embedding_model": "m1"}, seed=7)
        rec = self._record(out)
        assert rec["artifact_type"] == "dyf"
        assert rec["n_items"] == len(X)
        assert rec["params"]["seed"] == 7
        assert rec["params"]["fit_method"] == "itq"
        assert rec["params"]["embedding_model"] == "m1"

    def test_n_items_is_the_post_dedup_count(self, tmp_path):
        X = _corpus()
        X = np.vstack([X, X[:10]])  # 10 exact duplicates
        out = tmp_path / "d.dyf"
        finalize_index(X, out, stored_fields={}, metadata={"domain": "t"}, dedup=0.999)
        rec = self._record(out)
        assert rec["n_items"] < len(X)
        with LazyIndex(str(out)) as idx:
            assert idx.total_items == rec["n_items"]

    def test_params_hash_tracks_the_embedding_model(self, tmp_path):
        """A model change must invalidate the artifact even with identical tree params."""
        X = _corpus()
        a, b = tmp_path / "a.dyf", tmp_path / "b.dyf"
        finalize_index(X, a, stored_fields={}, metadata={"domain": "t", "embedding_model": "m1"})
        finalize_index(X, b, stored_fields={}, metadata={"domain": "t", "embedding_model": "m2"})
        assert self._record(a)["params_hash"] != self._record(b)["params_hash"]

    def test_source_hash_tracks_the_inputs(self, tmp_path):
        X = _corpus()
        f1 = tmp_path / "one.txt"
        f2 = tmp_path / "two.txt"
        f1.write_text("alpha")
        f2.write_text("beta")
        a, b, c = (tmp_path / n for n in ("a.dyf", "b.dyf", "c.dyf"))
        finalize_index(X, a, stored_fields={}, metadata={"domain": "t"}, source_paths=[f1])
        finalize_index(X, b, stored_fields={}, metadata={"domain": "t"}, source_paths=[f1])
        finalize_index(X, c, stored_fields={}, metadata={"domain": "t"}, source_paths=[f1, f2])
        ra, rb, rc = (self._record(p) for p in (a, b, c))
        assert ra["source_hash"] == rb["source_hash"]
        assert ra["source_hash"] != rc["source_hash"]

    def test_pipeline_reads_it(self, tmp_path):
        """The consumer this exists for: before, every .dyf was 'stale (no provenance)'."""
        from dyf.pipeline import Pipeline

        out = tmp_path / "q.dyf"
        finalize_index(_corpus(), out, stored_fields={}, metadata={"domain": "t"})
        prov = Pipeline._read_provenance(str(out))
        assert prov is not None
        assert prov.artifact_type == "dyf"
        assert prov.n_items == len(_corpus())

    def test_pipeline_stage_declaring_ingest_params_reads_fresh(self, tmp_path):
        """The contract: a Stage built with `ingest_params` matches the stamped hash.

        Without this, a stage wrapping an index-* command could only ever read
        'stale (params changed)', which is the state every .dyf was in before the stamp.
        """
        from dyf._ingest_common import ingest_params
        from dyf.pipeline import Pipeline, Stage

        out = tmp_path / "s.dyf"
        finalize_index(
            _corpus(), out, stored_fields={}, metadata={"domain": "t", "embedding_model": "m"}, seed=3, dedup=0.99
        )
        p = Pipeline()
        p.add(
            Stage(
                name="ingest",
                inputs=[],
                output=str(out),
                build_fn=lambda: None,
                params=ingest_params(seed=3, dedup=0.99, embedding_model="m", domain="t"),
            )
        )
        assert p._stage_status("ingest") == "fresh"
        p.stages["ingest"].params = ingest_params(seed=4, dedup=0.99, embedding_model="m", domain="t")
        assert p._stage_status("ingest") == "stale (params changed)"

    def test_info_reports_level_0(self, tmp_path):
        from dyf.info import _format_human, collect_info

        out = tmp_path / "r.dyf"
        finalize_index(_corpus(), out, stored_fields={}, metadata={"domain": "t"})
        info = collect_info(str(out))
        assert "0" in info["provenance"]
        assert "provenance       levels 0" in _format_human(info)
