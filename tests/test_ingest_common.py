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
