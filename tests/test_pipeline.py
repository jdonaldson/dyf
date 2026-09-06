"""Tests for dyf.pipeline — DAG pipeline runner."""

import json
import pickle

import pytest

from dyf.pipeline import Pipeline, Stage, _dyf_provenance_value
from dyf.provenance import create_provenance, provenance_to_dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_builder(path, params, n_items=100):
    """Return a build function that writes a pickle with provenance."""

    def build():
        data = {"result": "ok"}
        data["_provenance"] = provenance_to_dict(
            create_provenance(
                artifact_type="test",
                n_items=n_items,
                source_paths=["test_input"],
                params=params,
            )
        )
        with open(path, "wb") as f:
            pickle.dump(data, f)

    return build


# ---------------------------------------------------------------------------
# Topo sort
# ---------------------------------------------------------------------------


class TestTopoSort:
    def test_linear(self):
        p = Pipeline()
        p.add(Stage(name="a", inputs=[], output="/tmp/a.pkl", build_fn=lambda: None, params={}))
        p.add(Stage(name="b", inputs=["a"], output="/tmp/b.pkl", build_fn=lambda: None, params={}))
        p.add(Stage(name="c", inputs=["b"], output="/tmp/c.pkl", build_fn=lambda: None, params={}))

        order = p._topo_sort()
        assert order == ["a", "b", "c"]

    def test_diamond(self):
        p = Pipeline()
        p.add(Stage(name="a", inputs=[], output="/tmp/a.pkl", build_fn=lambda: None, params={}))
        p.add(Stage(name="b", inputs=["a"], output="/tmp/b.pkl", build_fn=lambda: None, params={}))
        p.add(Stage(name="c", inputs=["a"], output="/tmp/c.pkl", build_fn=lambda: None, params={}))
        p.add(Stage(name="d", inputs=["b", "c"], output="/tmp/d.pkl", build_fn=lambda: None, params={}))

        order = p._topo_sort()
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_target_prune(self):
        p = Pipeline()
        p.add(Stage(name="a", inputs=[], output="/tmp/a.pkl", build_fn=lambda: None, params={}))
        p.add(Stage(name="b", inputs=["a"], output="/tmp/b.pkl", build_fn=lambda: None, params={}))
        p.add(Stage(name="c", inputs=[], output="/tmp/c.pkl", build_fn=lambda: None, params={}))

        order = p._topo_sort(target="b")
        assert "a" in order
        assert "b" in order
        assert "c" not in order


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_missing_output(self, tmp_path):
        p = Pipeline()
        p.add(Stage(name="a", inputs=[], output=str(tmp_path / "missing.pkl"), build_fn=lambda: None, params={}))

        status = p.status()
        assert status["a"] == "missing"

    def test_fresh_after_build(self, tmp_path):
        out = str(tmp_path / "a.pkl")
        params = {"x": 1}
        p = Pipeline()
        p.add(Stage(name="a", inputs=[], output=out, build_fn=_make_builder(out, params), params=params))

        # Build it
        p.run()
        status = p.status()
        assert status["a"] == "fresh"

    def test_stale_params_change(self, tmp_path):
        out = str(tmp_path / "a.pkl")
        old_params = {"x": 1}
        new_params = {"x": 2}

        # Build with old params
        p = Pipeline()
        p.add(Stage(name="a", inputs=[], output=out, build_fn=_make_builder(out, old_params), params=old_params))
        p.run()

        # Change params
        p2 = Pipeline()
        p2.add(Stage(name="a", inputs=[], output=out, build_fn=_make_builder(out, new_params), params=new_params))

        status = p2.status()
        assert "stale" in status["a"]
        assert "params" in status["a"]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class TestRun:
    def test_builds_missing(self, tmp_path):
        out = str(tmp_path / "a.pkl")
        params = {"x": 1}
        p = Pipeline()
        p.add(Stage(name="a", inputs=[], output=out, build_fn=_make_builder(out, params), params=params))

        rebuilt = p.run()
        assert rebuilt == ["a"]
        assert (tmp_path / "a.pkl").exists()

    def test_skips_fresh(self, tmp_path):
        out = str(tmp_path / "a.pkl")
        params = {"x": 1}
        p = Pipeline()
        p.add(Stage(name="a", inputs=[], output=out, build_fn=_make_builder(out, params), params=params))

        p.run()
        rebuilt = p.run()
        assert rebuilt == []

    def test_dry_run(self, tmp_path):
        out = str(tmp_path / "a.pkl")
        params = {"x": 1}
        p = Pipeline()
        p.add(Stage(name="a", inputs=[], output=out, build_fn=_make_builder(out, params), params=params))

        rebuilt = p.run(dry_run=True)
        assert rebuilt == []
        assert not (tmp_path / "a.pkl").exists()

    def test_chain_rebuild(self, tmp_path):
        out_a = str(tmp_path / "a.pkl")
        out_b = str(tmp_path / "b.pkl")
        params_a = {"x": 1}
        params_b = {"y": 2}

        p = Pipeline()
        p.add(Stage(name="a", inputs=[], output=out_a, build_fn=_make_builder(out_a, params_a), params=params_a))
        p.add(Stage(name="b", inputs=["a"], output=out_b, build_fn=_make_builder(out_b, params_b), params=params_b))

        rebuilt = p.run(target="b")
        assert "a" in rebuilt
        assert "b" in rebuilt


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------


class TestExplain:
    def test_explain(self, tmp_path):
        p = Pipeline()
        p.add(Stage(name="a", inputs=[], output=str(tmp_path / "a.pkl"), build_fn=lambda: None, params={"x": 1}))
        p.add(Stage(name="b", inputs=["a"], output=str(tmp_path / "b.pkl"), build_fn=lambda: None, params={"y": 2}))

        text = p.explain("b")
        assert "a" in text
        assert "b" in text
        assert "missing" in text


# ---------------------------------------------------------------------------
# .dyf provenance
#
# This branch had no coverage at all. The tests above write `_provenance` into .pkl
# fixtures by hand, so none of them exercised the .dyf path — which read a key nothing
# writes, leaving every .dyf stage permanently "stale (no provenance)".
# ---------------------------------------------------------------------------


class TestDyfProvenance:
    def test_prefers_the_highest_enrichment_level(self):
        meta = {
            "_provenance_level_1": "one",
            "_provenance_level_3": "three",
            "_provenance_level_2": "two",
            "domain": "irrelevant",
        }
        assert _dyf_provenance_value(meta) == "three"

    def test_plain_key_wins_when_present(self):
        meta = {"_provenance": "direct", "_provenance_level_2": "two"}
        assert _dyf_provenance_value(meta) == "direct"

    def test_returns_none_when_no_provenance(self):
        assert _dyf_provenance_value({"domain": "x"}) is None

    def test_ignores_a_malformed_level_suffix(self):
        assert _dyf_provenance_value({"_provenance_level_abc": "junk"}) is None

    def test_reads_provenance_from_a_real_dyf_file(self, tmp_path):
        """End-to-end: the shape dyfviz actually writes must read back as fresh."""
        pytest.importorskip("flatbuffers")
        pytest.importorskip("dyf_rs")
        import numpy as np

        from dyf.dyf_tree import build_dyf_tree
        from dyf.lazy_index import write_lazy_index

        rng = np.random.default_rng(0)
        X = np.ascontiguousarray(rng.standard_normal((32, 8)).astype(np.float32))
        tree = build_dyf_tree(X, max_depth=2, num_bits=2, min_leaf_size=2, seed=42)

        params = {"k": 1}
        prov = provenance_to_dict(
            create_provenance(artifact_type="dyf", n_items=32, source_paths=["upstream"], params=params)
        )
        path = tmp_path / "enriched.dyf"
        write_lazy_index(tree, X, str(path), metadata={"_provenance_level_1": json.dumps(prov)})

        assert Pipeline._read_provenance(str(path)) is not None

        p = Pipeline()
        p.add(Stage(name="e", inputs=[], output=str(path), build_fn=lambda: None, params=params))
        assert p._stage_status("e") == "fresh"
