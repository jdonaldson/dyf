"""Tests for dyf.concept_graph — concept graph building and querying."""

import json
import os
import time

import pytest

from dyf.concept_graph import (
    ConceptGraphConfig,
    ConceptNode,
    ConfigError,
    MarkdownChunk,
    build_header_only_graph,
    check_staleness,
    chunk_markdown,
    fuzzy_match,
    load_graph,
    load_graph_meta,
    save_graph,
    slugify,
)

# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_basic(self):
        assert slugify("Hello World") == "hello-world"

    def test_special_chars(self):
        assert slugify("Debrief Pattern (v2)") == "debrief-pattern-v2"

    def test_emoji_stripped(self):
        result = slugify("🧠 File Organization")
        assert "file-organization" in result

    def test_truncation(self):
        long = "a" * 100
        assert len(slugify(long)) <= 50
        # `<= 50` alone is satisfied by the empty string — assert it truncates rather than
        # discards, and that the content that survives is the right content.
        assert slugify(long) == "a" * len(slugify(long))
        assert len(slugify(long)) > 0, "truncation returned nothing"

    def test_multiple_spaces(self):
        assert slugify("foo   bar") == "foo-bar"

    def test_leading_trailing_whitespace(self):
        assert slugify("  hello  ") == "hello"


# ---------------------------------------------------------------------------
# chunk_markdown
# ---------------------------------------------------------------------------

SAMPLE_MD = """\
# Top-Level Header

Intro text that is not a ## section.

## First Section

This is the first section body with enough content to pass min_length.

## Short

x

## Second Section

This is the second section body, also long enough to be kept.

## Third Section

Another section with sufficient content to be included in the chunks.
"""


class TestChunkMarkdown:
    def test_splits_on_h2(self):
        chunks = chunk_markdown(SAMPLE_MD, "test.md")
        headers = [c.header for c in chunks]
        assert "First Section" in headers
        assert "Second Section" in headers
        assert "Third Section" in headers

    def test_ignores_h1(self):
        chunks = chunk_markdown(SAMPLE_MD, "test.md")
        headers = [c.header for c in chunks]
        assert "Top-Level Header" not in headers

    def test_skips_short_sections(self):
        chunks = chunk_markdown(SAMPLE_MD, "test.md")
        headers = [c.header for c in chunks]
        assert "Short" not in headers

    def test_correct_line_numbers(self):
        chunks = chunk_markdown(SAMPLE_MD, "test.md")
        first = [c for c in chunks if c.header == "First Section"][0]
        # "## First Section" is on line 5
        assert first.line == 5

    def test_chunk_ids_namespaced(self):
        chunks = chunk_markdown(SAMPLE_MD, "/path/to/test.md")
        for c in chunks:
            assert c.id.startswith("test/")

    def test_custom_header_level(self):
        md = "### Sub A\n\nLong enough body for sub section A here.\n\n### Sub B\n\nAnother long enough body for sub section B.\n"
        chunks = chunk_markdown(md, "test.md", header_level=3)
        assert len(chunks) == 2
        assert chunks[0].header == "Sub A"

    def test_h3_not_split_at_h2_level(self):
        md = "### Not a split\n\nThis should not produce chunks at h2 level.\n"
        chunks = chunk_markdown(md, "test.md", header_level=2)
        assert len(chunks) == 0

    def test_empty_input(self):
        chunks = chunk_markdown("", "test.md")
        assert chunks == []

    def test_dedup_across_files(self):
        chunks_a = chunk_markdown(SAMPLE_MD, "fileA.md")
        chunks_b = chunk_markdown(SAMPLE_MD, "fileB.md")
        ids_a = {c.id for c in chunks_a}
        ids_b = {c.id for c in chunks_b}
        # IDs should be different because source prefix differs
        assert ids_a.isdisjoint(ids_b)


# ---------------------------------------------------------------------------
# fuzzy_match
# ---------------------------------------------------------------------------


class TestFuzzyMatch:
    def setup_method(self):
        self.graph = {
            "test/debrief-pattern": ConceptNode(
                header="Debrief Pattern",
                source="CLAUDE.md",
                line=39,
            ),
            "test/session-continuity": ConceptNode(
                header="Session Continuity",
                source="CLAUDE.md",
                line=79,
            ),
            "test/tmux-topic-trace": ConceptNode(
                header="Tmux Topic Trace",
                source="CLAUDE.md",
                line=50,
            ),
        }

    def test_exact_match(self):
        node_id, score = fuzzy_match("Debrief Pattern", self.graph)
        assert node_id == "test/debrief-pattern"
        assert score > 0.8

    def test_substring_match(self):
        node_id, score = fuzzy_match("debrief", self.graph)
        assert node_id == "test/debrief-pattern"
        assert score >= 0.4

    def test_below_threshold(self):
        node_id, score = fuzzy_match("xyzzy", self.graph, threshold=0.9)
        assert node_id is None

    def test_case_insensitive(self):
        node_id, _ = fuzzy_match("session continuity", self.graph)
        assert node_id == "test/session-continuity"

    def test_partial_match(self):
        node_id, _ = fuzzy_match("tmux trace", self.graph)
        assert node_id == "test/tmux-topic-trace"


# ---------------------------------------------------------------------------
# save_graph / load_graph
# ---------------------------------------------------------------------------


class TestGraphSerialization:
    def test_roundtrip(self, tmp_path):
        graph = {
            "test/node-a": ConceptNode(
                header="Node A",
                source="test.md",
                line=1,
                neighbors=[
                    {"id": "test/node-b", "header": "Node B", "similarity": 0.85, "source": "test.md", "line": 10},
                ],
            ),
            "test/node-b": ConceptNode(
                header="Node B",
                source="test.md",
                line=10,
                neighbors=[],
            ),
        }
        path = str(tmp_path / "graph.json")
        save_graph(graph, path)
        loaded = load_graph(path)

        assert set(loaded.keys()) == {"test/node-a", "test/node-b"}
        assert loaded["test/node-a"].header == "Node A"
        assert len(loaded["test/node-a"].neighbors) == 1
        assert loaded["test/node-a"].neighbors[0]["similarity"] == 0.85

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "graph.json")
        graph = {"test/n": ConceptNode(header="N", source="t.md", line=1)}
        save_graph(graph, path)
        assert os.path.exists(path)

    def test_json_valid(self, tmp_path):
        path = str(tmp_path / "graph.json")
        graph = {"test/n": ConceptNode(header="N", source="t.md", line=1)}
        save_graph(graph, path)
        with open(path) as f:
            data = json.load(f)
        assert "test/n" in data


# ---------------------------------------------------------------------------
# check_staleness
# ---------------------------------------------------------------------------


class TestCheckStaleness:
    def test_missing_graph_is_stale(self, tmp_path):
        config = ConceptGraphConfig(
            sources=[],
            output_path=str(tmp_path / "nonexistent.json"),
        )
        assert check_staleness(config) is True

    def test_fresh_graph_not_stale(self, tmp_path):
        # Create a source file
        src = tmp_path / "src.md"
        src.write_text("## Hello\n\nWorld\n")

        # Create a graph file that's newer
        graph_path = tmp_path / "graph.json"
        time.sleep(0.05)
        graph_path.write_text("{}")

        config = ConceptGraphConfig(
            sources=[str(src)],
            output_path=str(graph_path),
        )
        assert check_staleness(config) is False

    def test_stale_when_source_newer(self, tmp_path):
        # Create graph file first
        graph_path = tmp_path / "graph.json"
        graph_path.write_text("{}")

        # Then create a newer source file
        time.sleep(0.05)
        src = tmp_path / "src.md"
        src.write_text("## Hello\n\nWorld\n")

        config = ConceptGraphConfig(
            sources=[str(src)],
            output_path=str(graph_path),
        )
        assert check_staleness(config) is True


# ---------------------------------------------------------------------------
# ConceptGraphConfig
# ---------------------------------------------------------------------------


class TestConceptGraphConfig:
    def test_defaults(self):
        config = ConceptGraphConfig()
        assert config.top_k == 5
        assert config.similarity_threshold == 0.2
        assert config.embedder == "low"
        assert len(config.sources) > 0

    def test_load_missing_file(self, tmp_path):
        """CHANGED 2026-09-05: an explicitly requested config that is missing now raises.

        This previously returned defaults silently. That is a worse outcome than an
        error: the caller passes --config, believes their output_path is in effect, and
        the tool writes to ~/.dyf/ instead without saying anything. Pre-v1, so broken
        deliberately rather than preserved. The no-config-at-all case still falls back —
        see TestConfigLoading.test_default_path_missing_is_fine.
        """
        with pytest.raises(ConfigError):
            ConceptGraphConfig.load(str(tmp_path / "missing.json"))

    def test_load_from_file(self, tmp_path):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "sources": ["~/test.md"],
                    "output_path": "/tmp/test_graph.json",
                    "embeddings_cache_path": "/tmp/test_embeddings.npz",
                    "embedder": "medium",
                    "top_k": 3,
                    "similarity_threshold": 0.3,
                }
            )
        )
        config = ConceptGraphConfig.load(str(cfg_path))
        assert config.top_k == 3
        assert config.embedder == "medium"
        assert config.similarity_threshold == 0.3

    def test_expand_sources_filters_nonexistent(self, tmp_path):
        src = tmp_path / "exists.md"
        src.write_text("## Test\n\nContent\n")
        config = ConceptGraphConfig(
            sources=[str(src), str(tmp_path / "nope.md")],
        )
        paths = config.expand_sources()
        assert len(paths) == 1
        assert paths[0] == src


# ---------------------------------------------------------------------------
# MarkdownChunk dataclass
# ---------------------------------------------------------------------------


class TestMarkdownChunk:
    def test_fields(self):
        chunk = MarkdownChunk(
            id="test/hello",
            header="Hello",
            text="Hello: world",
            source="test.md",
            line=1,
        )
        assert chunk.id == "test/hello"
        assert chunk.metadata == {}

    def test_metadata(self):
        chunk = MarkdownChunk(
            id="test/hello",
            header="Hello",
            text="Hello: world",
            source="test.md",
            line=1,
            metadata={"project": "foo"},
        )
        assert chunk.metadata["project"] == "foo"


# ---------------------------------------------------------------------------
# JSON output
#
# The contract an agent depends on: stdout is parseable JSON, and the exit code means
# the same thing as it does on the human path.
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def _graph(self, tmp_path):
        path = str(tmp_path / "graph.json")
        save_graph(
            {
                "notes/alpha": ConceptNode(
                    header="Alpha",
                    source="a.md",
                    line=1,
                    neighbors=[{"id": "notes/beta", "header": "Beta", "similarity": 0.7, "source": "a.md", "line": 9}],
                ),
                "notes/beta": ConceptNode(header="Beta", source="a.md", line=9),
            },
            path,
            has_embeddings=True,
        )
        return path

    def _config(self, tmp_path):
        return ConceptGraphConfig(
            output_path=self._graph(tmp_path),
            embeddings_cache_path=str(tmp_path / "emb.npz"),
        )

    def test_query_json_is_parseable_and_versioned(self, tmp_path, capsys):
        import dyf.concept_graph as cg

        rc = cg._cmd_query(self._config(tmp_path), "Alpha", False, 5, as_json=True)
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert payload["schema_version"] == 0
        assert payload["match"]["header"] == "Alpha"
        assert payload["match"]["id"] == "notes/alpha"

    def test_query_json_includes_neighbors_with_scores(self, tmp_path, capsys):
        import dyf.concept_graph as cg

        cg._cmd_query(self._config(tmp_path), "Alpha", False, 5, as_json=True)
        payload = json.loads(capsys.readouterr().out)
        assert payload["neighbors"][0]["header"] == "Beta"
        assert payload["neighbors"][0]["similarity"] == 0.7

    def test_list_json_reports_every_node(self, tmp_path, capsys):
        import dyf.concept_graph as cg

        rc = cg._cmd_list(self._config(tmp_path), verbose=False, as_json=True)
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert {n["id"] for n in payload["nodes"]} == {"notes/alpha", "notes/beta"}
        # Non-verbose reports a neighbor *count*, so the payload stays small.
        assert payload["nodes"][0]["neighbors"] == 1

    def test_list_json_verbose_expands_neighbors(self, tmp_path, capsys):
        import dyf.concept_graph as cg

        cg._cmd_list(self._config(tmp_path), verbose=True, as_json=True)
        payload = json.loads(capsys.readouterr().out)
        alpha = next(n for n in payload["nodes"] if n["id"] == "notes/alpha")
        assert alpha["neighbors"][0]["header"] == "Beta"

    def test_header_only_graph_reports_it_and_exits_1(self, tmp_path, capsys):
        """A caller must be able to tell 'nothing matched' from 'this graph cannot match'."""
        import dyf.concept_graph as cg

        path = str(tmp_path / "hdr.json")
        save_graph(build_header_only_graph([]), path, has_embeddings=False)
        config = ConceptGraphConfig(output_path=path, embeddings_cache_path=str(tmp_path / "emb.npz"))

        rc = cg._cmd_query(config, "anything at all", False, 5, as_json=True)
        payload = json.loads(capsys.readouterr().out)
        assert rc == 1
        assert payload["match"] is None
        assert payload["graph"]["has_embeddings"] is False
        assert "remedy" in payload, "must say how to fix it, not just that it failed"


# ---------------------------------------------------------------------------
# Config loading
#
# The worst original behaviour was not the traceback on malformed JSON — it was that an
# explicitly requested config file that did not exist fell back to defaults in silence,
# so the tool wrote somewhere other than where the caller asked and said nothing.
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_explicit_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            ConceptGraphConfig.load(str(tmp_path / "absent.json"))

    def test_default_path_missing_is_fine(self, tmp_path, monkeypatch):
        """Having no config file at all is the normal case, not an error."""
        monkeypatch.setenv("HOME", str(tmp_path))
        config = ConceptGraphConfig.load(None)
        assert config.top_k == 5

    def test_malformed_json_names_the_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("this is not json")
        with pytest.raises(ConfigError, match="not valid JSON"):
            ConceptGraphConfig.load(str(path))

    def test_unknown_setting_is_rejected_and_lists_valid_ones(self, tmp_path):
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps({"top_k": 3, "nonsense": 1}))
        with pytest.raises(ConfigError) as exc:
            ConceptGraphConfig.load(str(path))
        assert "nonsense" in str(exc.value)
        assert "top_k" in str(exc.value), "should name the valid settings"

    def test_non_object_json_is_rejected(self, tmp_path):
        path = tmp_path / "arr.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ConfigError, match="JSON object"):
            ConceptGraphConfig.load(str(path))

    def test_valid_config_loads(self, tmp_path):
        path = tmp_path / "cfg.json"
        path.write_text(json.dumps({"top_k": 9, "similarity_threshold": 0.5}))
        config = ConceptGraphConfig.load(str(path))
        assert config.top_k == 9
        assert config.similarity_threshold == 0.5


# ---------------------------------------------------------------------------
# Header-only graphs
#
# The index half of this tool needs no model: fuzzy_match is pure SequenceMatcher, and
# only *neighbors* require embedding. Building without the multi-GB torch stack is what
# makes `dyf concepts` usable as the globally-invoked tool it is documented to be.
# ---------------------------------------------------------------------------


class TestHeaderOnlyGraph:
    def _chunks(self):
        return [
            MarkdownChunk(id="a/one", header="One", text="first", source="a.md", line=1, metadata={}),
            MarkdownChunk(id="a/two", header="Two", text="second", source="a.md", line=9, metadata={}),
        ]

    def test_builds_every_node_without_a_model(self):
        graph = build_header_only_graph(self._chunks())
        assert set(graph) == {"a/one", "a/two"}
        assert graph["a/one"].header == "One"
        assert graph["a/two"].line == 9

    def test_nodes_have_no_neighbors(self):
        graph = build_header_only_graph(self._chunks())
        assert all(node.neighbors == [] for node in graph.values())

    def test_fuzzy_match_still_works(self):
        """The query path must survive without embeddings — that is the whole point."""
        graph = build_header_only_graph(self._chunks())
        node_id, score = fuzzy_match("One", graph)
        assert node_id == "a/one"
        assert score > 0.5

    def test_roundtrip_records_that_embeddings_are_absent(self, tmp_path):
        path = str(tmp_path / "graph.json")
        save_graph(build_header_only_graph(self._chunks()), path, has_embeddings=False)

        assert load_graph_meta(path) == {"has_embeddings": False}
        loaded = load_graph(path)
        assert set(loaded) == {"a/one", "a/two"}, "the _meta key must not become a node"

    def test_meta_defaults_to_embeddings_present(self, tmp_path):
        path = str(tmp_path / "graph.json")
        save_graph(build_header_only_graph(self._chunks()), path)
        assert load_graph_meta(path) == {"has_embeddings": True}

    def test_build_refuses_to_downgrade_a_graph_with_neighbors(self, tmp_path, monkeypatch):
        """A lightweight install must not silently delete edges a heavier one computed.

        dyf is installed globally without the model while the project venv has it, so the
        same `dyf concepts build` means different things depending on PATH. Without this
        guard the lightweight one wipes every neighbor — measured at 710 edges lost.
        """
        import dyf.concept_graph as cg

        graph_path = tmp_path / "graph.json"
        source = tmp_path / "notes.md"
        source.write_text("## Alpha\n\nA body long enough to survive the min_length filter.\n")

        # An existing graph that has embeddings.
        save_graph(
            {"notes/alpha": ConceptNode(header="Alpha", source=str(source), line=1)},
            str(graph_path),
            has_embeddings=True,
        )

        def no_model(*args, **kwargs):
            raise ImportError("No module named 'sentence_transformers'")

        monkeypatch.setattr(cg, "build_concept_graph", no_model)

        config = ConceptGraphConfig(
            sources=[str(source)],
            output_path=str(graph_path),
            embeddings_cache_path=str(tmp_path / "emb.npz"),
        )

        assert cg._cmd_build(config, []) == 1, "should refuse rather than downgrade"
        assert load_graph_meta(str(graph_path)) == {"has_embeddings": True}, "graph was overwritten"

        # The explicit flag is the escape hatch and must still work.
        assert cg._cmd_build(config, [], no_embeddings=True) == 0
        assert load_graph_meta(str(graph_path)) == {"has_embeddings": False}

    def test_build_without_model_is_fine_when_no_graph_exists(self, tmp_path, monkeypatch):
        """The fresh-machine case must still produce a usable graph."""
        import dyf.concept_graph as cg

        graph_path = tmp_path / "graph.json"
        source = tmp_path / "notes.md"
        source.write_text("## Alpha\n\nA body long enough to survive the min_length filter.\n")

        def no_model(*args, **kwargs):
            raise ImportError("No module named 'sentence_transformers'")

        monkeypatch.setattr(cg, "build_concept_graph", no_model)

        config = ConceptGraphConfig(
            sources=[str(source)],
            output_path=str(graph_path),
            embeddings_cache_path=str(tmp_path / "emb.npz"),
        )
        assert cg._cmd_build(config, []) == 0
        assert load_graph_meta(str(graph_path)) == {"has_embeddings": False}
        assert set(load_graph(str(graph_path))) == {"notes/alpha"}

    def test_graph_without_meta_still_loads(self, tmp_path):
        """Graphs written before the _meta key existed must keep working."""
        path = tmp_path / "old.json"
        path.write_text(
            json.dumps(
                {
                    "a/one": {
                        "header": "One",
                        "source": "a.md",
                        "line": 1,
                        "metadata": {},
                        "neighbors": [],
                    }
                }
            )
        )
        assert set(load_graph(str(path))) == {"a/one"}
        assert load_graph_meta(str(path)) == {}
