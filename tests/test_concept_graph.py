"""Tests for dyf.concept_graph — concept graph building and querying."""

import json
import os
import time

from dyf.concept_graph import (
    ConceptGraphConfig,
    ConceptNode,
    MarkdownChunk,
    chunk_markdown,
    check_staleness,
    fuzzy_match,
    load_graph,
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
                    {"id": "test/node-b", "header": "Node B",
                     "similarity": 0.85, "source": "test.md", "line": 10},
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
        config = ConceptGraphConfig.load(str(tmp_path / "missing.json"))
        assert config.top_k == 5  # defaults

    def test_load_from_file(self, tmp_path):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({
            "sources": ["~/test.md"],
            "output_path": "/tmp/test_graph.json",
            "embeddings_cache_path": "/tmp/test_embeddings.npz",
            "embedder": "medium",
            "top_k": 3,
            "similarity_threshold": 0.3,
        }))
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
