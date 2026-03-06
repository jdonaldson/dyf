"""Tests for tree-sitter-based multi-language source chunking."""

import pytest
from pathlib import Path

from dyf.index_source import chunk_source_file, LANG_CONFIG, _EXT_TO_LANG


# ---------------------------------------------------------------------------
# Fixtures: small source snippets per language
# ---------------------------------------------------------------------------

PYTHON_SRC = """\
def greet(name):
    return f"Hello {name}"

class Calculator:
    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b
"""

JAVASCRIPT_SRC = """\
function greet(name) {
    return `Hello ${name}`;
}

class Calculator {
    add(a, b) {
        return a + b;
    }
}
"""

TYPESCRIPT_SRC = """\
function greet(name: string): string {
    return `Hello ${name}`;
}

class Calculator {
    add(a: number, b: number): number {
        return a + b;
    }
}
"""

RUST_SRC = """\
fn greet(name: &str) -> String {
    format!("Hello {}", name)
}

struct Calculator {
    value: i32,
}

impl Calculator {
    fn add(&self, a: i32, b: i32) -> i32 {
        a + b
    }
}

trait Greeter {
    fn say_hello(&self);
}
"""

GO_SRC = """\
package main

func greet(name string) string {
    return "Hello " + name
}

func (c *Calculator) Add(a, b int) int {
    return a + b
}

type Calculator struct {
    Value int
}
"""

JAVA_SRC = """\
public class Calculator {
    public Calculator() {
        // init
    }

    public int add(int a, int b) {
        return a + b;
    }
}
"""

C_SRC = """\
int add(int a, int b) {
    return a + b;
}

void greet(const char* name) {
    printf("Hello %s\\n", name);
}
"""

CPP_SRC = """\
class Calculator {
public:
    int add(int a, int b);
};

int multiply(int a, int b) {
    return a * b;
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_source(tmp_path):
    """Write a source file to tmp_path and return the path."""
    def _write(filename: str, content: str) -> Path:
        p = tmp_path / filename
        p.write_text(content)
        return p
    return _write


# ---------------------------------------------------------------------------
# Tests: config integrity
# ---------------------------------------------------------------------------

def test_all_extensions_mapped():
    """Every extension in LANG_CONFIG appears in _EXT_TO_LANG."""
    for lang, cfg in LANG_CONFIG.items():
        for ext in cfg["extensions"]:
            assert ext in _EXT_TO_LANG, f"{ext} not in _EXT_TO_LANG"
            assert _EXT_TO_LANG[ext][0] == lang


def test_no_duplicate_extensions():
    """No two languages claim the same extension."""
    seen = {}
    for lang, cfg in LANG_CONFIG.items():
        for ext in cfg["extensions"]:
            assert ext not in seen, f"{ext} claimed by both {seen[ext]} and {lang}"
            seen[ext] = lang


# ---------------------------------------------------------------------------
# Tests: Python
# ---------------------------------------------------------------------------

def test_python_chunks(tmp_source):
    p = tmp_source("example.py", PYTHON_SRC)
    chunks = chunk_source_file(p)

    titles = [c["title"] for c in chunks]
    assert "example.greet" in titles
    assert "example.Calculator" in titles
    assert "example.Calculator.add" in titles
    assert "example.Calculator.sub" in titles

    for c in chunks:
        assert c["language"] == "python"
        assert c["file"] == "example.py"
        assert isinstance(c["line"], int)
        assert c["line"] >= 1


def test_python_kinds(tmp_source):
    p = tmp_source("example.py", PYTHON_SRC)
    chunks = chunk_source_file(p)
    kind_map = {c["title"]: c["kind"] for c in chunks}

    assert kind_map["example.greet"] == "function"
    assert kind_map["example.Calculator"] == "class"
    assert kind_map["example.Calculator.add"] == "method" or kind_map["example.Calculator.add"] == "function"


def test_python_embed_prefix(tmp_source):
    p = tmp_source("example.py", PYTHON_SRC)
    chunks = chunk_source_file(p)
    for c in chunks:
        assert c["text"].startswith("search_document: python")


# ---------------------------------------------------------------------------
# Tests: JavaScript
# ---------------------------------------------------------------------------

def test_javascript_chunks(tmp_source):
    p = tmp_source("app.js", JAVASCRIPT_SRC)
    chunks = chunk_source_file(p)

    titles = [c["title"] for c in chunks]
    assert "app.greet" in titles
    assert "app.Calculator" in titles
    assert "app.Calculator.add" in titles

    for c in chunks:
        assert c["language"] == "javascript"


def test_jsx_extension(tmp_source):
    p = tmp_source("component.jsx", "function App() { return null; }")
    chunks = chunk_source_file(p)
    assert len(chunks) == 1
    assert chunks[0]["language"] == "javascript"


# ---------------------------------------------------------------------------
# Tests: TypeScript
# ---------------------------------------------------------------------------

def test_typescript_chunks(tmp_source):
    p = tmp_source("app.ts", TYPESCRIPT_SRC)
    chunks = chunk_source_file(p)

    titles = [c["title"] for c in chunks]
    assert "app.greet" in titles
    assert "app.Calculator" in titles
    assert "app.Calculator.add" in titles

    for c in chunks:
        assert c["language"] == "typescript"


def test_tsx_extension(tmp_source):
    p = tmp_source("component.tsx", "function App(): JSX.Element { return null; }")
    chunks = chunk_source_file(p)
    assert len(chunks) == 1
    assert chunks[0]["language"] == "typescript"


# ---------------------------------------------------------------------------
# Tests: Rust
# ---------------------------------------------------------------------------

def test_rust_chunks(tmp_source):
    p = tmp_source("lib.rs", RUST_SRC)
    chunks = chunk_source_file(p)

    titles = [c["title"] for c in chunks]
    assert "lib.greet" in titles
    assert "lib.Calculator" in titles  # struct and impl both produce this title
    assert "lib.Calculator.add" in titles  # impl method
    assert "lib.Greeter" in titles  # trait

    # Check kinds by finding specific chunks (struct and impl both named Calculator)
    kinds_by_title = {}
    for c in chunks:
        kinds_by_title.setdefault(c["title"], []).append(c["kind"])

    assert "struct" in kinds_by_title["lib.Calculator"]
    assert "impl" in kinds_by_title["lib.Calculator"]
    assert "trait" in kinds_by_title["lib.Greeter"]

    for c in chunks:
        assert c["language"] == "rust"


# ---------------------------------------------------------------------------
# Tests: Go
# ---------------------------------------------------------------------------

def test_go_chunks(tmp_source):
    p = tmp_source("main.go", GO_SRC)
    chunks = chunk_source_file(p)

    titles = [c["title"] for c in chunks]
    assert "main.greet" in titles
    assert "main.Add" in titles  # method_declaration
    assert "main.Calculator" in titles  # type_declaration

    for c in chunks:
        assert c["language"] == "go"


# ---------------------------------------------------------------------------
# Tests: Java
# ---------------------------------------------------------------------------

def test_java_chunks(tmp_source):
    p = tmp_source("Calculator.java", JAVA_SRC)
    chunks = chunk_source_file(p)

    titles = [c["title"] for c in chunks]
    assert "Calculator.Calculator" in titles  # class
    assert any("add" in t for t in titles)

    constructor_chunks = [c for c in chunks if c["kind"] == "constructor"]
    assert len(constructor_chunks) >= 1

    for c in chunks:
        assert c["language"] == "java"


# ---------------------------------------------------------------------------
# Tests: C
# ---------------------------------------------------------------------------

def test_c_chunks(tmp_source):
    p = tmp_source("math.c", C_SRC)
    chunks = chunk_source_file(p)

    assert len(chunks) == 2
    titles = [c["title"] for c in chunks]
    assert "math.add" in titles
    assert "math.greet" in titles

    for c in chunks:
        assert c["language"] == "c"
        assert c["kind"] == "function"


def test_c_header(tmp_source):
    p = tmp_source("math.h", "int add(int a, int b);")
    chunks = chunk_source_file(p)
    # Header declarations may or may not parse as function_definition
    # (they're usually declaration, not definition) — just verify no crash
    assert isinstance(chunks, list)
    for c in chunks:
        assert c["language"] == "c"


# ---------------------------------------------------------------------------
# Tests: C++
# ---------------------------------------------------------------------------

def test_cpp_chunks(tmp_source):
    p = tmp_source("calc.cpp", CPP_SRC)
    chunks = chunk_source_file(p)

    titles = [c["title"] for c in chunks]
    assert "calc.Calculator" in titles
    assert "calc.multiply" in titles

    kinds = {c["title"]: c["kind"] for c in chunks}
    assert kinds["calc.Calculator"] == "class"
    assert kinds["calc.multiply"] == "function"

    for c in chunks:
        assert c["language"] == "cpp"


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

def test_unsupported_extension(tmp_source):
    p = tmp_source("data.csv", "a,b,c\n1,2,3\n")
    chunks = chunk_source_file(p)
    assert chunks == []


def test_empty_file(tmp_source):
    p = tmp_source("empty.py", "")
    chunks = chunk_source_file(p)
    assert chunks == []


def test_syntax_error_still_parses(tmp_source):
    """tree-sitter is error-tolerant — it should still extract what it can."""
    p = tmp_source("broken.py", "def foo():\n    return 1\n\ndef bar(:\n    pass\n")
    chunks = chunk_source_file(p)
    # Should get at least foo, maybe bar with errors
    assert any("foo" in c["title"] for c in chunks)


def test_text_truncated_at_2000(tmp_source):
    """Embed text should be truncated to ~2000 chars of source."""
    long_fn = "def big():\n" + "    x = 1\n" * 500
    p = tmp_source("big.py", long_fn)
    chunks = chunk_source_file(p)
    assert len(chunks) == 1
    # The embed text includes a prefix, then source truncated at 2000
    source_part = chunks[0]["text"].split("\n", 1)[1] if "\n" in chunks[0]["text"] else ""
    # Total text length should be bounded
    assert len(chunks[0]["text"]) < 2200
