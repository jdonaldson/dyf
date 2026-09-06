# Agent Legibility TODO

Work queue for making dyf usable *by agents* — as a CLI they invoke, a package they
import, and a repo they read. Opened 2026-09-05.

## Relationship to the v1 heading

`CLAUDE.md`'s heading is **v1 quality, not more capability**, scored by "can someone
depend on this?". Everything below *is* that work: a mute CLI, a contract two modules
disagree on, a module named for something it isn't, and output no caller can parse are
all dependency failures. Agents are just the consumer that fails loudest, because they
cannot squint at a blank terminal and infer what happened.

**Scope decision, 2026-09-05.** The heading previously named only `sec10quant` and
`shortorder` as consumers — both of which import Python and never shell out. On the
strict reading the CLI was out of scope entirely, including the mute-logging bug. The
heading is now amended: **agents are a first-class consumer, via both the CLI and the
Python API.** The whole of this file is on-heading. P0→P3 is ordered by size and
blocking-ness, not by scope.

The guard that came with the amendment: agent work earns its place only where it makes an
*existing* surface dependable. Parseable output, loud failure, honest docs — yes. New
mechanisms to serve agents — no.

## Scope change: the tour split (2026-09-05)

`enrich/` (12 modules) and `tour.py` — 2,875 lines — moved to the new downstream
**`dyfviz`** package, so dyf is indexing and search. Every open item below that concerns
UMAP projection, Louvain clustering, LLM labelling, narration, TTS or the browser viewer
went with them and is now dyfviz's queue, specifically:

- `enrich audio` needing undeclared `kokoro` + `soundfile`, and its `SystemExit` from
  inside a library call
- `enrich reannotate` being dead against current output; the inert `--cluster-level` flag
- `fit_birch` / `merge_tiny_clusters` never called from `src/`
- `_scaffold._group_label_from_names` hardcoding medical-device vocabulary

**Measurements taken before the split are left as written** — they were true of dyf at
the time and are the evidence for the fixes that followed. The `enrich`/`tour` rows in
the P0 logging table are the clearest case: that bug was real and shared, and the fix
went to both packages.

Two couplings deliberately survive the split, because they are format conventions rather
than code dependencies:

- `lazy_index.detect_enrichment_level()` still names `edge_pairs` / `tour_narration`.
  dyf reads levels that dyfviz writes; that is the point of the ladder.
- `dyf info` still reports the level, and still excludes `tour_audio` from its summary.

## The organizing finding (2026-09-05)

The three standing audits pass cleanly at HEAD `707aebc`: public API **41/41 OK, 0
degenerate** with the canary intact; assertion audit **596 tests, 28 shape-only (5%),
zero in every other category**, selftest 34/34; threshold audit reproduces its table.

Every defect found in this review lives **outside the boundary those audits can see**:

| defect | why the audits miss it |
| --- | --- |
| CLI emits nothing | audits test the Python API, not `cli.py` |
| provenance key mismatch | it's a *contract between two modules*, not one callable |
| media ingest layer | not exported, so not reachable by `audit_public_api` |
| image/video enrichment degraded | returns correctly-shaped output; the failure is semantic |

The heading says *close the gap between shipped and validated surface*. This is where
that gap is. Note the consequence for the current top next-action: running the audits in
CI locks in the surface that is already clean and touches none of the above.

---

## P0 — CLOSED 2026-09-05: the CLI was mute, unreachable, and unbuildable

All six items done. What this section was, on the morning of 2026-09-05: the CLI printed
**zero bytes** from every subcommand; the `dyf` binary existed only inside this project's
`.venv` while `~/.claude/CLAUDE.md` told every agent to run it; the concept graph was six
months stale and could not be rebuilt without a multi-GB torch stack; and `Pipeline`
reported every `.dyf` artifact as stale forever.

Kept in full rather than deleted — each entry records what was measured, and three of
them describe bugs introduced *by an earlier fix in the same session*, which is the more
useful lesson.

- [x] **Add a log handler in `cli.py:main()`.** *(done 2026-09-05, `_configure_cli_logging`)*
      Scoped to the `dyf` logger, **not** `basicConfig` — the first attempt used
      `basicConfig`, which configures the *root* logger and made httpx dump every
      HuggingFace request during `concepts build`. Verified after: `concepts check` →
      ``STALE - run `dyf concepts build` ``; `concepts list` → 76 nodes;
      `index-source` → prints its header instead of nothing. 0 HTTP lines.

      **What it was.** `__init__.py:263` installs a `NullHandler` (correct for a library —
      it suppresses Python's handler of last resort), and **no `basicConfig` existed
      anywhere in the package**. Every `logger.info`/`logger.warning` in a subcommand was
      swallowed. Measured before the fix:

      | subcommand | logger calls | print calls | result |
      | --- | --- | --- | --- |
      | `concepts` | 19 | 0 | silent |
      | `index-source` | 17 | 0 | silent |
      | `index-images` | 21 | 0 | silent |
      | `index-video` | 21 | 0 | silent |
      | `enrich` | 102 across submodules | 25 | **mixed** |
      | `tour` | 0 | 7 | fine |

      Measured before: `dyf concepts list` → **0 bytes, rc 0** on a graph with 100+ nodes;
      `dyf concepts check` → **0 bytes, rc 1**.
      `enrich` is the worst case — `_narration`/`_splits`/`_reannotate` still `print`
      while `_cluster` (28), `_project` (18), `_viz` (16) are mute, so it looks alive
      while the progress reporting is gone. That also violates the *Inspectable
      Experiments* rule in `~/Projects/CLAUDE.md`: `index-video` on a long file now
      prints nothing until it exits.

- [x] **Make `dyf` reachable.** *(done 2026-09-05.)*
      `uv tool install --editable '~/Projects/dyf[lazy]'` → `~/.local/bin/dyf`, which is
      already on PATH and already how `harlequin` and `mflux` are installed here.
      Editable, so it tracks the working tree rather than freezing a copy. `[lazy]` only:
      that is what `dyf info` needs (flatbuffers + pyarrow), and it deliberately omits
      `[concepts]` so the global tool stays free of the multi-GB torch stack. Verified
      from `$HOME`: `dyf info` reads a 229k-item index, `dyf concepts check` and
      `query` both work.

      **Why the global tool still answers `concepts query` with neighbors** despite having
      no model: neighbors are computed at build time and saved into the graph, and
      `fuzzy_match` is pure `SequenceMatcher`. Read is dependency-free; only *write* needs
      the model. That asymmetry is what makes the whole arrangement work.

- [x] **Decide how a global tool gets its dependencies.** *(done 2026-09-05 — answered by
      the two items above, not by a packaging change.)* The tool does not need them for
      the common path: header-only build plus precomputed neighbors means `list`, `check`
      and `query <header>` all work with zero optional deps. The heavy stack is needed
      only to *rebuild* neighbors, which is a project-venv job.

      ⚠ That arrangement created a data-loss path, caught only by installing the tool and
      testing it: a global `build` silently replaced the full graph with a header-only
      one, **destroying 710 edges**. Now refused unless `--no-embeddings` is passed. The
      general shape is worth remembering — *the same command name resolving to two
      installs with different capabilities* is a hazard, and the fix is to make the
      lesser one refuse to clobber the greater one's output.

- [x] **Rebuild the concept graph.** *(done 2026-09-05 — 76 nodes → **142 nodes, 710
      edges**; `check` now returns `OK - graph is current`, rc 0.)* It had been built
      **2026-03-04**, six months stale: it still indexed `claude/tmux-topic-trace` where
      the section is now *Tmux Task Trace*, and had no node for the newer
      `Filenames are metadata` rule — which is why querying that phrase returned nothing
      even after the CLI could speak. Required installing the `[concepts]` extra first
      (below).

- [x] **`concepts build` crashed on a missing optional dep.**
      *(done 2026-09-05 — `build_header_only_graph`, automatic fallback, 7 tests.)*
      Found the moment the handler fix made the CLI audible: `check` said
      ``STALE - run `dyf concepts build` `` and `build` then died with a bare
      `ModuleNotFoundError: No module named 'sentence_transformers'` (`configs.py:138`) —
      the advice the tool gave was advice it could not itself follow, on the very first
      command after P0. Now degrades to a header-only graph with a loud warning, and
      `--semantic` refuses with a rebuild instruction instead of crashing.

- [x] **Fix the provenance key mismatch.** *(done 2026-09-05, `_dyf_provenance_value`,
      5 tests.)* `pipeline.py` read `_provenance`; the enrichment stages only ever wrote
      `_provenance_level_1/2/3`. Verified on a real artifact before the fix: a
      dyfviz-shaped `.dyf` read back as `None` and its stage status was
      `'stale (no provenance)'`; after, `FOUND` and `'fresh'`. So every `.dyf` stage
      rebuilt unconditionally and the caching this module exists for never engaged.

      Now reads `_provenance`, falling back to the **highest** `_provenance_level_N` —
      the most recent stage to touch the file, so its params hash is the right one to
      compare. Reading a key dyfviz writes is the same format convention
      `detect_enrichment_level` already uses, so this reduces inconsistency rather than
      adding coupling.

      ⚠ **The `.dyf` branch had no test coverage at all.** Every existing pipeline test
      hand-writes `_provenance` into a `.pkl` fixture, so the suite passed while the path
      that matters for real artifacts was broken — the degenerate-fixture pattern again,
      third instance today. Added 5 tests including one end-to-end on a real `.dyf`.

      - [ ] Still open, and now sharper after the split: **nothing in dyf writes
        provenance at all.** The ingest modules (`index_source`, `index_images`,
        `index_video`) stamp `build_params` but no provenance, and `Pipeline` never
        stamps after running a stage — it assumes `build_fn` did. So `provenance.py`
        exports 7 public symbols with no in-package producer. Either the ingest path
        should stamp, or the module should say plainly that stamping is the caller's job.

## P1 — give both surfaces a contract

An agent cannot distinguish *empty result* from *broken tool*. Every silent success path
is a place an agent will confidently report the wrong thing.

- [x] **`dyf info <file.dyf>` — describe an artifact without loading it.**
      *(done 2026-09-05, `src/dyf/info.py`, 10 tests in `tests/test_info.py`)*
      Built entirely over primitives that already existed — `total_items`
      (`lazy_index.py:1223`), `stored_field_names` (`:1235`), `tree_summary` (`:1249`),
      `detect_enrichment_level` (`:2077`). The architecture was already right; nothing
      exposed it, so the only way to learn what a `.dyf` held was to write Python.

      **Measured: 0.09 s on `sec10quant`'s real 479 MB / 229,243-item index**, including
      interpreter startup — the cheap-index-over-expensive-body claim holds on a real
      corpus, not just a fixture. Reports items, dim, leaves/nodes, build params, stored
      fields, domain, enrichment level, provenance stages.

- [x] **Structured output (`--json`), explicitly unstable.**
      *(done 2026-09-05 for `dyf info`.)* Emits `{"schema_version": 0, ...}` documented as
      carrying no compatibility guarantee before v1, which decouples *parseable* from
      *committed*. Known-broken fields are omitted rather than serialized — `gap_detected`
      is always `False` and `adaptive_nprobe` resolves to 5 for 85% of queries; publishing
      either into a contract agents will trust is worse than publishing nothing.
      - [x] `--json` for `concepts query` and `concepts list` *(done 2026-09-05.)*
        `query` reports the match with its fuzzy score, neighbors with similarities, and
        a `graph` block; `list` reports a neighbor count per node, expanding under `-v`.
        Exit codes verified identical between the human and JSON paths — `--json` must
        change how something is reported, never what happened.

        ⚠ **Stdout purity had to be enforced at the boundary.** `configs.py` alone has 21
        bare `print()` calls, and sentence-transformers and tqdm print too; measured, the
        semantic path emitted unparseable stdout. `_query_json` now runs the work under
        `contextlib.redirect_stdout` and replays the noise to stderr. Chasing 21 call
        sites would not have covered the third-party ones — and one stray line makes a
        whole payload unparseable, which is a worse failure than the silence it replaced.

      - [ ] Noticed while testing, not changed: **`semantic_search` has no similarity
        floor**, so a nonsense query returns `top_k` confident-looking results and exits
        0 on a full graph. The JSON carries the scores so a caller *can* judge, but a
        human reading the text output cannot. Design question, not a bug.

- [x] **CLI results go to stdout, problems to stderr.** *(done 2026-09-05.)* Caught while
      testing `dyf info`: `logging.StreamHandler` defaults to **stderr**, so the P0
      logging fix had made the CLI audible but put its answers on the wrong stream —
      `dyf info f.dyf > out.txt` would have written an empty file. Handlers are now split
      at `WARNING`. Verified by subprocess capture: stdout 573 bytes, stderr 0.
      ⚠ A shell check of this is easy to get wrong — zsh's MULTIOS made
      `2>&1 >/dev/null` appear to show the summary on stderr. Assert on captured streams
      in a subprocess, not on shell redirection.

- [ ] **Make the Python API discoverable.** `__all__` has ~109 entries in one flat list,
      dominated by tree/clustering/RAG primitives, with the entire media half absent
      (P2). An agent importing `dyf` has no cheap way to learn what exists or where to
      start. Same index/body split as `dyf info`: a grouped `__init__` docstring, or a
      one-line summary per public symbol, readable without opening 30 modules.

- [x] **Return-shape consistency for the search API.** *(done 2026-09-05, 11 tests.)*
      `LazyIndex.search` and `search_ivf` returned a `SearchResult`; `DenseSearchIndex.search`
      returned a bare `(indices, scores)` tuple, which cannot be introspected or extended
      and forces the caller to already know the arity. Both now return `SearchResult`, so
      the two index types are interchangeable at the call site.

      **Non-breaking**, because `SearchResult` already implements
      `__iter__`/`__getitem__`/`__len__` — someone had deliberately made it unpack as a
      2-tuple, which is what let this be reconciled without touching a single caller.
      Verified single and batched unpacking, positional indexing, and `len()`.

      ⚠ **`DenseSearchIndex` had no tests at all** despite being public, exported, and
      shown in the README — one of the 31 callables `audit_public_api.py` cannot exercise
      without a hand-written fixture. That set is where this class of defect lives.
      The 11 new tests assert *behaviour*, not shape: nearest neighbour of a vector is
      itself, scores rank descending, batched matches single, higher `nprobe` does not
      reduce recall.

      - [x] **All four retrievers now agree** *(2026-09-05)* — `LazyIndex.search`,
        `LazyIndex.search_ivf`, `DenseSearchIndex.search` and `BridgeIndex.query` return
        `SearchResult`; `query_batch` returns `list[SearchResult]`. Verified on real
        objects, all still unpacking.
      - [x] **`SearchResult.__len__` returned a hard-coded 2.** With the type now used by
        four entry points, `len(result)` reported **2 on a k=10 search** — and spreading
        the type is what widened that. Returns the hit count now. Safe because unpacking
        goes through `__iter__`, never `__len__` — checked directly, with a class whose
        `__len__` returns 99 that still unpacks fine.
      - [ ] Follow-up: `SearchResult` is defined in `lazy_index.py` but is now the shared
        return type of the whole search API. It arguably belongs in a neutral module.
        Not moved — that would break `from dyf.lazy_index import SearchResult` for anyone
        doing it, for no functional gain today.

- [ ] **Remaining return-shape work, from the full audit of all 109 exports (2026-09-05).**
      Ranked by "would freezing this into v1 be a mistake?". The retrieval API is done;
      these are not.

      1. **`embed_with_diagnostics` returns a 4-tuple** (`categorical.py:782`) whose 2nd
         and 3rd elements are *the same type* (`list[AxisDiagnostic]`) and differ only by
         semantics — before/after weighting. Swapping them is silent and type-checks
         clean. One return site (`:837`) already passes the same object for both.
      2. **`agglomerate_tree_leaves` (`agglomerate.py:477`) and `louvain_cluster_leaves`
         (`:553`) return identical unnamed 5-tuples**, documented as interchangeable — a
         de-facto struct with no name. Both return a degenerate
         `(None, {}, [], None, tree)` on the too-few-leaves path, an in-band `None`
         sentinel where the rest of the API raises `ValueError`.
      3. **`dedup_for_index` (`dedup.py:232`) is half-tuple, half-object** — element 3 is
         already a `DedupResult` while the other two stay positional, in the same module
         where `near_duplicate_clusters` returns a clean `DedupResult`.
      4. **Missing-item contract differs between adjacent methods**:
         `LazyIndex.get_item_vector` (`:1925`) raises `KeyError`;
         `LazyIndex.get_stored_fields` (`:1962`) fills `None` for the same condition.
      5. **"Not found" is `-1` in two places and `None` in two others** —
         `CategoryGraph.lca_depth` (`categorical.py:111`) and `CatalogSpace.get_lca_depth`
         (`catalog.py:1401`) vs `DAGTaxonomy.get_path` (`ontology.py:804`) and
         `ROGResult.get_layer_for_node` (`:1353`). The `-1` is worse: it does arithmetic
         silently.
      6. **`lazy_index.py` uses two "named result" idioms** — `TypedDict` for `TreeNode`
         and `ExtractedData`, `@dataclass` for `SearchResult` and `AdaptiveProbeConfig`.
         Consumers get `d["node_id"]` for one and `r.indices` for the other.
      7. **`LazyIndex.tree_summary` is the agent-facing call and the least typed thing in
         the file** — a nested untyped dict with a *conditional* schema (`pq` and
         `stored_fields` appear only sometimes), while the adjacent
         `get_tree_structure` returns `list[TreeNode]`. `dyf info` depends on it.
      8. **Tree constructors return structurally-typed dicts** — `build_pca_tree` yields
         `{left, right, ...}`, `build_dyf_tree` yields `{children, ...}`, and
         `cut_tree_to_labels` (`cut.py:20`) tells them apart by **sniffing for the
         `"children"` vs `"left"` key**. v1 would freeze that duck-typing as the contract.
      9. Raw dicts with fixed known keys that want small dataclasses — `refine_dyf_tree`
         (pure telemetry, and the project already has a `*Report` naming precedent),
         `cluster_quality`, `BridgeIndex.evaluate_recall`, `flatten_tree` (9 fixed arrays,
         exported, unpacked by key), and `compute_split_keywords` /
         `compute_embedding_keywords` — which have *identical* schemas, so one shared type
         would prove they are interchangeable.

      Cleanly typed already, for contrast: `catalog.py` and `ontology.py`. The untyped
      ones are `agglomerate.py`, `splits.py`, `categorical.py`.

- [ ] **Hidden requirements must be declarable before they fail.** `index-source` needs a
      live Ollama server at `localhost:11434` and only discovers this by raising mid-run
      (`index_source.py:247-272`, `raise_for_status`, no fallback). Agents need a
      preflight — a `--check` or a fast fail with the reason — not a stack trace after
      the embedding pass has started.

- [ ] **Cost preview before expensive work.** dyf's whole domain is corpora where the
      embedding pass is the expensive step, and `~/Projects/CLAUDE.md`'s *Sanity Check
      Before Deep Work* rule is the human version of this judgment ("don't re-embed 2.7M
      records when a regex gets 95%"). An agent has no way to apply it. A `--dry-run`
      reporting item count and estimated work is how that rule gets encoded rather than
      remembered.

- [x] **CLI smoke audit — the missing fourth audit.**
      *(done 2026-09-05, `benchmarks/audit_cli_surface.py`, now in the CLAUDE.md table.)*
      Runs a real invocation of every subcommand and classifies the result as
      OK / MUTE / TRACEBACK / CLEAN-FAIL / SKIP. `--selftest` checks the classifier
      against 9 hand-labelled cases plus a live canary that is deliberately mute — the
      lesson from `audit_test_assertions.py`, whose scanner had a 32% false-positive rate
      and had inflated every number it ever reported.

      ⚠ **Design finding: it must not test `--help`.** argparse prints help itself,
      writing to stdout without ever touching the logger, so **all 7 subcommands returned
      rc 0 with output for `--help` while 3 crashed on a real invocation**. A help-only
      smoke test would have been blind to the precise bug that motivated it.

      First run: **5 OK, 3 TRACEBACK, 1 CLEAN-FAIL, 1 SKIP — and 0 MUTE**, confirming the
      P0 fix holds across the whole surface, not just where it was spot-checked. After
      the ImportError fix below: **5 OK, 4 CLEAN-FAIL, 1 SKIP, 0 MUTE, 0 TRACEBACK**,
      audit exits 0.

- [x] **Optional dependencies fail as tracebacks, not as messages.**
      *(done 2026-09-05.)* Fixed once at the CLI boundary rather than at five call sites:
      `main()` catches `ImportError` around dispatch, prints the exception plus
      `pip install 'dyf[<extra>]'` from a `COMMAND_EXTRAS` map, and exits 3. Modules that
      already raise a good message keep their own wording — `index-source` said
      `Install it with: pip install "dyf[source]"` all along, buried under a traceback
      nobody caught. Measured before: `index-source`, `index-images` and `enrich project`
      all raised bare tracebacks; `index-video` was the one command already failing
      cleanly, so the in-repo example to copy existed.
      - [ ] Still open: `enrich audio` needs `kokoro` + `soundfile`, declared in **no**
        extra, so there is no `dyf[...]` to name. Either add the extra or drop the path.
      - [ ] Still open: `index-source` also needs a live Ollama server and only finds out
        mid-run (`index_source.py:247-272`). A missing *service* is not an ImportError,
        so the boundary fix does not cover it.
- [x] **`concepts query` with no match must say so.** *(done 2026-09-05.)* It fell
      through to semantic search and could return zero lines at rc 0. Now exits 1 when
      nothing matched — grep's convention, since finding nothing is a negative answer
      rather than a failure — a header-only graph says it has no semantic fallback
      instead of attempting one, and an empty semantic result says so explicitly.

- [x] **Audit exit codes across `concepts`.** *(done 2026-09-05, documented in the module
      docstring as a contract.)* `0` success, `1` normal negative answer (no match, or
      `check` finding the graph stale), `2` bad request (missing/malformed `--config`),
      `3` missing dependency. Verified identical between human and `--json` paths.
      - [ ] Still to do: the same pass over `info` and the `index-*` commands. `info`
        already uses 1/2/3 consistently; `index-*` have not been audited.

- [x] **Config errors were tracebacks — and worse, silence.** *(done 2026-09-05,
      `ConfigError`, 6 tests.)* The reported symptom was a raw `JSONDecodeError` from
      `--config /dev/null`. The bug next to it was worse: **an explicitly requested config
      that did not exist fell back to defaults silently**, so a caller passing `--config`
      believed their `output_path` was in effect while the tool wrote to `~/.dyf/`.
      Missing, malformed, non-object and unknown-key configs now all exit 2 naming the
      file and the problem — and for unknown keys, the valid settings.
      ⚠ This changed behaviour a test had pinned (`test_load_missing_file`). Broken
      deliberately per the pre-v1 rule, with the reason recorded in the test.

- [ ] **Make the remaining error messages actionable.** `concepts` is done; `index-*` and
      `info` still have paths that say only what went wrong, not what to do.

- [ ] **Optional dependencies fail as tracebacks, not as messages.** One generator, five
      instances — worth one fix at the CLI boundary rather than five patches:

      | site | today |
      | --- | --- |
      | `concepts build` | raw `ModuleNotFoundError: sentence_transformers` (`configs.py:138`), no guidance |
      | `index-source` | *has* a good `ImportError: tree-sitter-language-pack is required for source indexing.` — buried under a full traceback |
      | `index-source` | also needs a live Ollama server; `raise_for_status`, no fallback (`index_source.py:247-272`) |
      | `enrich audio` | needs `kokoro` + `soundfile`, declared in **no** pyproject extra |
      | `tests/test_dedup.py` | **fails** rather than skips when `[lazy]`/`flatbuffers` is absent, unlike the 70 tests that skip cleanly |

      Fix: catch `ImportError` in `cli.main()`, print `<message> — pip install 'dyf[<extra>]'`,
      exit non-zero. An agent can act on that line; it cannot act on a traceback.

## P2 — make the package self-describing

- [ ] **Three modules are named for things they are not.** `catalog.py` (1446 lines) is a
      multi-catalog *taxonomy matcher* for UNSPSC/BroadJump/Curvo with **zero connection
      to the ingest path** — its only in-repo consumer is `tests/test_catalog.py`.
      `ontology.py` is *discovered* structure in embedding space, not a declared schema.
      `provenance.py` is real but partial. A reader scanning the module list builds the
      wrong model of the package. Fix by renaming, or by a one-line module docstring that
      contradicts the name loudly.
- [ ] **Decide what dyf is, then say it once.** `README.md` says "discover structure in
      embedding spaces" (Dense/Bridge/Orphan topology). The working one-liner is
      "generalized embeddings space over arbitrary media". These describe different
      products. Pick one; the other is at best a section.
- [ ] **The media half is absent from `__init__.py`.** `index_images`, `index_video`,
      `index_source` and all of `enrich/` are CLI-only — there is no `dyf.index_images(...)`.
      The ~109-entry `__all__` is dominated by tree/clustering/RAG primitives. Either
      export it or state plainly that it is CLI-only.
- [ ] **Document that image/video enrichment is degraded, not equivalent.** Enrichment is
      media-agnostic only because it treats everything as *text titles*: images get the
      filename (`index_images.py:163`), video gets `"Scene 3 at 1:24"`
      (`index_video.py:219`). LLM labeling and TF-IDF then run over filenames and scene
      strings, never pixels. Video titles will likely trip `assess_text_diversity`
      (`_cluster.py:140`) into the frequency-label fallback. Silent quality cliff.
- [ ] **Cross-reference this file from `KNOWN_ISSUES.md`** so there is one entry point to
      the open queue rather than two.
- [ ] **Prune or fence the dead paths** (each is a trap for an agent reading the code as
      documentation):
      - `enrich reannotate` is dead against current output — it discovers
        `cluster_<k>_2d/_3d` fields that only `demo/dyfviz.py` ever wrote, and
        `enrich_cluster` actively deletes them as stale (`_cluster.py:443-455`).
      - `--cluster-level` is inert; `_viz.py:128-158`'s legacy branch only fires when
        fields the current pipeline always writes are missing.
      - `fit_birch` (`_cluster.py:19`) and `merge_tiny_clusters` (`_cluster.py:53`) are
        never called from `src/`. `merge_tiny_clusters` has tests exercising code the
        product does not use.
      - `_scaffold._group_label_from_names` (`_scaffold.py:24-33`) hardcodes
        medical-device vocabulary and brand names (`cardinal`, `medline`, `baxter`) in a
        general module; on any non-GUDID corpus every group scores `None` and labels
        silently fall through to the largest member's name.
      - `enrich audio` needs `kokoro` + `soundfile`, declared in no pyproject extra.

## P3 — larger work, still bounded by the heading

The heading was amended 2026-09-05 to make agents a first-class consumer, so these no
longer conflict with it. They are P3 because they are *large*, not because they are out
of scope. The unchanged guard: they must make an existing surface dependable, not add a
new mechanism.

- [ ] **Unify the three ingest scripts behind one interface.** No base class, registry or
      protocol exists today; each of `index_images`/`index_video`/`index_source`
      independently repeats the same normalize → `build_dyf_tree` → `write_lazy_index`
      tail plus its own argparse block. The copy-paste already costs something concrete:
      **`--dedup` exists only for source** (`index_source.py:284`), while near-identical
      video keyframes are the textbook case for it.
- [ ] **`llms.txt` for dyf.io** plus heading anchors, so agents can cite and retrieve a
      fragment rather than a page.

## Deliberately not doing

- Making `catalog.py` part of the ingest path. It is a separate product that happens to
  live here; the fix is naming and docs, not integration.
- Real audio ingestion. `enrich/_audio.py` is Kokoro TTS *output* for tour narration —
  the opposite direction. Adding an audio encoder is a project, not a cleanup.
