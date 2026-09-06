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

## P0 — the CLI produces no output at all

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

- [ ] **Make `dyf` reachable.** It exists only at `.venv/bin/dyf`. `~/.claude/CLAUDE.md`
      instructs every agent, every session, to run `dyf concepts check` before editing
      CLAUDE.md sections or memory — those agents get `command not found`.

- [x] **Rebuild the concept graph.** *(done 2026-09-05 — 76 nodes → **142 nodes, 710
      edges**; `check` now returns `OK - graph is current`, rc 0.)* It had been built
      **2026-03-04**, six months stale: it still indexed `claude/tmux-topic-trace` where
      the section is now *Tmux Task Trace*, and had no node for the newer
      `Filenames are metadata` rule — which is why querying that phrase returned nothing
      even after the CLI could speak. Required installing the `[concepts]` extra first
      (below).

- [ ] **`concepts build` crashes on a missing optional dep.** Found the moment the
      handler fix made the CLI audible: `check` says ``STALE - run `dyf concepts build` ``,
      and `build` then dies with a bare
      `ModuleNotFoundError: No module named 'sentence_transformers'`
      (`configs.py:138`). The advice the tool gives is advice it cannot itself follow.
      Catch the import and say `pip install 'dyf[concepts]'`. This is the P1
      *actionable error messages* item, hit on the very first command after P0.

- [ ] **Decide how a global tool gets its dependencies.** `~/.claude/CLAUDE.md` tells
      every agent in every session to run `dyf concepts`, but the graph lives at
      `~/.dyf/` while the only `dyf` binary lives in this project's `.venv`, and building
      requires the `[concepts]` extra — which pulls **torch + transformers, multi-GB**,
      absent here today. Note the asymmetry: `fuzzy_match` (`concept_graph.py:294`) is
      pure `SequenceMatcher` and needs nothing, so *querying by header is free* — but you
      cannot obtain a graph at all without the expensive path. Options: a header-only
      build mode with no embeddings (loses semantic neighbors, keeps the index), a
      lighter embedder, or ship the tool separately from the research package.

- [ ] **Fix the provenance key mismatch.** `pipeline.py:164` reads metadata key
      `"_provenance"`; the enrich stages only ever write `_provenance_level_1/2/3`
      (`_project.py:190`, `_cluster.py:432`, `_viz.py:176`). The pipeline runner cannot
      read the provenance its own package writes and reports every `.dyf` as
      `"stale (no provenance)"`. The ingest modules stamp none at all. This is the
      mechanism behind the *DAG-Oriented Task Flow* section's "is a cached intermediate
      still valid?" — currently it always answers no.

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
      - [ ] Still to do: `--json` for `concepts query` and `concepts list`. (`enrich` and
        `index-*` emit progress, which agents do not need to parse — this helps two or
        three commands well, not the whole surface.)

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

- [ ] **Audit return-shape consistency.** `LazyIndex.search` returns a result object with
      `.indices`/`.scores`/`.fields`; `DenseSearchIndex.search` returns a bare
      `(indices, scores)` tuple. A tuple cannot be introspected or extended — an agent
      has to already know the arity. Pick one convention before v1 freezes both.

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
- [ ] **`concepts query` with no match must say so.** It currently falls through to
      semantic search and can return zero lines at rc 0.
- [ ] **Audit exit codes across subcommands.** `check` returning rc 1 for STALE is
      correct and useful; confirm the rest are meaningful and documented.
- [ ] **Make error messages actionable.** An error string is documentation delivered
      exactly when it's needed at zero context cost until then. Prefer
      `no graph at ~/.dyf/concept_graph.json — run: dyf concepts build` over `not found`.

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
