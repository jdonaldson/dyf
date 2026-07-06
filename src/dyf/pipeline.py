"""Lightweight DAG pipeline runner for the DYF visualization pipeline.

Each ``Stage`` knows its inputs, output, and build function.  The runner
checks timestamps + provenance hashes to decide what to rebuild.

Example usage::

    from dyf.pipeline import Pipeline, Stage

    p = Pipeline()
    p.add(Stage(
        name="embed",
        inputs=["data/gudid_full.json"],
        output="demo/gudid_energy_devices.parquet",
        build_fn=lambda: embed_devices(...),
        params={"model": "nomic-ai/nomic-embed-text-v1.5"},
    ))
    p.add(Stage(
        name="rog_cache",
        inputs=["embed"],
        output="demo/gudid_energy_devices_rog_cache.pkl",
        build_fn=lambda: run_rog_preprocess(...),
        params={"sample": 0, "bridge_level": 100},
    ))

    p.status()       # {"embed": "fresh", "rog_cache": "stale ..."}
    p.run("viz")     # rebuild stale stages in topo order
"""

from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

from .provenance import (
    Provenance,
    params_hash,
    provenance_from_dict,
)


@dataclass
class Stage:
    """A single pipeline stage."""

    name: str
    inputs: list[str]  # file paths or stage names
    output: str  # output file path
    build_fn: Callable[[], None]  # function to call
    params: dict = field(default_factory=dict)  # build parameters (for hash)


class Pipeline:
    """Lightweight make-style DAG runner."""

    def __init__(self) -> None:
        self.stages: dict[str, Stage] = {}

    def add(self, stage: Stage) -> None:
        """Register a stage."""
        self.stages[stage.name] = stage

    # ------------------------------------------------------------------
    # DAG helpers
    # ------------------------------------------------------------------

    def _resolve_input(self, inp: str) -> str:
        """Resolve an input to a file path.

        If *inp* matches a stage name, returns that stage's output path.
        Otherwise returns *inp* as-is (assumed to be a file path).
        """
        if inp in self.stages:
            return self.stages[inp].output
        return inp

    def _topo_sort(self, target: str | None = None) -> list[str]:
        """Topological sort of stages (Kahn's algorithm).

        If *target* is given, only stages required to build that target
        are returned.
        """
        if target and target not in self.stages:
            raise KeyError(f"Unknown stage: {target!r}")

        # Build adjacency + in-degree
        deps: dict[str, set[str]] = {name: set() for name in self.stages}
        for name, stage in self.stages.items():
            for inp in stage.inputs:
                if inp in self.stages:
                    deps[name].add(inp)

        # If targeting a specific stage, prune to reachable ancestors
        if target:
            needed: set[str] = set()
            stack = [target]
            while stack:
                cur = stack.pop()
                if cur in needed:
                    continue
                needed.add(cur)
                stack.extend(deps.get(cur, set()))
            deps = {k: v for k, v in deps.items() if k in needed}

        # Kahn's
        in_degree = {name: len(d) for name, d in deps.items()}
        queue = [n for n, d in in_degree.items() if d == 0]
        order: list[str] = []
        while queue:
            queue.sort()  # deterministic order
            node = queue.pop(0)
            order.append(node)
            for name, d in deps.items():
                if node in d:
                    d.discard(node)
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)

        if len(order) != len(deps):
            raise ValueError("Cycle detected in pipeline DAG")

        return order

    # ------------------------------------------------------------------
    # Provenance I/O per artifact type
    # ------------------------------------------------------------------

    @staticmethod
    def _read_provenance(path: str) -> Provenance | None:
        """Try to read provenance from a file, return None if absent."""
        p = Path(path)
        if not p.exists():
            return None

        suffix = p.suffix.lower()
        try:
            if suffix == ".pkl":
                with open(p, "rb") as f:
                    data = pickle.load(f)
                raw = data.get("_provenance") if isinstance(data, dict) else None
                if raw:
                    return provenance_from_dict(raw)

            elif suffix == ".json":
                with open(p) as f:
                    data = json.load(f)
                raw = data.get("_provenance") if isinstance(data, dict) else None
                if raw:
                    return provenance_from_dict(raw)

            elif suffix == ".dyf":
                # Lazy import to avoid circular / heavy deps
                from .lazy_index import LazyIndex

                with LazyIndex(str(p)) as idx:
                    meta = idx._get_metadata()
                    raw_str = meta.get("_provenance")
                    if raw_str:
                        return provenance_from_dict(json.loads(raw_str))
        except Exception as e:
            logger.debug("Could not read provenance from %s: %s", path, e)

        return None

    # ------------------------------------------------------------------
    # Status / staleness
    # ------------------------------------------------------------------

    def _stage_status(self, name: str) -> str:
        """Return status for a single stage."""
        stage = self.stages[name]
        out = Path(stage.output)
        if not out.exists():
            return "missing"

        # Check if params hash matches
        prov = self._read_provenance(stage.output)
        if prov is None:
            return "stale (no provenance)"

        current_ph = params_hash(stage.params)
        if prov.params_hash != current_ph:
            return "stale (params changed)"

        # Check if any input is newer than output
        out_mtime = out.stat().st_mtime
        for inp in stage.inputs:
            inp_path = Path(self._resolve_input(inp))
            if inp_path.exists() and inp_path.stat().st_mtime > out_mtime:
                return f"stale (input {inp} newer)"

        # Check if upstream stages are stale
        for inp in stage.inputs:
            if inp in self.stages:
                upstream_status = self._stage_status(inp)
                if upstream_status != "fresh":
                    return f"stale (depends on {inp})"

        return "fresh"

    def status(self) -> dict[str, str]:
        """Return ``{stage_name: status}`` for all stages."""
        return {name: self._stage_status(name) for name in self.stages}

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, target: str | None = None, dry_run: bool = False) -> list[str]:
        """Topological-sort stages, rebuild stale ones.

        Returns list of stage names that were rebuilt.
        """
        order = self._topo_sort(target)
        rebuilt: list[str] = []

        for name in order:
            st = self._stage_status(name)
            if st == "fresh":
                logger.info(f"  [{name}] fresh — skipping")
                continue

            stage = self.stages[name]
            logger.info(f"  [{name}] {st} — {'would rebuild' if dry_run else 'rebuilding'}...")

            if not dry_run:
                stage.build_fn()
                rebuilt.append(name)

        return rebuilt

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def explain(self, target: str) -> str:
        """Human-readable explanation of what would be rebuilt and why."""
        order = self._topo_sort(target)
        lines: list[str] = []
        for name in order:
            stage = self.stages[name]
            st = self._stage_status(name)
            inputs_str = ", ".join(stage.inputs)
            lines.append(f"  {name}: {st}")
            lines.append(f"    inputs: [{inputs_str}]")
            lines.append(f"    output: {stage.output}")
        return "\n".join(lines)
