"""`--dry-run` previews for the `dyf index-*` commands.

Every ingest path here is cheap-then-expensive: scan, then parse or decode, then embed.
The embedding pass dominates — an external service round trip per batch for source, a GPU
pass for images and video — and it is the one you cannot take back once started.

`~/Projects/CLAUDE.md`'s *Sanity Check Before Deep Work* rule is the human form of the
judgement this supports: "don't re-embed 2.7M records when a regex on category labels
gets you 95% of the way in 30 seconds". A person applies that by eyeballing the corpus
first. Anything driving dyf as a tool had no way to look before leaping, so the rule was
something to remember rather than something the tool afforded.

Two rules this module holds itself to:

**A dry run stays cheap.** It runs the free and moderate stages and stops *before* the
expensive one. Where even the counting is expensive — video scene detection is a full
decode — it reports the count as unknown rather than doing the work it exists to help you
avoid.

**No invented time estimates.** It reports exact counts: files, chunks, batches. It does
not multiply them by a throughput number, because there is no measured one, and a
plausible fabricated duration is worse than an honest count — it would be believed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

SCHEMA_VERSION = 0


@dataclass
class IngestPreview:
    """What an ingest run would do, without doing the expensive part.

    Attributes:
        command: The subcommand this previews, e.g. "index-source".
        source: Input path.
        output: Where the .dyf would be written.
        model: Embedding model that would be used.
        counts: Exact unit counts discovered cheaply — files, chunks, images.
            A value of None means "not knowable without doing the expensive work".
        batch_size: Embedding batch size.
        batches: Number of embedding calls, or None when the unit count is unknown.
        service: Preflight result for an external service, if the command needs one.
        notes: Anything the caller should weigh before committing to the real run.
    """

    command: str
    source: str
    output: str
    model: str
    batch_size: int
    counts: dict[str, int | None] = field(default_factory=dict)
    batches: int | None = None
    service: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "dry_run": True,
            "command": self.command,
            "source": self.source,
            "output": self.output,
            "model": self.model,
            "counts": self.counts,
            "batch_size": self.batch_size,
            "embedding_batches": self.batches,
            "service": self.service,
            "notes": self.notes,
        }

    def render(self) -> str:
        lines = [
            f"DRY RUN — {self.command} would do the following, and has done nothing.",
            "",
            f"  source        {self.source}",
            f"  output        {self.output}",
            f"  model         {self.model}",
        ]
        if self.service is not None:
            lines.append(f"  service       {self.service}")

        lines.append("")
        for label, value in self.counts.items():
            shown = f"{value:,}" if isinstance(value, int) else "unknown"
            lines.append(f"  {label:<13} {shown}")

        if self.batches is not None:
            lines.append(f"  embed calls   {self.batches:,}  (batch size {self.batch_size})")
        else:
            lines.append(f"  embed calls   unknown  (batch size {self.batch_size})")

        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"  note: {note}")

        lines.append("")
        lines.append("  No embeddings were computed and no file was written.")
        lines.append("  Re-run without --dry-run to proceed.")
        return "\n".join(lines)

    def emit(self, as_json: bool, logger) -> int:
        """Print the preview. Returns the exit code (always 0 — a preview is not a verdict)."""
        if as_json:
            print(json.dumps(self.as_dict(), indent=2, default=str))
        else:
            logger.info("%s", self.render())
        return 0


def batches_for(n_units: int | None, batch_size: int) -> int | None:
    """Number of embedding calls for `n_units`, or None when the count is unknown."""
    if n_units is None or batch_size <= 0:
        return None
    return -(-n_units // batch_size)  # ceil division
