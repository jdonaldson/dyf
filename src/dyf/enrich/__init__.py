"""
dyf enrich — Enrich a .dyf file through progressive levels.

    Level 0 → 1 (project): Add UMAP coordinates as stored fields
    Level 1 → 2 (cluster): Add Louvain cluster labels + dendrogram
    Level 2 → 3 (viz):     Add bridge edges + tour narration

Usage:
    dyf enrich project demo/gudid_50k_titled.dyf
    dyf enrich cluster demo/gudid_50k_titled.dyf
    dyf enrich viz demo/gudid_50k_titled.dyf
    dyf enrich all demo/gudid_50k_titled.dyf
"""

import argparse
import sys


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="dyf enrich",
        description="Enrich a .dyf file through progressive levels")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # project
    p_proj = subparsers.add_parser(
        "project", help="Level 0→1: Add UMAP coordinates")
    p_proj.add_argument("dyf_path", help="Path to .dyf file")
    p_proj.add_argument("--n-components", type=int, default=3,
                        help="UMAP dimensions (default: 3)")
    p_proj.add_argument("--densmap", action="store_true",
                        help="Use densMAP")
    p_proj.add_argument("--fisher-col", default=None,
                        help="Column name for Fisher dimension weighting")
    p_proj.add_argument("--fisher-parquet", default=None,
                        help="Parquet file containing the Fisher column")
    p_proj.add_argument("--diagnose-parquet", default=None,
                        help="Parquet file for axis diagnostics sanity check")
    p_proj.add_argument("-o", "--output", default=None,
                        help="Output path (default: overwrite input)")

    # cluster
    p_clust = subparsers.add_parser(
        "cluster", help="Level 1→2: Louvain clustering + dendrogram")
    p_clust.add_argument("dyf_path", help="Path to .dyf file")
    p_clust.add_argument("--model", default="gpt-oss:20b",
                         help="Ollama model for labeling")
    p_clust.add_argument("--force", action="store_true",
                         help="Re-run even if already at level 2+")
    p_clust.add_argument("--domain", default=None,
                         help="Domain description for LLM prompts")
    p_clust.add_argument("--resolution", type=float, default=1.0,
                         help="Louvain resolution (default: 1.0)")
    p_clust.add_argument("-o", "--output", default=None)

    # viz
    p_viz = subparsers.add_parser(
        "viz", help="Level 2→3: Add bridge edges + narration")
    p_viz.add_argument("dyf_path", help="Path to .dyf file")
    p_viz.add_argument("--cluster-level", type=int, default=None,
                       help="Which cluster level for edges (default: auto)")
    p_viz.add_argument("--model", default="gpt-oss:20b")
    p_viz.add_argument("--title", default=None,
                       help="Title for intro narration")
    p_viz.add_argument("--force", action="store_true",
                       help="Re-run even if already at level 3")
    p_viz.add_argument("--domain", default=None,
                       help="Domain description for narration prompts")
    p_viz.add_argument("-o", "--output", default=None)

    # tree
    p_tree = subparsers.add_parser(
        "tree", help="Add tree-based hierarchical labels")
    p_tree.add_argument("dyf_path", help="Path to .dyf file")
    p_tree.add_argument("--depth", type=int, default=3,
                        help="Tree depth for branches (default: 3)")
    p_tree.add_argument("--samples", type=int, default=8,
                        help="Titles to sample per child (default: 8)")
    p_tree.add_argument("--model", default="gpt-oss:20b",
                        help="Ollama model for labeling")
    p_tree.add_argument("-o", "--output", default=None)

    # splits
    p_splits = subparsers.add_parser(
        "splits", help="Compute tree split keywords (no LLM)")
    p_splits.add_argument("dyf_path", help="Path to .dyf file")
    p_splits.add_argument("--depth", type=int, default=3,
                          help="Max depth from root (default: 3)")
    p_splits.add_argument("--bigram-check", action="store_true",
                          help="Enable PMI-based compound meaning detection")
    p_splits.add_argument("-o", "--output", default=None)

    # reannotate
    p_reann = subparsers.add_parser(
        "reannotate", help="Re-run glyph annotations without re-clustering")
    p_reann.add_argument("dyf_path", help="Path to .dyf file")
    p_reann.add_argument("-o", "--output", default=None)

    # audio
    p_audio = subparsers.add_parser(
        "audio", help="Generate Kokoro TTS audio for tour narration")
    p_audio.add_argument("dyf_path", help="Path to .dyf file")
    p_audio.add_argument("--voice", default="bf_emma",
                         help="Kokoro voice ID (default: bf_emma)")
    p_audio.add_argument("--speed", type=float, default=1.0,
                         help="Playback speed (default: 1.0)")
    p_audio.add_argument("-o", "--output", default=None)

    # all
    p_all = subparsers.add_parser(
        "all", help="Run all enrichment levels in sequence")
    p_all.add_argument("dyf_path", help="Path to .dyf file")
    p_all.add_argument("--model", default="gpt-oss:20b")
    p_all.add_argument("--title", default=None)
    p_all.add_argument("--domain", default=None,
                       help="Domain description for LLM prompts")
    p_all.add_argument("--fisher-col", default=None,
                       help="Column name for Fisher dimension weighting")
    p_all.add_argument("--fisher-parquet", default=None,
                       help="Parquet file containing the Fisher column")
    p_all.add_argument("--diagnose-parquet", default=None,
                       help="Parquet file for axis diagnostics sanity check")
    p_all.add_argument("-o", "--output", default=None)

    args = parser.parse_args(argv)

    if args.command == "project":
        from ._project import enrich_project
        enrich_project(args.dyf_path, n_components=args.n_components,
                       densmap=args.densmap, output_path=args.output,
                       fisher_col=args.fisher_col,
                       fisher_parquet=args.fisher_parquet,
                       diagnose_parquet=args.diagnose_parquet)

    elif args.command == "cluster":
        from ._cluster import enrich_cluster
        enrich_cluster(args.dyf_path, model=args.model,
                       output_path=args.output, force=args.force,
                       domain=args.domain, resolution=args.resolution)

    elif args.command == "viz":
        from ._viz import enrich_viz
        enrich_viz(args.dyf_path, cluster_level=args.cluster_level,
                   model=args.model, title=args.title,
                   output_path=args.output, force=args.force,
                   domain=args.domain)

    elif args.command == "splits":
        from ._splits import enrich_splits
        enrich_splits(args.dyf_path, max_depth=args.depth,
                      bigram_check=args.bigram_check,
                      output_path=args.output)

    elif args.command == "reannotate":
        from ._reannotate import reannotate
        reannotate(args.dyf_path, output_path=args.output)

    elif args.command == "tree":
        from ._tree import enrich_tree
        enrich_tree(args.dyf_path, model=args.model,
                    target_depth=args.depth,
                    samples_per_child=args.samples,
                    output_path=args.output)

    elif args.command == "audio":
        from ._audio import generate_tour_audio
        generate_tour_audio(args.dyf_path, voice=args.voice,
                            speed=args.speed, output_path=args.output)

    elif args.command == "all":
        from ._cluster import enrich_cluster
        from ._project import enrich_project
        from ._viz import enrich_viz
        out = args.output or args.dyf_path
        enrich_project(args.dyf_path, output_path=out,
                       fisher_col=getattr(args, 'fisher_col', None),
                       fisher_parquet=getattr(args, 'fisher_parquet', None),
                       diagnose_parquet=getattr(args, 'diagnose_parquet', None))
        enrich_cluster(out, model=args.model,
                       output_path=out, domain=args.domain)
        enrich_viz(out, cluster_level=None,
                   model=args.model, title=args.title, output_path=out,
                   domain=args.domain)
        # Generate TTS audio if kokoro is available
        try:
            from ._audio import generate_tour_audio
            generate_tour_audio(out, output_path=out)
        except SystemExit:
            print("  Skipping audio generation (kokoro not installed)")
