"""
DYF CLI — entry point for `dyf` command.

Usage:
    dyf concepts build     # build concept graph
    dyf concepts query ... # query concepts
    dyf concepts check     # check staleness
    dyf concepts list      # list all nodes
"""

import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "concepts":
        from .concept_graph import main as concepts_main
        sys.exit(concepts_main(sys.argv[2:]))
    else:
        print("Usage: dyf <command>")
        print()
        print("Commands:")
        print("  concepts  Build and query concept graphs from markdown files")
        sys.exit(1)


if __name__ == "__main__":
    main()
