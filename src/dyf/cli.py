"""
DYF CLI — entry point for `dyf` command.

Usage:
    dyf concepts build     # build concept graph
    dyf concepts query ... # query concepts
    dyf concepts check     # check staleness
    dyf concepts list      # list all nodes
    dyf index-source dir/  # index source code into .dyf
"""

import sys


def main():
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "concepts":
            from .concept_graph import main as concepts_main
            sys.exit(concepts_main(sys.argv[2:]))
        elif cmd == "index-source":
            from .index_source import main as index_main
            sys.exit(index_main(sys.argv[2:]))

    print("Usage: dyf <command>")
    print()
    print("Commands:")
    print("  concepts      Build and query concept graphs from markdown files")
    print("  index-source  Index source code into a .dyf file (Python, JS, TS, Rust, Go, Java, C, C++)")
    sys.exit(1)


if __name__ == "__main__":
    main()
