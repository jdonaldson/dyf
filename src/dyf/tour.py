"""
dyf tour — Launch the browser viewer with tour autoplay.

Usage:
    dyf tour demo/gudid_50k_louvain_r4.dyf
    dyf tour haxe_compiler.dyf --port 8800
"""

import argparse
import atexit
import signal
import subprocess
import sys
import webbrowser
from pathlib import Path


def _find_demo_dir():
    """Locate the demo/ directory relative to this package."""
    # Walk up from src/dyf/tour.py to find the project root
    pkg_dir = Path(__file__).resolve().parent  # src/dyf/
    # Try: pkg_dir/../../demo (source install)
    candidate = pkg_dir.parent.parent / "demo"
    if candidate.is_dir() and (candidate / "dyf_viewer.html").exists():
        return candidate
    # Try: site-packages install — look for demo relative to cwd
    cwd_demo = Path.cwd() / "demo"
    if cwd_demo.is_dir() and (cwd_demo / "dyf_viewer.html").exists():
        return cwd_demo
    return None


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="dyf tour",
        description="Launch the browser viewer with tour autoplay")
    parser.add_argument("dyf_path", help="Path to .dyf file")
    parser.add_argument("--port", type=int, default=8766,
                        help="Server port (default: 8766)")
    parser.add_argument("--no-autoplay", action="store_true",
                        help="Don't auto-start the tour")
    args = parser.parse_args(argv)

    dyf_file = Path(args.dyf_path).resolve()
    if not dyf_file.exists():
        print(f"Error: {dyf_file} not found")
        sys.exit(1)

    demo_dir = _find_demo_dir()
    if demo_dir is None:
        print("Error: could not locate demo/ directory with dyf_viewer.html")
        sys.exit(1)
    demo_dir = demo_dir.resolve()

    # Symlink the .dyf file into demo/ if not already there
    symlink_path = None
    if dyf_file.parent != demo_dir:
        symlink_path = demo_dir / dyf_file.name
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(dyf_file)

        def _cleanup():
            if symlink_path and symlink_path.is_symlink():
                symlink_path.unlink(missing_ok=True)
        atexit.register(_cleanup)

    # Build viewer URL
    autoplay = "" if args.no_autoplay else "&autoplay=1"
    url = (f"http://localhost:{args.port}/dyf_viewer.html"
           f"?file={dyf_file.name}{autoplay}")

    # Start viz_server.py
    viz_server = demo_dir / "viz_server.py"
    if not viz_server.exists():
        print(f"Error: {viz_server} not found")
        sys.exit(1)

    print(f"Starting viewer on port {args.port}...")
    print(f"  File: {dyf_file}")
    print(f"  URL: {url}")

    proc = subprocess.Popen(
        [sys.executable, str(viz_server),
         "--port", str(args.port),
         "--dir", str(demo_dir),
         "--no-browser"],
        cwd=str(demo_dir),
    )

    def _kill_server(*_args):
        proc.terminate()
        proc.wait(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, _kill_server)
    signal.signal(signal.SIGTERM, _kill_server)

    # Give server a moment to start, then open browser
    import time
    time.sleep(1.0)
    webbrowser.open(url)

    print("Press Ctrl+C to stop.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        _kill_server()
