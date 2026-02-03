"""
Combined HTTP + WebSocket server for controlling pydeck visualizations.

Server mode (default):
    python demo/viz_server.py [--port 8766] [--dir demo]

    Serves static HTML files via HTTP and runs a WebSocket endpoint at /ws.
    Browser clients connect on page load and receive commands as JSON.

Command mode:
    python demo/viz_server.py --cmd '{"cmd": "isolate", "cluster": 5}'

    Connects as a WebSocket client, sends the command to the running server,
    and exits. Use this from the CLI or from Claude's Bash tool.

Commands:
    {"cmd": "hide", "cluster": 5}         Hide a cluster
    {"cmd": "show", "cluster": 5}         Show a hidden cluster
    {"cmd": "isolate", "cluster": 5}      Show only this cluster
    {"cmd": "show_all"}                   Show all clusters
    {"cmd": "reset_view"}                 Reset camera to default
    {"cmd": "point_size", "size": 3.0}    Set point size
    {"cmd": "labels", "visible": true}    Toggle label visibility
    {"cmd": "highlight", "indices": [...]} Flash specific points
"""

import argparse
import json
import os
import webbrowser

import tornado.ioloop
import tornado.web
import tornado.websocket
from tornado.websocket import websocket_connect


# All connected browser clients
_clients = set()


class WSHandler(tornado.websocket.WebSocketHandler):
    """WebSocket endpoint for browser clients."""

    def check_origin(self, origin):
        return True  # allow local connections from any origin

    def open(self, *args, **kwargs):
        _clients.add(self)
        print(f"[ws] Client connected ({len(_clients)} total)")

    def on_message(self, message):
        # Browser clients can also send commands (relay to all others)
        for c in _clients:
            if c is not self:
                try:
                    c.write_message(message)
                except tornado.websocket.WebSocketClosedError:
                    pass

    def on_close(self):
        _clients.discard(self)
        print(f"[ws] Client disconnected ({len(_clients)} total)")


class InjectHandler(tornado.web.StaticFileHandler):
    """Serves static files, but injects WS client JS into .html files."""

    def set_extra_headers(self, path):
        # Prevent caching during development
        self.set_header("Cache-Control", "no-cache, no-store, must-revalidate")

    def get_content_type(self):
        if self.absolute_path.endswith(".html"):
            return "text/html"
        return super().get_content_type()


def broadcast(message):
    """Send a JSON message to all connected browser clients."""
    text = message if isinstance(message, str) else json.dumps(message)
    for c in list(_clients):
        try:
            c.write_message(text)
        except tornado.websocket.WebSocketClosedError:
            _clients.discard(c)


def start_server(port=8766, static_dir="demo", open_browser=True):
    """Start the HTTP + WebSocket server."""
    static_dir = os.path.abspath(static_dir)

    app = tornado.web.Application([
        (r"/ws", WSHandler),
        (r"/(.*)", InjectHandler, {"path": static_dir, "default_filename": "index.html"}),
    ])
    app.listen(port)
    print(f"[viz_server] Serving {static_dir} on http://localhost:{port}")
    print(f"[viz_server] WebSocket at ws://localhost:{port}/ws")
    print("[viz_server] Press Ctrl+C to stop")

    if open_browser:
        # Try to open a default HTML file
        for name in ["rog_3d_birch_clusters.html", "rog_3d_dyf_tree_clusters.html"]:
            if os.path.exists(os.path.join(static_dir, name)):
                webbrowser.open(f"http://localhost:{port}/{name}")
                break

    tornado.ioloop.IOLoop.current().start()


async def send_command(cmd, port=8766):
    """Connect as a WS client, send a command, and disconnect."""
    url = f"ws://localhost:{port}/ws"
    text = cmd if isinstance(cmd, str) else json.dumps(cmd)
    try:
        conn = await websocket_connect(url, connect_timeout=5)
        await conn.write_message(text)
        conn.close()
        print(f"[viz_server] Sent: {text}")
    except Exception as e:
        print(f"[viz_server] Failed to connect to {url}: {e}")
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Pydeck visualization server")
    parser.add_argument("--port", type=int, default=8766, help="Port (default: 8766)")
    parser.add_argument("--dir", default="demo", help="Static file directory (default: demo)")
    parser.add_argument("--cmd", type=str, default=None,
                        help="Send a JSON command to a running server and exit")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open browser on start")
    args = parser.parse_args()

    if args.cmd:
        # Command mode: fire-and-forget
        import asyncio
        try:
            cmd = json.loads(args.cmd)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            raise SystemExit(1)
        asyncio.run(send_command(cmd, port=args.port))
    else:
        # Server mode
        start_server(port=args.port, static_dir=args.dir,
                     open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
