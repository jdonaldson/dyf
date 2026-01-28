"""
ROG Browser - Bokeh implementation with multi-level hierarchical clustering.

Run preprocessing first:
  python demo/rog_preprocess.py demo/wiki_simple_50k.parquet --sample 10000

Then start server:
  bokeh serve demo/rog_panel.py --port 5007 --args demo/wiki_simple_50k_rog_cache.pkl
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, LabelSet, Div, Toggle
from bokeh.layouts import column, row
import colorcet as cc

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CLUSTER_LEVELS = (8, 15, 30)
ZOOM_THRESHOLDS = (1.5, 3.0)
COLORS_30 = cc.glasbey_light[:30]

# -----------------------------------------------------------------------------
# Bokeh Application
# -----------------------------------------------------------------------------

class ROGBrowser:
    def __init__(self, coords_2d: np.ndarray, titles: list[str], cluster_result: dict, bridge_edges: dict):
        self.coords_2d = coords_2d
        self.titles = titles
        self.cluster_result = cluster_result
        self.current_level = CLUSTER_LEVELS[0]

        # Store original extent for zoom ratio calculation
        self.x_min, self.x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        self.y_min, self.y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()
        self.original_width = self.x_max - self.x_min

        # Build color arrays for each level
        self.colors = {}
        for level in CLUSTER_LEVELS:
            labels = cluster_result['labels'][level]
            self.colors[level] = [COLORS_30[l % 30] for l in labels]

        # Create data source
        self.source = ColumnDataSource(data={
            'x': coords_2d[:, 0],
            'y': coords_2d[:, 1],
            'title': titles,
            'color': self.colors[self.current_level],
        })

        # Create label source
        self._update_label_source()

        # Process bundled bridge edges into multi_line format for each level
        self.bridge_edges = {}
        for level, bundled in bridge_edges.items():
            self.bridge_edges[level] = self._process_bundled_edges(bundled)

        # Initialize edge source with current level's bridges
        xs, ys = self.bridge_edges[self.current_level]
        self.edge_source = ColumnDataSource(data={'xs': xs, 'ys': ys})

        # Create figure with WebGL
        self.figure = figure(
            width=1200,
            height=800,
            tools='pan,wheel_zoom,reset',
            active_scroll='wheel_zoom',
            output_backend='webgl',
            title=f"ROG Browser - {len(titles):,} points - Level: {self.current_level} clusters",
        )

        # Add bundled edges (rendered first, behind points)
        self.edge_renderer = self.figure.multi_line(
            'xs', 'ys',
            source=self.edge_source,
            line_color='#4488cc',
            line_alpha=0.15,
            line_width=1,
        )

        # Add points
        self.scatter_renderer = self.figure.scatter(
            'x', 'y',
            source=self.source,
            size=4,
            color='color',
            alpha=0.7,
        )

        # Add hover tool targeting only the scatter points
        from bokeh.models import HoverTool
        hover = HoverTool(tooltips=[("Title", "@title")], renderers=[self.scatter_renderer])
        self.figure.add_tools(hover)

        # Add cluster labels
        self.labels = LabelSet(
            x='x', y='y', text='label',
            source=self.label_source,
            text_font_size='10pt',
            text_color='white',
            background_fill_color='rgba(0,0,0,0.7)',
            background_fill_alpha=0.7,
            text_align='center',
        )
        self.figure.add_layout(self.labels)

        # Toggle for edges
        self.edge_toggle = Toggle(label="Show Edges", active=True, width=100)
        self.edge_toggle.on_change('active', self._on_edge_toggle)

        # Status display
        self.status = Div(text=self._get_status_html(), width=200)

        # Set up zoom detection
        self.figure.x_range.on_change('start', self._on_zoom)
        self.figure.x_range.on_change('end', self._on_zoom)

    def _update_label_source(self):
        """Update the label data source for current level."""
        centroids = self.cluster_result['centroids'][self.current_level]
        names = self.cluster_result['names'][self.current_level]

        self.label_source = ColumnDataSource(data={
            'x': centroids[:, 0],
            'y': centroids[:, 1],
            'label': names,
        })

    def _process_bundled_edges(self, bundled: pd.DataFrame) -> tuple[list, list]:
        """Convert hammer_bundle output to multi_line format."""
        if bundled.empty:
            return [], []

        xs, ys = [], []
        current_x, current_y = [], []

        for _, row in bundled.iterrows():
            if pd.isna(row['x']) or pd.isna(row['y']):
                if current_x:
                    xs.append(current_x)
                    ys.append(current_y)
                    current_x, current_y = [], []
            else:
                current_x.append(row['x'])
                current_y.append(row['y'])

        if current_x:
            xs.append(current_x)
            ys.append(current_y)

        return xs, ys

    def _on_edge_toggle(self, attr, old, new):
        """Toggle edge visibility."""
        self.edge_renderer.visible = new

    def _get_status_html(self) -> str:
        """Generate status HTML."""
        return f"""
        <div style="font-family: sans-serif; padding: 10px;">
            <h3>Cluster Level: {self.current_level}</h3>
            <p>Zoom to change detail level:</p>
            <ul>
                <li>Zoomed out: 8 clusters</li>
                <li>Mid zoom: 15 clusters</li>
                <li>Zoomed in: 30 clusters</li>
            </ul>
        </div>
        """

    def _get_zoom_ratio(self) -> float:
        """Calculate current zoom ratio."""
        x_start = self.figure.x_range.start
        x_end = self.figure.x_range.end

        if x_start is None or x_end is None:
            return 1.0

        current_width = x_end - x_start
        if current_width <= 0:
            return 1.0

        return self.original_width / current_width

    def _get_level_for_zoom(self, zoom_ratio: float) -> int:
        """Map zoom ratio to cluster level."""
        if zoom_ratio < ZOOM_THRESHOLDS[0]:
            return 8
        elif zoom_ratio < ZOOM_THRESHOLDS[1]:
            return 15
        else:
            return 30

    def _on_zoom(self, attr, old, new):
        """Handle zoom changes."""
        zoom_ratio = self._get_zoom_ratio()
        new_level = self._get_level_for_zoom(zoom_ratio)

        if new_level != self.current_level:
            print(f"Zoom ratio: {zoom_ratio:.2f} -> switching to {new_level} clusters")
            self.current_level = new_level
            self._update_view()

    def _update_view(self):
        """Update points, labels, and bridge edges for current level."""
        # Update point colors
        self.source.data['color'] = self.colors[self.current_level]

        # Update labels
        centroids = self.cluster_result['centroids'][self.current_level]
        names = self.cluster_result['names'][self.current_level]

        self.label_source.data = {
            'x': centroids[:, 0],
            'y': centroids[:, 1],
            'label': names,
        }

        # Update bridge edges for current level
        xs, ys = self.bridge_edges[self.current_level]
        self.edge_source.data = {'xs': xs, 'ys': ys}

        # Update title and status
        self.figure.title.text = f"ROG Browser - {len(self.titles):,} points - Level: {self.current_level} clusters"
        self.status.text = self._get_status_html()

    def layout(self):
        """Return the complete layout."""
        controls = column(self.edge_toggle, self.status, width=200)
        return row(self.figure, controls)

    # -------------------------------------------------------------------------
    # Control API (for MCP server)
    # -------------------------------------------------------------------------

    def zoom_to(self, x: float, y: float, radius: float = 5.0):
        """Zoom to center on a point."""
        self.figure.x_range.start = x - radius
        self.figure.x_range.end = x + radius
        self.figure.y_range.start = y - radius
        self.figure.y_range.end = y + radius

    def reset_view(self):
        """Reset to original view."""
        padding = self.original_width * 0.05
        self.figure.x_range.start = self.x_min - padding
        self.figure.x_range.end = self.x_max + padding
        self.figure.y_range.start = self.y_min - padding
        self.figure.y_range.end = self.y_max + padding

    def set_level(self, level: int):
        """Set cluster level manually."""
        if level in CLUSTER_LEVELS:
            self.current_level = level
            self._update_view()

    def set_edges_visible(self, visible: bool):
        """Show or hide edges."""
        self.edge_renderer.visible = visible
        self.edge_toggle.active = visible

    def highlight_points(self, indices: list[int]):
        """Highlight specific points by making others semi-transparent."""
        n = len(self.coords_2d)
        alphas = [0.1] * n
        sizes = [3] * n
        for i in indices:
            if 0 <= i < n:
                alphas[i] = 1.0
                sizes[i] = 8
        self.source.data['alpha'] = alphas
        self.source.data['size'] = sizes

    def clear_highlight(self):
        """Clear highlighting."""
        n = len(self.coords_2d)
        self.source.data['alpha'] = [0.7] * n
        self.source.data['size'] = [4] * n


# Global browser instance for control API
BROWSER = None


# -----------------------------------------------------------------------------
# Control HTTP Handler
# -----------------------------------------------------------------------------

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading


class ControlHandler(BaseHTTPRequestHandler):
    """HTTP handler for control commands from MCP server."""

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        global BROWSER

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        if BROWSER is None:
            self.wfile.write(json.dumps({"error": "Browser not initialized"}).encode())
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            action = data.get("action")
            params = data.get("params", {})

            # Schedule the action on Bokeh's document
            doc = curdoc()

            if action == "zoom_to":
                doc.add_next_tick_callback(
                    lambda: BROWSER.zoom_to(params["x"], params["y"], params.get("radius", 5))
                )
                result = {"status": "ok", "action": "zoom_to"}
            elif action == "reset":
                doc.add_next_tick_callback(BROWSER.reset_view)
                result = {"status": "ok", "action": "reset"}
            elif action == "set_level":
                doc.add_next_tick_callback(lambda: BROWSER.set_level(params["level"]))
                result = {"status": "ok", "action": "set_level", "level": params["level"]}
            elif action == "toggle_edges":
                doc.add_next_tick_callback(lambda: BROWSER.set_edges_visible(params["visible"]))
                result = {"status": "ok", "action": "toggle_edges", "visible": params["visible"]}
            elif action == "highlight":
                doc.add_next_tick_callback(lambda: BROWSER.highlight_points(params["indices"]))
                result = {"status": "ok", "action": "highlight", "count": len(params["indices"])}
            elif action == "clear_highlight":
                doc.add_next_tick_callback(BROWSER.clear_highlight)
                result = {"status": "ok", "action": "clear_highlight"}
            else:
                result = {"error": f"Unknown action: {action}"}

            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self.wfile.write(json.dumps({"error": str(e)}).encode())


def start_control_server(port: int = 5008):
    """Start the control HTTP server in a background thread."""
    server = HTTPServer(("localhost", port), ControlHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Control server: http://localhost:{port}/control")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    global BROWSER

    # Get cache path from command line args
    args = sys.argv[1:] if len(sys.argv) > 1 else []

    # Find the cache file argument
    cache_path = None
    for arg in args:
        if arg.endswith('.pkl'):
            cache_path = arg
            break

    if not cache_path:
        # Default cache path
        cache_path = "demo/wiki_simple_50k_rog_cache.pkl"

    cache_file = Path(cache_path)

    if not cache_file.exists():
        print(f"Cache file not found: {cache_file}")
        print("Run preprocessing first:")
        print(f"  python demo/rog_preprocess.py demo/wiki_simple_50k.parquet --sample 10000")
        return

    print(f"Loading cache from {cache_file}...")
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)

    print("Creating browser...")
    BROWSER = ROGBrowser(
        cache['coords_2d'],
        cache['titles'],
        cache['cluster_result'],
        cache['bridge_edges'],
    )

    curdoc().add_root(BROWSER.layout())
    curdoc().title = "ROG Browser"

    # Start control server for MCP
    start_control_server(port=5008)

    print("Ready!")


main()
