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
from bokeh.models import ColumnDataSource, LabelSet, Div, Toggle, TextInput, Button, TapTool
from bokeh.layouts import column, row
import colorcet as cc

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CLUSTER_LEVELS = (5, 12, 25, 50)
ZOOM_THRESHOLDS = (1.5, 3.0, 6.0)
COLORS_50 = cc.glasbey_light[:50]

# -----------------------------------------------------------------------------
# Bokeh Application
# -----------------------------------------------------------------------------

class ROGBrowser:
    def __init__(self, coords_2d: np.ndarray, titles: list[str], cluster_result: dict, bridge_edges: dict):
        self.coords_2d = coords_2d
        self.titles = titles
        self.cluster_result = cluster_result
        self.current_level = CLUSTER_LEVELS[0]
        self._programmatic_zoom = False  # Flag to skip _on_zoom during programmatic changes
        self.last_command = None  # Track last command received
        self.connected = True  # Connection status

        # Animation state
        self._animation = None  # Current animation target
        self._animation_callback = None  # Periodic callback for animation

        # Highlight animation state
        self._highlighted_indices = []
        self._pulse_callback = None
        self._pulse_phase = 0

        # Store original extent for zoom ratio calculation
        self.x_min, self.x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        self.y_min, self.y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()
        self.original_width = self.x_max - self.x_min

        # Build color arrays for each level
        self.colors = {}
        for level in CLUSTER_LEVELS:
            labels = cluster_result['labels'][level]
            self.colors[level] = [COLORS_50[l % 50] for l in labels]

        # Create data source
        n = len(coords_2d)
        self.source = ColumnDataSource(data={
            'x': coords_2d[:, 0],
            'y': coords_2d[:, 1],
            'title': titles,
            'color': self.colors[self.current_level],
            'alpha': [0.7] * n,
            'size': [4] * n,
        })

        # Create label source
        self._update_label_source()

        # Process bundled bridge edges into multi_line format for each level
        self.bridge_edges = {}
        for level, bundled in bridge_edges.items():
            self.bridge_edges[level] = self._process_bundled_edges(bundled)

        # Initialize edge source with current level's bridges
        xs, ys = self.bridge_edges[self.current_level]
        n_edges = len(xs)
        self.edge_source = ColumnDataSource(data={
            'xs': xs, 'ys': ys,
            'line_color': ['#4488cc'] * n_edges,
            'line_alpha': [0.15] * n_edges,
            'line_width': [1] * n_edges,
        })

        # Create figure with WebGL and explicit ranges (no auto-ranging)
        from bokeh.models import Range1d
        padding = self.original_width * 0.05
        self.figure = figure(
            width=1200,
            height=800,
            tools='pan,wheel_zoom,reset',
            background_fill_color='#2a2a2a',
            active_scroll='wheel_zoom',
            output_backend='webgl',
            title=f"ROG Browser - {len(titles):,} points - Level: {self.current_level} clusters",
            x_range=Range1d(self.x_min - padding, self.x_max + padding),
            y_range=Range1d(self.y_min - padding, self.y_max + padding),
        )

        # Add bundled edges (rendered first, behind points)
        self.edge_renderer = self.figure.multi_line(
            'xs', 'ys',
            source=self.edge_source,
            line_color='line_color',
            line_alpha='line_alpha',
            line_width='line_width',
        )
        self._default_edge_color = '#4488cc'
        self._default_edge_alpha = 0.15
        self._default_edge_width = 1

        # Add points
        self.scatter_renderer = self.figure.scatter(
            'x', 'y',
            source=self.source,
            size='size',
            color='color',
            alpha='alpha',
        )

        # Add hover tool targeting only the scatter points
        from bokeh.models import HoverTool
        hover = HoverTool(tooltips=[("Title", "@title")], renderers=[self.scatter_renderer])
        self.figure.add_tools(hover)

        # Add tap tool for clicking to highlight
        tap = TapTool(renderers=[self.scatter_renderer])
        self.figure.add_tools(tap)
        self.source.selected.on_change('indices', self._on_tap_select)

        # Add cluster labels with styled background
        self.labels = LabelSet(
            x='x', y='y', text='label',
            source=self.label_source,
            text_font_size='11pt',
            text_font_style='bold',
            text_color='white',
            background_fill_color='#2d2d2d',
            background_fill_alpha=0.9,
            border_line_color='#666666',
            border_line_width=1,
            border_line_alpha=0.8,
            text_align='center',
        )
        self.figure.add_layout(self.labels)

        # Toggle for edges
        self.edge_toggle = Toggle(label="Show Edges", active=True, width=100)
        self.edge_toggle.on_change('active', self._on_edge_toggle)

        # Search box for finding articles
        self.search_input = TextInput(title="Search articles:", placeholder="Type to search...", width=180)
        self.search_input.on_change('value', self._on_search)

        # Clear highlight button
        self.clear_btn = Button(label="Clear Highlight", button_type="default", width=100)
        self.clear_btn.on_click(self._on_clear_click)

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

    def _on_tap_select(self, attr, old, new):
        """Handle tap selection on points."""
        if new:
            # Highlight selected points and their neighbors
            self.highlight_points(list(new))

    def _on_search(self, attr, old, new):
        """Search for articles by title."""
        if not new or len(new) < 2:
            return
        query = new.lower()
        matches = []
        for i, title in enumerate(self.titles):
            if query in title.lower():
                matches.append(i)
                if len(matches) >= 50:  # Limit matches
                    break
        if matches:
            self.highlight_points(matches)
            # Zoom to first match
            x, y = self.coords_2d[matches[0]]
            self.zoom_to(x, y, radius=3)

    def _on_clear_click(self):
        """Clear highlight button handler."""
        self.clear_highlight()
        self.source.selected.indices = []  # Clear selection too

    def _get_status_html(self) -> str:
        """Generate status HTML."""
        status_color = "#28a745" if self.connected else "#dc3545"
        status_text = "Connected" if self.connected else "Disconnected"
        last_cmd = self.last_command or "None"
        return f"""
        <div style="font-family: sans-serif; padding: 10px;">
            <div style="margin-bottom: 10px; padding: 8px; background: #f8f9fa; border-radius: 4px;">
                <span style="display: inline-block; width: 10px; height: 10px;
                       background: {status_color}; border-radius: 50%; margin-right: 6px;"></span>
                <strong>Control API:</strong> {status_text}<br>
                <small style="color: #666;">Last: {last_cmd}</small>
            </div>
            <h3>Cluster Level: {self.current_level}</h3>
            <p>Zoom to change detail level:</p>
            <ul>
                <li>Zoomed out: 5 clusters</li>
                <li>Light zoom: 12 clusters</li>
                <li>Mid zoom: 25 clusters</li>
                <li>Zoomed in: 50 clusters</li>
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
            return 5
        elif zoom_ratio < ZOOM_THRESHOLDS[1]:
            return 12
        elif zoom_ratio < ZOOM_THRESHOLDS[2]:
            return 25
        else:
            return 50

    def _on_zoom(self, attr, old, new):
        """Handle zoom changes."""
        if self._programmatic_zoom:
            return  # Skip during programmatic zoom changes
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
        n_edges = len(xs)
        self.edge_source.data = {
            'xs': xs, 'ys': ys,
            'line_color': ['#4488cc'] * n_edges,
            'line_alpha': [0.15] * n_edges,
            'line_width': [1] * n_edges,
        }

        # Update title and status
        self.figure.title.text = f"ROG Browser - {len(self.titles):,} points - Level: {self.current_level} clusters"
        self.status.text = self._get_status_html()

    def layout(self):
        """Return the complete layout."""
        controls = column(
            self.search_input,
            self.clear_btn,
            self.edge_toggle,
            self.status,
            width=200
        )
        return row(self.figure, controls)

    # -------------------------------------------------------------------------
    # Control API (for MCP server)
    # -------------------------------------------------------------------------

    def zoom_to(self, x: float, y: float, radius: float = 5.0, animate: bool = True, duration: int = 500):
        """Zoom to center on a point with optional animation.

        Args:
            x, y: Target center coordinates
            radius: Target view radius
            animate: Whether to animate the transition
            duration: Animation duration in milliseconds
        """
        from bokeh.io import curdoc

        target_x_start, target_x_end = x - radius, x + radius
        target_y_start, target_y_end = y - radius, y + radius

        if not animate:
            # Instant zoom
            self._programmatic_zoom = True
            self.figure.x_range.update(start=target_x_start, end=target_x_end)
            self.figure.y_range.update(start=target_y_start, end=target_y_end)
            self._update_zoom_level()
            def clear_flag():
                self._programmatic_zoom = False
            curdoc().add_timeout_callback(clear_flag, 100)
            return

        # Animated zoom
        self._programmatic_zoom = True

        # Cancel any existing animation
        if self._animation_callback:
            try:
                curdoc().remove_periodic_callback(self._animation_callback)
            except:
                pass

        # Animation parameters
        fps = 60
        frames = max(1, int(duration * fps / 1000))
        frame = [0]  # Use list for mutable closure

        # Capture starting positions
        start_x_start = float(self.figure.x_range.start)
        start_x_end = float(self.figure.x_range.end)
        start_y_start = float(self.figure.y_range.start)
        start_y_end = float(self.figure.y_range.end)

        def ease_out_cubic(t):
            """Cubic ease-out for smooth deceleration."""
            return 1 - (1 - t) ** 3

        def animate_step():
            frame[0] += 1
            t = ease_out_cubic(frame[0] / frames)

            # Interpolate all range values
            new_x_start = start_x_start + (target_x_start - start_x_start) * t
            new_x_end = start_x_end + (target_x_end - start_x_end) * t
            new_y_start = start_y_start + (target_y_start - start_y_start) * t
            new_y_end = start_y_end + (target_y_end - start_y_end) * t

            self.figure.x_range.update(start=new_x_start, end=new_x_end)
            self.figure.y_range.update(start=new_y_start, end=new_y_end)

            # Check for level changes during animation
            self._update_zoom_level()

            if frame[0] >= frames:
                # Animation complete
                try:
                    curdoc().remove_periodic_callback(self._animation_callback)
                except:
                    pass
                self._animation_callback = None
                self._programmatic_zoom = False

        self._animation_callback = curdoc().add_periodic_callback(animate_step, int(1000 / fps))

    def _update_zoom_level(self):
        """Update cluster level based on current zoom."""
        zoom_ratio = self._get_zoom_ratio()
        new_level = self._get_level_for_zoom(zoom_ratio)
        if new_level != self.current_level:
            self.current_level = new_level
            self._update_view()

    def reset_view(self, animate: bool = True):
        """Reset to original view with optional animation."""
        padding = self.original_width * 0.05
        center_x = (self.x_min + self.x_max) / 2
        center_y = (self.y_min + self.y_max) / 2
        radius_x = (self.x_max - self.x_min) / 2 + padding
        radius_y = (self.y_max - self.y_min) / 2 + padding
        radius = max(radius_x, radius_y)
        self.zoom_to(center_x, center_y, radius, animate=animate)

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
        """Highlight specific points with pulsing animation."""
        from bokeh.io import curdoc
        import math

        n = len(self.coords_2d)
        self._highlighted_indices = [i for i in indices if 0 <= i < n]

        # Set initial highlight state
        alphas = [0.1] * n
        sizes = [3] * n
        for i in self._highlighted_indices:
            alphas[i] = 1.0
            sizes[i] = 12
        self.source.data['alpha'] = alphas
        self.source.data['size'] = sizes

        # Brighten edges during highlight
        n_edges = len(self.edge_source.data['xs'])
        self.edge_source.data['line_color'] = ['#66aaff'] * n_edges
        self.edge_source.data['line_alpha'] = [0.4] * n_edges
        self.edge_source.data['line_width'] = [1.5] * n_edges

        # Stop existing pulse animation
        if self._pulse_callback:
            try:
                curdoc().remove_periodic_callback(self._pulse_callback)
            except:
                pass

        # Start pulsing animation
        self._pulse_phase = 0

        def pulse():
            if not self._highlighted_indices:
                return
            self._pulse_phase += 0.15
            # Pulse size between 10 and 16
            pulse_size = 13 + 3 * math.sin(self._pulse_phase)
            sizes = [3] * n
            for i in self._highlighted_indices:
                sizes[i] = pulse_size
            self.source.data['size'] = sizes

        self._pulse_callback = curdoc().add_periodic_callback(pulse, 50)  # 20 FPS

    def clear_highlight(self):
        """Clear highlighting and stop animation."""
        from bokeh.io import curdoc

        # Stop pulse animation
        if self._pulse_callback:
            try:
                curdoc().remove_periodic_callback(self._pulse_callback)
            except:
                pass
            self._pulse_callback = None

        self._highlighted_indices = []
        n = len(self.coords_2d)
        self.source.data['alpha'] = [0.7] * n
        self.source.data['size'] = [4] * n

        # Reset edges to default
        n_edges = len(self.edge_source.data['xs'])
        self.edge_source.data['line_color'] = ['#4488cc'] * n_edges
        self.edge_source.data['line_alpha'] = [0.15] * n_edges
        self.edge_source.data['line_width'] = [1] * n_edges


# Shared state container (persists across Bokeh sessions)
class AppState:
    browser = None
    doc = None
    command_queue = []

STATE = AppState()


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
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        if STATE.browser is None or STATE.doc is None:
            self.wfile.write(json.dumps({"error": "Browser not initialized"}).encode())
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            action = data.get("action")
            params = data.get("params", {})

            # Add command to queue - will be processed by Bokeh's periodic callback
            STATE.command_queue.append({"action": action, "params": params})
            print(f"HTTP: Queued {action}, queue id={id(STATE.command_queue)}, len={len(STATE.command_queue)}", flush=True)
            result = {"status": "ok", "action": action, "queued": True}

            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            self.wfile.write(json.dumps({"error": str(e)}).encode())


_control_server_started = False


def start_control_server(port: int = 5008):
    """Start the control HTTP server in a background thread (only once)."""
    global _control_server_started
    if _control_server_started:
        print(f"Control server already running on port {port}")
        return

    try:
        server = HTTPServer(("localhost", port), ControlHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        _control_server_started = True
        print(f"Control server: http://localhost:{port}/control")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"Control server port {port} already in use (likely from previous session)")
            _control_server_started = True
        else:
            raise


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
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
    STATE.browser = ROGBrowser(
        cache['coords_2d'],
        cache['titles'],
        cache['cluster_result'],
        cache['bridge_edges'],
    )

    # Store document reference for control API
    STATE.doc = curdoc()
    STATE.doc.add_root(STATE.browser.layout())
    STATE.doc.title = "ROG Browser"

    # Periodic callback to process command queue
    def process_commands():
        if STATE.command_queue:
            print(f"BOKEH: Processing {len(STATE.command_queue)} commands, queue id={id(STATE.command_queue)}", flush=True)
        while STATE.command_queue:
            cmd = STATE.command_queue.pop(0)
            action = cmd["action"]
            params = cmd.get("params", {})
            print(f"  Executing: {action} with {params}")
            try:
                if action == "zoom_to":
                    STATE.browser.zoom_to(params["x"], params["y"], params.get("radius", 5))
                elif action == "reset":
                    STATE.browser.reset_view()
                elif action == "set_level":
                    STATE.browser.set_level(params["level"])
                elif action == "toggle_edges":
                    STATE.browser.set_edges_visible(params["visible"])
                elif action == "highlight":
                    STATE.browser.highlight_points(params["indices"])
                elif action == "clear_highlight":
                    STATE.browser.clear_highlight()
                # Update last command and refresh status
                STATE.browser.last_command = action
                STATE.browser.status.text = STATE.browser._get_status_html()
            except Exception as e:
                print(f"Error processing command {action}: {e}")

    STATE.doc.add_periodic_callback(process_commands, 100)  # Check every 100ms

    # Start control server for MCP
    start_control_server(port=5008)

    print("Ready!")


main()
