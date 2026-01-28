"""
ROG Browser - Bokeh implementation with multi-level hierarchical clustering.

Run preprocessing first:
  python demo/rog_preprocess.py demo/wiki_simple_50k.parquet --sample 10000

Then start server:
  bokeh serve demo/rog_panel.py --port 5007 --args demo/wiki_simple_50k_rog_cache.pkl
"""

import sys
import pickle
import json
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, LabelSet, Div, Toggle, TextInput, Button, TapTool
from bokeh.layouts import column, row
import colorcet as cc
from scipy.cluster.hierarchy import fcluster

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
    def __init__(self, coords_2d: np.ndarray, titles: list[str], cluster_result: dict,
                 bridge_edges: dict, edge_indices: list[tuple[int, int]] | None = None,
                 cluster_pairs: dict | None = None):
        self.coords_2d = coords_2d
        self.titles = titles
        self.cluster_result = cluster_result
        self.edge_indices = edge_indices or []  # (source, target) pairs for edge highlighting

        # Check if dendrogram data is available for dynamic cutting
        self.dendrogram = cluster_result.get('dendrogram')
        self.use_dynamic_dendrogram = self.dendrogram is not None
        if self.use_dynamic_dendrogram:
            print("Dendrogram data available - using dynamic cutting", flush=True)
            self._dendrogram_cache = {}  # Cache for computed cuts: n_clusters -> (labels, centroids, names)
        else:
            self._dendrogram_cache = {}

        # Store ordered list of cluster pairs for edge highlighting
        # Edge i corresponds to cluster_pair_list[i]
        # IMPORTANT: Must be sorted by count descending to match preprocessing order
        self.cluster_pairs = cluster_pairs or {}
        if cluster_pairs:
            sorted_pairs = sorted(cluster_pairs.items(), key=lambda x: -x[1])
            self.cluster_pair_list = [pair for pair, count in sorted_pairs]
        else:
            self.cluster_pair_list = []

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

        # Build STABLE hierarchical colors:
        # - Top-level cluster (5) determines base hue
        # - Micro-cluster determines variation within that hue
        # This keeps boundaries visible while preventing color flashing
        self.stable_colors = self._build_hierarchical_colors(cluster_result)

        # Legacy: keep colors dict for compatibility but won't be used for updates
        self.colors = {}
        for level in CLUSTER_LEVELS:
            labels = cluster_result['labels'][level]
            self.colors[level] = [COLORS_50[l % 50] for l in labels]

        # Create data source with stable colors
        n = len(coords_2d)
        self.source = ColumnDataSource(data={
            'x': coords_2d[:, 0],
            'y': coords_2d[:, 1],
            'title': titles,
            'color': self.stable_colors,  # Use stable colors, not level-dependent
            'alpha': [0.7] * n,
            'size': [4] * n,
            'line_color': ['rgba(0,0,0,0)'] * n,  # Transparent outline by default
            'line_width': [1] * n,
        })

        # Create label source
        self._update_label_source()

        # Bridge edges are pre-processed into (xs, ys) format during preprocessing
        self.bridge_edges = bridge_edges

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
            line_color='line_color',
            line_width='line_width',
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

        # Set up zoom/pan detection (both x and y ranges)
        self.figure.x_range.on_change('start', self._on_zoom)
        self.figure.x_range.on_change('end', self._on_zoom)
        self.figure.y_range.on_change('start', self._on_zoom)
        self.figure.y_range.on_change('end', self._on_zoom)

    def _build_hierarchical_colors(self, cluster_result: dict) -> list[str]:
        """Build stable colors with hierarchical structure.

        Top-level cluster (5) determines base hue, micro-cluster adds variation.
        This makes cluster boundaries visible while keeping colors stable.
        """
        import colorsys

        n_points = len(self.coords_2d)

        # Get top-level (5) cluster assignments for base hue
        top_labels = cluster_result['labels'][5]

        # Get micro-cluster assignments for variation
        if self.use_dynamic_dendrogram:
            micro_labels = self.dendrogram['kmeans_labels']
            n_micro = self.dendrogram['n_micro']
        else:
            micro_labels = cluster_result['labels'][50]
            n_micro = 50

        # Define 5 distinct base hues (well-separated on color wheel)
        base_hues = [0.0, 0.15, 0.35, 0.55, 0.75]  # Red, Orange, Green, Cyan, Purple

        colors = []
        for i in range(n_points):
            top_cluster = int(top_labels[i])
            micro_cluster = int(micro_labels[i])

            # Base hue from top-level cluster
            base_hue = base_hues[top_cluster % len(base_hues)]

            # Small hue variation from micro-cluster (±0.05)
            hue_variation = ((micro_cluster % 10) - 5) * 0.01
            hue = (base_hue + hue_variation) % 1.0

            # Saturation/lightness variation from micro-cluster
            sat = 0.5 + (micro_cluster % 7) * 0.07  # 0.5-0.92
            light = 0.45 + (micro_cluster % 5) * 0.08  # 0.45-0.77

            # Convert HSL to RGB
            r, g, b = colorsys.hls_to_rgb(hue, light, sat)
            colors.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')

        return colors

    def _update_label_source(self):
        """Update the label data source for current level."""
        centroids = self.cluster_result['centroids'][self.current_level]
        names = self.cluster_result['names'][self.current_level]
        # Add padding spaces around label text
        padded_names = [f'  {name}  ' for name in names]

        self.label_source = ColumnDataSource(data={
            'x': centroids[:, 0],
            'y': centroids[:, 1],
            'label': padded_names,
        })

    def _cut_dendrogram_dynamic(self, n_clusters: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Dynamically cut the dendrogram to get n_clusters.

        Uses caching to avoid recomputation for the same n_clusters.

        Returns:
            labels: Array of cluster assignments per point
            centroids: Array of cluster centroids (n_clusters, 2)
            names: List of cluster names
        """
        if not self.use_dynamic_dendrogram:
            # Fallback to fixed levels
            level = min(CLUSTER_LEVELS, key=lambda l: abs(l - n_clusters))
            return (
                self.cluster_result['labels'][level],
                self.cluster_result['centroids'][level],
                self.cluster_result['names'][level],
            )

        # Check cache
        if n_clusters in self._dendrogram_cache:
            return self._dendrogram_cache[n_clusters]

        # Check if we have this level pre-computed
        if n_clusters in self.cluster_result['labels']:
            result = (
                self.cluster_result['labels'][n_clusters],
                self.cluster_result['centroids'][n_clusters],
                self.cluster_result['names'][n_clusters],
            )
            self._dendrogram_cache[n_clusters] = result
            return result

        # Cut dendrogram dynamically
        Z = self.dendrogram['Z']
        kmeans_labels = self.dendrogram['kmeans_labels']
        node_labels = self.dendrogram['node_labels']
        node_points = self.dendrogram['node_points']
        node_centroids = self.dendrogram['node_centroids']

        # Cut at n_clusters
        micro_cluster_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
        point_labels = np.array([micro_cluster_labels[m] for m in kmeans_labels])

        # Find representative nodes and compute names/centroids
        from scipy.cluster.hierarchy import to_tree
        tree = to_tree(Z, rd=True)
        root, nodes = tree

        cluster_names = [f"Cluster {i}" for i in range(n_clusters)]
        cluster_centroids = []

        for cluster_id in range(n_clusters):
            mask = point_labels == cluster_id
            cluster_point_indices = np.where(mask)[0]

            if len(cluster_point_indices) == 0:
                cluster_centroids.append(np.array([0.0, 0.0]))
                continue

            # Find best matching internal node
            best_node = None
            best_size = float('inf')
            cluster_pts = set(cluster_point_indices)

            for node in nodes:
                if node.is_leaf():
                    continue
                node_pts = set(node_points[node.id])
                if cluster_pts.issubset(node_pts) and len(node_pts) < best_size:
                    best_size = len(node_pts)
                    best_node = node

            if best_node is not None and best_node.id in node_labels:
                cluster_names[cluster_id] = node_labels[best_node.id]

            # Compute centroid
            if best_node is not None and best_node.id in node_centroids:
                cx, cy = node_centroids[best_node.id]
                cluster_centroids.append(np.array([cx, cy]))
            else:
                centroid = self.coords_2d[cluster_point_indices].mean(axis=0)
                cluster_centroids.append(centroid)

        result = (point_labels, np.array(cluster_centroids), cluster_names)
        self._dendrogram_cache[n_clusters] = result
        return result

    def _update_labels_for_viewport(self):
        """Update labels based on viewport - show 2-12 labels, aggregating as needed.

        If dendrogram data is available, uses dynamic cutting to find the optimal
        number of clusters (any value from 2-100) that results in 2-12 visible labels.
        """
        # Get current viewport bounds
        x_start = self.figure.x_range.start
        x_end = self.figure.x_range.end
        y_start = self.figure.y_range.start
        y_end = self.figure.y_range.end

        if x_start is None or x_end is None or y_start is None or y_end is None:
            return

        cx_view = (x_start + x_end) / 2
        cy_view = (y_start + y_end) / 2

        def count_in_view(centroids):
            """Count centroids visible in viewport."""
            return sum(1 for cx, cy in centroids
                      if x_start <= cx <= x_end and y_start <= cy <= y_end)

        if self.use_dynamic_dendrogram:
            # Dynamic dendrogram: binary search for optimal n_clusters
            # that gives us 2-12 visible labels
            min_k, max_k = 2, 100
            best_k = 5
            best_count = 0

            # Binary search for the largest k that gives <= 12 visible
            while min_k <= max_k:
                mid_k = (min_k + max_k) // 2
                _, centroids, _ = self._cut_dendrogram_dynamic(mid_k)
                visible_count = count_in_view(centroids)

                if visible_count <= 12:
                    best_k = mid_k
                    best_count = visible_count
                    min_k = mid_k + 1  # Try more clusters
                else:
                    max_k = mid_k - 1  # Try fewer clusters

            # Get the chosen cut
            labels, centroids, names = self._cut_dendrogram_dynamic(best_k)
            chosen_level = best_k
        else:
            # Fixed levels: use traditional approach
            level_counts = {}
            for level in CLUSTER_LEVELS:
                centroids = self.cluster_result['centroids'][level]
                level_counts[level] = count_in_view(centroids)

            # Choose finest level where count <= 12
            chosen_level = 50
            for level in CLUSTER_LEVELS:
                if level_counts[level] <= 12:
                    chosen_level = level
                else:
                    break

            centroids = self.cluster_result['centroids'][chosen_level]
            names = self.cluster_result['names'][chosen_level]
            labels = self.cluster_result['labels'][chosen_level]

        # Get centroids in view
        in_view = [(i, cx, cy) for i, (cx, cy) in enumerate(centroids)
                   if x_start <= cx <= x_end and y_start <= cy <= y_end]

        # If < 2 in view, add nearest centroids to get at least 2
        if len(in_view) < 2:
            in_view_ids = {i for i, _, _ in in_view}
            distances = [(i, (cx - cx_view)**2 + (cy - cy_view)**2, cx, cy)
                        for i, (cx, cy) in enumerate(centroids)
                        if i not in in_view_ids]
            distances.sort(key=lambda x: x[1])
            needed = 2 - len(in_view)
            for i, _, cx, cy in distances[:needed]:
                in_view.append((i, cx, cy))

        # Update label source with chosen labels
        label_x = [cx for _, cx, _ in in_view]
        label_y = [cy for _, _, cy in in_view]
        label_text = [f'  {names[i]}  ' for i, _, _ in in_view]

        self.label_source.data = {
            'x': label_x,
            'y': label_y,
            'label': label_text,
        }

        # Update current level (colors stay stable - only labels change)
        if chosen_level != self.current_level:
            self.current_level = chosen_level
            self.figure.title.text = f"ROG Browser - {len(self.titles):,} points - Level: {self.current_level} clusters"

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
        """Handle zoom changes - update labels based on viewport."""
        if self._programmatic_zoom:
            return  # Skip during programmatic zoom changes
        # Use viewport-aware label aggregation
        self._update_labels_for_viewport()

    def _update_view(self):
        """Update labels and bridge edges for current level (colors stay stable)."""
        # Note: Colors are stable based on micro-clusters, so no color update needed

        # Update labels using viewport-aware aggregation
        self._update_labels_for_viewport()

        # Update bridge edges for current level
        # If there's an active highlight, reapply it instead of resetting to default
        if self._highlighted_indices:
            self.highlight_points(self._highlighted_indices)
        else:
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
        highlighted_set = set(self._highlighted_indices)

        # Set initial highlight state
        alphas = [0.1] * n
        sizes = [3] * n
        line_colors = ['rgba(0,0,0,0)'] * n  # Transparent by default
        line_widths = [1] * n
        for i in self._highlighted_indices:
            alphas[i] = 1.0
            sizes[i] = 14
            line_colors[i] = '#ffaa33'  # Orange outline (same as edge highlight)
            line_widths[i] = 3
        self.source.data['alpha'] = alphas
        self.source.data['size'] = sizes
        self.source.data['line_color'] = line_colors
        self.source.data['line_width'] = line_widths

        # Find which buckets the highlighted points belong to (at 50-cluster level)
        highlighted_buckets = set()
        if 50 in self.cluster_result['labels'] and highlighted_set:
            labels_50 = self.cluster_result['labels'][50]
            for idx in self._highlighted_indices:
                highlighted_buckets.add(int(labels_50[idx]))

        # Highlight bucket-to-bucket edges connected to highlighted buckets
        # Use original edge count, not filtered edge_source
        orig_xs, orig_ys = self.bridge_edges[self.current_level]
        n_edges = len(orig_xs)

        # When highlighting, dim non-connected edges (but keep them visible)
        if highlighted_buckets:
            edge_colors = ['#666666'] * n_edges  # Grey for non-connected
            edge_alphas = [0.08] * n_edges  # Very faint
            edge_widths = [0.5] * n_edges
        else:
            edge_colors = [self._default_edge_color] * n_edges
            edge_alphas = [self._default_edge_alpha] * n_edges
            edge_widths = [self._default_edge_width] * n_edges

        matched_edges = 0
        if self.cluster_pair_list and highlighted_buckets:
            for edge_idx, (c1, c2) in enumerate(self.cluster_pair_list):
                if edge_idx < n_edges and (int(c1) in highlighted_buckets or int(c2) in highlighted_buckets):
                    edge_colors[edge_idx] = '#ffaa33'  # Orange for connected edges
                    edge_alphas[edge_idx] = 0.9
                    edge_widths[edge_idx] = 3.0
                    matched_edges += 1

        # Show all edges - highlighted ones in orange, others dimmed
        self.edge_source.data = {
            'xs': list(orig_xs),
            'ys': list(orig_ys),
            'line_color': edge_colors,
            'line_alpha': edge_alphas,
            'line_width': edge_widths,
        }

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
        self.source.data['line_color'] = ['rgba(0,0,0,0)'] * n
        self.source.data['line_width'] = [1] * n

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
        cache.get('edge_indices'),  # May be None for old caches
        cache.get('cluster_pairs'),  # (bucket1, bucket2) -> count for edge highlighting
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
