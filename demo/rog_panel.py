"""
ROG (Recursive Ontological Graph) Browser - Bokeh visualization with multi-level hierarchical clustering.

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
from bokeh.models import ColumnDataSource, CustomJS, Label, LabelSet, Div, Toggle, TextInput, Button, TapTool
from bokeh.layouts import column, row
import colorcet as cc
from scipy.cluster.hierarchy import fcluster
from rog_preprocess import disambiguate_cluster_names

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CLUSTER_LEVELS = (12, 25, 50, 100)
ZOOM_THRESHOLDS = (1.5, 3.0, 6.0)
COLORS_50 = cc.glasbey_light[:50]

# -----------------------------------------------------------------------------
# Bokeh Application
# -----------------------------------------------------------------------------

class ROGBrowser:
    def __init__(self, coords_2d: np.ndarray, titles: list[str], cluster_result: dict,
                 bridge_edges: dict, edge_indices: list[tuple[int, int]] | None = None,
                 cluster_pairs: dict | None = None, lsh_data: dict | None = None,
                 unified_edge_types: dict | None = None,
                 unified_edge_mutual: dict | None = None):
        self.coords_2d = coords_2d
        self.titles = titles
        self.cluster_result = cluster_result
        self.edge_indices = edge_indices or []  # (source, target) pairs for edge highlighting

        # Unified edge metadata (None for old caches without unified bundling)
        self.edge_types = unified_edge_types
        self.edge_mutual = unified_edge_mutual
        self.has_unified_edges = unified_edge_types is not None
        if self.has_unified_edges:
            print("Unified edge bundling available - bridge + gradient edges in single pass", flush=True)

        # Check if dendrogram data is available for dynamic cutting
        self.dendrogram = cluster_result.get('dendrogram')
        self.use_dynamic_dendrogram = self.dendrogram is not None
        if self.use_dynamic_dendrogram:
            print("Dendrogram data available - using dynamic cutting", flush=True)
            self._dendrogram_cache = {}  # Cache for computed cuts: n_clusters -> (labels, centroids, names)
        else:
            self._dendrogram_cache = {}

        # LSH visualization data
        self.lsh_data = lsh_data
        self.lsh_mode = False  # Toggle between cluster view and LSH view
        self.lsh_density_mode = False  # Show density coloring instead of bucket coloring
        self.lsh_recovery_mode = False  # Show recovery depth coloring
        self.lsh_persistence_mode = False  # Show bridge persistence coloring
        if lsh_data:
            print(f"LSH data available - {lsh_data['num_bits']} bits, {len(lsh_data['bucket_centroids_2d'])} buckets", flush=True)
            # Build LSH bucket colors
            self.lsh_colors = self._build_lsh_colors(lsh_data)
            # Build LSH density colors (by bucket size)
            self.lsh_density_colors, self.lsh_density_sizes = self._build_lsh_density_colors(lsh_data)
            # Build recovery colors (if multi-resolution data available)
            if 'recovery_depth' in lsh_data:
                self.lsh_recovery_colors, self.lsh_recovery_sizes = self._build_recovery_colors(lsh_data)
                print(f"  Recovery data available - threshold={lsh_data.get('mra_dense_threshold', '?')}", flush=True)
            # Build persistence colors (if bridge persistence data available)
            if 'bridge_persistence' in lsh_data:
                self.lsh_persistence_colors, self.lsh_persistence_sizes = self._build_persistence_colors(lsh_data)
                print(f"  Persistence data available", flush=True)

        # Chunk analysis modes
        self.chunk_dedup_mode = False
        self.chunk_redundancy_mode = False
        self.chunk_spread_mode = False
        if lsh_data and 'chunk_data' in lsh_data:
            chunk_data = lsh_data['chunk_data']
            self.chunk_dedup_mask = chunk_data['dedup_mask']
            self.chunk_redundancy_alphas = self._build_redundancy_alphas(chunk_data)
            self.chunk_spread_colors, self.chunk_spread_sizes = self._build_spread_colors(chunk_data)
            kept = int(chunk_data['dedup_mask'].sum())
            print(f"  Chunk data available - dedup keeps {kept}/{len(chunk_data['dedup_mask'])}", flush=True)

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
        self._toggling_lsh = False  # Guard: prevent sub-toggle cascades during LSH toggle
        self._label_update_callback = None  # Pending debounced label update
        self.last_command = None  # Track last command received
        self.connected = True  # Connection status

        # Animation state
        self._animation = None  # Current animation target
        self._animation_callback = None  # Periodic callback for animation

        # Highlight animation state
        self._highlighted_indices = []
        self._pulse_callback = None
        self._pulse_phase = 0

        # Annotation state
        self._annotation_callbacks = []  # Pending timeout callbacks for auto-clear


        # Store original extent for zoom ratio calculation
        self.x_min, self.x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        self.y_min, self.y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()
        self.original_width = self.x_max - self.x_min

        # Build STABLE hierarchical colors:
        # - Coarsest cluster level determines base hue (golden ratio spacing)
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

        # Cap edge count for rendering performance.
        # For unified edges: cap bridge edges but always keep all gradient edges.
        if self.has_unified_edges:
            self.bridge_edges = self._simplify_unified_edges(
                bridge_edges, self.edge_types, self.edge_mutual)
        else:
            self.bridge_edges = self._simplify_bridge_edges(bridge_edges)

        # Initialize edge source with current level's edges (type-aware if unified)
        xs, ys = self.bridge_edges[self.current_level]
        n_edges = len(xs)
        if self.has_unified_edges:
            types = self.edge_types.get(self.current_level, [])
            mutual = self.edge_mutual.get(self.current_level, [])
            colors, alphas, widths = self._style_edges_by_type(
                types[:n_edges], mutual[:n_edges], gradient_mode=False)
        else:
            colors = ['#4488cc'] * n_edges
            alphas = [0.15] * n_edges
            widths = [1] * n_edges
        self.edge_source = ColumnDataSource(data={
            'xs': xs, 'ys': ys,
            'line_color': colors,
            'line_alpha': alphas,
            'line_width': widths,
        })

        # Create figure with WebGL and explicit ranges (no auto-ranging)
        from bokeh.models import Range1d
        padding = self.original_width * 0.05

        # Broadcast title-safe zone (10% inset from each edge)
        self.plot_width = 1200
        self.plot_height = 800
        self.title_safe_x = int(self.plot_width * 0.10)   # 120px
        self.title_safe_y = int(self.plot_height * 0.90)   # 720px (from bottom)

        # Lock ranges so hidden renderers (hyperplane lines) can't expand the view
        max_extent = self.original_width * 2
        x_range = Range1d(self.x_min - padding, self.x_max + padding)
        x_range.bounds = (self.x_min - max_extent, self.x_max + max_extent)
        x_range.reset_start = self.x_min - padding
        x_range.reset_end = self.x_max + padding
        y_range = Range1d(self.y_min - padding, self.y_max + padding)
        y_range.bounds = (self.y_min - max_extent, self.y_max + max_extent)
        y_range.reset_start = self.y_min - padding
        y_range.reset_end = self.y_max + padding
        self.figure = figure(
            width=self.plot_width,
            height=self.plot_height,
            tools='pan,wheel_zoom,reset',
            background_fill_color='#2a2a2a',
            active_scroll='wheel_zoom',
            output_backend='webgl',
            title=f"ROG Browser - {len(titles):,} points - Level: {self.current_level} clusters",
            x_range=x_range,
            y_range=y_range,
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
        # border_width, border_color, font_size are per-label fields
        # driven by cluster level (coarser = more prominent)
        self.labels = LabelSet(
            x='x', y='y', text='label',
            source=self.label_source,
            text_font_size='font_size',
            text_font_style='bold',
            text_color='white',
            background_fill_color='bg_color',
            background_fill_alpha=0.9,
            border_line_color='border_color',
            border_line_width='border_width',
            border_line_alpha=0.8,
            text_align='center',
        )
        self.figure.add_layout(self.labels)

        # --- Annotation layer (ephemeral overlays driven by control API) ---

        # Circle annotations (unfilled rings with optional labels)
        self.annotation_source = ColumnDataSource(data={
            'x': [], 'y': [], 'width': [], 'height': [],
            'line_color': [], 'line_width': [], 'fill_alpha': [],
            'label': [], 'label_y': [],
            'label': [], 'label_y': [],
        })
        self.figure.ellipse(
            'x', 'y', 'width', 'height',
            source=self.annotation_source,
            line_color='line_color',
            line_width='line_width',
            fill_alpha='fill_alpha',
            fill_color=None,
        )
        self.annotation_circle_labels = LabelSet(
            x='x', y='label_y', text='label',
            source=self.annotation_source,
            text_font_size='13pt',
            text_font_style='bold',
            text_color='#ffdd44',
            text_align='center',
        )
        self.figure.add_layout(self.annotation_circle_labels)

        # Title overlay (Label at data coordinates, upper-left of plot)
        self.annotation_title = Label(
            x=self.title_safe_x, y=self.title_safe_y,
            x_units='screen', y_units='screen',
            text='',
            text_font_size='28pt',
            text_font_style='bold',
            text_color='white',
            text_baseline='top',
            background_fill_color='#1a1a1a',
            background_fill_alpha=0.85,
        )
        self.figure.add_layout(self.annotation_title)

        # Sketch annotation layer (hand-drawn strokes via multi_line)
        self.sketch_source = ColumnDataSource(data={
            'xs': [], 'ys': [],
            'line_color': [], 'line_alpha': [], 'line_width': [],
        })
        self.sketch_renderer = self.figure.multi_line(
            'xs', 'ys',
            source=self.sketch_source,
            line_color='line_color',
            line_alpha='line_alpha',
            line_width='line_width',
            line_cap='round',
            line_join='round',
        )
        self._sketch_callback = None  # Periodic callback for draw animation

        # Toggle for edges
        self.edge_toggle = Toggle(label="Show Edges", active=True, width=100)
        self.edge_toggle.on_change('active', self._on_edge_toggle)

        # Hide edges during any interaction (pan or zoom), fade back in after idle
        hide_edges_js = CustomJS(
            args=dict(renderer=self.edge_renderer, toggle=self.edge_toggle,
                      source=self.edge_source),
            code="""
                if (!toggle.active) return;
                renderer.visible = false;
                clearTimeout(window._edge_show_timer);
                clearInterval(window._edge_fade_interval);
                window._edge_show_timer = setTimeout(() => {
                    if (!toggle.active) return;
                    // Fade in: start at 0 alpha, step up over ~250ms
                    const n = source.data['line_alpha'].length;
                    const targets = source.data['line_alpha'].slice();
                    for (let i = 0; i < n; i++) source.data['line_alpha'][i] = 0;
                    renderer.visible = true;
                    source.change.emit();
                    let step = 0;
                    const steps = 5;
                    window._edge_fade_interval = setInterval(() => {
                        step++;
                        const frac = step / steps;
                        for (let i = 0; i < n; i++)
                            source.data['line_alpha'][i] = targets[i] * frac;
                        source.change.emit();
                        if (step >= steps) clearInterval(window._edge_fade_interval);
                    }, 50);
                }, 200);
            """,
        )
        self.figure.x_range.js_on_change('start', hide_edges_js)
        self.figure.y_range.js_on_change('start', hide_edges_js)

        # Toggle for LSH mode (if LSH data available)
        self.lsh_toggle = Toggle(label="LSH Mode", active=False, width=100)
        # Debug: log browser zoom changes on toggle
        zoom_debug_js = CustomJS(code="""
            const z = Math.round(window.devicePixelRatio * 100);
            const iw = window.innerWidth;
            const ow = window.outerWidth;
            console.log(`[LSH toggle] DPR=${window.devicePixelRatio} zoom~${z}% inner=${iw} outer=${ow}`);
            // Watch for changes over next 2 seconds
            let checks = 0;
            const timer = setInterval(() => {
                const z2 = Math.round(window.devicePixelRatio * 100);
                const iw2 = window.innerWidth;
                if (z2 !== z || iw2 !== iw) {
                    console.warn(`[LSH toggle] ZOOM CHANGED! DPR=${window.devicePixelRatio} zoom~=${z2}% inner=${iw2}`);
                }
                if (++checks > 20) clearInterval(timer);
            }, 100);
        """)
        self.lsh_toggle.js_on_change('active', zoom_debug_js)
        self.lsh_toggle.on_change('active', self._on_lsh_toggle)
        if not self.lsh_data:
            self.lsh_toggle.disabled = True

        # Toggle for LSH density coloring (only active in LSH mode)
        self.density_toggle = Toggle(label="Density", active=False, width=80)
        self.density_toggle.on_change('active', self._on_density_toggle)
        self.density_toggle.disabled = True  # Enable only when LSH mode is active

        # Toggle for recovery depth coloring (only active in LSH mode, if data available)
        self.recovery_toggle = Toggle(label="Recovery", active=False, width=80)
        self.recovery_toggle.on_change('active', self._on_recovery_toggle)
        has_recovery = lsh_data and 'recovery_depth' in lsh_data
        self.recovery_toggle.disabled = not has_recovery

        # Toggle for bridge persistence coloring (only active in LSH mode, if data available)
        self.persistence_toggle = Toggle(label="Persistence", active=False, width=90)
        self.persistence_toggle.on_change('active', self._on_persistence_toggle)
        has_persistence = lsh_data and 'bridge_persistence' in lsh_data
        self.persistence_toggle.disabled = not has_persistence

        # Chunk analysis toggles
        has_chunk_data = lsh_data and 'chunk_data' in lsh_data
        self.dedup_toggle = Toggle(label="Dedup", active=False, width=80)
        self.dedup_toggle.on_change('active', self._on_dedup_toggle)
        self.dedup_toggle.disabled = not has_chunk_data

        self.redundancy_toggle = Toggle(label="Redundancy", active=False, width=90)
        self.redundancy_toggle.on_change('active', self._on_redundancy_toggle)
        self.redundancy_toggle.disabled = not has_chunk_data

        self.spread_toggle = Toggle(label="Doc Spread", active=False, width=90)
        self.spread_toggle.on_change('active', self._on_spread_toggle)
        self.spread_toggle.disabled = not has_chunk_data

        # Add hyperplane lines (hidden by default, shown in LSH mode)
        self.hyperplane_source = ColumnDataSource(data={'xs': [], 'ys': []})
        self.hyperplane_renderer = self.figure.multi_line(
            'xs', 'ys',
            source=self.hyperplane_source,
            line_color='#ff6600',
            line_alpha=0.6,
            line_width=2,
            line_dash='dashed',
        )
        self.hyperplane_renderer.visible = False

        # --- Gradient DAG overlay ---
        self.gradient_dag_mode = False
        self._gradient_dag_data = {}  # level -> dag source dicts
        # Gradient detection requires embeddings (done in preprocessing)
        if 'gradients' not in cluster_result:
            print("Gradient data not in cache — re-run preprocessing to enable.", flush=True)

        has_gradients = bool(cluster_result.get('gradients'))
        if has_gradients:
            self._build_gradient_dag_sources(cluster_result)

        self.gradient_dag_toggle = Toggle(label="Gradients", active=False, width=90)
        self.gradient_dag_toggle.on_change('active', self._on_gradient_dag_toggle)
        self.gradient_dag_toggle.disabled = not has_gradients

        # DAG edge lines (multi_line with per-edge color/alpha)
        self.gradient_edge_source = ColumnDataSource(data={
            'xs': [], 'ys': [],
            'line_color': [], 'line_alpha': [], 'line_width': [],
        })
        self.gradient_edge_renderer = self.figure.multi_line(
            'xs', 'ys',
            source=self.gradient_edge_source,
            line_color='line_color',
            line_alpha='line_alpha',
            line_width='line_width',
            line_cap='round',
        )
        self.gradient_edge_renderer.visible = False

        # DAG endpoint nodes (scatter with per-node color)
        self.gradient_node_source = ColumnDataSource(data={
            'x': [], 'y': [], 'color': [], 'size': [],
            'line_color': [], 'label': [], 'keywords': [],
            'marker': [],
        })
        self.gradient_node_renderer = self.figure.scatter(
            'x', 'y',
            source=self.gradient_node_source,
            size='size',
            color='color',
            marker='marker',
            line_color='line_color',
            line_width=2,
            alpha=0.9,
        )
        self.gradient_node_renderer.visible = False

        # Hover tooltip for gradient nodes
        from bokeh.models import HoverTool as HT
        gradient_hover = HT(
            tooltips=[("Cluster", "@label"), ("Keywords", "@keywords")],
            renderers=[self.gradient_node_renderer],
        )
        self.figure.add_tools(gradient_hover)

        # DAG node labels (color-coded: red=start, teal=end)
        self.gradient_label_source = ColumnDataSource(data={
            'x': [], 'y': [], 'label': [], 'text_color': [],
        })
        self.gradient_labels = LabelSet(
            x='x', y='y', text='label',
            source=self.gradient_label_source,
            text_font_size='9pt',
            text_font_style='bold',
            text_color='text_color',
            text_align='left',
            x_offset=10, y_offset=-3,
        )
        self.figure.add_layout(self.gradient_labels)
        self.gradient_labels.visible = False

        # DAG arrowheads (small triangles at edge midpoints pointing in flow direction)
        self.gradient_arrow_source = ColumnDataSource(data={
            'x': [], 'y': [], 'angle': [], 'size': [], 'color': [],
        })
        self.gradient_arrow_renderer = self.figure.triangle(
            'x', 'y',
            source=self.gradient_arrow_source,
            size='size',
            angle='angle',
            fill_color='color',
            fill_alpha=0.7,
            line_color=None,
        )
        self.gradient_arrow_renderer.visible = False

        # Gradient ellipse overlays (replace gradient intra-cluster edges)
        from bokeh.models.glyphs import Ellipse
        self.gradient_ellipse_source = ColumnDataSource(data={
            'x': [], 'y': [], 'width': [], 'height': [], 'angle': [],
            'fill_alpha': [], 'line_alpha': [], 'cluster_name': [],
            'gradient_info': [],
        })
        ellipse_glyph = Ellipse(
            x='x', y='y', width='width', height='height', angle='angle',
            fill_color='white', fill_alpha='fill_alpha',
            line_color='white', line_alpha='line_alpha',
            line_width=1, line_dash='dashed',
        )
        self.gradient_ellipse_renderer = self.figure.add_glyph(
            self.gradient_ellipse_source, ellipse_glyph
        )
        self.gradient_ellipse_renderer.visible = False

        # Hover tooltip for ellipses
        ellipse_hover = HT(
            tooltips=[
                ("Cluster", "@cluster_name"),
                ("Gradient", "@gradient_info"),
            ],
            renderers=[self.gradient_ellipse_renderer],
        )
        self.figure.add_tools(ellipse_hover)

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

    def _build_lsh_colors(self, lsh_data: dict) -> list[str]:
        """Build colors based on LSH bucket assignments."""
        import colorsys

        bucket_ids = lsh_data['bucket_ids']
        unique_buckets = sorted(set(bucket_ids))
        n_buckets = len(unique_buckets)
        bucket_to_idx = {b: i for i, b in enumerate(unique_buckets)}

        colors = []
        for bid in bucket_ids:
            idx = bucket_to_idx[int(bid)]
            # Use golden ratio for hue distribution
            hue = (idx * 0.618033988749895) % 1.0
            r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.7)
            colors.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')

        return colors

    def _build_lsh_density_colors(self, lsh_data: dict) -> tuple[list[str], list[float]]:
        """Build colors based on LSH bucket density (size).

        Returns colors and point sizes based on bucket density.
        Uses a perceptual colormap: sparse (blue) → medium (green) → dense (yellow).
        """
        bucket_ids = lsh_data['bucket_ids']
        bucket_sizes = lsh_data['bucket_sizes']

        # Get size for each point
        point_sizes = [bucket_sizes.get(int(bid), 1) for bid in bucket_ids]

        # Compute log-scale density for better visual distribution
        log_sizes = [np.log1p(s) for s in point_sizes]
        min_log, max_log = min(log_sizes), max(log_sizes)
        range_log = max_log - min_log if max_log > min_log else 1.0

        # Viridis-inspired colormap (sparse → dense)
        # Purple/blue for sparse, green for medium, yellow for dense
        def density_to_color(normalized: float) -> str:
            """Map normalized density [0,1] to color."""
            if normalized < 0.25:
                # Dark blue to blue
                t = normalized / 0.25
                r, g, b = int(68 + t * 20), int(1 + t * 80), int(84 + t * 80)
            elif normalized < 0.5:
                # Blue to teal/green
                t = (normalized - 0.25) / 0.25
                r, g, b = int(88 - t * 55), int(81 + t * 90), int(164 - t * 50)
            elif normalized < 0.75:
                # Green to yellow-green
                t = (normalized - 0.5) / 0.25
                r, g, b = int(33 + t * 100), int(171 + t * 30), int(114 - t * 60)
            else:
                # Yellow-green to yellow
                t = (normalized - 0.75) / 0.25
                r, g, b = int(133 + t * 120), int(201 + t * 50), int(54 - t * 30)

            return f'#{r:02x}{g:02x}{b:02x}'

        colors = []
        sizes = []
        for log_size in log_sizes:
            normalized = (log_size - min_log) / range_log
            colors.append(density_to_color(normalized))
            # Size: sparse = 4, dense = 12
            sizes.append(4 + normalized * 8)

        return colors, sizes

    def _build_recovery_colors(self, lsh_data: dict) -> tuple[list[str], list[float]]:
        """Build colors based on multi-resolution recovery depth.

        Colors: grey (already dense, depth=0) -> green (early recovery) ->
                orange (late recovery) -> red (never recovered).
        Size scales with recovery_ratio.
        """
        recovery_depth = lsh_data['recovery_depth']
        recovery_ratio = lsh_data['recovery_ratio']
        num_bits = lsh_data['num_bits']

        colors = []
        sizes = []
        for depth, ratio in zip(recovery_depth, recovery_ratio):
            if depth == 0:
                # Already dense at full resolution - grey
                colors.append('#aaaaaa')
                sizes.append(4)
            elif depth > num_bits:
                # Never recovered - red
                colors.append('#cc2222')
                sizes.append(6)
            else:
                # Recovered at some depth - green to orange gradient
                # depth 1 = earliest recovery (best) -> green
                # depth num_bits = latest recovery (worst) -> orange
                t = (depth - 1) / max(num_bits - 1, 1)  # 0=early, 1=late
                if t < 0.5:
                    # Green to yellow
                    s = t / 0.5
                    r = int(40 + s * 180)
                    g = int(180 - s * 40)
                    b = int(40)
                else:
                    # Yellow to orange
                    s = (t - 0.5) / 0.5
                    r = int(220 + s * 30)
                    g = int(140 - s * 80)
                    b = int(40 - s * 20)
                colors.append(f'#{r:02x}{g:02x}{b:02x}')
                # Size based on recovery ratio (larger = denser than expected)
                sizes.append(4 + min(ratio, 5.0) * 2)

        return colors, sizes

    def _build_persistence_colors(self, lsh_data: dict) -> tuple[list[str], list[float]]:
        """Build colors based on bridge persistence across depths.

        Uses relative threshold detection: connectors are items where
        max_other_sim / own_sim >= threshold at some depth.

        Colors: grey (never connector) -> blue (low persistence) ->
                purple (medium) -> red (high persistence).
        Size scales with bridge_ratio (peak relative similarity).
        """
        persistence = lsh_data['bridge_persistence']
        bridge_ratio = lsh_data.get('bridge_ratio', [0.0] * len(persistence))
        num_bits = lsh_data['num_bits']

        colors = []
        sizes = []
        for p, ratio in zip(persistence, bridge_ratio):
            if p == 0:
                # Never a connector - grey
                colors.append('#aaaaaa')
                sizes.append(3)
            else:
                # Connector at p depths - blue to purple to red gradient
                t = (p - 1) / max(num_bits - 1, 1)  # 0=low persistence, 1=high
                if t < 0.33:
                    # Blue
                    s = t / 0.33
                    r = int(40 + s * 60)
                    g = int(80 + s * 20)
                    b = int(220 - s * 40)
                elif t < 0.66:
                    # Blue to purple
                    s = (t - 0.33) / 0.33
                    r = int(100 + s * 80)
                    g = int(100 - s * 60)
                    b = int(180 - s * 30)
                else:
                    # Purple to red
                    s = (t - 0.66) / 0.34
                    r = int(180 + s * 60)
                    g = int(40 - s * 20)
                    b = int(150 - s * 120)
                colors.append(f'#{r:02x}{g:02x}{b:02x}')
                # Size: 5-12 scaling with peak ratio (stronger connector = larger)
                sizes.append(5 + min(ratio, 1.5) / 1.5 * 7)

        return colors, sizes

    def _build_redundancy_alphas(self, chunk_data: dict) -> list[float]:
        """Build alpha values based on chunk redundancy (sibling count).

        0 siblings → alpha 0.7 (unique, bright), increasing → down to 0.15 (dim).
        """
        redundancy = chunk_data['redundancy']
        max_r = max(int(redundancy.max()), 1)
        alphas = []
        for r in redundancy:
            # Linear mapping: 0 → 0.7, max → 0.15
            t = int(r) / max_r
            alphas.append(0.7 - t * 0.55)
        return alphas

    def _build_spread_colors(self, chunk_data: dict) -> tuple[list[str], list[float]]:
        """Build colors based on document spread (concentration).

        Low concentration (bridge docs spanning many buckets) → warm red/orange.
        High concentration (focused docs in few buckets) → cool green/blue.
        Size scales with n_buckets.
        """
        concentration = chunk_data['concentration']
        n_buckets = chunk_data['n_buckets']

        # Normalize concentration to [0, 1]
        c_min = float(concentration[concentration > 0].min()) if (concentration > 0).any() else 0.0
        c_max = float(concentration.max()) if concentration.max() > 0 else 1.0
        c_range = c_max - c_min if c_max > c_min else 1.0

        max_n_buckets = max(int(n_buckets.max()), 1)

        colors = []
        sizes = []
        for c, nb in zip(concentration, n_buckets):
            if c == 0 and nb == 0:
                # Single-chunk doc or no data — grey
                colors.append('#aaaaaa')
                sizes.append(3)
                continue

            t = (float(c) - c_min) / c_range  # 0 = low concentration (bridge), 1 = high (focused)
            t = max(0.0, min(1.0, t))

            if t < 0.33:
                # Warm: red/orange (bridge docs)
                s = t / 0.33
                r = int(220 - s * 40)
                g = int(60 + s * 60)
                b = int(30 + s * 20)
            elif t < 0.66:
                # Transition: yellow/green
                s = (t - 0.33) / 0.33
                r = int(180 - s * 100)
                g = int(120 + s * 60)
                b = int(50 + s * 40)
            else:
                # Cool: green/blue (focused docs)
                s = (t - 0.66) / 0.34
                r = int(80 - s * 40)
                g = int(180 - s * 40)
                b = int(90 + s * 100)
            colors.append(f'#{r:02x}{g:02x}{b:02x}')

            # Size: 3 + (n_buckets / max_n_buckets) * 7
            sizes.append(3 + (int(nb) / max_n_buckets) * 7)

        return colors, sizes

    def _build_hierarchical_colors(self, cluster_result: dict) -> list[str]:
        """Build stable colors with hierarchical structure.

        Coarsest-level cluster determines base hue (golden ratio spacing),
        micro-cluster adds variation. This makes cluster boundaries visible
        while keeping colors stable across zoom levels.
        """
        import colorsys

        n_points = len(self.coords_2d)

        # Get coarsest-level cluster assignments for base hue
        coarsest_level = min(cluster_result['labels'].keys())
        top_labels = cluster_result['labels'][coarsest_level]
        n_top = len(set(top_labels))

        # Get micro-cluster assignments for variation
        if self.use_dynamic_dendrogram:
            micro_labels = self.dendrogram['kmeans_labels']
        else:
            finest_level = max(cluster_result['labels'].keys())
            micro_labels = cluster_result['labels'][finest_level]

        # Golden ratio hues for top-level clusters (well-distributed)
        base_hues = [(i * 0.618033988749895) % 1.0 for i in range(n_top)]

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

    @staticmethod
    def _darken_hex(hex_color: str, factor: float = 0.5) -> str:
        """Darken a hex color by the given factor (0=black, 1=unchanged)."""
        hex_color = hex_color.lstrip('#')
        r = int(int(hex_color[0:2], 16) * factor)
        g = int(int(hex_color[2:4], 16) * factor)
        b = int(int(hex_color[4:6], 16) * factor)
        return f'#{r:02x}{g:02x}{b:02x}'

    def _get_cluster_bg_colors(self, cluster_ids: list[int], labels: np.ndarray) -> list[str]:
        """Get darkened background colors for cluster labels.

        For each cluster_id, finds a representative point and darkens its stable color.
        """
        bg_colors = []
        for cid in cluster_ids:
            # Find first point in this cluster
            match = np.where(labels == cid)[0]
            if len(match) > 0:
                point_color = self.stable_colors[match[0]]
                bg_colors.append(self._darken_hex(point_color))
            else:
                bg_colors.append('#2d2d2d')
        return bg_colors

    def _update_label_source(self):
        """Update the label data source for current level."""
        centroids = self.cluster_result['centroids'][self.current_level]
        names = self.cluster_result['names'][self.current_level]
        labels = self.cluster_result['labels'][self.current_level]
        padded_names = [f'  {name}  ' for name in names]
        bg_colors = self._get_cluster_bg_colors(list(range(len(names))), labels)
        n = len(names)

        self.label_source = ColumnDataSource(data={
            'x': centroids[:, 0],
            'y': centroids[:, 1],
            'label': padded_names,
            'bg_color': bg_colors,
            'border_width': [2] * n,
            'border_color': ['#999999'] * n,
            'font_size': ['11pt'] * n,
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

        # Cut at n_clusters
        micro_cluster_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
        point_labels = np.array([micro_cluster_labels[m] for m in kmeans_labels])

        # Compute centroids from point positions
        cluster_centroids = []
        for cluster_id in range(n_clusters):
            mask = point_labels == cluster_id
            pts = np.where(mask)[0]
            if len(pts) == 0:
                cluster_centroids.append(np.array([0.0, 0.0]))
            else:
                cluster_centroids.append(self.coords_2d[pts].mean(axis=0))

        # Transfer labels from nearest pre-computed level by majority overlap
        ref_level = min(CLUSTER_LEVELS, key=lambda l: abs(l - n_clusters))
        ref_labels = self.cluster_result['labels'][ref_level]
        ref_names = self.cluster_result['names'][ref_level]

        cluster_names = []
        for cluster_id in range(n_clusters):
            mask = point_labels == cluster_id
            if not mask.any():
                cluster_names.append(f"Cluster {cluster_id}")
                continue
            # Find which pre-computed cluster has the most overlap
            ref_subset = ref_labels[mask]
            counts = np.bincount(ref_subset, minlength=len(ref_names))
            best_ref = int(np.argmax(counts))
            cluster_names.append(ref_names[best_ref])

        # Disambiguate duplicate cluster names with TF-IDF keywords
        cluster_names = disambiguate_cluster_names(cluster_names, self.titles, point_labels)

        result = (point_labels, np.array(cluster_centroids), cluster_names)
        self._dendrogram_cache[n_clusters] = result
        return result

    # Visual style per cluster level (coarser = more prominent)
    @staticmethod
    def _label_data(x, y, label, bg_color=None, border_width=1,
                    border_color='#888888', font_size='10pt'):
        """Build a complete label_source.data dict with all required columns.

        Bokeh's LabelSet references font_size, border_color, and border_width
        columns; omitting them causes a render crash.
        """
        n = len(x)
        if bg_color is None:
            bg_color = ['#2d2d2d'] * n
        if isinstance(border_width, (int, float)):
            border_width = [border_width] * n
        if isinstance(border_color, str):
            border_color = [border_color] * n
        if isinstance(font_size, str):
            font_size = [font_size] * n
        return {
            'x': x, 'y': y, 'label': label, 'bg_color': bg_color,
            'border_width': border_width, 'border_color': border_color,
            'font_size': font_size,
        }

    _LEVEL_STYLE = {
        12:  {'border_width': 3, 'border_color': '#ffffff', 'font_size': '13pt'},
        25:  {'border_width': 2, 'border_color': '#bbbbbb', 'font_size': '11pt'},
        50:  {'border_width': 1, 'border_color': '#888888', 'font_size': '10pt'},
        100: {'border_width': 1, 'border_color': '#666666', 'font_size': '9pt'},
    }

    def _update_labels_for_viewport(self):
        """Update labels based on viewport with multi-level visual hierarchy.

        Shows labels from multiple pre-computed levels simultaneously.
        Coarser levels are placed first with thicker borders and larger font.
        Finer levels fill in detail, skipping positions too close to existing labels.
        """
        # Suppress cluster labels in LSH mode — bucket coloring replaces them
        if self.lsh_mode:
            return

        x_start = self.figure.x_range.start
        x_end = self.figure.x_range.end
        y_start = self.figure.y_range.start
        y_end = self.figure.y_range.end

        if x_start is None or x_end is None or y_start is None or y_end is None:
            return

        viewport_width = x_end - x_start
        viewport_height = y_end - y_start
        # Minimum separation between labels (fraction of viewport diagonal)
        min_sep_sq = (0.08 * (viewport_width**2 + viewport_height**2))

        # Max labels on screen: scale from 6 (zoomed out) to 18 (zoomed in)
        zoom_frac = max(0.0, min(1.0, 1.0 - viewport_width / (self.original_width * 1.1)))
        max_labels = int(6 + 12 * zoom_frac)

        # Collect labels from all levels, coarsest first.
        # Coarse labels anchor the layout; finer labels fill gaps.
        placed = []  # (x, y, name, level, cluster_id, labels_array)

        for level in CLUSTER_LEVELS:
            centroids = self.cluster_result['centroids'][level]
            names = self.cluster_result['names'][level]
            labels = self.cluster_result['labels'][level]

            # Positions from coarser levels already placed
            coarser = [(px, py) for px, py, _, lvl, _, _ in placed]

            for i, (cx, cy) in enumerate(centroids):
                if not (x_start <= cx <= x_end and y_start <= cy <= y_end):
                    continue

                # Only check separation against coarser-level labels.
                # Within the same level, show all visible labels.
                too_close = False
                for px, py in coarser:
                    if (cx - px)**2 + (cy - py)**2 < min_sep_sq:
                        too_close = True
                        break

                if not too_close:
                    placed.append((cx, cy, names[i], level, i, labels))

            if len(placed) >= max_labels:
                break

        # If < 2 labels, add nearest from coarsest level
        if len(placed) < 2:
            cx_view = (x_start + x_end) / 2
            cy_view = (y_start + y_end) / 2
            coarsest = CLUSTER_LEVELS[0]
            centroids = self.cluster_result['centroids'][coarsest]
            names = self.cluster_result['names'][coarsest]
            labels = self.cluster_result['labels'][coarsest]
            placed_set = {(p[0], p[1]) for p in placed}
            dists = [(i, (cx - cx_view)**2 + (cy - cy_view)**2)
                     for i, (cx, cy) in enumerate(centroids)
                     if (cx, cy) not in placed_set]
            dists.sort(key=lambda x: x[1])
            for i, _ in dists[:2 - len(placed)]:
                cx, cy = centroids[i]
                placed.append((cx, cy, names[i], coarsest, i, labels))

        # Build data source with visual hierarchy
        label_x, label_y, label_text = [], [], []
        border_widths, border_colors, font_sizes, bg_colors = [], [], [], []

        for cx, cy, name, level, cid, lbls in placed:
            label_x.append(cx)
            label_y.append(cy)
            label_text.append(f'  {name}  ')
            style = self._LEVEL_STYLE.get(level, self._LEVEL_STYLE[100])
            border_widths.append(style['border_width'])
            border_colors.append(style['border_color'])
            font_sizes.append(style['font_size'])
            bg_colors.append(self._get_cluster_bg_colors([cid], lbls)[0])

        self.label_source.data = {
            'x': label_x,
            'y': label_y,
            'label': label_text,
            'bg_color': bg_colors,
            'border_width': border_widths,
            'border_color': border_colors,
            'font_size': font_sizes,
        }

        # Count levels represented
        levels_shown = sorted(set(p[3] for p in placed))
        level_str = '/'.join(str(l) for l in levels_shown)
        self.figure.title.text = (
            f"ROG Browser - {len(self.titles):,} points - "
            f"Levels: {level_str} ({len(placed)} labels)"
        )

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

    def _on_lsh_toggle(self, attr, old, new):
        """Toggle between cluster view and LSH bucket view."""
        self.lsh_mode = new
        # Guard: prevent sub-toggle callbacks from doing redundant updates
        self._toggling_lsh = True

        if new and self.lsh_data:
            # Switch to LSH mode
            # Enable density toggle
            self.density_toggle.disabled = False
            # Enable recovery toggle if data available
            if 'recovery_depth' in self.lsh_data:
                self.recovery_toggle.disabled = False
            # Enable persistence toggle if data available
            if 'bridge_persistence' in self.lsh_data:
                self.persistence_toggle.disabled = False

            # Update colors to LSH bucket colors (or density if enabled)
            if self.lsh_density_mode:
                self.source.data['color'] = self.lsh_density_colors
                self.source.data['size'] = self.lsh_density_sizes
            else:
                self.source.data['color'] = self.lsh_colors

            # Show hyperplane lines
            hyperplanes = self.lsh_data['hyperplanes_2d']
            xs = [hp['x'] for hp in hyperplanes]
            ys = [hp['y'] for hp in hyperplanes]
            self.hyperplane_source.data = {'xs': xs, 'ys': ys}
            self.hyperplane_renderer.visible = True

            # Update labels to show bucket info
            bucket_centroids = self.lsh_data['bucket_centroids_2d']
            bucket_sizes = self.lsh_data['bucket_sizes']
            # Show top 10 largest buckets
            top_buckets = sorted(bucket_sizes.items(), key=lambda x: -x[1])[:12]
            label_x = [bucket_centroids[bid][0] for bid, _ in top_buckets if bid in bucket_centroids]
            label_y = [bucket_centroids[bid][1] for bid, _ in top_buckets if bid in bucket_centroids]
            label_text = [f'  B{bid} ({sz})  ' for bid, sz in top_buckets if bid in bucket_centroids]
            self.label_source.data = self._label_data(label_x, label_y, label_text)

            # Hide cluster edges, show boundary points
            self.edge_renderer.visible = False
            self.edge_toggle.active = False

            self.figure.title.text = f"LSH Mode - {self.lsh_data['num_bits']} bits, {len(bucket_centroids)} buckets"
        else:
            # Switch back to cluster mode
            # Disable and reset sub-toggles (guard prevents their callbacks from firing updates)
            self.density_toggle.disabled = True
            self.density_toggle.active = False
            self.lsh_density_mode = False
            self.recovery_toggle.disabled = True
            self.recovery_toggle.active = False
            self.lsh_recovery_mode = False
            self.persistence_toggle.disabled = True
            self.persistence_toggle.active = False
            self.lsh_persistence_mode = False

            self.source.data['color'] = self.stable_colors
            # Reset point sizes to default
            self.source.data['size'] = [4] * len(self.titles)
            self.hyperplane_renderer.visible = False
            self._update_labels_for_viewport()
            self.figure.title.text = f"ROG Browser - {len(self.titles):,} points - Level: {self.current_level} clusters"

        self._toggling_lsh = False

    def _on_density_toggle(self, attr, old, new):
        """Toggle between bucket coloring and density coloring in LSH mode."""
        self.lsh_density_mode = new
        if self._toggling_lsh:
            return

        if not self.lsh_mode or not self.lsh_data:
            return

        # Deactivate recovery, persistence, and spread if density is turned on
        if new and self.lsh_recovery_mode:
            self.recovery_toggle.active = False
            self.lsh_recovery_mode = False
        if new and self.lsh_persistence_mode:
            self.persistence_toggle.active = False
            self.lsh_persistence_mode = False
        if new and self.chunk_spread_mode:
            self.spread_toggle.active = False
            self.chunk_spread_mode = False

        bucket_centroids = self.lsh_data['bucket_centroids_2d']
        bucket_sizes = self.lsh_data['bucket_sizes']

        if new:
            # Switch to density coloring
            self.source.data['color'] = self.lsh_density_colors
            self.source.data['size'] = self.lsh_density_sizes

            # Update labels to show density distribution
            # Sort by size and show range
            sizes = list(bucket_sizes.values())
            min_sz, max_sz = min(sizes), max(sizes)
            median_sz = sorted(sizes)[len(sizes)//2]

            # Show sparse, medium, and dense bucket examples
            sorted_buckets = sorted(bucket_sizes.items(), key=lambda x: x[1])
            sparse_buckets = sorted_buckets[:3]  # Smallest
            dense_buckets = sorted_buckets[-5:]  # Largest

            label_x, label_y, label_text = [], [], []
            for bid, sz in dense_buckets:
                if bid in bucket_centroids:
                    label_x.append(bucket_centroids[bid][0])
                    label_y.append(bucket_centroids[bid][1])
                    label_text.append(f'  Dense: {sz}  ')

            self.label_source.data = self._label_data(label_x, label_y, label_text)
            self.figure.title.text = f"LSH Density - Range: {min_sz} to {max_sz} (median: {median_sz})"
        else:
            # Switch back to bucket coloring
            self.source.data['color'] = self.lsh_colors
            self.source.data['size'] = [4] * len(self.titles)

            # Restore bucket labels
            top_buckets = sorted(bucket_sizes.items(), key=lambda x: -x[1])[:12]
            label_x = [bucket_centroids[bid][0] for bid, _ in top_buckets if bid in bucket_centroids]
            label_y = [bucket_centroids[bid][1] for bid, _ in top_buckets if bid in bucket_centroids]
            label_text = [f'  B{bid} ({sz})  ' for bid, sz in top_buckets if bid in bucket_centroids]
            self.label_source.data = self._label_data(label_x, label_y, label_text)
            self.figure.title.text = f"LSH Mode - {self.lsh_data['num_bits']} bits, {len(bucket_centroids)} buckets"

    def _on_recovery_toggle(self, attr, old, new):
        """Toggle between bucket/density coloring and recovery depth coloring in LSH mode."""
        self.lsh_recovery_mode = new
        if self._toggling_lsh:
            return

        if not self.lsh_data or 'recovery_depth' not in self.lsh_data:
            return

        if new:
            # Switch to recovery coloring - deactivate density, persistence, and spread if on
            if self.lsh_density_mode:
                self.density_toggle.active = False
                self.lsh_density_mode = False
            if self.lsh_persistence_mode:
                self.persistence_toggle.active = False
                self.lsh_persistence_mode = False
            if self.chunk_spread_mode:
                self.spread_toggle.active = False
                self.chunk_spread_mode = False

            self.source.data['color'] = self.lsh_recovery_colors
            self.source.data['size'] = self.lsh_recovery_sizes

            # Update labels to show recovery stats
            recovery_depth = self.lsh_data['recovery_depth']
            num_bits = self.lsh_data['num_bits']
            already_dense = sum(1 for d in recovery_depth if d == 0)
            recovered = sum(1 for d in recovery_depth if 0 < d <= num_bits)
            never = sum(1 for d in recovery_depth if d > num_bits)

            self.label_source.data = self._label_data([], [], [])
            self.figure.title.text = (
                f"Recovery View - Dense: {already_dense}, "
                f"Recovered: {recovered}, Never: {never} "
                f"(threshold={self.lsh_data.get('mra_dense_threshold', '?')})"
            )
        else:
            # Switch back to current LSH mode (bucket or density)
            if self.lsh_mode:
                self.source.data['color'] = self.lsh_colors
                self.source.data['size'] = [4] * len(self.titles)

                bucket_centroids = self.lsh_data['bucket_centroids_2d']
                bucket_sizes = self.lsh_data['bucket_sizes']
                top_buckets = sorted(bucket_sizes.items(), key=lambda x: -x[1])[:12]
                label_x = [bucket_centroids[bid][0] for bid, _ in top_buckets if bid in bucket_centroids]
                label_y = [bucket_centroids[bid][1] for bid, _ in top_buckets if bid in bucket_centroids]
                label_text = [f'  B{bid} ({sz})  ' for bid, sz in top_buckets if bid in bucket_centroids]
                self.label_source.data = self._label_data(label_x, label_y, label_text)
                self.figure.title.text = f"LSH Mode - {self.lsh_data['num_bits']} bits, {len(bucket_centroids)} buckets"
            else:
                self.source.data['color'] = self.stable_colors
                self.source.data['size'] = [4] * len(self.titles)
                self._update_labels_for_viewport()

    def _on_persistence_toggle(self, attr, old, new):
        """Toggle between bucket/density coloring and bridge persistence coloring in LSH mode."""
        self.lsh_persistence_mode = new
        if self._toggling_lsh:
            return

        if not self.lsh_data or 'bridge_persistence' not in self.lsh_data:
            return

        if new:
            # Switch to persistence coloring - deactivate density, recovery, and spread if on
            if self.lsh_density_mode:
                self.density_toggle.active = False
                self.lsh_density_mode = False
            if self.lsh_recovery_mode:
                self.recovery_toggle.active = False
                self.lsh_recovery_mode = False
            if self.chunk_spread_mode:
                self.spread_toggle.active = False
                self.chunk_spread_mode = False

            self.source.data['color'] = self.lsh_persistence_colors
            self.source.data['size'] = self.lsh_persistence_sizes

            # Update title with persistence stats
            persistence = self.lsh_data['bridge_persistence']
            total_bridges = sum(1 for p in persistence if p > 0)
            max_p = max(persistence) if persistence else 0
            num_bits = self.lsh_data['num_bits']

            self.label_source.data = self._label_data([], [], [])
            self.figure.title.text = (
                f"Persistence View - Bridges: {total_bridges}, "
                f"Max depth span: {max_p}/{num_bits}"
            )
        else:
            # Switch back to current LSH mode (bucket or density)
            if self.lsh_mode:
                self.source.data['color'] = self.lsh_colors
                self.source.data['size'] = [4] * len(self.titles)

                bucket_centroids = self.lsh_data['bucket_centroids_2d']
                bucket_sizes = self.lsh_data['bucket_sizes']
                top_buckets = sorted(bucket_sizes.items(), key=lambda x: -x[1])[:12]
                label_x = [bucket_centroids[bid][0] for bid, _ in top_buckets if bid in bucket_centroids]
                label_y = [bucket_centroids[bid][1] for bid, _ in top_buckets if bid in bucket_centroids]
                label_text = [f'  B{bid} ({sz})  ' for bid, sz in top_buckets if bid in bucket_centroids]
                self.label_source.data = self._label_data(label_x, label_y, label_text)
                self.figure.title.text = f"LSH Mode - {self.lsh_data['num_bits']} bits, {len(bucket_centroids)} buckets"
            else:
                self.source.data['color'] = self.stable_colors
                self.source.data['size'] = [4] * len(self.titles)
                self._update_labels_for_viewport()

    def _on_dedup_toggle(self, attr, old, new):
        """Toggle chunk deduplication overlay. Hides non-representative points."""
        self.chunk_dedup_mode = new

        if not self.lsh_data or 'chunk_data' not in self.lsh_data:
            return

        n = len(self.coords_2d)

        if new:
            # Hide non-representative points (alpha=0, size=0)
            mask = self.chunk_dedup_mask
            kept = int(mask.sum())

            alphas = list(self.source.data['alpha'])
            sizes = list(self.source.data['size'])
            for i in range(n):
                if not mask[i]:
                    alphas[i] = 0.0
                    sizes[i] = 0
            self.source.data['alpha'] = alphas
            self.source.data['size'] = sizes

            self.figure.title.text = (
                f"Dedup View - {kept}/{n} kept ({100*kept/n:.1f}%)"
            )
        else:
            # Restore: apply current base alphas/sizes
            self._restore_chunk_state()

    def _on_redundancy_toggle(self, attr, old, new):
        """Toggle chunk redundancy overlay. Dims points by sibling count."""
        self.chunk_redundancy_mode = new

        if not self.lsh_data or 'chunk_data' not in self.lsh_data:
            return

        n = len(self.coords_2d)

        if new:
            # Apply redundancy alphas, respecting dedup overlay
            alphas = list(self.chunk_redundancy_alphas)
            if self.chunk_dedup_mode:
                mask = self.chunk_dedup_mask
                for i in range(n):
                    if not mask[i]:
                        alphas[i] = 0.0
            self.source.data['alpha'] = alphas

            redundancy = self.lsh_data['chunk_data']['redundancy']
            max_r = int(redundancy.max())
            unique = int((redundancy == 0).sum())
            self.figure.title.text = (
                f"Redundancy View - {unique} unique, max siblings={max_r}"
            )
        else:
            # Restore base alphas
            self._restore_chunk_state()

    def _on_spread_toggle(self, attr, old, new):
        """Toggle doc spread color mode. Mutually exclusive with density/recovery/persistence."""
        self.chunk_spread_mode = new

        if not self.lsh_data or 'chunk_data' not in self.lsh_data:
            return

        n = len(self.coords_2d)

        if new:
            # Deactivate mutually exclusive color modes
            if self.lsh_density_mode:
                self.density_toggle.active = False
            if self.lsh_recovery_mode:
                self.recovery_toggle.active = False
            if self.lsh_persistence_mode:
                self.persistence_toggle.active = False

            # Apply spread colors and sizes
            colors = list(self.chunk_spread_colors)
            sizes = list(self.chunk_spread_sizes)

            # Respect dedup overlay
            if self.chunk_dedup_mode:
                mask = self.chunk_dedup_mask
                for i in range(n):
                    if not mask[i]:
                        sizes[i] = 0

            self.source.data['color'] = colors
            self.source.data['size'] = sizes

            chunk_data = self.lsh_data['chunk_data']
            n_unique = chunk_data['n_unique_docs']
            self.label_source.data = self._label_data([], [], [])
            self.figure.title.text = (
                f"Doc Spread View - {n_unique} unique docs, "
                f"warm=bridge, cool=focused"
            )
        else:
            # Restore to current base mode
            if self.lsh_mode:
                self.source.data['color'] = self.lsh_colors
                self.source.data['size'] = [4] * n
            else:
                self.source.data['color'] = self.stable_colors
                self.source.data['size'] = [4] * n
                self._update_labels_for_viewport()

            # Re-apply dedup overlay if active
            if self.chunk_dedup_mode:
                mask = self.chunk_dedup_mask
                sizes = list(self.source.data['size'])
                alphas = list(self.source.data['alpha'])
                for i in range(n):
                    if not mask[i]:
                        alphas[i] = 0.0
                        sizes[i] = 0
                self.source.data['alpha'] = alphas
                self.source.data['size'] = sizes

    def _restore_chunk_state(self):
        """Restore alpha/size arrays based on active chunk modes."""
        n = len(self.coords_2d)

        # Determine base alphas
        if self.chunk_redundancy_mode:
            alphas = list(self.chunk_redundancy_alphas)
        else:
            alphas = [0.7] * n

        # Determine base sizes
        if self.chunk_spread_mode:
            sizes = list(self.chunk_spread_sizes)
        else:
            sizes = [4] * n

        # Apply dedup overlay on top
        if self.chunk_dedup_mode:
            mask = self.chunk_dedup_mask
            for i in range(n):
                if not mask[i]:
                    alphas[i] = 0.0
                    sizes[i] = 0

        self.source.data['alpha'] = alphas
        self.source.data['size'] = sizes

    def _build_gradient_dag_sources(self, cluster_result: dict):
        """Pre-build gradient DAG overlay data for each cluster level."""
        from rog_preprocess import detect_cluster_gradients, build_gradient_network, label_gradient_endpoints
        import math

        gradients_dict = cluster_result.get('gradients', {})
        network_dict = cluster_result.get('gradient_network', {})

        # Ensure gradient endpoints have descriptive labels
        for n_cl, grads in gradients_dict.items():
            if grads and 'start_label' not in next(iter(grads.values())):
                labs = cluster_result['labels'][n_cl]
                label_gradient_endpoints(grads, self.coords_2d, labs, cluster_result, n_cl)

        for n_clusters in CLUSTER_LEVELS:
            gradients = gradients_dict.get(n_clusters, {})
            if not gradients:
                self._gradient_dag_data[n_clusters] = None
                continue

            names = cluster_result['names'][n_clusters]
            edges = network_dict.get(n_clusters, [])

            # If no precomputed network, build it now
            if not edges and len(gradients) >= 2:
                edges = build_gradient_network(gradients, names, k=1)

            # Build endpoint data
            endpoints = []  # (cid, pole, x, y, keywords, pole_label)
            ep_lookup = {}
            for cid, g in gradients.items():
                s_label = g.get('start_label', ', '.join(g['start_keywords'][:3]))
                e_label = g.get('end_label', ', '.join(g['end_keywords'][:3]))
                ep_lookup[(cid, 'start')] = len(endpoints)
                endpoints.append((cid, 'start', g['start_point'][0], g['start_point'][1],
                                  g['start_keywords'][:3], s_label))
                ep_lookup[(cid, 'end')] = len(endpoints)
                endpoints.append((cid, 'end', g['end_point'][0], g['end_point'][1],
                                  g['end_keywords'][:3], e_label))

            # Orient edges as DAG using PCA global axis
            from sklearn.decomposition import PCA
            pts = np.array([[ep[2], ep[3]] for ep in endpoints])
            if len(pts) >= 2:
                pca = PCA(n_components=1)
                pca.fit(pts)
                global_axis = pca.components_[0]
            else:
                global_axis = np.array([1.0, 0.0])

            projections = {i: endpoints[i][2] * global_axis[0] + endpoints[i][3] * global_axis[1]
                           for i in range(len(endpoints))}

            # Build DAG edges: intra-cluster + inter-cluster (oriented by projection)
            dag_edges = []
            # Intra-cluster
            for cid, g in gradients.items():
                s_idx = ep_lookup[(cid, 'start')]
                e_idx = ep_lookup[(cid, 'end')]
                if projections[s_idx] <= projections[e_idx]:
                    dag_edges.append((s_idx, e_idx, 'intra'))
                else:
                    dag_edges.append((e_idx, s_idx, 'intra'))

            # Inter-cluster from network edges (already computed as k=1 nearest)
            # Orient by projection
            for edge in edges:
                src_cid, src_label = edge['source']
                dst_cid, dst_label = edge['target']
                s_key = (src_cid, src_label)
                t_key = (dst_cid, dst_label)
                if s_key not in ep_lookup or t_key not in ep_lookup:
                    continue
                s_idx = ep_lookup[s_key]
                t_idx = ep_lookup[t_key]
                is_mutual = edge.get('mutual', False)
                etype = 'mutual' if is_mutual else 'directed'
                if projections[s_idx] <= projections[t_idx]:
                    dag_edges.append((s_idx, t_idx, etype))
                else:
                    dag_edges.append((t_idx, s_idx, etype))

            # Build multi_line data for edges
            edge_xs, edge_ys = [], []
            edge_colors, edge_alphas, edge_widths = [], [], []

            for s_idx, t_idx, etype in dag_edges:
                sx, sy = endpoints[s_idx][2], endpoints[s_idx][3]
                tx, ty = endpoints[t_idx][2], endpoints[t_idx][3]
                edge_xs.append([sx, tx])
                edge_ys.append([sy, ty])

                if etype == 'intra':
                    edge_colors.append('#ffffff')
                    edge_alphas.append(0.5)
                    edge_widths.append(2.0)
                elif etype == 'mutual':
                    edge_colors.append('#4ecdc4')
                    edge_alphas.append(0.7)
                    edge_widths.append(2.0)
                else:
                    edge_colors.append('#4ecdc4')
                    edge_alphas.append(0.25)
                    edge_widths.append(1.0)

            # Build node data — two poles per cluster, distinguished by color
            # Use cluster's own color (from COLORS_50) with brightness shift for the two poles
            node_x, node_y, node_color, node_size = [], [], [], []
            node_line_color, node_label, node_keywords, node_marker = [], [], [], []
            for i, (cid, pole, x, y, kw, pole_label) in enumerate(endpoints):
                node_x.append(x)
                node_y.append(y)
                base_color = COLORS_50[cid % 50]
                # Lighten for one pole, keep saturated for other
                if pole == 'start':
                    node_color.append(base_color)
                    node_marker.append('square')
                else:
                    # Lighten: blend toward white
                    r, g, b = int(base_color[1:3], 16), int(base_color[3:5], 16), int(base_color[5:7], 16)
                    r2 = min(255, r + (255 - r) // 2)
                    g2 = min(255, g + (255 - g) // 2)
                    b2 = min(255, b + (255 - b) // 2)
                    node_color.append(f'#{r2:02x}{g2:02x}{b2:02x}')
                    node_marker.append('circle')
                node_size.append(14)
                node_line_color.append('#ffffff')
                name = names[cid] if cid < len(names) else f'C{cid}'
                node_label.append(f'{name}: {pole_label}')
                node_keywords.append(', '.join(kw))

            # Build label data: use the descriptive pole label
            label_x, label_y, label_text, label_color = [], [], [], []
            for i, (cid, pole, x, y, kw, pole_label) in enumerate(endpoints):
                label_x.append(x)
                label_y.append(y)
                label_text.append(pole_label)
                label_color.append('#ffffff')

            # Build ellipse overlay data for each elongated cluster
            ellipse_x, ellipse_y = [], []
            ellipse_width, ellipse_height, ellipse_angle = [], [], []
            ellipse_fill_alpha, ellipse_line_alpha = [], []
            ellipse_cluster_name, ellipse_gradient_info = [], []

            for cid, g in gradients.items():
                cx, cy = g['center']
                dx, dy = g['major_axis']
                sx, sy = g['start_point']
                ex, ey = g['end_point']

                # Semi-major: half start-to-end distance, scaled to cover ~90% of cluster
                half_len = math.sqrt((ex - sx)**2 + (ey - sy)**2) / 2
                semi_major = half_len * 1.3  # start/end are at 10th percentile bands

                # Semi-minor: from embedding elongation ratio
                semi_minor = semi_major / math.sqrt(max(g['elongation'], 1.0))

                # Rotation angle from major axis unit vector
                angle = math.atan2(dy, dx)

                ellipse_x.append(cx)
                ellipse_y.append(cy)
                ellipse_width.append(semi_major * 2)
                ellipse_height.append(semi_minor * 2)
                ellipse_angle.append(angle)
                ellipse_fill_alpha.append(0.04)
                ellipse_line_alpha.append(0.25)

                name = names[cid] if cid < len(names) else f'C{cid}'
                ellipse_cluster_name.append(name)

                # Build info string for tooltip
                s_label = g.get('start_label', ', '.join(g['start_keywords'][:3]))
                e_label = g.get('end_label', ', '.join(g['end_keywords'][:3]))
                info_parts = [f"{s_label} → {e_label}",
                              f"elongation={g['elongation']:.2f}"]
                emb_ev = g.get('eigenvalues_emb', [])
                if emb_ev:
                    top = emb_ev[0]
                    if len(emb_ev) >= 2 and emb_ev[1] > 0.15:
                        info_parts.append(
                            f"PC1={top:.0%}, PC2={emb_ev[1]:.0%}")
                    else:
                        info_parts.append(f"PC1={top:.0%}")
                ellipse_gradient_info.append(' | '.join(info_parts))

            ellipses_data = {
                'x': ellipse_x, 'y': ellipse_y,
                'width': ellipse_width, 'height': ellipse_height,
                'angle': ellipse_angle,
                'fill_alpha': ellipse_fill_alpha, 'line_alpha': ellipse_line_alpha,
                'cluster_name': ellipse_cluster_name,
                'gradient_info': ellipse_gradient_info,
            }

            dag_entry = {
                'nodes': {
                    'x': node_x, 'y': node_y, 'color': node_color,
                    'size': node_size, 'line_color': node_line_color,
                    'label': node_label, 'keywords': node_keywords,
                    'marker': node_marker,
                },
                'labels': {
                    'x': label_x, 'y': label_y, 'label': label_text,
                    'text_color': label_color,
                },
                'ellipses': ellipses_data,
                'n_elongated': len(gradients),
                'n_edges': len(dag_edges),
            }
            # Only include separate edge data for legacy (non-unified) mode
            if not self.has_unified_edges:
                dag_entry['edges'] = {
                    'xs': edge_xs, 'ys': edge_ys,
                    'line_color': edge_colors, 'line_alpha': edge_alphas,
                    'line_width': edge_widths,
                }
            self._gradient_dag_data[n_clusters] = dag_entry

        print(f"Gradient DAG data built for levels: "
              f"{[k for k, v in self._gradient_dag_data.items() if v]}", flush=True)

    def _on_gradient_dag_toggle(self, attr, old, new):
        """Toggle gradient DAG overlay visibility."""
        self.gradient_dag_mode = new

        if new:
            self._show_gradient_dag()
        else:
            self._hide_gradient_dag()

    def _show_gradient_dag(self):
        """Show gradient DAG overlay — dispatches to unified or legacy."""
        if self.has_unified_edges:
            self._show_gradient_dag_unified()
        else:
            self._show_gradient_dag_legacy()

    def _hide_gradient_dag(self):
        """Hide gradient DAG overlay — dispatches to unified or legacy."""
        if self.has_unified_edges:
            self._hide_gradient_dag_unified()
        else:
            self._hide_gradient_dag_legacy()

    def _show_gradient_dag_unified(self):
        """Show gradient overlay with ellipses and endpoint nodes."""
        # Find DAG data for current level (or fall back)
        dag_data = self._gradient_dag_data.get(self.current_level)
        if not dag_data:
            for level in CLUSTER_LEVELS:
                if self._gradient_dag_data.get(level):
                    dag_data = self._gradient_dag_data[level]
                    break
        if not dag_data:
            return

        # Show ellipse overlays
        if 'ellipses' in dag_data:
            self.gradient_ellipse_source.data = dag_data['ellipses']
            self.gradient_ellipse_renderer.visible = True

        # Show endpoint nodes + labels
        self.gradient_node_source.data = dag_data['nodes']
        self.gradient_label_source.data = dag_data['labels']
        self.gradient_node_renderer.visible = True
        self.gradient_labels.visible = True

        # Dim background points
        n = len(self.coords_2d)
        self.source.data['alpha'] = [0.15] * n
        self.source.data['size'] = [2] * n

        # Dim bridge edges
        xs, ys = self.bridge_edges[self.current_level]
        types = self.edge_types.get(self.current_level, [])
        mutual = self.edge_mutual.get(self.current_level, [])
        if types:
            vxs, vys, vtypes, vmutual = self._cull_edges_to_viewport_typed(
                xs, ys, types, mutual)
            colors, alphas, widths = self._style_edges_by_type(
                vtypes, vmutual, gradient_mode=True)
            self.edge_source.data = {
                'xs': vxs, 'ys': vys,
                'line_color': colors,
                'line_alpha': alphas,
                'line_width': widths,
            }

        n_ellipses = len(dag_data.get('ellipses', {}).get('x', []))
        self.figure.title.text = (
            f"Gradient overlays - {n_ellipses} elongated clusters"
        )

    def _hide_gradient_dag_unified(self):
        """Hide gradient overlay — ellipses, endpoint nodes, labels."""
        # Hide ellipse overlay
        self.gradient_ellipse_renderer.visible = False

        # Hide endpoint nodes + labels
        self.gradient_node_renderer.visible = False
        self.gradient_labels.visible = False

        # Restore point appearance
        self._restore_chunk_state()

        # Restore bridge edges
        if not self._highlighted_indices:
            xs, ys = self.bridge_edges[self.current_level]
            types = self.edge_types.get(self.current_level, [])
            mutual = self.edge_mutual.get(self.current_level, [])

            if types:
                vxs, vys, vtypes, vmutual = self._cull_edges_to_viewport_typed(
                    xs, ys, types, mutual)
                colors, alphas, widths = self._style_edges_by_type(
                    vtypes, vmutual, gradient_mode=False)
                self.edge_source.data = {
                    'xs': vxs, 'ys': vys,
                    'line_color': colors,
                    'line_alpha': alphas,
                    'line_width': widths,
                }
            else:
                vxs, vys = self._cull_edges_to_viewport(xs, ys)
                n_edges = len(vxs)
                self.edge_source.data = {
                    'xs': vxs, 'ys': vys,
                    'line_color': ['#4488cc'] * n_edges,
                    'line_alpha': [0.15] * n_edges,
                    'line_width': [1] * n_edges,
                }

        self.figure.title.text = (
            f"ROG Browser - {len(self.titles):,} points - "
            f"Level: {self.current_level} clusters"
        )

    def _show_gradient_dag_legacy(self):
        """Legacy: show gradient DAG with separate straight-line renderer."""
        dag_data = self._gradient_dag_data.get(self.current_level)
        if not dag_data:
            for level in CLUSTER_LEVELS:
                if self._gradient_dag_data.get(level):
                    dag_data = self._gradient_dag_data[level]
                    break
        if not dag_data:
            return

        self.gradient_edge_source.data = dag_data['edges']
        self.gradient_node_source.data = dag_data['nodes']
        self.gradient_label_source.data = dag_data['labels']
        self.gradient_edge_renderer.visible = True
        self.gradient_node_renderer.visible = True
        self.gradient_labels.visible = True

        # Save and hide bridge edges so they don't clash with DAG edges
        self._pre_gradient_edges_visible = self.edge_renderer.visible
        self.edge_renderer.visible = False

        # Dim background points to make DAG visible
        n = len(self.coords_2d)
        self.source.data['alpha'] = [0.15] * n
        self.source.data['size'] = [2] * n

        self.figure.title.text = (
            f"Gradient DAG - {dag_data['n_elongated']} elongated clusters, "
            f"{dag_data['n_edges']} edges"
        )

    def _hide_gradient_dag_legacy(self):
        """Legacy: hide gradient DAG and restore normal view."""
        self.gradient_edge_renderer.visible = False
        self.gradient_node_renderer.visible = False
        self.gradient_labels.visible = False

        # Restore bridge edge visibility
        was_visible = getattr(self, '_pre_gradient_edges_visible', self.edge_toggle.active)
        self.edge_renderer.visible = was_visible

        # Restore point appearance
        self._restore_chunk_state()

        # Refresh bridge edges for current viewport
        if not self._highlighted_indices:
            xs, ys = self.bridge_edges[self.current_level]
            vxs, vys = self._cull_edges_to_viewport(xs, ys)
            n_edges = len(vxs)
            self.edge_source.data = {
                'xs': vxs, 'ys': vys,
                'line_color': ['#4488cc'] * n_edges,
                'line_alpha': [0.15] * n_edges,
                'line_width': [1] * n_edges,
            }

        self.figure.title.text = (
            f"ROG Browser - {len(self.titles):,} points - "
            f"Level: {self.current_level} clusters"
        )

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
                <li>Zoomed out: 12 clusters</li>
                <li>Light zoom: 25 clusters</li>
                <li>Mid zoom: 50 clusters</li>
                <li>Zoomed in: 100 clusters</li>
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
            return 12
        elif zoom_ratio < ZOOM_THRESHOLDS[1]:
            return 25
        elif zoom_ratio < ZOOM_THRESHOLDS[2]:
            return 50
        else:
            return 100

    def _schedule_label_update(self):
        """Debounce label updates — only process the latest viewport."""
        if self._label_update_callback is not None:
            try:
                curdoc().remove_timeout_callback(self._label_update_callback)
            except ValueError:
                pass
        self._label_update_callback = curdoc().add_timeout_callback(
            self._do_label_update, 150
        )

    def _do_label_update(self):
        self._label_update_callback = None
        self._update_labels_for_viewport()

    def _schedule_view_update(self):
        """Debounce full view updates (level change) — only process the latest."""
        if getattr(self, '_view_update_callback', None) is not None:
            try:
                curdoc().remove_timeout_callback(self._view_update_callback)
            except ValueError:
                pass
        self._view_update_callback = curdoc().add_timeout_callback(
            self._do_view_update, 150
        )

    def _do_view_update(self):
        self._view_update_callback = None
        self._update_view()

    def _on_zoom(self, attr, old, new):
        """Handle zoom changes - update level, labels, and gradients."""
        if self._programmatic_zoom:
            return  # Skip during programmatic zoom changes
        # Check if zoom crossed a level threshold
        zoom_ratio = self._get_zoom_ratio()
        new_level = self._get_level_for_zoom(zoom_ratio)
        if new_level != self.current_level:
            self.current_level = new_level
            self._schedule_view_update()
        else:
            # Same level — just debounce label + edge update
            self._schedule_label_update()

    @staticmethod
    def _simplify_bridge_edges(bridge_edges, max_edges=500):
        """Cap edge count for rendering performance.

        Edges are hidden during interaction and only rendered on idle,
        so full control points are fine — just limit total edge count.
        """
        simplified = {}
        for level, (xs, ys) in bridge_edges.items():
            simplified[level] = (xs[:max_edges], ys[:max_edges])
        return simplified

    @staticmethod
    def _simplify_unified_edges(bridge_edges, edge_types, edge_mutual,
                                max_bridge_edges=500):
        """Cap bridge edges but always keep all gradient edges.

        Reorders edges so capped bridges come first, then all gradient edges.
        Mutates edge_types and edge_mutual dicts in place to match.
        """
        simplified = {}
        for level, (xs, ys) in bridge_edges.items():
            types = edge_types.get(level, [])
            mutual = edge_mutual.get(level, [])

            # Separate bridge and gradient edges
            bridge_xs, bridge_ys, bridge_t, bridge_m = [], [], [], []
            grad_xs, grad_ys, grad_t, grad_m = [], [], [], []
            for i in range(len(xs)):
                t = types[i] if i < len(types) else 'bridge'
                m = mutual[i] if i < len(mutual) else False
                if t == 'bridge':
                    bridge_xs.append(xs[i])
                    bridge_ys.append(ys[i])
                    bridge_t.append(t)
                    bridge_m.append(m)
                else:
                    grad_xs.append(xs[i])
                    grad_ys.append(ys[i])
                    grad_t.append(t)
                    grad_m.append(m)

            # Cap bridge edges, keep all gradient edges
            capped_bx = bridge_xs[:max_bridge_edges]
            capped_by = bridge_ys[:max_bridge_edges]
            capped_bt = bridge_t[:max_bridge_edges]
            capped_bm = bridge_m[:max_bridge_edges]

            # Reassemble: capped bridges + all gradients
            simplified[level] = (capped_bx + grad_xs, capped_by + grad_ys)
            edge_types[level] = capped_bt + grad_t
            edge_mutual[level] = capped_bm + grad_m

        return simplified

    def _cull_edges_to_viewport(self, xs, ys):
        """Filter edges to only those intersecting the current viewport."""
        x0 = self.figure.x_range.start
        x1 = self.figure.x_range.end
        y0 = self.figure.y_range.start
        y1 = self.figure.y_range.end
        # Pad viewport by 20% to keep edges that are partially visible
        dx, dy = (x1 - x0) * 0.2, (y1 - y0) * 0.2
        x0, x1, y0, y1 = x0 - dx, x1 + dx, y0 - dy, y1 + dy

        vxs, vys = [], []
        for ex, ey in zip(xs, ys):
            # Check if any control point of this edge is within the padded viewport
            for px, py in zip(ex, ey):
                if x0 <= px <= x1 and y0 <= py <= y1:
                    vxs.append(ex)
                    vys.append(ey)
                    break
        return vxs, vys

    def _cull_edges_to_viewport_typed(self, xs, ys, types, mutual):
        """Filter edges to viewport, preserving parallel type/mutual arrays.

        Gradient edges are never culled — there are few of them and their
        visibility is controlled by _style_edges_by_type (alpha=0 when off).
        Only bridge edges are viewport-culled for performance.
        """
        x0 = self.figure.x_range.start
        x1 = self.figure.x_range.end
        y0 = self.figure.y_range.start
        y1 = self.figure.y_range.end
        dx, dy = (x1 - x0) * 0.2, (y1 - y0) * 0.2
        x0, x1, y0, y1 = x0 - dx, x1 + dx, y0 - dy, y1 + dy

        vxs, vys, vtypes, vmutual = [], [], [], []
        for ex, ey, et, em in zip(xs, ys, types, mutual):
            if et != 'bridge':
                # Always include gradient edges (skip viewport culling)
                vxs.append(ex)
                vys.append(ey)
                vtypes.append(et)
                vmutual.append(em)
                continue
            for px, py in zip(ex, ey):
                if x0 <= px <= x1 and y0 <= py <= y1:
                    vxs.append(ex)
                    vys.append(ey)
                    vtypes.append(et)
                    vmutual.append(em)
                    break
        return vxs, vys, vtypes, vmutual

    def _style_edges_by_type(self, types, mutual, gradient_mode=False):
        """Map edge types + gradient_mode to per-edge color/alpha/width arrays.

        Args:
            types: list of 'bridge'|'gradient_intra'|'gradient_inter'
            mutual: list of bool (mutual flag per edge)
            gradient_mode: if True, show gradient edges; if False, hide them

        Returns:
            (colors, alphas, widths) — parallel lists for ColumnDataSource
        """
        colors, alphas, widths = [], [], []
        for etype, is_mutual in zip(types, mutual):
            if etype == 'bridge':
                if gradient_mode:
                    colors.append('#4488cc')
                    alphas.append(0.05)
                    widths.append(0.5)
                else:
                    colors.append('#4488cc')
                    alphas.append(0.15)
                    widths.append(1)
            elif etype == 'gradient_intra':
                # Hidden — replaced by ellipse overlays
                colors.append('#ffffff')
                alphas.append(0)
                widths.append(0)
            elif etype == 'gradient_inter':
                if gradient_mode:
                    if is_mutual:
                        colors.append('#4ecdc4')
                        alphas.append(0.7)
                        widths.append(2)
                    else:
                        colors.append('#4ecdc4')
                        alphas.append(0.25)
                        widths.append(1)
                else:
                    colors.append('#4ecdc4')
                    alphas.append(0.0)
                    widths.append(0)
            else:
                # Unknown type — treat as bridge
                colors.append('#4488cc')
                alphas.append(0.15 if not gradient_mode else 0.05)
                widths.append(1 if not gradient_mode else 0.5)
        return colors, alphas, widths

    def _update_view(self):
        """Update labels and bridge edges for current level (colors stay stable)."""
        # Note: Colors are stable based on micro-clusters, so no color update needed

        # Update labels using viewport-aware aggregation
        self._update_labels_for_viewport()

        # Update bridge edges for current level (viewport-culled)
        # If there's an active highlight, reapply it instead of resetting to default
        if self._highlighted_indices:
            self.highlight_points(self._highlighted_indices)
        elif self.has_unified_edges:
            # Unified edges: use type-aware styling
            xs, ys = self.bridge_edges[self.current_level]
            types = self.edge_types.get(self.current_level, [])
            mutual = self.edge_mutual.get(self.current_level, [])
            if types:
                vxs, vys, vtypes, vmutual = self._cull_edges_to_viewport_typed(
                    xs, ys, types, mutual)
                colors, alphas, widths = self._style_edges_by_type(
                    vtypes, vmutual, gradient_mode=self.gradient_dag_mode)
                self.edge_source.data = {
                    'xs': vxs, 'ys': vys,
                    'line_color': colors,
                    'line_alpha': alphas,
                    'line_width': widths,
                }
            else:
                vxs, vys = self._cull_edges_to_viewport(xs, ys)
                n_edges = len(vxs)
                self.edge_source.data = {
                    'xs': vxs, 'ys': vys,
                    'line_color': ['#4488cc'] * n_edges,
                    'line_alpha': [0.15] * n_edges,
                    'line_width': [1] * n_edges,
                }
        else:
            xs, ys = self.bridge_edges[self.current_level]
            # Viewport culling: only include edges that intersect the visible area
            vxs, vys = self._cull_edges_to_viewport(xs, ys)
            n_edges = len(vxs)
            self.edge_source.data = {
                'xs': vxs, 'ys': vys,
                'line_color': ['#4488cc'] * n_edges,
                'line_alpha': [0.15] * n_edges,
                'line_width': [1] * n_edges,
            }

        # Refresh gradient DAG if active
        if self.gradient_dag_mode:
            self._show_gradient_dag()
        else:
            # Update title and status
            self.figure.title.text = f"ROG Browser - {len(self.titles):,} points - Level: {self.current_level} clusters"
        self.status.text = self._get_status_html()

    def layout(self):
        """Return the complete layout."""
        controls = column(
            self.search_input,
            self.clear_btn,
            self.edge_toggle,
            row(self.lsh_toggle, self.density_toggle),
            row(self.recovery_toggle, self.persistence_toggle),
            row(self.dedup_toggle, self.redundancy_toggle),
            self.spread_toggle,
            self.gradient_dag_toggle,
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

        # Capture starting positions — fall back to instant zoom if NaN
        import math
        start_x_start = float(self.figure.x_range.start)
        start_x_end = float(self.figure.x_range.end)
        start_y_start = float(self.figure.y_range.start)
        start_y_end = float(self.figure.y_range.end)

        if any(math.isnan(v) for v in [start_x_start, start_x_end, start_y_start, start_y_end]):
            print(f"  zoom_to: NaN starting positions, falling back to instant zoom", flush=True)
            self.figure.x_range.update(start=target_x_start, end=target_x_end)
            self.figure.y_range.update(start=target_y_start, end=target_y_end)
            self._update_zoom_level()
            def clear_flag_nan():
                self._programmatic_zoom = False
            curdoc().add_timeout_callback(clear_flag_nan, 100)
            return

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

        # Find which clusters the highlighted points belong to (at finest level)
        highlighted_buckets = set()
        finest_level = max(self.cluster_result['labels'].keys())
        if finest_level in self.cluster_result['labels'] and highlighted_set:
            labels_finest = self.cluster_result['labels'][finest_level]
            for idx in self._highlighted_indices:
                highlighted_buckets.add(int(labels_finest[idx]))

        # Highlight bucket-to-bucket edges connected to highlighted buckets
        # Use original edge count, not filtered edge_source
        orig_xs, orig_ys = self.bridge_edges[self.current_level]
        n_edges = len(orig_xs)

        if self.has_unified_edges:
            # Unified edges: apply type-aware base styling, then highlight bridge edges
            types = self.edge_types.get(self.current_level, [])
            mutual = self.edge_mutual.get(self.current_level, [])
            edge_colors, edge_alphas, edge_widths = self._style_edges_by_type(
                types[:n_edges], mutual[:n_edges],
                gradient_mode=self.gradient_dag_mode)

            # Dim all edges when highlighting points
            if highlighted_buckets:
                for i in range(n_edges):
                    if types[i] == 'bridge':
                        edge_colors[i] = '#666666'
                        edge_alphas[i] = 0.08
                        edge_widths[i] = 0.5
                    # Gradient edges keep their type-based styling (already dim or hidden)

            # Highlight bridge edges connected to highlighted clusters
            matched_edges = 0
            if self.cluster_pair_list and highlighted_buckets:
                for edge_idx, (c1, c2) in enumerate(self.cluster_pair_list):
                    if (edge_idx < n_edges
                            and types[edge_idx] == 'bridge'
                            and (int(c1) in highlighted_buckets or int(c2) in highlighted_buckets)):
                        edge_colors[edge_idx] = '#ffaa33'
                        edge_alphas[edge_idx] = 0.9
                        edge_widths[edge_idx] = 3.0
                        matched_edges += 1
        else:
            # Legacy: all edges are bridge edges
            if highlighted_buckets:
                edge_colors = ['#666666'] * n_edges
                edge_alphas = [0.08] * n_edges
                edge_widths = [0.5] * n_edges
            else:
                edge_colors = [self._default_edge_color] * n_edges
                edge_alphas = [self._default_edge_alpha] * n_edges
                edge_widths = [self._default_edge_width] * n_edges

            matched_edges = 0
            if self.cluster_pair_list and highlighted_buckets:
                for edge_idx, (c1, c2) in enumerate(self.cluster_pair_list):
                    if edge_idx < n_edges and (int(c1) in highlighted_buckets or int(c2) in highlighted_buckets):
                        edge_colors[edge_idx] = '#ffaa33'
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

        # Restore alphas based on active chunk modes
        if self.chunk_redundancy_mode and hasattr(self, 'chunk_redundancy_alphas'):
            alphas = list(self.chunk_redundancy_alphas)
        else:
            alphas = [0.7] * n

        # Restore sizes based on active chunk modes
        if self.chunk_spread_mode and hasattr(self, 'chunk_spread_sizes'):
            sizes = list(self.chunk_spread_sizes)
        else:
            sizes = [4] * n

        # Apply dedup overlay
        if self.chunk_dedup_mode and hasattr(self, 'chunk_dedup_mask'):
            mask = self.chunk_dedup_mask
            for i in range(n):
                if not mask[i]:
                    alphas[i] = 0.0
                    sizes[i] = 0

        self.source.data['alpha'] = alphas
        self.source.data['size'] = sizes
        self.source.data['line_color'] = ['rgba(0,0,0,0)'] * n
        self.source.data['line_width'] = [1] * n

        # Reset edges to default (type-aware for unified edges)
        n_edges = len(self.edge_source.data['xs'])
        if self.has_unified_edges:
            types = self.edge_types.get(self.current_level, [])
            mutual = self.edge_mutual.get(self.current_level, [])
            colors, alphas, widths = self._style_edges_by_type(
                types[:n_edges], mutual[:n_edges],
                gradient_mode=self.gradient_dag_mode)
            self.edge_source.data['line_color'] = colors
            self.edge_source.data['line_alpha'] = alphas
            self.edge_source.data['line_width'] = widths
        else:
            self.edge_source.data['line_color'] = ['#4488cc'] * n_edges
            self.edge_source.data['line_alpha'] = [0.15] * n_edges
            self.edge_source.data['line_width'] = [1] * n_edges

    # -------------------------------------------------------------------------
    # Annotation API (ephemeral overlays)
    # -------------------------------------------------------------------------

    def annotate(self, annotation_type: str, params: dict):
        """Add an ephemeral annotation.

        Types:
            title  — HTML title bar above plot (params: text, duration_ms=3000)
            circle — dashed ring with optional label (params: x, y, radius, label="", color="#ffaa33", duration_ms=5000)
            clear  — remove all annotations immediately
        """
        if annotation_type == "clear":
            self._clear_annotations()
            return

        if annotation_type == "title":
            text = params.get("text", "")
            duration_ms = params.get("duration_ms", 3000)
            self.annotation_title.text = f"  {text}  " if text else ""
            if duration_ms > 0:
                self._schedule_annotation_clear(duration_ms, "title")

        elif annotation_type == "circle":
            x = float(params["x"])
            y = float(params["y"])
            radius = float(params.get("radius", 1.0))
            color = params.get("color", "#ffaa33")
            label = params.get("label", "")
            duration_ms = params.get("duration_ms", 5000)
            # Replace previous circles
            self.annotation_source.data = {
                'x': [x], 'y': [y],
                'width': [radius * 2], 'height': [radius * 2],
                'line_color': [color], 'line_width': [2],
                'fill_alpha': [0],
                'label': [label], 'label_y': [y + radius],
            }
            if duration_ms > 0:
                self._schedule_annotation_clear(duration_ms, "circle")

        elif annotation_type == "sketch_circle":
            x = float(params["x"])
            y = float(params["y"])
            radius = float(params.get("radius", 1.0))
            color = params.get("color", "#ffff44")
            width = float(params.get("width", 12.0))
            draw_ms = int(params.get("draw_ms", 600))
            hold_ms = int(params.get("hold_ms", 4000))
            label = params.get("label", "")
            self.sketch_circle(x, y, radius, color=color, width=width,
                               duration_ms=draw_ms, hold_ms=hold_ms, label=label)

        elif annotation_type == "sketch_arc":
            x = float(params["x"])
            y = float(params["y"])
            radius = float(params.get("radius", 1.0))
            start_angle = float(params.get("start_angle", 0))
            end_angle = float(params.get("end_angle", np.pi))
            color = params.get("color", "#ffff44")
            width = float(params.get("width", 12.0))
            draw_ms = int(params.get("draw_ms", 400))
            hold_ms = int(params.get("hold_ms", 0))
            self.sketch_arc(x, y, radius, start_angle, end_angle,
                            color=color, width=width, duration_ms=draw_ms,
                            hold_ms=hold_ms)

        elif annotation_type == "sketch_line":
            x1 = float(params["x1"])
            y1 = float(params["y1"])
            x2 = float(params["x2"])
            y2 = float(params["y2"])
            color = params.get("color", "#ffff44")
            width = float(params.get("width", 12.0))
            draw_ms = int(params.get("draw_ms", 300))
            hold_ms = int(params.get("hold_ms", 0))
            self.sketch_line(x1, y1, x2, y2, color=color, width=width,
                             duration_ms=draw_ms, hold_ms=hold_ms)

        elif annotation_type == "sketch_dot":
            x = float(params["x"])
            y = float(params["y"])
            radius = float(params.get("radius", 0.15))
            color = params.get("color", "#ffff44")
            width = float(params.get("width", 12.0))
            draw_ms = int(params.get("draw_ms", 300))
            hold_ms = int(params.get("hold_ms", 0))
            # A dot is just a tiny filled circle
            self.sketch_circle(x, y, radius, color=color, width=width,
                               duration_ms=draw_ms, hold_ms=hold_ms)

    def _clear_annotations(self):
        """Clear all annotation renderers."""
        self.annotation_title.text = ""
        self.annotation_source.data = {
            'x': [], 'y': [], 'width': [], 'height': [],
            'line_color': [], 'line_width': [], 'fill_alpha': [],
            'label': [], 'label_y': [],
        }
        self._clear_sketch()
        # Cancel all pending auto-clear callbacks
        for cb in self._annotation_callbacks:
            try:
                curdoc().remove_timeout_callback(cb)
            except (ValueError, AttributeError):
                pass
        self._annotation_callbacks = []

    def _clear_sketch(self):
        """Clear sketch strokes and stop any draw animation."""
        if self._sketch_callback is not None:
            try:
                curdoc().remove_periodic_callback(self._sketch_callback)
            except (ValueError, AttributeError):
                pass
            self._sketch_callback = None
        self.sketch_source.data = {
            'xs': [], 'ys': [],
            'line_color': [], 'line_alpha': [], 'line_width': [],
        }

    def _generate_brush_circle(self, x, y, radius, n_points=100, overshoot=0.12,
                               base_width=12.0):
        """Generate a hand-drawn circle as segments with varying brush width.

        Returns (seg_xs, seg_ys, seg_widths) — per-segment 2-point polylines.
        Use with line_alpha=1.0 so overlapping round caps don't cause darkening.
        """
        total_angle = 2 * np.pi * (1 + overshoot)
        angles = np.linspace(0, total_angle, n_points)

        # Random walk for radius drift (organic, not periodic)
        steps = np.random.randn(n_points) * 0.006
        drift = np.cumsum(steps)
        kernel = np.ones(7) / 7
        drift = np.convolve(drift, kernel, mode='same')
        drift *= np.exp(-0.3 * np.abs(drift))

        radii = radius * (1 + drift)

        # Slight angular speed variation (hand doesn't move at constant speed)
        speed_drift = np.cumsum(np.random.randn(n_points) * 0.003)
        speed_drift = np.convolve(speed_drift, kernel, mode='same')
        adjusted_angles = angles + speed_drift

        pts_x = x + radii * np.cos(adjusted_angles)
        pts_y = y + radii * np.sin(adjusted_angles)

        # Width envelope: ease in, sustained, ease out + pressure variation
        seg_xs, seg_ys, seg_widths = [], [], []
        t = np.linspace(0, 1, n_points)
        envelope = np.clip(np.minimum(t * 6, (1 - t) * 5), 0, 1)
        pressure = 1.0 + 0.3 * np.convolve(
            np.random.randn(n_points), np.ones(5) / 5, mode='same'
        )
        widths = base_width * envelope * pressure

        for i in range(n_points - 1):
            seg_xs.append([float(pts_x[i]), float(pts_x[i + 1])])
            seg_ys.append([float(pts_y[i]), float(pts_y[i + 1])])
            seg_widths.append(float(widths[i]))

        return seg_xs, seg_ys, seg_widths

    def _generate_brush_arc(self, x, y, radius, start_angle, end_angle,
                            n_points=60, base_width=12.0):
        """Generate a hand-drawn arc as segments with varying brush width."""
        angles = np.linspace(start_angle, end_angle, n_points)

        # Random walk drift
        steps = np.random.randn(n_points) * 0.006
        drift = np.cumsum(steps)
        kernel = np.ones(7) / 7
        drift = np.convolve(drift, kernel, mode='same')
        drift *= np.exp(-0.3 * np.abs(drift))

        radii = radius * (1 + drift)

        pts_x = x + radii * np.cos(angles)
        pts_y = y + radii * np.sin(angles)

        seg_xs, seg_ys, seg_widths = [], [], []
        t = np.linspace(0, 1, n_points)
        envelope = np.clip(np.minimum(t * 6, (1 - t) * 5), 0, 1)
        pressure = 1.0 + 0.3 * np.convolve(
            np.random.randn(n_points), np.ones(5) / 5, mode='same'
        )
        widths = base_width * envelope * pressure

        for i in range(n_points - 1):
            seg_xs.append([float(pts_x[i]), float(pts_x[i + 1])])
            seg_ys.append([float(pts_y[i]), float(pts_y[i + 1])])
            seg_widths.append(float(widths[i]))

        return seg_xs, seg_ys, seg_widths

    def sketch_arc(self, x, y, radius, start_angle, end_angle,
                   color='#ffff44', width=12.0, duration_ms=400,
                   hold_ms=0, label=''):
        """Draw an animated hand-sketched arc annotation."""
        from bokeh.io import curdoc

        # Stop existing draw animation but keep existing strokes
        if self._sketch_callback is not None:
            try:
                curdoc().remove_periodic_callback(self._sketch_callback)
            except (ValueError, AttributeError):
                pass
            self._sketch_callback = None

        seg_xs, seg_ys, seg_widths = self._generate_brush_arc(
            x, y, radius, start_angle, end_angle, base_width=width
        )
        n_total = len(seg_xs)

        fps = 30
        n_frames = max(1, int(duration_ms * fps / 1000))
        segs_per_frame = max(1, n_total / n_frames)
        frame = [0]

        prev_xs = list(self.sketch_source.data['xs'])
        prev_ys = list(self.sketch_source.data['ys'])
        prev_colors = list(self.sketch_source.data['line_color'])
        prev_alphas = list(self.sketch_source.data['line_alpha'])
        prev_widths = list(self.sketch_source.data['line_width'])

        def draw_step():
            frame[0] += 1
            n_reveal = min(n_total, int(frame[0] * segs_per_frame))

            self.sketch_source.data = {
                'xs': prev_xs + seg_xs[:n_reveal],
                'ys': prev_ys + seg_ys[:n_reveal],
                'line_color': prev_colors + [color] * n_reveal,
                'line_alpha': prev_alphas + [1.0] * n_reveal,
                'line_width': prev_widths + seg_widths[:n_reveal],
            }

            if n_reveal >= n_total:
                try:
                    curdoc().remove_periodic_callback(self._sketch_callback)
                except (ValueError, AttributeError):
                    pass
                self._sketch_callback = None

                if hold_ms > 0:
                    self._schedule_annotation_clear(hold_ms, "sketch")

        interval_ms = int(1000 / fps)
        self._sketch_callback = curdoc().add_periodic_callback(draw_step, interval_ms)

    def _generate_brush_line(self, x1, y1, x2, y2, n_points=40, base_width=12.0):
        """Generate a hand-drawn line as segments with varying brush width."""
        t = np.linspace(0, 1, n_points)

        # Random walk perpendicular drift
        steps = np.random.randn(n_points) * 0.004
        drift = np.cumsum(steps)
        kernel = np.ones(5) / 5
        drift = np.convolve(drift, kernel, mode='same')
        drift *= np.exp(-0.3 * np.abs(drift))

        # Direction vector and perpendicular
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        if length > 0:
            px, py = -dy / length, dx / length  # perpendicular
        else:
            px, py = 0, 1

        pts_x = x1 + t * dx + drift * length * px
        pts_y = y1 + t * dy + drift * length * py

        seg_xs, seg_ys, seg_widths = [], [], []
        envelope = np.clip(np.minimum(t * 6, (1 - t) * 5), 0, 1)
        pressure = 1.0 + 0.25 * np.convolve(
            np.random.randn(n_points), np.ones(5) / 5, mode='same'
        )
        widths = base_width * envelope * pressure

        for i in range(n_points - 1):
            seg_xs.append([float(pts_x[i]), float(pts_x[i + 1])])
            seg_ys.append([float(pts_y[i]), float(pts_y[i + 1])])
            seg_widths.append(float(widths[i]))

        return seg_xs, seg_ys, seg_widths

    def sketch_line(self, x1, y1, x2, y2, color='#ffff44', width=12.0,
                    duration_ms=300, hold_ms=0):
        """Draw an animated hand-sketched line."""
        from bokeh.io import curdoc

        if self._sketch_callback is not None:
            try:
                curdoc().remove_periodic_callback(self._sketch_callback)
            except (ValueError, AttributeError):
                pass
            self._sketch_callback = None

        seg_xs, seg_ys, seg_widths = self._generate_brush_line(
            x1, y1, x2, y2, base_width=width
        )
        n_total = len(seg_xs)

        fps = 30
        n_frames = max(1, int(duration_ms * fps / 1000))
        segs_per_frame = max(1, n_total / n_frames)
        frame = [0]

        prev_xs = list(self.sketch_source.data['xs'])
        prev_ys = list(self.sketch_source.data['ys'])
        prev_colors = list(self.sketch_source.data['line_color'])
        prev_alphas = list(self.sketch_source.data['line_alpha'])
        prev_widths = list(self.sketch_source.data['line_width'])

        def draw_step():
            frame[0] += 1
            n_reveal = min(n_total, int(frame[0] * segs_per_frame))

            self.sketch_source.data = {
                'xs': prev_xs + seg_xs[:n_reveal],
                'ys': prev_ys + seg_ys[:n_reveal],
                'line_color': prev_colors + [color] * n_reveal,
                'line_alpha': prev_alphas + [1.0] * n_reveal,
                'line_width': prev_widths + seg_widths[:n_reveal],
            }

            if n_reveal >= n_total:
                try:
                    curdoc().remove_periodic_callback(self._sketch_callback)
                except (ValueError, AttributeError):
                    pass
                self._sketch_callback = None

                if hold_ms > 0:
                    self._schedule_annotation_clear(hold_ms, "sketch")

        interval_ms = int(1000 / fps)
        self._sketch_callback = curdoc().add_periodic_callback(draw_step, interval_ms)

    def sketch_circle(self, x, y, radius, color='#ffff44', width=12.0,
                      duration_ms=600, hold_ms=4000, label=''):
        """Draw an animated hand-sketched circle annotation.

        Args:
            x, y: Center of the circle
            radius: Radius in data coordinates
            color: Stroke color
            width: Stroke width
            duration_ms: Time to draw the circle
            hold_ms: Time to hold before fading (0 = persistent)
            label: Optional text label above the circle
        """
        from bokeh.io import curdoc

        # Stop any in-progress draw animation, but keep existing strokes
        if self._sketch_callback is not None:
            try:
                curdoc().remove_periodic_callback(self._sketch_callback)
            except (ValueError, AttributeError):
                pass
            self._sketch_callback = None

        seg_xs, seg_ys, seg_widths = self._generate_brush_circle(
            x, y, radius, base_width=width
        )
        n_total = len(seg_xs)

        fps = 30
        n_frames = max(1, int(duration_ms * fps / 1000))
        segs_per_frame = max(1, n_total / n_frames)
        frame = [0]

        # Show label immediately if provided
        if label:
            self.annotation_source.data = {
                'x': [x], 'y': [y],
                'width': [0], 'height': [0],
                'line_color': ['rgba(0,0,0,0)'], 'line_width': [0],
                'fill_alpha': [0],
                'label': [label], 'label_y': [y + radius * 1.15],
            }

        # Snapshot existing strokes so new ones accumulate
        prev_xs = list(self.sketch_source.data['xs'])
        prev_ys = list(self.sketch_source.data['ys'])
        prev_colors = list(self.sketch_source.data['line_color'])
        prev_alphas = list(self.sketch_source.data['line_alpha'])
        prev_widths = list(self.sketch_source.data['line_width'])

        def draw_step():
            frame[0] += 1
            n_reveal = min(n_total, int(frame[0] * segs_per_frame))

            self.sketch_source.data = {
                'xs': prev_xs + seg_xs[:n_reveal],
                'ys': prev_ys + seg_ys[:n_reveal],
                'line_color': prev_colors + [color] * n_reveal,
                'line_alpha': prev_alphas + [1.0] * n_reveal,
                'line_width': prev_widths + seg_widths[:n_reveal],
            }

            if n_reveal >= n_total:
                try:
                    curdoc().remove_periodic_callback(self._sketch_callback)
                except (ValueError, AttributeError):
                    pass
                self._sketch_callback = None

                if hold_ms > 0:
                    self._schedule_annotation_clear(hold_ms, "sketch")

        interval_ms = int(1000 / fps)
        self._sketch_callback = curdoc().add_periodic_callback(draw_step, interval_ms)

    def _schedule_annotation_clear(self, duration_ms: int, annotation_type: str):
        """Schedule auto-clear of a specific annotation type after duration_ms."""
        def do_clear():
            if annotation_type == "title":
                self.annotation_title.text = ""
            elif annotation_type == "circle":
                self.annotation_source.data = {
                    'x': [], 'y': [], 'width': [], 'height': [],
                    'line_color': [], 'line_width': [], 'fill_alpha': [],
                }
            elif annotation_type == "sketch":
                self._clear_sketch()
                # Also clear associated label
                self.annotation_source.data = {
                    'x': [], 'y': [], 'width': [], 'height': [],
                    'line_color': [], 'line_width': [], 'fill_alpha': [],
                    'label': [], 'label_y': [],
                }

        cb = curdoc().add_timeout_callback(do_clear, duration_ms)
        self._annotation_callbacks.append(cb)

    def animate_lsh_explanation(self, step_delay_ms: int = 1500):
        """Animate LSH explanation by adding hyperplanes one at a time.

        Shows how each hyperplane bisects the space and creates new buckets.
        """
        from bokeh.io import curdoc
        import colorsys

        if not self.lsh_data:
            return

        # Stop any existing animation
        if hasattr(self, '_lsh_anim_callback') and self._lsh_anim_callback:
            try:
                curdoc().remove_periodic_callback(self._lsh_anim_callback)
            except:
                pass

        # Get LSH data
        bucket_ids = self.lsh_data['bucket_ids']
        hyperplanes = self.lsh_data['hyperplanes_2d']
        num_bits = self.lsh_data['num_bits']
        n = len(self.coords_2d)

        # Animation state
        self._lsh_anim_step = 0
        self._lsh_anim_max = num_bits

        # Reset view
        self.reset_view(animate=False)

        # Hide cluster edges
        self.edge_renderer.visible = False
        self.edge_toggle.active = False

        # Start with all points grey
        self.source.data['color'] = ['#888888'] * n
        self.source.data['alpha'] = [0.6] * n
        self.source.data['size'] = [5] * n

        # Clear hyperplane display
        self.hyperplane_source.data = {'xs': [], 'ys': []}
        self.hyperplane_renderer.visible = True

        # Clear labels initially
        self.label_source.data = self._label_data([], [], [])

        self.figure.title.text = "LSH Explanation - Starting with all points"

        def animation_step():
            step = self._lsh_anim_step

            if step >= self._lsh_anim_max:
                # Animation complete - show final state
                self.figure.title.text = f"LSH Complete - {num_bits} bits = {len(set(bucket_ids))} buckets"
                try:
                    curdoc().remove_periodic_callback(self._lsh_anim_callback)
                except:
                    pass
                self._lsh_anim_callback = None
                return

            # Compute partial bucket IDs using low (step+1) bits (LSB-first).
            # Bit 0 (LSB) = first PCA component = highest variance, so masking
            # low bits gives the most informative bits first.
            mask = (1 << (step + 1)) - 1  # e.g., step=0 -> mask=1, step=1 -> mask=3
            partial_ids = [int(bid) & mask for bid in bucket_ids]

            # Count unique partial buckets
            unique_partial = len(set(partial_ids))

            # Color points by partial bucket ID
            colors = []
            for pid in partial_ids:
                # Use golden ratio for distinct colors
                hue = (pid * 0.618033988749895) % 1.0
                r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.8)
                colors.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')

            self.source.data['color'] = colors
            self.source.data['alpha'] = [0.7] * n

            # Show hyperplanes up to current step
            xs = [hyperplanes[i]['x'] for i in range(step + 1)]
            ys = [hyperplanes[i]['y'] for i in range(step + 1)]
            self.hyperplane_source.data = {'xs': xs, 'ys': ys}

            # Update title
            bit_str = f"Bit {step}" if step == 0 else f"Bits 0-{step}"
            self.figure.title.text = f"LSH Explanation - {bit_str} → {unique_partial} buckets"

            # Highlight the newest hyperplane by making it brighter
            # (This would require multiple renderers, skip for now)

            self._lsh_anim_step += 1

        # Start animation
        self._lsh_anim_callback = curdoc().add_periodic_callback(animation_step, step_delay_ms)

        # Run first step immediately
        animation_step()


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
    """HTTP handler for control commands from MCP server.

    All methods use local imports and self.server for state access because
    bokeh serve re-executes the script in fresh exec() namespaces per session,
    which clears the globals dict that this class's methods reference.
    """

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        import json as _json

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            state = getattr(self.server, "app_state", None)
            if state is None:
                self.wfile.write(_json.dumps({"status": "no_state"}).encode())
                return

            status = {
                "status": "ok" if (state.browser and state.doc) else "not_ready",
                "browser_initialized": state.browser is not None,
                "doc_initialized": state.doc is not None,
                "queue_length": len(state.command_queue),
            }

            if state.browser:
                b = state.browser
                status["sketch_strokes"] = len(b.sketch_source.data.get("xs", []))
                status["annotations"] = len(b.annotation_source.data.get("x", []))
                status["sketch_animating"] = b._sketch_callback is not None
                status["edges_visible"] = b.edge_renderer.visible if hasattr(b, "edge_renderer") else None
                status["current_level"] = getattr(b, "current_level", None)
                # View bounds for zoom verification
                try:
                    status["x_range"] = {"start": float(b.figure.x_range.start), "end": float(b.figure.x_range.end)}
                    status["y_range"] = {"start": float(b.figure.y_range.start), "end": float(b.figure.y_range.end)}
                except Exception:
                    pass

            self.wfile.write(_json.dumps(status, indent=2).encode())
        except Exception as e:
            import traceback
            print(f"STATUS ERROR: {e}", flush=True)
            traceback.print_exc()
            self.wfile.write(_json.dumps({"error": str(e)}).encode())

    def do_POST(self):
        import json as _json

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        state = getattr(self.server, "app_state", None)
        if state is None or state.browser is None or state.doc is None:
            self.wfile.write(_json.dumps({"error": "Browser not initialized"}).encode())
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = _json.loads(body)
            action = data.get("action")
            params = data.get("params", {})

            # Add command to queue - will be processed by Bokeh's periodic callback
            state.command_queue.append({"action": action, "params": params})
            print(f"HTTP: Queued {action}, queue id={id(state.command_queue)}, len={len(state.command_queue)}", flush=True)
            result = {"status": "ok", "action": action, "queued": True}

            self.wfile.write(_json.dumps(result).encode())
        except Exception as e:
            self.wfile.write(_json.dumps({"error": str(e)}).encode())


def start_control_server(state, port: int = 5008):
    """Start the control HTTP server in a background thread (only once).

    Uses sys._rog_control_server to persist the HTTPServer reference across
    bokeh serve's per-session exec() calls. The sys module is a real import
    that survives namespace resets. State is stored on the HTTPServer instance
    so the handler accesses it via self.server.app_state.
    """
    import sys as _sys

    existing = getattr(_sys, '_rog_control_server', None)

    if existing is not None:
        # Server already running from a previous session — update its state
        existing.app_state = state
        print(f"Control server already running on port {port} (state updated)")
        return

    try:
        server = HTTPServer(("localhost", port), ControlHandler)
        server.app_state = state
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        _sys._rog_control_server = server
        print(f"Control server: http://localhost:{port}/control")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"Control server port {port} already in use (stale from previous process)")
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
        cache.get('lsh_data'),  # LSH bucket visualization data
        cache.get('unified_edge_types'),  # {level: list[str]} per-edge type
        cache.get('unified_edge_mutual'),  # {level: list[bool]} mutual flag
    )

    # Store document reference for control API
    STATE.doc = curdoc()
    STATE.doc.add_root(STATE.browser.layout())
    STATE.doc.title = "ROG Browser"

    # Prevent browser zoom from being hijacked by Bokeh scroll/gesture events
    from bokeh.models import Div
    anti_zoom = Div(text="""
    <style>
        html { touch-action: manipulation; }
        body { touch-action: manipulation; overflow: hidden; }
    </style>
    <script>
        // Block Ctrl+scroll and Cmd+scroll browser zoom over the page
        document.addEventListener('wheel', function(e) {
            if (e.ctrlKey || e.metaKey) { e.preventDefault(); }
        }, {passive: false});
        // Block gesturestart/gesturechange (Safari pinch-zoom)
        document.addEventListener('gesturestart', function(e) { e.preventDefault(); });
        document.addEventListener('gesturechange', function(e) { e.preventDefault(); });
    </script>
    """, visible=False)
    STATE.doc.add_root(anti_zoom)

    # Periodic callback to process command queue
    def process_commands():
        if STATE.command_queue:
            print(f"BOKEH: Processing {len(STATE.command_queue)} commands, queue id={id(STATE.command_queue)}", flush=True)
        while STATE.command_queue:
            cmd = STATE.command_queue.pop(0)
            action = cmd["action"]
            params = cmd.get("params", {})
            print(f"  Executing: {action} with {params}", flush=True)
            try:
                if action == "zoom_to":
                    STATE.browser.zoom_to(params["x"], params["y"], params.get("radius", 5))
                elif action == "reset":
                    STATE.browser.reset_view()
                elif action == "set_level":
                    STATE.browser.set_level(params["level"])
                elif action == "toggle_edges":
                    STATE.browser.set_edges_visible(params["visible"])
                elif action == "set_lsh_mode":
                    STATE.browser.lsh_toggle.active = params.get("active", True)
                elif action == "set_recovery_mode":
                    STATE.browser.recovery_toggle.active = params.get("active", True)
                elif action == "set_persistence_mode":
                    STATE.browser.persistence_toggle.active = params.get("active", True)
                elif action == "animate_lsh":
                    delay = params.get("step_delay_ms", 1500)
                    STATE.browser.animate_lsh_explanation(step_delay_ms=delay)
                elif action == "highlight":
                    STATE.browser.highlight_points(params["indices"])
                elif action == "clear_highlight":
                    STATE.browser.clear_highlight()
                elif action == "set_dedup_mode":
                    STATE.browser.dedup_toggle.active = params.get("active", True)
                elif action == "set_redundancy_mode":
                    STATE.browser.redundancy_toggle.active = params.get("active", True)
                elif action == "set_spread_mode":
                    STATE.browser.spread_toggle.active = params.get("active", True)
                elif action == "annotate":
                    STATE.browser.annotate(params.get("type", "title"), params)
                    print(f"  Sketch source has {len(STATE.browser.sketch_source.data['xs'])} entries", flush=True)
                    print(f"  Annotation source has {len(STATE.browser.annotation_source.data['x'])} items", flush=True)
                # Update last command and refresh status
                STATE.browser.last_command = action
                STATE.browser.status.text = STATE.browser._get_status_html()
            except Exception as e:
                import traceback
                print(f"Error processing command {action}: {e}", flush=True)
                traceback.print_exc()

    STATE.doc.add_periodic_callback(process_commands, 100)  # Check every 100ms

    # Start control server for MCP
    start_control_server(STATE, port=5008)

    print("Ready!")


main()
