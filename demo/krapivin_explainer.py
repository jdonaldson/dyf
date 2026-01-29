"""
Krapivin Hash Table Explainer - Interactive visualization of hierarchical hashing

Demonstrates:
1. Hierarchical levels with geometric sizing
2. Single hash + arithmetic probing across levels
3. How items find slots and overflow to lower levels
4. O(log² δ⁻¹) probe complexity visualization

Run: bokeh serve demo/krapivin_explainer.py --port 5009
"""

from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource, Div, Button, Slider,
    LabelSet, Arrow, VeeHead, Spacer
)
from bokeh.layouts import column, row
from bokeh.palettes import Spectral6
import numpy as np
from dataclasses import dataclass
from typing import Optional
import asyncio

# ============================================================================
# Krapivin Hash Table Simulation
# ============================================================================

@dataclass
class KrapivinTable:
    """Simplified Krapivin hash table for visualization"""
    capacity: int
    delta: float  # empty fraction

    def __post_init__(self):
        # α = O(log δ⁻¹) levels
        self.alpha = max(1, int(np.log2(1.0 / self.delta)))

        # Create levels with geometric sizing
        self.levels = []
        remaining = self.capacity
        for i in range(self.alpha):
            level_size = max(1, remaining // (2 ** (self.alpha - i - 1)))
            self.levels.append([None] * level_size)
            remaining -= level_size

        # Overflow level
        if remaining > 0:
            self.levels.append([None] * remaining)

        self.num_levels = len(self.levels)

    def probe_sequence(self, hash_val: int, max_probes: int = 20):
        """Generate probe sequence across levels"""
        probes = []
        for level_idx, level in enumerate(self.levels):
            level_size = len(level)
            for j in range(min(max_probes, 5)):  # Limit probes per level for viz
                # Quadratic probing with level offset
                offset = level_idx * 1000 + j * j
                slot = (hash_val + offset) % level_size
                probes.append((level_idx, slot))
        return probes

    def insert(self, key: str, hash_val: int):
        """Insert and return probe path taken"""
        path = []
        for level_idx, slot in self.probe_sequence(hash_val):
            path.append((level_idx, slot, self.levels[level_idx][slot] is None))
            if self.levels[level_idx][slot] is None:
                self.levels[level_idx][slot] = key
                return path
            elif self.levels[level_idx][slot] == key:
                return path  # Already exists
        return path  # Table full

    def clear(self):
        """Clear all slots"""
        for level in self.levels:
            for i in range(len(level)):
                level[i] = None


# ============================================================================
# Visualization
# ============================================================================

class KrapivinExplainer:
    def __init__(self):
        # Table parameters
        self.capacity = 64
        self.delta = 0.25
        self.table = KrapivinTable(self.capacity, self.delta)

        # Visual layout constants
        self.level_height = 40
        self.level_gap = 20
        self.slot_padding = 2
        self.max_width = 800

        # Animation state
        self.probe_path = []
        self.current_probe_idx = 0
        self.animating = False
        self.items_inserted = 0

        # Build visualization
        self._build_plot()
        self._build_controls()
        self._update_table_view()

    def _build_plot(self):
        """Create the main visualization figure"""
        total_height = self.table.num_levels * (self.level_height + self.level_gap) + 100

        self.plot = figure(
            width=900, height=max(400, total_height + 150),
            x_range=(-50, self.max_width + 50),
            y_range=(-50, total_height + 50),
            tools="",
            toolbar_location=None,
            title="Krapivin Hash Table: Hierarchical Levels with Geometric Sizing"
        )
        self.plot.grid.visible = False
        self.plot.axis.visible = False

        # Data sources for slots
        self.slot_source = ColumnDataSource(data=dict(
            x=[], y=[], width=[], height=[], color=[], line_color=[], alpha=[]
        ))

        # Draw slots as rectangles
        self.plot.rect(
            x='x', y='y', width='width', height='height',
            fill_color='color', line_color='line_color', fill_alpha='alpha',
            source=self.slot_source
        )

        # Level labels
        self.level_label_source = ColumnDataSource(data=dict(
            x=[], y=[], text=[]
        ))
        level_labels = LabelSet(
            x='x', y='y', text='text', source=self.level_label_source,
            text_font_size='12pt', text_color='#333333',
            x_offset=-45, y_offset=-5
        )
        self.plot.add_layout(level_labels)

        # Slot labels (for filled slots)
        self.slot_label_source = ColumnDataSource(data=dict(
            x=[], y=[], text=[]
        ))
        slot_labels = LabelSet(
            x='x', y='y', text='text', source=self.slot_label_source,
            text_font_size='8pt', text_color='white',
            text_align='center', text_baseline='middle'
        )
        self.plot.add_layout(slot_labels)

        # Probe arrow
        self.arrow_source = ColumnDataSource(data=dict(
            x_start=[], y_start=[], x_end=[], y_end=[]
        ))

        # Probe indicator (circle showing current probe location)
        self.probe_source = ColumnDataSource(data=dict(
            x=[], y=[]
        ))
        self.plot.circle(
            x='x', y='y', size=20, fill_color='red', fill_alpha=0.7,
            line_color='darkred', line_width=3, source=self.probe_source
        )

        # Hash value display
        self.hash_label_source = ColumnDataSource(data=dict(
            x=[self.max_width / 2], y=[self.table.num_levels * (self.level_height + self.level_gap) + 30],
            text=['']
        ))
        hash_label = LabelSet(
            x='x', y='y', text='text', source=self.hash_label_source,
            text_font_size='14pt', text_color='#666666',
            text_align='center'
        )
        self.plot.add_layout(hash_label)

        # Complexity annotation
        complexity_text = f"α = {self.table.num_levels} levels (α = O(log δ⁻¹) where δ = {self.delta})"
        self.plot.add_layout(LabelSet(
            x='x', y='y', text='text',
            source=ColumnDataSource(data=dict(x=[self.max_width/2], y=[-30], text=[complexity_text])),
            text_font_size='11pt', text_color='#888888', text_align='center'
        ))

    def _build_controls(self):
        """Build control widgets"""
        self.insert_btn = Button(label="Insert Random Item", button_type="success", width=150)
        self.insert_btn.on_click(self._on_insert)

        self.step_btn = Button(label="Step Probe", button_type="primary", width=120)
        self.step_btn.on_click(self._on_step)
        self.step_btn.disabled = True

        self.auto_btn = Button(label="Auto Animate", button_type="warning", width=120)
        self.auto_btn.on_click(self._on_auto)

        self.clear_btn = Button(label="Clear Table", button_type="danger", width=100)
        self.clear_btn.on_click(self._on_clear)

        self.fill_btn = Button(label="Fill 50%", button_type="default", width=80)
        self.fill_btn.on_click(self._on_fill)

        # Info panel
        self.info_div = Div(
            text=self._get_info_html(),
            width=400, height=300
        )

        # Explanation panel
        self.explanation_div = Div(
            text=self._get_explanation_html(),
            width=450, height=300
        )

    def _get_info_html(self):
        """Generate info panel HTML"""
        filled = sum(1 for level in self.table.levels for slot in level if slot is not None)
        total = sum(len(level) for level in self.table.levels)

        level_info = ""
        for i, level in enumerate(self.table.levels):
            filled_in_level = sum(1 for s in level if s is not None)
            level_info += f"<li>Level {i}: {len(level)} slots ({filled_in_level} filled)</li>"

        return f"""
        <div style="font-family: sans-serif; font-size: 12px;">
            <h3 style="margin-top:0">Table State</h3>
            <p><b>Capacity:</b> {self.capacity} slots</p>
            <p><b>Delta (δ):</b> {self.delta} (empty fraction target)</p>
            <p><b>Levels (α):</b> {self.table.num_levels}</p>
            <p><b>Load:</b> {filled}/{total} ({100*filled/total:.1f}%)</p>
            <h4>Level Sizes (geometric):</h4>
            <ul style="margin:0; padding-left:20px;">{level_info}</ul>
        </div>
        """

    def _get_explanation_html(self):
        """Generate explanation panel HTML"""
        return """
        <div style="font-family: sans-serif; font-size: 12px;">
            <h3 style="margin-top:0">How Krapivin Hashing Works</h3>

            <p><b>Key insight:</b> Instead of one flat table, use multiple
            <span style="color:#e74c3c">hierarchical levels</span> with
            <span style="color:#3498db">geometric sizing</span>.</p>

            <h4>Probe Sequence:</h4>
            <ol style="margin:0; padding-left:20px;">
                <li><b>Hash once</b> → get a number</li>
                <li><b>Try Level 0</b> (largest) with quadratic probing</li>
                <li><b>Overflow</b> to Level 1, 2... using arithmetic offsets</li>
                <li><b>Stop</b> when empty slot found</li>
            </ol>

            <h4>Why O(log² δ⁻¹)?</h4>
            <ul style="margin:0; padding-left:20px;">
                <li>α = O(log δ⁻¹) levels</li>
                <li>~O(log δ⁻¹) probes per level</li>
                <li>Total: O(log² δ⁻¹) probes worst case</li>
            </ul>

            <h4>Color Key:</h4>
            <ul style="margin:0; padding-left:20px;">
                <li><span style="background:#2ecc71; color:white; padding:2px 6px;">Green</span> = Empty slot</li>
                <li><span style="background:#3498db; color:white; padding:2px 6px;">Blue</span> = Filled slot</li>
                <li><span style="background:#e74c3c; color:white; padding:2px 6px;">Red dot</span> = Current probe</li>
                <li><span style="background:#f39c12; color:white; padding:2px 6px;">Yellow</span> = Collision (move on)</li>
            </ul>
        </div>
        """

    def _get_slot_geometry(self, level_idx: int, slot_idx: int):
        """Calculate x, y, width for a slot"""
        level = self.table.levels[level_idx]
        level_size = len(level)

        # Scale width to fit max_width
        slot_width = min(30, (self.max_width - 20) / level_size - self.slot_padding)
        total_level_width = level_size * (slot_width + self.slot_padding)

        # Center the level
        x_offset = (self.max_width - total_level_width) / 2

        # Y position (level 0 at top)
        y = (self.table.num_levels - 1 - level_idx) * (self.level_height + self.level_gap)

        x = x_offset + slot_idx * (slot_width + self.slot_padding) + slot_width / 2

        return x, y, slot_width

    def _update_table_view(self):
        """Update the visual representation of the table"""
        x, y, width, height, color, line_color, alpha = [], [], [], [], [], [], []
        label_x, label_y, label_text = [], [], []
        slot_label_x, slot_label_y, slot_label_text = [], [], []

        for level_idx, level in enumerate(self.table.levels):
            for slot_idx, slot_val in enumerate(level):
                sx, sy, sw = self._get_slot_geometry(level_idx, slot_idx)

                x.append(sx)
                y.append(sy)
                width.append(sw)
                height.append(self.level_height - 4)

                if slot_val is None:
                    color.append('#27ae60')  # Green for empty
                    alpha.append(0.3)
                    line_color.append('#1e8449')
                else:
                    color.append('#3498db')  # Blue for filled
                    alpha.append(0.8)
                    line_color.append('#2471a3')
                    # Add label for filled slot
                    slot_label_x.append(sx)
                    slot_label_y.append(sy)
                    slot_label_text.append(slot_val[:3] if len(slot_val) > 3 else slot_val)

            # Level label
            first_x, first_y, _ = self._get_slot_geometry(level_idx, 0)
            label_x.append(0)
            label_y.append(first_y)
            label_text.append(f"L{level_idx}")

        self.slot_source.data = dict(
            x=x, y=y, width=width, height=height,
            color=color, line_color=line_color, alpha=alpha
        )
        self.level_label_source.data = dict(x=label_x, y=label_y, text=label_text)
        self.slot_label_source.data = dict(x=slot_label_x, y=slot_label_y, text=slot_label_text)

        # Update info panel
        self.info_div.text = self._get_info_html()

    def _highlight_probe(self, level_idx: int, slot_idx: int, is_empty: bool):
        """Highlight a probed slot"""
        x, y, _ = self._get_slot_geometry(level_idx, slot_idx)

        # Move probe indicator
        self.probe_source.data = dict(x=[x], y=[y])

        # Update hash label with probe info
        self.hash_label_source.data['text'] = [
            f"Probing Level {level_idx}, Slot {slot_idx} → {'EMPTY (insert here!)' if is_empty else 'COLLISION (continue...)'}"
        ]

        # Highlight the slot
        colors = list(self.slot_source.data['color'])
        alphas = list(self.slot_source.data['alpha'])

        # Calculate flat index
        flat_idx = sum(len(self.table.levels[i]) for i in range(level_idx)) + slot_idx

        if is_empty:
            colors[flat_idx] = '#2ecc71'  # Bright green
            alphas[flat_idx] = 1.0
        else:
            colors[flat_idx] = '#f39c12'  # Orange for collision
            alphas[flat_idx] = 1.0

        self.slot_source.data['color'] = colors
        self.slot_source.data['alpha'] = alphas

    def _on_insert(self):
        """Start inserting a new item"""
        self.items_inserted += 1
        key = f"item{self.items_inserted}"
        hash_val = hash(key) & 0xFFFFFFFF

        # Get probe path
        self.probe_path = self.table.probe_sequence(hash_val)
        self.current_probe_idx = 0

        # Store the key and hash for insertion
        self._pending_key = key
        self._pending_hash = hash_val

        # Update hash display
        self.hash_label_source.data = dict(
            x=[self.max_width / 2],
            y=[self.table.num_levels * (self.level_height + self.level_gap) + 30],
            text=[f"Inserting '{key}' | hash = {hash_val} (mod table sizes)"]
        )

        # Enable step button
        self.step_btn.disabled = False

        # Show first probe
        level_idx, slot_idx = self.probe_path[0]
        is_empty = self.table.levels[level_idx][slot_idx] is None
        self._highlight_probe(level_idx, slot_idx, is_empty)

    def _on_step(self):
        """Step through probe sequence"""
        if self.current_probe_idx >= len(self.probe_path):
            self.step_btn.disabled = True
            return

        level_idx, slot_idx = self.probe_path[self.current_probe_idx]
        is_empty = self.table.levels[level_idx][slot_idx] is None

        if is_empty:
            # Insert here
            self.table.levels[level_idx][slot_idx] = self._pending_key
            self._update_table_view()
            self.probe_source.data = dict(x=[], y=[])
            self.hash_label_source.data['text'] = [f"Inserted '{self._pending_key}' at Level {level_idx}, Slot {slot_idx}"]
            self.step_btn.disabled = True
        else:
            # Collision, move to next probe
            self.current_probe_idx += 1
            if self.current_probe_idx < len(self.probe_path):
                next_level, next_slot = self.probe_path[self.current_probe_idx]
                next_empty = self.table.levels[next_level][next_slot] is None
                self._highlight_probe(next_level, next_slot, next_empty)
            else:
                self.hash_label_source.data['text'] = ['Table full!']
                self.step_btn.disabled = True

    def _on_auto(self):
        """Auto-animate insertion"""
        self._on_insert()
        self._auto_step()

    def _auto_step(self):
        """Recursively step with delay"""
        if self.step_btn.disabled:
            return
        self._on_step()
        if not self.step_btn.disabled:
            curdoc().add_timeout_callback(self._auto_step, 400)

    def _on_clear(self):
        """Clear the table"""
        self.table.clear()
        self.items_inserted = 0
        self.probe_source.data = dict(x=[], y=[])
        self.hash_label_source.data['text'] = ['']
        self.step_btn.disabled = True
        self._update_table_view()

    def _on_fill(self):
        """Fill table to ~50%"""
        target = self.capacity // 2
        for i in range(target):
            key = f"x{i}"
            hash_val = hash(key) & 0xFFFFFFFF
            self.table.insert(key, hash_val)
        self.items_inserted = target
        self.probe_source.data = dict(x=[], y=[])
        self._update_table_view()

    def layout(self):
        """Return the complete layout"""
        controls = row(
            self.insert_btn, self.step_btn, self.auto_btn,
            Spacer(width=20), self.fill_btn, self.clear_btn
        )
        info_row = row(self.info_div, Spacer(width=20), self.explanation_div)
        return column(controls, self.plot, info_row)


# ============================================================================
# Main
# ============================================================================

explainer = KrapivinExplainer()
curdoc().add_root(explainer.layout())
curdoc().title = "Krapivin Hash Explainer"
