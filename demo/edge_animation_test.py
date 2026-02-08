#!/usr/bin/env python3
"""Minimal test: PathLayer edges with animation in pydeck OrbitView."""

import json
import pydeck as pdk
import numpy as np
from pathlib import Path

# Generate point cloud with many clusters
np.random.seed(42)
n_points = 100000
n_clusters = 100  # 10x10 grid of clusters

# Create cluster centers in a grid
centroids = []
for row in range(10):
    for col in range(10):
        centroids.append((col * 2 - 9, row * 2 - 9, 0))

# Generate points
points = []
for i in range(n_points):
    cluster = i % n_clusters
    cx, cy, cz = centroids[cluster]
    # Color based on position in grid
    r = int(50 + (cluster % 10) * 20)
    g = int(50 + (cluster // 10) * 20)
    b = int(150 + np.random.rand() * 50)
    points.append({
        "x": cx + np.random.randn() * 0.4,
        "y": cy + np.random.randn() * 0.4,
        "z": np.random.randn() * 0.3,
        "r": r,
        "g": g,
        "b": b,
        "cluster": cluster,
    })

# Generate catenary edges between nearby cluster centers
def catenary(p1, p2, n_pts=20, sag=0.3):
    """Generate catenary curve between two 3D points."""
    path = []
    dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    sag_amount = sag * dist / 2  # Scale sag with distance
    for i in range(n_pts):
        t = i / (n_pts - 1)
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        z = p1[2] + t * (p2[2] - p1[2]) - sag_amount * np.sin(t * np.pi)
        path.append([x, y, z])
    return path

# Connect clusters that are close (grid neighbors)
edges = []
for i in range(n_clusters):
    for j in range(i + 1, n_clusters):
        c1, c2 = centroids[i], centroids[j]
        dist = np.sqrt((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2)
        if dist < 3:  # Only connect nearby clusters
            edges.append({
                "path": catenary(c1, c2),
                "color": [255, 140, 0, 200],
                "width": 0.03,
            })

print(f"Generated {len(points)} points, {len(edges)} edges")

# Create layers
point_layer = pdk.Layer(
    "PointCloudLayer",
    data=points,
    get_position=["x", "y", "z"],
    get_color=["r", "g", "b", 200],
    get_normal=[0, 0, 1],
    point_size=5,
    pickable=True,
)

edge_layer = pdk.Layer(
    "PathLayer",
    data=edges,
    get_path="path",
    get_color="color",
    get_width="width",
    width_scale=1,
    width_min_pixels=2,
    width_max_pixels=8,
    pickable=False,
)

view_state = pdk.ViewState(
    target=[0, 0, 0],
    controller=True,
    rotation_x=15,
    rotation_orbit=30,
    zoom=4,
)

view = pdk.View(type="OrbitView", controller=True)

deck = pdk.Deck(
    layers=[point_layer, edge_layer],
    initial_view_state=view_state,
    views=[view],
)

out_path = Path(__file__).parent / "edge_animation_test.html"
deck.to_html(str(out_path), open_browser=False)

# Inject test controls
html = out_path.read_text()
html = html.replace(
    "const deckInstance = createDeck(",
    "window.deckInstance = createDeck(",
)

edges_json = json.dumps([e["path"] for e in edges])

test_script = f"""
<style>
  #controls {{ position: fixed; top: 10px; left: 10px; z-index: 1000;
              background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px; color: white; }}
  #controls button {{ margin: 5px; padding: 8px 16px; cursor: pointer; }}
  #controls label {{ display: block; margin: 8px 0; }}
  #log {{ position: fixed; bottom: 10px; left: 10px; z-index: 1000;
         background: rgba(0,0,0,0.8); padding: 10px; border-radius: 8px;
         color: #0f0; font-family: monospace; font-size: 12px; max-height: 200px; overflow-y: auto; }}
</style>
<div id="controls">
  <div><b>Edge Physics Test</b></div>
  <button id="btn-start">Start Physics</button>
  <button id="btn-stop">Stop</button>
  <button id="btn-impulse">Apply Impulse</button>
  <label><input type="checkbox" id="chk-edges" checked> Show Edges</label>
  <label>Stiffness: <input type="range" id="stiffness" min="1" max="50" value="15" style="width:80px"> <span id="stiffness-val">15</span></label>
  <label>Damping: <input type="range" id="damping" min="1" max="50" value="25" style="width:80px"> <span id="damping-val">25</span></label>
  <label>Mass: <input type="range" id="mass" min="1" max="20" value="5" style="width:80px"> <span id="mass-val">5</span></label>
  <hr style="border-color:#555; margin:10px 0">
  <div><b>Test sag direction:</b></div>
  <div style="display:flex; flex-wrap:wrap; gap:3px; margin:5px 0;">
    <button class="sag-btn" data-sag="1,0,0">+X</button>
    <button class="sag-btn" data-sag="-1,0,0">-X</button>
    <button class="sag-btn" data-sag="0,1,0">+Y</button>
    <button class="sag-btn" data-sag="0,-1,0">-Y</button>
    <button class="sag-btn" data-sag="0,0,1">+Z</button>
    <button class="sag-btn" data-sag="0,0,-1">-Z</button>
    <button class="sag-btn" data-sag="auto">Auto</button>
  </div>
  <hr style="border-color:#555; margin:10px 0">
  <div style="font-family:monospace; font-size:11px;">
    <div>orbit: <span id="dbg-orbit">-</span>° pitch: <span id="dbg-pitch">-</span>°</div>
    <div>sag: <span id="dbg-gx">-</span></div>
    <div>mode: <span id="dbg-gy">-</span></div>
    <div>current: [<span id="dbg-sx">-</span>, <span id="dbg-sy">-</span>]</div>
    <div>sagZ: <span id="dbg-vx">-</span> | <span id="dbg-vy">-</span></div>
    <div>viewState: <span id="dbg-vs">-</span></div>
  </div>
</div>
<div id="log"></div>
<script>
(function() {{
  var logEl = document.getElementById("log");
  function log(msg) {{
    var line = document.createElement("div");
    line.textContent = new Date().toISOString().substr(11, 8) + " " + msg;
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
    if (logEl.children.length > 30) logEl.removeChild(logEl.firstChild);
  }}

  function getDeck() {{
    var d = window.deckInstance;
    return d && d.deck ? d.deck : d || null;
  }}

  // Original edge paths
  var origEdgePaths = {edges_json};
  var edgesVisible = true;

  // Cache original layers
  var _origPointLayer = null;
  var _origEdgeLayer = null;

  function cacheOriginalLayers() {{
    var dk = getDeck();
    if (dk && dk.props && dk.props.layers && !_origPointLayer) {{
      _origPointLayer = dk.props.layers[0];
      _origEdgeLayer = dk.props.layers[1] || null;
      log("Cached layers");
    }}
  }}

  // ── Physics state ──
  // 2D displacement and velocity for directional sway
  var swayPos = [0, 0];      // current displacement [x, y]
  var swayVel = [0, 0];      // current velocity [x, y]

  // Track mouse drag
  var isDragging = false;
  var lastMouseX = 0;
  var lastMouseY = 0;
  var mouseForce = [0, 0];   // accumulated force from drag

  var physicsTimer = null;
  var lastTime = performance.now();

  function getParams() {{
    return {{
      stiffness: parseFloat(document.getElementById("stiffness").value),
      damping: parseFloat(document.getElementById("damping").value),
      mass: parseFloat(document.getElementById("mass").value)
    }};
  }}

  // Mouse tracking on canvas
  function setupMouseTracking() {{
    var canvas = document.querySelector("#deck-container canvas");
    if (!canvas) {{
      setTimeout(setupMouseTracking, 500);
      return;
    }}
    log("Mouse tracking enabled");

    canvas.addEventListener("mousedown", function(e) {{
      isDragging = true;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    }});

    window.addEventListener("mouseup", function() {{
      isDragging = false;
    }});

    canvas.addEventListener("mousemove", function(e) {{
      if (!isDragging) return;
      var dx = e.clientX - lastMouseX;
      var dy = e.clientY - lastMouseY;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
      // Apply force in drag direction (screen Y flipped for world coords)
      mouseForce[0] = dx * 0.8;
      mouseForce[1] = -dy * 0.8;
    }});

    // Touch support
    canvas.addEventListener("touchstart", function(e) {{
      isDragging = true;
      lastMouseX = e.touches[0].clientX;
      lastMouseY = e.touches[0].clientY;
    }});

    window.addEventListener("touchend", function() {{
      isDragging = false;
    }});

    canvas.addEventListener("touchmove", function(e) {{
      if (!isDragging || !e.touches.length) return;
      var dx = e.touches[0].clientX - lastMouseX;
      var dy = e.touches[0].clientY - lastMouseY;
      lastMouseX = e.touches[0].clientX;
      lastMouseY = e.touches[0].clientY;
      mouseForce[0] = dx * 0.8;
      mouseForce[1] = -dy * 0.8;
    }});
  }}

  // Track view state via onViewStateChange hook
  var liveViewState = {{ orbit: 30, pitch: 15 }};  // initial values

  function hookViewStateChange() {{
    var dk = getDeck();
    if (!dk) {{
      setTimeout(hookViewStateChange, 500);
      return;
    }}

    // Wrap the existing onViewStateChange if any
    var origCallback = dk.props.onViewStateChange;
    dk.setProps({{
      onViewStateChange: function(params) {{
        var vs = params.viewState;
        if (vs) {{
          liveViewState.orbit = vs.rotationOrbit !== undefined ? vs.rotationOrbit : liveViewState.orbit;
          liveViewState.pitch = vs.rotationX !== undefined ? vs.rotationX : liveViewState.pitch;
        }}
        if (origCallback) origCallback(params);
      }}
    }});
    log("Hooked onViewStateChange");
  }}

  // Get current view angles from live tracking
  function getViewAngles() {{
    document.getElementById("dbg-orbit").textContent = liveViewState.orbit.toFixed(1);
    document.getElementById("dbg-pitch").textContent = liveViewState.pitch.toFixed(1);
    document.getElementById("dbg-vs").textContent = "live ✓";
    return liveViewState;
  }}

  // Compute target sag direction from camera orientation
  // Use viewport's unproject to find screen-down in world coords
  function getTargetSag() {{
    // If manual override is set, use that
    if (manualSag) {{
      document.getElementById("dbg-gx").textContent = manualSag.join(",") + " (manual)";
      return manualSag;
    }}

    var dk = getDeck();
    if (!dk) return [0, 0, -1];

    // Get the viewport
    var viewports = dk.getViewports ? dk.getViewports() : null;
    if (!viewports || !viewports.length) {{
      document.getElementById("dbg-gx").textContent = "no viewport";
      return [0, 0, -1];
    }}
    var vp = viewports[0];

    // Get screen center and a point below it
    var w = vp.width || 800;
    var h = vp.height || 600;
    var centerX = w / 2;
    var centerY = h / 2;

    try {{
      // Unproject screen center and a point 100px below
      var p1 = vp.unproject([centerX, centerY]);
      var p2 = vp.unproject([centerX, centerY + 100]);

      if (!p1 || !p2) {{
        document.getElementById("dbg-gx").textContent = "unproject failed";
        return [0, 0, -1];
      }}

      // Direction from p1 to p2 is "screen down" in world coords
      var dx = p2[0] - p1[0];
      var dy = p2[1] - p1[1];
      var dz = (p2[2] || 0) - (p1[2] || 0);

      // Normalize
      var len = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1;
      var sagX = dx / len;
      var sagY = dy / len;
      var sagZ = dz / len;

      document.getElementById("dbg-gx").textContent =
        sagX.toFixed(2) + "," + sagY.toFixed(2) + "," + sagZ.toFixed(2);

      return [sagX, sagY, sagZ];
    }} catch(e) {{
      document.getElementById("dbg-gx").textContent = "error: " + e.message;
      return [0, 0, -1];
    }}
  }}

  // 3D sag direction with smooth interpolation
  var currentSag = [0, 0, -1];  // current sag direction [x, y, z]
  var sagVelocity = [0, 0, 0];  // velocity for smooth transitions

  function physicsStep(dt) {{
    var p = getParams();
    var k = p.stiffness * 0.5;  // spring pulls toward target
    var c = p.damping * 0.3;    // damping
    var m = p.mass;

    // Get target sag direction from camera
    var target = getTargetSag();

    // Decay mouse force when not dragging
    if (!isDragging) {{
      mouseForce[0] *= 0.85;
      mouseForce[1] *= 0.85;
    }}

    // For each axis (x, y, z) - spring toward target sag direction
    for (var axis = 0; axis < 3; axis++) {{
      var x = currentSag[axis];
      var v = sagVelocity[axis];
      var targetVal = target[axis];

      // Mouse force only affects X and Y
      var ext = (axis < 2) ? mouseForce[axis] * 0.3 : 0;

      // Spring pulls toward target direction
      var springF = k * (targetVal - x);
      var dampF = -c * v;
      var totalF = springF + dampF + ext;

      var a = totalF / m;
      v += a * dt;
      x += v * dt;

      sagVelocity[axis] = v;
      currentSag[axis] = x;
    }}

    // Normalize sag direction
    var len = Math.sqrt(currentSag[0]*currentSag[0] + currentSag[1]*currentSag[1] + currentSag[2]*currentSag[2]) || 1;
    // Don't fully normalize - allow magnitude to vary for effect
    // currentSag[0] /= len; currentSag[1] /= len; currentSag[2] /= len;

    // Update debug display
    document.getElementById("dbg-sx").textContent = currentSag[0].toFixed(3);
    document.getElementById("dbg-sy").textContent = currentSag[1].toFixed(3);
    document.getElementById("dbg-vx").textContent = currentSag[2].toFixed(3);
    document.getElementById("dbg-vy").textContent = "t:" + target[2].toFixed(2);
  }}

  // Pre-compute random offsets per edge for variation
  var edgeRandom = [];
  for (var _e = 0; _e < origEdgePaths.length; _e++) {{
    edgeRandom.push({{
      phase1: Math.random() * Math.PI * 2,
      phase2: Math.random() * Math.PI * 2,
      phase3: Math.random() * Math.PI * 2,
      amp: 0.4 + Math.random() * 1.2,       // 0.4 to 1.6
      freq1: 0.5 + Math.random() * 1.0,     // 0.5 to 1.5
      freq2: 0.3 + Math.random() * 0.8,     // 0.3 to 1.1 (slower secondary)
      freq3: 1.0 + Math.random() * 2.0,     // 1.0 to 3.0 (faster tertiary)
      wobbleAmp: 0.1 + Math.random() * 0.4, // 0.1 to 0.5
      delay: Math.random() * 2.0,           // reaction delay
      damping: 0.5 + Math.random() * 0.5    // individual damping
    }});
  }}

  var swayTime = 0;

  function computeSwayedPaths() {{
    var swayed = [];
    swayTime += 0.05;

    // Sag direction in 3D from physics
    var sagX = currentSag[0];
    var sagY = currentSag[1];
    var sagZ = currentSag[2];

    // Motion intensity based on velocity magnitude - squared for faster falloff
    var velMag = Math.sqrt(sagVelocity[0]*sagVelocity[0] + sagVelocity[1]*sagVelocity[1] + sagVelocity[2]*sagVelocity[2]);
    var motionIntensity = Math.min(1, velMag * velMag * 0.3);  // squared = faster decay to zero

    for (var i = 0; i < origEdgePaths.length; i++) {{
      var path = origEdgePaths[i];
      var p0 = path[0], pn = path[path.length - 1];
      var edgeLen = Math.sqrt(
        Math.pow(pn[0]-p0[0], 2) + Math.pow(pn[1]-p0[1], 2) + Math.pow(pn[2]-p0[2], 2)
      ) || 1;

      var r = edgeRandom[i];
      var t_delayed = Math.max(0, swayTime - r.delay);

      // Wobbles are scaled by motion intensity - no motion = no wobble
      var wobble1 = Math.sin(t_delayed * r.freq1 + r.phase1) * r.wobbleAmp * motionIntensity;
      var wobble2 = Math.sin(t_delayed * r.freq2 + r.phase2) * r.wobbleAmp * 0.7 * motionIntensity;
      var wobble3 = Math.sin(t_delayed * r.freq3 + r.phase3) * r.wobbleAmp * 0.3 * motionIntensity;
      var totalWobble = wobble1 + wobble2 + wobble3;

      // Perpendicular directions
      var perpX = -sagY;
      var perpY = sagX;
      // Second perpendicular (cross product for Z component variation)
      var perp2X = sagZ * sagY;
      var perp2Y = -sagZ * sagX;
      var perp2Z = sagX * sagX + sagY * sagY;

      var newPath = [];
      for (var j = 0; j < path.length; j++) {{
        var t = j / (path.length - 1);
        // Sine envelope: max sag at middle, zero at endpoints
        var envelope = Math.sin(t * Math.PI);

        // Curve noise also scaled by motion
        var curveNoise = Math.sin(t * 3.14159 * 2 + r.phase3) * 0.2 * motionIntensity;

        var sagAmount = envelope * edgeLen * 0.25 * r.amp;
        var wobbleAmount = envelope * totalWobble * edgeLen * 0.15;
        var wobble2Amount = envelope * curveNoise * wobble2 * edgeLen * 0.1;

        newPath.push([
          path[j][0] + sagX * sagAmount + perpX * wobbleAmount + perp2X * wobble2Amount,
          path[j][1] + sagY * sagAmount + perpY * wobbleAmount + perp2Y * wobble2Amount,
          path[j][2] + sagZ * sagAmount + perp2Z * wobble2Amount * 0.3
        ]);
      }}
      swayed.push(newPath);
    }}
    return swayed;
  }}

  function rebuildEdges() {{
    var dk = getDeck();
    if (!dk || !_origPointLayer || !_origEdgeLayer) return;
    if (!edgesVisible) {{
      dk.setProps({{ layers: [_origPointLayer] }});
      return;
    }}

    var swayedPaths = computeSwayedPaths();
    var edgeData = swayedPaths.map(function(path) {{
      return {{ path: path, color: [255, 140, 0, 255], width: 0.05 }};
    }});

    var newEdgeLayer = _origEdgeLayer.clone({{
      data: edgeData,
      getPath: function(d) {{ return d.path; }},
      getColor: function(d) {{ return d.color; }},
      getWidth: function(d) {{ return d.width; }}
    }});

    dk.setProps({{ layers: [_origPointLayer, newEdgeLayer] }});
  }}

  function physicsLoop() {{
    var now = performance.now();
    var dt = Math.min((now - lastTime) / 1000, 0.05);  // cap at 50ms
    lastTime = now;

    physicsStep(dt);
    rebuildEdges();
  }}

  function startPhysics() {{
    if (physicsTimer) return;
    log("Starting physics simulation");
    cacheOriginalLayers();
    lastTime = performance.now();
    physicsTimer = setInterval(physicsLoop, 16);  // ~60fps
  }}

  function stopPhysics() {{
    if (physicsTimer) {{
      clearInterval(physicsTimer);
      physicsTimer = null;
      log("Stopped physics");
    }}
  }}

  function applyImpulse() {{
    log("Impulse!");
    // Random 2D impulse
    sagVelocity[0] += (Math.random() - 0.5) * 15;
    sagVelocity[1] += (Math.random() - 0.5) * 15;
    sagVelocity[2] += (Math.random() - 0.5) * 15;
  }}

  // Manual sag direction override for testing
  var manualSag = null;  // null = auto, or [x,y,z]

  function setManualSag(sag) {{
    manualSag = sag;
    document.getElementById("dbg-gy").textContent = sag ? "manual " + sag.join(",") : "auto";
    log(sag ? "Manual sag: " + sag.join(",") : "Auto sag");
  }}

  // Setup sag test buttons
  function setupSagButtons() {{
    document.querySelectorAll(".sag-btn").forEach(function(btn) {{
      btn.onclick = function() {{
        var val = btn.getAttribute("data-sag");
        if (val === "auto") {{
          setManualSag(null);
        }} else {{
          var parts = val.split(",").map(Number);
          setManualSag(parts);
        }}
      }};
    }});
  }}

  // UI bindings
  document.getElementById("btn-start").onclick = startPhysics;
  document.getElementById("btn-stop").onclick = stopPhysics;
  document.getElementById("btn-impulse").onclick = applyImpulse;
  document.getElementById("chk-edges").onchange = function() {{
    edgesVisible = this.checked;
    rebuildEdges();
  }};

  // Slider value displays
  ["stiffness", "damping", "mass"].forEach(function(id) {{
    var el = document.getElementById(id);
    var valEl = document.getElementById(id + "-val");
    el.oninput = function() {{ valEl.textContent = el.value; }};
  }});

  // Init
  setTimeout(function() {{
    cacheOriginalLayers();
    setupMouseTracking();
    hookViewStateChange();
    setupSagButtons();
    log("Ready - drag to rotate, edges will react");
  }}, 1000);
}})();
</script>
"""

html = html.replace("</body>", test_script + "\n</body>")
out_path.write_text(html)
print(f"Wrote {out_path}")
