import { tableFromIPC } from "https://cdn.jsdelivr.net/npm/apache-arrow@18.1.0/+esm";

(async function() {
  var dd = window.__DYF_DATA__ || null;

  // ── Base64 → Uint8Array, then gzip decompress ─────────────────────
  function b64toBytes(b64) {
    var bin = atob(b64), n = bin.length, u8 = new Uint8Array(n);
    for (var i = 0; i < n; i++) u8[i] = bin.charCodeAt(i);
    return u8;
  }
  async function ungzip(bytes) {
    var ds = new DecompressionStream("gzip");
    var writer = ds.writable.getWriter();
    writer.write(bytes);
    writer.close();
    return new Uint8Array(await new Response(ds.readable).arrayBuffer());
  }

  // ── Generate and display session ID ───────────────────────────────
  var sessionId = Math.random().toString(36).substring(2, 8).toUpperCase();
  var sessionEl = document.getElementById("session-id");
  if (sessionEl) sessionEl.textContent = "Session: " + sessionId;

  // ── Multi-level cluster metadata and color functions ──────────────
  var clusterMeta = null;  // legacy: unused in dendrogram mode
  var currentLevelKey = "dendro";
  var defaultLevelKey = "dendro";

  // HLS→RGB for color map generation; precomputed spatial maps preferred,
  // golden ratio fallback when no precomputed map exists for a level
  function _hlsHelper(m1, m2, hue) {
    hue = hue % 1.0;
    if (hue < 1/6) return m1 + (m2 - m1) * hue * 6;
    if (hue < 0.5) return m2;
    if (hue < 2/3) return m1 + (m2 - m1) * (2/3 - hue) * 6;
    return m1;
  }
  function hlsToRgb(h, l, s) {
    if (s === 0) return [Math.round(l*255), Math.round(l*255), Math.round(l*255)];
    var m2 = l <= 0.5 ? l * (1 + s) : l + s - l * s;
    var m1 = 2 * l - m2;
    return [
      Math.round(_hlsHelper(m1, m2, h + 1/3) * 255),
      Math.round(_hlsHelper(m1, m2, h) * 255),
      Math.round(_hlsHelper(m1, m2, h - 1/3) * 255)
    ];
  }
  function buildColorMap(uniqueCids, levelKey) {
    // Use LSH bucket colors in bucket mode
    if (_colorMode === "bucket" && _lshBucketColors && Object.keys(_lshBucketColors).length > 0) {
      var map = {};
      for (var i = 0; i < uniqueCids.length; i++) {
        var cid = uniqueCids[i];
        map[cid] = _lshBucketColors[String(cid)] || hlsToRgb((i * 0.618033988749895) % 1.0, 0.45, 0.6);
      }
      return map;
    }
    // Use dendrogram community colors if available
    if (_communityColors && Object.keys(_communityColors).length > 0) {
      var map = {};
      for (var i = 0; i < uniqueCids.length; i++) {
        var cid = uniqueCids[i];
        map[cid] = _communityColors[String(cid)] || hlsToRgb((i * 0.618033988749895) % 1.0, 0.45, 0.6);
      }
      return map;
    }
    // Fallback: golden ratio
    var sorted = uniqueCids.slice().sort(function(a,b) { return a - b; });
    var map = {};
    for (var i = 0; i < sorted.length; i++) {
      var hue = (i * 0.618033988749895) % 1.0;
      map[sorted[i]] = hlsToRgb(hue, 0.45, 0.6);
    }
    return map;
  }

  // ── Reconstruct point data from DYF or gzipped Arrow IPC ──────────
  var _pt, _x, _y, _z, _a, _titles, _nPts;
  if (dd && dd.x) {
    // DYF mode: typed arrays provided directly
    _x = dd.x; _y = dd.y; _z = dd.z;
    _nPts = dd.nPts;
    _a = dd.a || new Uint8Array(_nPts).fill(255);
    var _titlesArr = dd.titles || [];
    _titles = { get: function(i) { return _titlesArr[i]; } };
  } else if (dd && dd.pointsIpcB64) {
    // Baked mode via __DYF_DATA__
    _pt = tableFromIPC(await ungzip(b64toBytes(dd.pointsIpcB64)));
    _x = _pt.getChild("x").toArray();
    _y = _pt.getChild("y").toArray();
    _z = _pt.getChild("z").toArray();
    _a = _pt.getChild("a").toArray();
    _titles = _pt.getChild("title");
    _nPts = _pt.numRows;
  } else {
    // Legacy template mode
    _pt = tableFromIPC(await ungzip(b64toBytes("__POINTS_IPC_B64__")));
    _x = _pt.getChild("x").toArray();
    _y = _pt.getChild("y").toArray();
    _z = _pt.getChild("z").toArray();
    _a = _pt.getChild("a").toArray();
    _titles = _pt.getChild("title");
    _nPts = _pt.numRows;
  }
  var allPoints = new Array(_nPts);

  // ── Dendrogram-based continuous cluster slider ────────────────────────
  // The dendrogram stores ~50 leaf→community mappings + a merge linkage
  // matrix. Cutting at any height yields any k from 1 to natural_k.
  var _dendro = dd ? dd.dendrogram : null;
  var _pointToCommunity = null;  // Int32Array(N): point → natural community ID
  var _communityColors = {};     // {cid: [r,g,b]} for natural communities
  var _communityNames = {};      // {cid: "name"} for natural communities
  var _communityCentroids = {};  // {cid: [x,y,z]} for natural communities
  var _communitySizes = {};      // {cid: int} for natural communities
  var _dendroZ = null;           // linkage matrix (k-1 × 4)
  var _naturalK = 0;
  var _currentMergedIds = null;  // Int32Array(N): point → merged cluster ID at current cut

  if (_dendro) {
    _dendroZ = _dendro.Z;
    _naturalK = _dendro.naturalK;
    _communityColors = _dendro.communityColors || {};
    _communityNames = _dendro.communityNames || {};
    _communityCentroids = _dendro.communityCentroids || {};
    _communitySizes = _dendro.communitySizes || {};

    // Build point → community mapping: prefer community_id stored field
    // (correct post-reassignment labels) over leaf-map reconstruction
    if (_dendro.communityIds) {
      _pointToCommunity = new Int32Array(_dendro.communityIds);
    } else {
      _pointToCommunity = new Int32Array(_nPts).fill(-1);
      var lim = _dendro.leafItemMap;
      var l2c = _dendro.leafToCommunity;
      for (var leafIdx in lim) {
        var commId = l2c[leafIdx];
        if (commId === undefined) continue;
        var items = lim[leafIdx];
        for (var j = 0; j < items.length; j++) {
          var itemIdx = items[j];
          if (itemIdx < _nPts) _pointToCommunity[itemIdx] = commId;
        }
      }
      // Handle unassigned (shouldn't happen but be safe)
      for (var i = 0; i < _nPts; i++) {
        if (_pointToCommunity[i] < 0) _pointToCommunity[i] = 0;
      }
    }

    // Initialize points with natural community colors
    var _uniqueNatural = [];
    var _seenNat = {};
    for (var i = 0; i < _nPts; i++) {
      var c = _pointToCommunity[i];
      if (!(c in _seenNat)) { _seenNat[c] = true; _uniqueNatural.push(c); }
    }
    for (var i = 0; i < _nPts; i++) {
      var cid = _pointToCommunity[i];
      var rgb = _communityColors[String(cid)] || [128, 128, 128];
      allPoints[i] = {
        x: _x[i], y: _y[i], z: 0,
        r: rgb[0], g: rgb[1], b: rgb[2], a: _a[i],
        title: _titles.get(i), cluster: cid,
        clusters: {}
      };
    }
    _currentMergedIds = new Int32Array(_pointToCommunity);  // start at natural k

    // Recompute community centroids with actual 3D z-coordinates from UMAP
    // (the .dyf metadata centroids only have 2D x,y with z=0)
    var _centSums = {};  // {cid: {sx, sy, sz, count}}
    for (var i = 0; i < _nPts; i++) {
      var cid = _pointToCommunity[i];
      if (!_centSums[cid]) _centSums[cid] = { sx: 0, sy: 0, sz: 0, count: 0 };
      _centSums[cid].sx += _x[i];
      _centSums[cid].sy += _y[i];
      _centSums[cid].sz += _z[i];
      _centSums[cid].count++;
    }
    for (var cid in _centSums) {
      var s = _centSums[cid];
      _communityCentroids[cid] = [s.sx / s.count, s.sy / s.count, s.sz / s.count];
    }
  } else {
    // No dendrogram: grey points
    for (var _i = 0; _i < _nPts; _i++) {
      allPoints[_i] = {
        x: _x[_i], y: _y[_i], z: 0,
        r: 128, g: 128, b: 128, a: _a[_i],
        title: _titles.get(_i), cluster: 0,
        clusters: {}
      };
    }
  }

  // ── Union-Find for dendrogram cutting ──────────────────────────────────
  function _ufFind(parent, x) {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];  // path compression
      x = parent[x];
    }
    return x;
  }

  /**
   * Generic linkage cut: given a linkage matrix Z (n-1 × 4) and n leaves,
   * cut at height → { mergedMap: {leafIdx → groupId}, k: numGroups }
   */
  function cutLinkage(Z, n, height) {
    if (!Z || !n) return null;
    var totalNodes = n + Z.length;
    var parent = new Int32Array(totalNodes);
    var size = new Int32Array(totalNodes);
    for (var i = 0; i < totalNodes; i++) { parent[i] = i; size[i] = 1; }
    for (var i = 0; i < Z.length; i++) {
      if (Z[i][2] > height) continue;
      var a = Math.round(Z[i][0]);
      var b = Math.round(Z[i][1]);
      var mergeNode = n + i;
      var ra = _ufFind(parent, a);
      var rb = _ufFind(parent, b);
      parent[ra] = mergeNode;
      parent[rb] = mergeNode;
      size[mergeNode] = size[ra] + size[rb];
    }
    var roots = {};
    var rootList = [];
    var mergedMap = {};
    for (var i = 0; i < n; i++) {
      var root = _ufFind(parent, i);
      if (!(root in roots)) {
        roots[root] = rootList.length;
        rootList.push(root);
      }
      mergedMap[i] = roots[root];
    }
    return { mergedMap: mergedMap, k: rootList.length };
  }

  /** Cut the community dendrogram at a given height. */
  function cutDendrogram(height) {
    return cutLinkage(_dendroZ, _naturalK, height);
  }

  /**
   * Build a complete-linkage agglomerative clustering on centroids.
   * centroids: [[x,y,z], ...], returns linkage matrix Z (n-1 × 4).
   */
  function buildLinkage(centroids) {
    var n = centroids.length;
    if (n <= 1) return [];

    // Precompute pairwise distances
    var pairDist = {};
    for (var i = 0; i < n; i++) {
      for (var j = i + 1; j < n; j++) {
        var dx = centroids[i][0] - centroids[j][0];
        var dy = centroids[i][1] - centroids[j][1];
        var dz = (centroids[i][2] || 0) - (centroids[j][2] || 0);
        pairDist[i + "," + j] = Math.sqrt(dx * dx + dy * dy + dz * dz);
      }
    }

    // Track which original points are in each active cluster
    var members = {};
    for (var i = 0; i < n; i++) members[i] = [i];
    var active = new Set();
    for (var i = 0; i < n; i++) active.add(i);

    var Z = [];
    var nextId = n;

    for (var step = 0; step < n - 1; step++) {
      // Find closest pair (complete linkage: max pairwise distance)
      var bestA = -1, bestB = -1, bestD = Infinity;
      var arr = Array.from(active);
      for (var ii = 0; ii < arr.length; ii++) {
        for (var jj = ii + 1; jj < arr.length; jj++) {
          var a = arr[ii], b = arr[jj];
          // Complete linkage: max distance between any pair of original members
          var maxD = 0;
          for (var mi = 0; mi < members[a].length; mi++) {
            for (var mj = 0; mj < members[b].length; mj++) {
              var pi = members[a][mi], pj = members[b][mj];
              var key = Math.min(pi, pj) + "," + Math.max(pi, pj);
              if (pairDist[key] > maxD) maxD = pairDist[key];
            }
          }
          if (maxD < bestD) { bestD = maxD; bestA = a; bestB = b; }
        }
      }

      Z.push([bestA, bestB, bestD, members[bestA].length + members[bestB].length]);
      members[nextId] = members[bestA].concat(members[bestB]);
      delete members[bestA];
      delete members[bestB];
      active.delete(bestA);
      active.delete(bestB);
      active.add(nextId);
      nextId++;
    }
    return Z;
  }

  /**
   * Get merged colors: each merged group inherits the color of
   * its largest child community.
   */
  function getMergedColors(mergedMap) {
    // For each merged group, find the community with the largest size
    var groupBest = {};  // {groupId: {cid, size}}
    for (var cidStr in mergedMap) {
      var cid = parseInt(cidStr);
      var group = mergedMap[cidStr];
      var sz = _communitySizes[String(cid)] || 0;
      if (!(group in groupBest) || sz > groupBest[group].size) {
        groupBest[group] = { cid: cid, size: sz };
      }
    }
    var colors = {};
    for (var gid in groupBest) {
      colors[gid] = _communityColors[String(groupBest[gid].cid)] || [128, 128, 128];
    }
    return colors;
  }

  /**
   * Get merged names: each merged group inherits the name of
   * its largest child community.
   */
  function getMergedNames(mergedMap) {
    var groupBest = {};
    for (var cidStr in mergedMap) {
      var cid = parseInt(cidStr);
      var group = mergedMap[cidStr];
      var sz = _communitySizes[String(cid)] || 0;
      if (!(group in groupBest) || sz > groupBest[group].size) {
        groupBest[group] = { cid: cid, size: sz };
      }
    }
    var names = {};
    for (var gid in groupBest) {
      names[gid] = _communityNames[String(groupBest[gid].cid)] || ("Cluster " + gid);
    }
    return names;
  }

  /**
   * Get merged centroids: weighted average of child community centroids.
   */
  function getMergedCentroids(mergedMap) {
    var groupMembers = {};  // {groupId: [{cid, size, x, y, z}]}
    for (var cidStr in mergedMap) {
      var cid = parseInt(cidStr);
      var group = mergedMap[cidStr];
      var cent = _communityCentroids[String(cid)] || [0, 0, 0];
      var sz = _communitySizes[String(cid)] || 0;
      if (!groupMembers[group]) groupMembers[group] = [];
      groupMembers[group].push({ x: cent[0], y: cent[1], z: cent[2] || 0, size: sz });
    }
    var centroids = {};
    for (var gid in groupMembers) {
      var members = groupMembers[gid];
      var totalSize = 0, wx = 0, wy = 0, wz = 0;
      for (var i = 0; i < members.length; i++) {
        totalSize += members[i].size;
        wx += members[i].x * members[i].size;
        wy += members[i].y * members[i].size;
        wz += members[i].z * members[i].size;
      }
      if (totalSize > 0) {
        centroids[gid] = [wx / totalSize, wy / totalSize, wz / totalSize];
      } else {
        centroids[gid] = [0, 0, 0];
      }
    }
    return centroids;
  }

  /**
   * Apply a dendrogram cut: recolor all points and update labels.
   * height=0 → natural k (no merging), height=maxHeight → k=1.
   */
  function applyDendrogramCut(height) {
    if (!_dendro || !_pointToCommunity) return;

    var cut;
    if (height <= 0) {
      // Natural k: identity mapping
      cut = { mergedMap: {}, k: _naturalK };
      // Identity: each community maps to itself
      for (var cid in _communityNames) {
        cut.mergedMap[cid] = parseInt(cid);
      }
    } else {
      cut = cutDendrogram(height);
      if (!cut) return;
    }

    var mergedColors = getMergedColors(cut.mergedMap);
    var mergedNames = getMergedNames(cut.mergedMap);
    var mergedCentroids = getMergedCentroids(cut.mergedMap);

    // Compute merged sizes
    var mergedSizes = {};
    for (var cidStr in cut.mergedMap) {
      var group = cut.mergedMap[cidStr];
      var sz = _communitySizes[String(cidStr)] || 0;
      mergedSizes[group] = (mergedSizes[group] || 0) + sz;
    }

    // Recolor points
    for (var i = 0; i < _nPts; i++) {
      var natCid = _pointToCommunity[i];
      var mergedGroup = cut.mergedMap[String(natCid)];
      if (mergedGroup === undefined) mergedGroup = natCid;
      _currentMergedIds[i] = mergedGroup;
      var rgb = mergedColors[mergedGroup] || [128, 128, 128];
      allPoints[i].r = rgb[0];
      allPoints[i].g = rgb[1];
      allPoints[i].b = rgb[2];
      allPoints[i].cluster = mergedGroup;
    }

    // Update labels
    var newLabels = [];
    for (var gid in mergedNames) {
      var cent = mergedCentroids[gid] || [0, 0, 0];
      var rgb = mergedColors[gid] || [128, 128, 128];
      newLabels.push({
        cid: parseInt(gid),
        text: mergedNames[gid],
        x: cent[0], y: cent[1], z: cent[2] || 0,
        size: mergedSizes[gid] || 0,
        leaf_cids: [parseInt(gid)],
        color: rgb,
      });
    }
    labels = newLabels;
    labelLevels = {};
    labelLevels["dendro"] = newLabels;
    currentLevelKey = "dendro";

    // Update the zoom-driven label system to use the new cut labels
    allLabelLevels = {};
    allLabelLevels["dendro"] = newLabels;
    allLevelKeys = ["dendro"];

    // Rebuild the cluster list and layer
    rebuildClusterList();
    rebuildLayer();
  }

  // ── LSH bucket color mode state ─────────────────────────────────────────
  var _colorMode = "cluster";  // "cluster" or "bucket"
  var _lshBucketIds = dd ? dd.lshBucketIds : null;  // Int32Array(N) or null
  var _lshBucketColors = dd ? (dd.lshBucketColors || {}) : {};
  var _lshBucketNames = dd ? (dd.lshBucketNames || {}) : {};
  var _lshBucketCentroids = dd ? (dd.lshBucketCentroids || {}) : {};

  // Show color-by buttons if LSH data is available
  if (_lshBucketIds) {
    var colorSection = document.getElementById("color-mode-section");
    if (colorSection) colorSection.style.display = "";
  }

  // Backup original Z for 3D toggle (points start flat in 2D)
  var _zBackupInit = new Float32Array(_nPts);
  for (var _i = 0; _i < _nPts; _i++) _zBackupInit[_i] = _z[_i];

  // ── Reconstruct edge paths (2D bundled + 3D catenary) from Arrow IPC ──
  async function loadEdges(b64) {
    var paths = [], weights = [];
    if (b64 && b64.length > 0) {
      var tbl = tableFromIPC(await ungzip(b64toBytes(b64)));
      var pathCol = tbl.getChild("path");
      var weightCol = tbl.getChild("weight");
      for (var j = 0; j < tbl.numRows; j++) {
        var flat = pathCol.get(j).toArray();
        var path = [];
        for (var k = 0; k < flat.length; k += 3) {
          path.push([flat[k], flat[k+1], flat[k+2]]);
        }
        paths.push(path);
        weights.push(weightCol ? weightCol.get(j) : 0.5);
      }
    }
    return { paths: paths, weights: weights };
  }
  var _edges2d, _edges3d;
  if (dd && dd.edges2d) {
    _edges2d = dd.edges2d;
    _edges3d = dd.edges3d || { paths: [], weights: [] };
  } else if (dd) {
    _edges2d = await loadEdges(dd.edges2dIpcB64 || "");
    _edges3d = await loadEdges(dd.edges3dIpcB64 || "");
  } else {
    _edges2d = await loadEdges("__EDGES_2D_IPC_B64__");
    _edges3d = await loadEdges("__EDGES_3D_IPC_B64__");
  }
  var edgePaths2d = _edges2d.paths;
  var edgePaths3d = _edges3d.paths;
  var edgeWeights = _edges2d.weights.length ? _edges2d.weights : _edges3d.weights;

  var labels, labelLevels, edgePairs, tourNarration, tourCallouts;
  if (dd) {
    edgePairs = dd.edgePairs || [];
    tourNarration = dd.tourNarration || {};
    tourCallouts = dd.tourCallouts || {};
    // Build initial labels from dendrogram natural communities
    if (_dendro) {
      var initLabels = [];
      for (var cidStr in _communityNames) {
        var cent = _communityCentroids[cidStr] || [0, 0, 0];
        initLabels.push({
          cid: parseInt(cidStr),
          text: _communityNames[cidStr],
          x: cent[0], y: cent[1], z: cent[2] || 0,
          size: _communitySizes[cidStr] || 0,
          leaf_cids: [parseInt(cidStr)],
        });
      }
      labels = initLabels;
      labelLevels = { "dendro": initLabels };
    } else {
      labels = [];
      labelLevels = {};
    }
  } else {
    labels = __LABEL_JSON__;
    labelLevels = __LEVELS_JSON__;
    edgePairs = __EDGE_PAIRS_JSON__;
    tourNarration = __NARRATION_JSON__;
    tourCallouts = __CALLOUTS_JSON__;
  }

  // Build cluster centroid map from 3D catenary endpoints (exact cluster centroids).
  // For 2D mode we flatten z to 0 — x,y are the same in both modes since 2D just
  // zeroes z on the shared 3D UMAP layout.  We do NOT use hammer_bundle (2D edge)
  // endpoints because the bundling algorithm can shift them from the true centroids.
  var edgeCentroids3d = {};
  edgePairs.forEach(function(pair, idx) {
    var path3 = edgePaths3d[idx];
    if (path3 && path3.length >= 2) {
      var c1 = pair[0], c2 = pair[1];
      if (!(c1 in edgeCentroids3d)) edgeCentroids3d[c1] = path3[0];
      if (!(c2 in edgeCentroids3d)) edgeCentroids3d[c2] = path3[path3.length - 1];
    }
  });

  function getEdgeCentroid(cid) {
    var c = edgeCentroids3d[cid];
    if (!c) return null;
    return (viewMode === "2d") ? [c[0], c[1], 0] : c;
  }

  var labelsVisible = true;
  var edgesVisible = true;
  var highlightedEdgeClusters = new Set();  // cluster IDs whose edges to highlight
  var edgeFadeAlpha = 0;      // current fade state (0 = normal, 1 = fully highlighted)
  var edgeFadeTarget = 0;     // target fade state
  var edgeFadeSpeed = 0.05;   // fade speed per frame (~60fps = ~0.3s fade)
  var lastHighlightedKey = "";  // track which clusters are highlighted to detect changes

  function setEdgeHighlight(clusterIds) {
    // Build a key to detect changes
    var newKey = Array.from(clusterIds).sort().join(",");
    if (newKey !== lastHighlightedKey) {
      // Cluster set changed - reset fade to animate in
      edgeFadeAlpha = 0;
      lastHighlightedKey = newKey;
    }
    highlightedEdgeClusters = new Set(clusterIds);
    edgeFadeTarget = highlightedEdgeClusters.size > 0 ? 1 : 0;
  }

  // Cluster visibility state: null=all visible, Set=only those visible
  var hiddenClusters = new Set();
  var isolatedCluster = null;  // null=normal mode, number=isolated cluster
  var outlierClusterIds = new Set();  // populated after labelClusterIds

  // 2D/3D mode state
  var viewMode = "2d";
  var currentTheme = "dark";
  var zBackup = Array.from(_zBackupInit);

  // Global flag: pause all animations while user is dragging
  var userDragging = false;
  document.addEventListener("pointerdown", function() { userDragging = true; });
  document.addEventListener("pointerup", function() { userDragging = false; });
  document.addEventListener("pointercancel", function() { userDragging = false; });

  // ── Edge sway physics (3D gravity-based sag) ────────────────────────
  var currentSag = [0, 0, -1];   // Current sag direction [x, y, z]
  var sagVelocity = [0, 0, 0];   // Velocity for smooth transitions
  var lastSwayTime = 0;
  var swayDamping = 12;          // Damping (lower = more bounce)
  var swayStiffness = 80;        // Stiffness (higher = faster snap to target)
  var swayMass = 1.5;            // Mass (lower = snappier response)
  var swayActive = false;
  var swayThreshold = 0.002;
  var swayTime = 0;              // For per-edge oscillation phase

  // Pre-compute random offsets per edge for organic variation
  var edgeRandom = [];
  (function() {
    for (var _e = 0; _e < edgePaths3d.length; _e++) {
      edgeRandom.push({
        phase1: Math.random() * Math.PI * 2,
        phase2: Math.random() * Math.PI * 2,
        phase3: Math.random() * Math.PI * 2,
        amp: 0.4 + Math.random() * 1.2,
        freq1: 0.5 + Math.random() * 1.0,
        freq2: 0.3 + Math.random() * 0.8,
        freq3: 1.0 + Math.random() * 2.0,
        wobbleAmp: 0.1 + Math.random() * 0.4,
        delay: Math.random() * 2.0
      });
    }
  })();

  // Get target sag direction using viewport unproject
  function getTargetSag() {
    var dk = getDeck();
    if (!dk) return [0, 0, -1];

    var viewports = dk.getViewports ? dk.getViewports() : null;
    if (!viewports || !viewports.length) return [0, 0, -1];
    var vp = viewports[0];

    var w = vp.width || 800;
    var h = vp.height || 600;
    var centerX = w / 2;
    var centerY = h / 2;

    try {
      var p1 = vp.unproject([centerX, centerY]);
      var p2 = vp.unproject([centerX, centerY + 100]);
      if (!p1 || !p2) return [0, 0, -1];

      var dx = p2[0] - p1[0];
      var dy = p2[1] - p1[1];
      var dz = (p2[2] || 0) - (p1[2] || 0);
      var len = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1;
      return [dx/len, dy/len, dz/len];
    } catch(e) {
      return [0, 0, -1];
    }
  }

  function updateSway(timestamp) {
    if (!swayActive) return;
    if (viewMode === "2d") { swayActive = false; return; }

    var dk = getDeck();
    if (!dk) { requestAnimationFrame(updateSway); return; }

    // Delta time
    if (!lastSwayTime) lastSwayTime = timestamp;
    var dt = Math.min(0.05, (timestamp - lastSwayTime) / 1000);
    lastSwayTime = timestamp;
    swayTime += dt;

    // Get target sag from viewport
    var target = getTargetSag();

    // 3D spring-damper physics toward target sag direction
    var k = swayStiffness * 0.5;
    var c = swayDamping * 0.3;
    var m = swayMass;

    for (var axis = 0; axis < 3; axis++) {
      var x = currentSag[axis];
      var v = sagVelocity[axis];
      var springF = k * (target[axis] - x);
      var dampF = -c * v;
      var a = (springF + dampF) / m;
      v += a * dt;
      x += v * dt;
      sagVelocity[axis] = v;
      currentSag[axis] = x;
    }

    // Check energy to stop
    var velMag = Math.sqrt(sagVelocity[0]*sagVelocity[0] + sagVelocity[1]*sagVelocity[1] + sagVelocity[2]*sagVelocity[2]);
    if (velMag < swayThreshold && !userDragging) {
      currentSag = target.slice();
      sagVelocity = [0, 0, 0];
      swayActive = false;
      rebuildLayer();
      return;
    }

    rebuildLayer();
    requestAnimationFrame(updateSway);
  }

  function startSway() {
    if (swayActive || viewMode === "2d") return;
    swayActive = true;
    lastSwayTime = 0;
    requestAnimationFrame(updateSway);
  }

  // Start sway when user interacts
  document.addEventListener("pointerdown", function() { startSway(); });
  document.addEventListener("wheel", function() { startSway(); }, { passive: true });

  // ── Compute data extent and optimal zoom ────────────────────────────
  // Use stddev-based extent (2.5 sigma) to ignore outliers for zoom
  var xVals = allPoints.map(function(p) { return p.x; });
  var yVals = allPoints.map(function(p) { return p.y; });
  var zVals = allPoints.map(function(p) { return p.z; });
  var xMin = Math.min.apply(null, xVals);
  var xMax = Math.max.apply(null, xVals);
  var yMin = Math.min.apply(null, yVals);
  var yMax = Math.max.apply(null, yVals);
  var zMin = Math.min.apply(null, zVals);
  var zMax = Math.max.apply(null, zVals);
  var xRange = xMax - xMin || 1;
  var yRange = yMax - yMin || 1;
  var zRange = zMax - zMin || 1;

  function meanStd(vals) {
    var n = vals.length;
    var sum = 0; for (var i = 0; i < n; i++) sum += vals[i];
    var mu = sum / n;
    var ss = 0; for (var i = 0; i < n; i++) { var d = vals[i] - mu; ss += d * d; }
    return { mean: mu, std: Math.sqrt(ss / n) };
  }
  var xStat = meanStd(xVals), yStat = meanStd(yVals), zStat = meanStd(zVals);
  var SIGMA = 1.5;
  var xExtent = 2 * SIGMA * xStat.std;
  var yExtent = 2 * SIGMA * yStat.std;
  var zExtent = 2 * SIGMA * zStat.std;
  var maxExtent = Math.max(xExtent, yExtent, zExtent) || 1;

  // Compute zoom to fill viewport: use container size for accurate fit
  // OrbitView at zoom Z shows roughly (baseSize / 2^Z) world units
  var container = document.getElementById("deckgl-wrapper");
  var vpSize = container ? Math.min(container.clientWidth, container.clientHeight) : 800;
  var defaultZoom = Math.log2(vpSize * 1.3 / maxExtent);
  defaultZoom = Math.max(4, Math.min(12, defaultZoom));
  console.log("[dyfviz] stddev extent:", maxExtent.toFixed(2), "sigma:", SIGMA, "vpSize:", vpSize, "zoom:", defaultZoom.toFixed(2));

  // ── Specular sweep animation (traveling highlight for orientation) ──
  var sheenEnabled = false;
  var sheenPhase = 0;
  var sheenLastTime = 0;
  // Normalized X position for each point (0 to 1)
  var sheenXNorm = allPoints.map(function(p) { return (p.x - xMin) / xRange; });

  function updateSheen(timestamp) {
    if (!sheenEnabled) return;
    if (userDragging) { requestAnimationFrame(updateSheen); return; }

    // Time-based animation (not frame-based)
    if (!sheenLastTime) sheenLastTime = timestamp;
    var dt = (timestamp - sheenLastTime) / 1000;  // seconds
    sheenLastTime = timestamp;

    var dk = getDeck();
    if (!dk || !dk.props) return;
    var layers = dk.props.layers;
    if (!layers || !layers.length) return;

    sheenPhase += dt * 0.35;  // ~5 sec per full sweep
    if (sheenPhase > 1.3) sheenPhase = -0.3;
    var is2d = (viewMode === "2d");
    var animated = allPoints.map(function(p, i) {
      if (!isClusterVisible(p.cluster)) return null;
      // Alpha wave sweeping across
      var dist = Math.abs(sheenXNorm[i] - sheenPhase);
      // Gaussian falloff for wave
      var wave = Math.exp(-dist * dist / 0.04);
      // Alpha: base 40% + up to 60% in wave
      var alpha = Math.round(255 * (0.4 + 0.6 * wave));
      if (is2d) {
        return { x: p.x, y: p.y, z: 0, r: p.r, g: p.g, b: p.b,
                  a: alpha, title: p.title, cluster: p.cluster };
      }
      return { x: p.x, y: p.y, z: p.z, r: p.r, g: p.g, b: p.b,
                a: alpha, title: p.title, cluster: p.cluster };
    }).filter(function(p) { return p !== null; });

    var newLayers = [];
    var basePointLayer = layers[0];

    // Main points layer with animation
    var newLayer = basePointLayer.clone({ data: animated });
    newLayers.push(newLayer);

    // Add remaining layers (edges)
    for (var li = 1; li < layers.length; li++) {
      newLayers.push(layers[li]);
    }
    dk.setProps({ layers: newLayers });

    // Continue animation loop
    if (sheenEnabled) {
      requestAnimationFrame(updateSheen);
    }
  }

  function startSheen() {
    if (sheenEnabled) return;
    sheenEnabled = true;
    sheenPhase = -0.4;
    sheenLastTime = 0;
    requestAnimationFrame(updateSheen);
  }

  function stopSheen() {
    sheenEnabled = false;
    sheenLastTime = 0;
    // Restore original colors
    rebuildLayer();
  }

  // ── Auto-orbit animation ───────────────────────────────────────────
  var orbitEnabled = false;
  var orbitTimer = null;
  var orbitAngle = 30;  // starting angle (matches initial view)
  var orbitZoom = defaultZoom;  // track zoom separately
  var orbitPaused = false;  // true when user is interacting
  var orbitResumeTimer = null;

  function pauseOrbitAndResume(delayMs) {
    if (!orbitEnabled) return;
    orbitPaused = true;
    if (orbitResumeTimer) clearTimeout(orbitResumeTimer);
    orbitResumeTimer = setTimeout(function() {
      // Sync angle and zoom to current view before resuming
      try {
        var dk = getDeck();
        var vs = dk.viewManager.getViewState();
        if (vs) {
          if (typeof vs.rotationOrbit === "number") orbitAngle = vs.rotationOrbit;
          if (typeof vs.zoom === "number") orbitZoom = vs.zoom;
        }
      } catch(e) {}
      orbitPaused = false;
    }, delayMs);
  }

  // Pause orbit while user drags
  document.addEventListener("pointerdown", function(e) {
    pauseOrbitAndResume(800);
  });
  document.addEventListener("pointerup", function(e) {
    pauseOrbitAndResume(500);
  });

  // Update zoom while orbiting (don't pause, just adjust zoom level)
  document.addEventListener("wheel", function(e) {
    if (orbitEnabled) {
      // Adjust zoom based on wheel delta (negative = zoom in, positive = zoom out)
      var delta = e.deltaY > 0 ? -0.15 : 0.15;
      orbitZoom = Math.max(1, Math.min(12, orbitZoom + delta));
    }
  }, { passive: true });

  function updateOrbit() {
    if (!orbitEnabled || orbitPaused) return;
    var dk = getDeck();
    if (!dk || !dk.setProps) return;
    orbitAngle += 0.3;  // degrees per frame
    if (orbitAngle >= 360) orbitAngle -= 360;
    dk.setProps({ initialViewState: {
      target: [0, 0, 0],
      rotationX: 15,
      rotationOrbit: orbitAngle,
      zoom: orbitZoom,
      transitionDuration: 0
    } });
  }

  function stopOrbit() {
    orbitEnabled = false;
    if (orbitTimer) {
      clearInterval(orbitTimer);
      orbitTimer = null;
    }
    var toggle = document.getElementById("toggle-orbit");
    if (toggle) toggle.checked = false;
  }

  function clearTourCallouts() {
    tourActiveCallouts = [];
    tourCalloutHighlightSet = new Set();
    var container = document.getElementById("tour-callout-labels");
    container.innerHTML = "";
    container.style.display = "none";
  }

  function fadeTourCallouts() {
    // Fade out labels (CSS transition handles animation), keep DOM for positioning
    var container = document.getElementById("tour-callout-labels");
    var divs = container.querySelectorAll(".tour-callout-label");
    divs.forEach(function(d) { d.classList.remove("visible"); });
  }

  function showTourCallouts(cid) {
    clearTourCallouts();
    var callout = tourCallouts[String(cid)];
    if (!callout || !callout.indices || !callout.indices.length) return;

    var container = document.getElementById("tour-callout-labels");
    container.style.display = "block";

    // Build callout data from point positions
    callout.indices.forEach(function(ptIdx, i) {
      if (ptIdx >= 0 && ptIdx < allPoints.length) {
        var p = allPoints[ptIdx];
        tourActiveCallouts.push({
          index: ptIdx,
          label: callout.labels[i] || p.title,
          pos: [p.x, p.y, p.z]
        });
        tourCalloutHighlightSet.add(ptIdx);
      }
    });

    // Create label divs with staggered reveal
    tourActiveCallouts.forEach(function(co, i) {
      var div = document.createElement("div");
      div.className = "tour-callout-label" + (i < 3 ? " core" : " outlier");
      div.textContent = co.label;
      container.appendChild(div);
      // Stagger appearance (phase runner calls showTourCallouts at the right time)
      setTimeout(function() { div.classList.add("visible"); }, 300 * (i + 1));
    });

    // Rebuild layer to apply callout highlighting
    rebuildLayer();
  }

  function stopTourMode() {
    if (!tourRunning) return;
    tourRunning = false;
    if (tourTimerId) { clearTimeout(tourTimerId); tourTimerId = null; }
    stopTourProgress();
    stopNarration();
    tourCentroid = null;
    tourConnected = [];
    tourPhase = "";
    clearTourCallouts();
    clearTourCircles();
    tourRevealedCids.clear();
    setEdgeHighlight([]);
    document.getElementById("tour-btn").textContent = "▶ Start Tour";
    document.getElementById("tour-label").style.display = "none";
    document.getElementById("tour-edge-labels").style.display = "none";
    document.getElementById("camera-debug").style.display = "none";
    // Restore panel after tour
    if (window.panelHidden) window.togglePanel();
    var tourListEl = document.getElementById("tour-list");
    if (tourListEl) {
      var items = tourListEl.querySelectorAll(".tour-item");
      items.forEach(function(el) { el.classList.remove("active"); });
    }
  }

  function stopAmbientMode() {
    if (!ambientRunning) return;
    ambientRunning = false;
    setEdgeHighlight([]);
  }

  function stopAllAnimations() {
    stopOrbit();
    stopTourMode();
    stopAmbientMode();
    rebuildLayer();
  }

  function startOrbit() {
    if (orbitTimer) return;
    // Cancel other animations first
    stopTourMode();
    stopAmbientMode();
    rebuildLayer();
    orbitEnabled = true;
    orbitTimer = setInterval(updateOrbit, 50);  // 20 FPS
  }

  // ── Pre-rendered audio for tour narration ─────────────────────────
  var tourAudio;
  if (dd) { tourAudio = dd.tourAudio || {}; }
  else { tourAudio = __AUDIO_JSON__; }
  var audioContext = null;
  var currentAudioSource = null;
  var currentAudioDuration = 10000;  // duration of current clip in ms

  // Eagerly unlock AudioContext on first user gesture (needed for autoplay policy)
  document.addEventListener("click", function initAudio() {
    if (!audioContext) audioContext = new AudioContext();
    if (audioContext.state === "suspended") audioContext.resume();
    document.removeEventListener("click", initAudio);
  }, { once: true });

  function playClusterAudio(cid) {
    var entry = tourAudio[String(cid)];
    if (!entry || !entry.data) return 0;
    // Stop any currently playing audio
    if (currentAudioSource) {
      try { currentAudioSource.stop(); } catch(e) {}
      currentAudioSource = null;
    }
    // Decode base64 to ArrayBuffer
    var binary = atob(entry.data);
    var len = binary.length;
    var bytes = new Uint8Array(len);
    for (var i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
    // Play audio
    if (!audioContext) audioContext = new AudioContext();
    if (audioContext.state === "suspended") audioContext.resume();
    audioContext.decodeAudioData(bytes.buffer).then(function(buffer) {
      var source = audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(audioContext.destination);
      source.start(0);
      currentAudioSource = source;
    }).catch(function(e) {
      console.error("[Audio] Playback failed:", e);
    });
    return entry.duration || 10000;
  }

  function getAudioDuration(cid) {
    var entry = tourAudio[String(cid)];
    return (entry && entry.duration) ? entry.duration : 10000;
  }

  function stopNarration() {
    if (currentAudioSource) {
      try { currentAudioSource.stop(); } catch(e) {}
      currentAudioSource = null;
    }
  }

  // ── Tour animation utilities ───────────────────────────────────────
  function easeInOutQuad(t) { return t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t + 2, 2)/2; }
  function easeOutQuad(t) { return 1 - Math.pow(1 - t, 2); }
  function lerpZoom(a, b, t) { return Math.log2(Math.pow(2, a) + (Math.pow(2, b) - Math.pow(2, a)) * t); }
  function lerpScalar(a, b, t) { return a + (b - a) * t; }
  function lerpArray(a, b, t) { return a.map(function(v, i) { return v + (b[i] - v) * t; }); }
  function makeCamState(target, orbit, pitch, zoom) {
    return { target: target, orbit: orbit, pitch: pitch, zoom: zoom };
  }
  function lerpCamState(a, b, t) {
    return makeCamState(
      lerpArray(a.target, b.target, t),
      lerpScalar(a.orbit, b.orbit, t),
      lerpScalar(a.pitch, b.pitch, t),
      lerpZoom(a.zoom, b.zoom, t)
    );
  }
  function applyCamState(dk, state) {
    dk.setProps({ initialViewState: {
      target: state.target,
      rotationOrbit: state.orbit,
      rotationX: state.pitch,
      zoom: state.zoom,
      transitionDuration: 0
    } });
  }

  // Sequential phase runner: animation phases (RAF-driven) and hold phases (setTimeout)
  // Each phase: {name, duration, ease?, from?, to?, onStart?, onEnd?}
  //   - If from/to provided: interpolates camera state via lerpCamState
  //   - If neither from/to: hold phase (just waits duration ms)
  function runPhases(phases, gen, onAllComplete) {
    var idx = 0;
    function next() {
      if (!tourRunning || gen !== tourGeneration) return;
      if (idx >= phases.length) { if (onAllComplete) onAllComplete(); return; }
      var phase = phases[idx++];
      tourPhase = phase.name || "";
      if (phase.onStart) phase.onStart();
      var dur = phase.duration || 0;
      var easeFn = phase.ease || easeInOutQuad;
      if (phase.from && phase.to) {
        // RAF-driven camera interpolation
        var dk = getDeck();
        var startTime = performance.now();
        function tick() {
          if (!tourRunning || gen !== tourGeneration) return;
          var t = Math.min(1, (performance.now() - startTime) / dur);
          var et = easeFn(t);
          applyCamState(dk, lerpCamState(phase.from, phase.to, et));
          if (phase.onTick) phase.onTick(et);
          if (t < 1) { requestAnimationFrame(tick); }
          else { if (phase.onEnd) phase.onEnd(); next(); }
        }
        requestAnimationFrame(tick);
      } else {
        // Hold phase — just wait
        setTimeout(function() {
          if (!tourRunning || gen !== tourGeneration) return;
          if (phase.onEnd) phase.onEnd();
          next();
        }, dur);
      }
    }
    next();
  }

  // ── Cluster tour ───────────────────────────────────────────────────
  var tourRunning = false;
  var tourGeneration = 0;   // incremented on each tour start; stale closures bail out
  var tourIndex = 0;
  var tourTimerId = null;   // setTimeout ID for next visit
  var tourProgressStart = 0; // timestamp when current stop started
  var tourProgressRAF = null; // requestAnimationFrame ID for progress bar
  var tourStopDuration = 10000; // duration of current stop in ms (from audio)
  var tourCentroid = null;  // current centroid for label positioning
  var tourConnected = [];   // connected cluster objects for edge labels
  var tourPhase = "";       // current phase name (e.g. "panToWide", "holdClose")
  var tourRevealedCids = new Set();  // cluster IDs revealed so far during tour
  var tourActiveCallouts = [];  // active callout point objects
  var tourCalloutHighlightSet = new Set();  // indices of points to highlight during callouts

  // Build label-to-clusterID mapping (same logic as rowClusterIds)
  var labelClusterIds = (function() {
    var uniqueCids = [];
    var cset = {};
    allPoints.forEach(function(p) {
      if (!(p.cluster in cset)) { cset[p.cluster] = true; uniqueCids.push(p.cluster); }
    });
    uniqueCids.sort(function(a,b) { return a - b; });
    return labels.map(function(c, i) {
      return i < uniqueCids.length ? uniqueCids[i] : i;
    });
  })();

  // Detect spatially distant outlier clusters via MAD (median absolute deviation)
  // MAD is more robust than IQR for small n and skewed distributions
  (function() {
    var centroidDists = [];
    labels.forEach(function(c) {
      var d = Math.sqrt(c.x * c.x + c.y * c.y + (c.z || 0) * (c.z || 0));
      centroidDists.push(d);
    });
    if (centroidDists.length < 4) return;
    var sorted = centroidDists.slice().sort(function(a, b) { return a - b; });
    var median = sorted[Math.floor(sorted.length / 2)];
    // Compute MAD (median absolute deviation)
    var absDevs = centroidDists.map(function(d) { return Math.abs(d - median); });
    absDevs.sort(function(a, b) { return a - b; });
    var mad = absDevs[Math.floor(absDevs.length / 2)];
    // Modified z-score: flag clusters with z > 2.0 (standard MAD outlier threshold)
    // The 0.6745 factor normalizes MAD to match std for normal distributions
    var madThreshold = 1.5;
    if (mad < 0.001) return;  // all clusters equidistant
    // Tiny-cluster threshold: clusters with < 0.5% of total points
    var totalPts = allPoints.length;
    var tinyThreshold = totalPts * 0.005;
    for (var i = 0; i < centroidDists.length; i++) {
      var z = 0.6745 * (centroidDists[i] - median) / mad;
      var isTiny = labels[i].size < tinyThreshold && centroidDists[i] > median;
      if (z > madThreshold || isTiny) {
        var cid = labels[i].cid !== undefined ? labels[i].cid : (i < labelClusterIds.length ? labelClusterIds[i] : i);
        outlierClusterIds.add(cid);
      }
    }
    // Show outliers by default (toggle checkbox hides them)
    if (outlierClusterIds.size > 0) {
      console.log("[outliers] Detected " + outlierClusterIds.size + " outlier clusters (MAD z>" + madThreshold + " or size<" + Math.round(tinyThreshold) + " & dist>median, median=" + median.toFixed(2) + ", MAD=" + mad.toFixed(2) + "):");
      outlierClusterIds.forEach(function(cid) {
        var lbl = labels.find(function(c) { return c.cid === cid; });
        if (lbl) console.log("  [" + cid + "] " + lbl.text + " (dist=" + Math.sqrt(lbl.x*lbl.x + lbl.y*lbl.y + (lbl.z||0)*(lbl.z||0)).toFixed(2) + ")");
      });
    }
  })();

  // Debug overlay for camera state during tour
  var debugEl = document.getElementById("camera-debug");
  function updateCameraDebug(centroid, targetOrbit, targetPitch, curOrbit, curPitch, curZoom, phase) {
    if (!debugEl) return;
    // Compute centroid angle (from +X axis, standard math convention)
    var centroidAngle = Math.atan2(centroid[1], centroid[0]) * 180 / Math.PI;
    // Compute alignment error: how far off is the camera from pointing at centroid?
    // If orbit=0 means camera at +Y, then camera direction at orbit θ might be:
    // Option A: camera at angle (90 - θ) from +X  => alignment = centroidAngle - (90 - curOrbit)
    // Option B: camera at angle θ from +Y => θ - 90 from +X => alignment = centroidAngle - (curOrbit - 90)
    // Option C: camera at angle -θ from +Y
    // Let's compute several and show which is closest to 0 or 180
    var errA = centroidAngle - (90 - curOrbit);
    var errB = centroidAngle - (curOrbit - 90);
    var errC = centroidAngle - (-curOrbit + 90);
    var errD = centroidAngle - curOrbit;
    // Normalize to -180..180
    function norm(a) { while(a>180) a-=360; while(a<-180) a+=360; return a; }
    errA = norm(errA); errB = norm(errB); errC = norm(errC); errD = norm(errD);
    var lines = [
      "Centroid: [" + centroid[0].toFixed(2) + ", " + centroid[1].toFixed(2) + "] angle=" + centroidAngle.toFixed(1) + "°",
      "Orbit:    target=" + targetOrbit.toFixed(1) + "° cur=" + curOrbit.toFixed(1) + "°",
      "ErrA(90-θ):" + errA.toFixed(1) + "° ErrB(θ-90):" + errB.toFixed(1) + "°",
      "ErrC(-θ+90):" + errC.toFixed(1) + "° ErrD(θ):" + errD.toFixed(1) + "°",
      "Phase: " + phase
    ];
    debugEl.textContent = lines.join("\\n");
  }

  function startTourProgress() {
    var tourListEl = document.getElementById("tour-list");
    if (!tourListEl) return;
    var activeItem = tourListEl.querySelector(".tour-item.active");
    if (!activeItem) return;
    activeItem.style.setProperty("--progress", "0%");
    tourProgressStart = performance.now();
    if (tourProgressRAF) cancelAnimationFrame(tourProgressRAF);
    function animateProgress() {
      if (!tourRunning) return;
      var item = document.querySelector(".tour-item.active");
      if (!item) return;
      var elapsed = performance.now() - tourProgressStart;
      var pct = Math.min(100, (elapsed / tourStopDuration) * 100);
      item.style.setProperty("--progress", pct + "%");
      if (pct < 100) {
        tourProgressRAF = requestAnimationFrame(animateProgress);
      }
    }
    tourProgressRAF = requestAnimationFrame(animateProgress);
  }

  function stopTourProgress() {
    if (tourProgressRAF) { cancelAnimationFrame(tourProgressRAF); tourProgressRAF = null; }
    var items = document.querySelectorAll(".tour-item");
    items.forEach(function(el) { el.style.setProperty("--progress", "0%"); });
  }

  function runTour() {
    var tourListEl = document.getElementById("tour-list");
    if (tourRunning) {
      // Stop tour — bump generation so all stale animation closures bail out
      tourRunning = false;
      tourGeneration++;
      if (tourTimerId) { clearTimeout(tourTimerId); tourTimerId = null; }
      stopTourProgress();
      stopNarration();
      tourCentroid = null;
      tourConnected = [];
      clearTourCallouts();
      setEdgeHighlight([]);
      rebuildLayer();
      document.getElementById("tour-btn").textContent = "▶ Start Tour";
      document.getElementById("tour-label").style.display = "none";
      document.getElementById("tour-edge-labels").style.display = "none";
      document.getElementById("tour-edge-labels").innerHTML = "";
      document.getElementById("camera-debug").style.display = "none";
      // Restore panel after tour
      if (window.panelHidden) window.togglePanel();
      // Clear tour list highlights
      if (tourListEl) {
        var items = tourListEl.querySelectorAll(".tour-item");
        items.forEach(function(el) { el.classList.remove("active"); });
      }
      return;
    }
    if (labels.length === 0) return;

    // Cancel other animations first
    stopOrbit();
    stopAmbientMode();

    tourRunning = true;
    tourGeneration++;
    document.getElementById("tour-btn").textContent = "◼ Stop Tour";
    // Clear all highlights and annotations at tour start
    annotations.length = 0;
    setEdgeHighlight([]);
    rebuildLayer();
    // Hide panel during tour
    if (!window.panelHidden) window.togglePanel();
    // Debug panel disabled: document.getElementById("camera-debug").style.display = "block";

    // Sort labels by size (largest first), keeping track of cluster IDs
    // Filter out hidden clusters (e.g. spatial outliers) from tour
    var sortedWithIds = labels.map(function(c, i) {
      return { label: c, cid: c.cid !== undefined ? c.cid : labelClusterIds[i] };
    }).filter(function(item) {
      return !hiddenClusters.has(item.cid);
    }).sort(function(a, b) {
      return (a.label.x || 0) - (b.label.x || 0);
    });

    // Check if intro/outro audio exists
    var hasIntro = tourAudio["intro"] && tourAudio["intro"].data;
    var hasOutro = tourAudio["outro"] && tourAudio["outro"].data;

    // Start at -1 for intro if available, otherwise start at 0
    tourIndex = hasIntro ? -1 : 0;

    // Populate tour list with intro, clusters, and outro
    if (tourListEl) {
      tourListEl.innerHTML = "";
      // Add intro item
      if (hasIntro) {
        var introDiv = document.createElement("div");
        introDiv.className = "tour-item";
        introDiv.setAttribute("data-idx", "-1");
        introDiv.textContent = "▶ Introduction";
        introDiv.style.padding = "2px 4px";
        introDiv.style.borderRadius = "3px";
        introDiv.style.cursor = "pointer";
        introDiv.style.fontStyle = "italic";
        introDiv.onclick = function() {
          if (tourTimerId) { clearTimeout(tourTimerId); tourTimerId = null; }
          tourGeneration++;
          clearTourCallouts();
          tourIndex = -1;
          visitNext();
        };
        tourListEl.appendChild(introDiv);
      }
      // Add cluster items
      sortedWithIds.forEach(function(item, idx) {
        var div = document.createElement("div");
        div.className = "tour-item";
        div.setAttribute("data-idx", idx);
        div.textContent = (idx + 1) + ". " + (item.label.text || "Cluster " + item.cid);
        div.style.padding = "2px 4px";
        div.style.borderRadius = "3px";
        div.style.cursor = "pointer";
        div.onclick = function() {
          if (tourTimerId) { clearTimeout(tourTimerId); tourTimerId = null; }
          tourGeneration++;
          clearTourCallouts();
          tourIndex = idx;
          visitNext();
        };
        tourListEl.appendChild(div);
      });
      // Add outro item
      if (hasOutro) {
        var outroDiv = document.createElement("div");
        outroDiv.className = "tour-item";
        outroDiv.setAttribute("data-idx", "outro");
        outroDiv.textContent = "◀ Conclusion";
        outroDiv.style.padding = "2px 4px";
        outroDiv.style.borderRadius = "3px";
        outroDiv.style.cursor = "pointer";
        outroDiv.style.fontStyle = "italic";
        outroDiv.onclick = function() {
          if (tourTimerId) { clearTimeout(tourTimerId); tourTimerId = null; }
          tourGeneration++;
          clearTourCallouts();
          tourIndex = sortedWithIds.length;  // outro index
          visitNext();
        };
        tourListEl.appendChild(outroDiv);
      }
    }

    var tourLabelEl = document.getElementById("tour-label");

    function visitNext() {
      var gen = tourGeneration;  // capture so stale closures bail out
      // Highlight current item in tour list based on data-idx attribute
      if (tourListEl) {
        var items = tourListEl.querySelectorAll(".tour-item");
        var targetIdx = (tourIndex === sortedWithIds.length) ? "outro" : String(tourIndex);
        items.forEach(function(el) {
          if (el.getAttribute("data-idx") === targetIdx) {
            el.classList.add("active");
            el.scrollIntoView({ block: "nearest" });
          } else {
            el.classList.remove("active");
          }
        });
      }

      // Handle intro (tourIndex == -1)
      if (tourIndex === -1) {
        if (!tourRunning || gen !== tourGeneration) return;
        tourLabelEl.textContent = dd ? (dd.tourTitle || "") : __TOUR_TITLE_JSON__;
        tourLabelEl.classList.add("hero");
        tourLabelEl.style.display = "block";
        document.getElementById("tour-edge-labels").style.display = "none";

        // Play intro audio
        var introDuration = 5000;
        if (tourAudio["intro"] && tourAudio["intro"].data) {
          introDuration = playClusterAudio("intro") + 2000;
        }
        tourStopDuration = introDuration;
        startTourProgress();

        // Animate to overview position
        var dk = getDeck();
        var is2d = (viewMode === "2d");
        if (dk && dk.setProps) {
          dk.setProps({ initialViewState: {
            target: [0, 0, 0],
            rotationX: is2d ? 90 : 25,
            rotationOrbit: 0,
            zoom: defaultZoom,
            transitionDuration: 2000,
            transitionInterpolator: new deck.LinearInterpolator(['target', 'zoom', 'rotationOrbit', 'rotationX'])
          } });
        }

        tourIndex++;
        tourTimerId = setTimeout(visitNext, tourStopDuration);
        return;
      }

      // Handle outro (tourIndex == sortedWithIds.length, exactly once)
      if (tourIndex === sortedWithIds.length && hasOutro) {
        if (!tourRunning || gen !== tourGeneration) return;

        tourLabelEl.textContent = "Thank You";
        tourLabelEl.classList.add("hero");
        tourLabelEl.style.display = "block";
        setEdgeHighlight([]);
        rebuildLayer();
        document.getElementById("tour-edge-labels").style.display = "none";

        // Play outro audio
        var outroDuration = playClusterAudio("outro") + 2000;
        tourStopDuration = outroDuration;
        startTourProgress();

        // Animate to wide view
        var dk = getDeck();
        var is2d = (viewMode === "2d");
        if (dk && dk.setProps) {
          dk.setProps({ initialViewState: {
            target: [0, 0, 0],
            rotationX: is2d ? 90 : 15,
            rotationOrbit: is2d ? 0 : 180,
            zoom: defaultZoom - 0.3,
            transitionDuration: 2000,
            transitionInterpolator: new deck.LinearInterpolator(['target', 'zoom', 'rotationOrbit', 'rotationX'])
          } });
        }

        // Schedule tour end after outro
        tourTimerId = setTimeout(function() {
          tourIndex++;  // Move past outro (to sortedWithIds.length + 1)
          visitNext();  // This will hit the tour complete logic
        }, outroDuration);
        return;
      }

      // Tour complete (no outro, or after outro played)
      if (tourIndex >= sortedWithIds.length) {
        tourRunning = false;
        if (tourTimerId) { clearTimeout(tourTimerId); tourTimerId = null; }
        stopTourProgress();
        stopNarration();
        tourCentroid = null;
        tourConnected = [];
        clearTourCallouts();
        setEdgeHighlight([]);
        rebuildLayer();
        document.getElementById("tour-btn").textContent = "▶ Start Tour";
        tourLabelEl.style.display = "none";
        document.getElementById("tour-edge-labels").style.display = "none";
        document.getElementById("camera-debug").style.display = "none";
        document.getElementById("tour-edge-labels").innerHTML = "";
        // Panel stays hidden after tour (user can toggle manually)
        // Clear tour list highlights
        if (tourListEl) {
          var items = tourListEl.querySelectorAll(".tour-item");
          items.forEach(function(el) { el.classList.remove("active"); });
        }
        // Return to default view
        var dk = getDeck();
        var is2d = (viewMode === "2d");
        if (dk && dk.setProps) {
          dk.setProps({ initialViewState: {
            target: [0, 0, 0],
            rotationX: is2d ? 90 : 15,
            rotationOrbit: is2d ? 0 : 30,
            zoom: defaultZoom,
            transitionDuration: 1000,
            transitionInterpolator: new deck.LinearInterpolator(['target', 'zoom', 'rotationOrbit'])
          } });
        }
        return;
      }

      var item = sortedWithIds[tourIndex];
      var cluster = item.label;
      var cid = item.cid;

      // Hide the big cluster label during visits — the ring identifies the cluster
      tourLabelEl.classList.remove("hero");
      tourLabelEl.style.display = "none";

      // Get audio duration (playback deferred to holdClose phase after zoom settles)
      tourStopDuration = getAudioDuration(cid) + 2000 + 4000;  // panZoomIn + audio + 4s buffer

      // No edge highlighting during tour — zoom and ring identify the active cluster
      setEdgeHighlight([]);
      rebuildLayer();

      // Start progress bar for this stop
      startTourProgress();

      var dk = getDeck();
      if (dk && dk.setProps) {
        // Use edge centroid if available (matches where edges connect)
        // Fall back to computing from points if cluster has no edges
        var centroid = getEdgeCentroid(cid);
        if (!centroid) {
          var clusterPts = allPoints.filter(function(p) { return p.cluster === cid; });
          centroid = [0, 0, 0];
          if (clusterPts.length > 0) {
            clusterPts.forEach(function(p) {
              centroid[0] += p.x; centroid[1] += p.y; centroid[2] += p.z;
            });
            centroid[0] /= clusterPts.length;
            centroid[1] /= clusterPts.length;
            centroid[2] /= clusterPts.length;
          }
        }

        // Compute bounding box for zoom calculation
        var clusterPts = allPoints.filter(function(p) { return p.cluster === cid; });
        var minX = Infinity, maxX = -Infinity;
        var minY = Infinity, maxY = -Infinity;
        var minZ = Infinity, maxZ = -Infinity;
        clusterPts.forEach(function(p) {
          if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
          if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
          if (p.z < minZ) minZ = p.z; if (p.z > maxZ) maxZ = p.z;
        });

        // Store centroid for label positioning
        tourCentroid = centroid;
        tourConnected = [];
        document.getElementById("tour-edge-labels").style.display = "none";

        // Compute zoom level for the cluster
        var dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
        var is2d = (viewMode === "2d");
        var clusterExtent = is2d ? Math.max(dx, dy) || 0.5 : Math.max(dx, dy, dz) || 0.5;
        var container = document.getElementById("deckgl-wrapper");
        var vpSize = container ? Math.min(container.clientWidth, container.clientHeight) : 800;

        // Zoom to see cluster detail
        var closeZoom = Math.log2(vpSize * 0.6 / clusterExtent);
        closeZoom = Math.max(defaultZoom + 0.5, Math.min(14, closeZoom));

        // Get current state
        var curState = dk.viewManager ? dk.viewManager.getViewState() : {};
        var startZoom = curState.zoom || defaultZoom;

        // === Build camera states for declarative phase runner ===
        var phases;

        if (is2d) {
          // 2D: pan directly to cluster and zoom in — stay close
          var startTarget = curState.target || [0, 0, 0];
          var targetXY = [centroid[0], centroid[1], 0];

          var startState  = makeCamState(startTarget, 0, 90, startZoom);
          var closeState  = makeCamState(targetXY, 0, 90, closeZoom);

          phases = [
            { name: "panZoomIn",  duration: 2000, from: startState, to: closeState,
               onStart: function() {
                 addClusterCircle(cid);
               } },
            { name: "holdClose",  duration: tourStopDuration - 2500,
               onStart: function() { playClusterAudio(cid); showTourCallouts(cid); } },
            { name: "settle",     duration: 500,
               onStart: function() { fadeTourCallouts(); clearTourCircles(); rebuildLayer(); } }
          ];

        } else {
          // 3D: vary orbit/pitch, fix target=[0,0,0]
          var centroidAngle = Math.atan2(centroid[1], centroid[0]) * 180 / Math.PI;
          var targetOrbit = -centroidAngle - 90;
          var xyDist = Math.sqrt(centroid[0]*centroid[0] + centroid[1]*centroid[1]);
          var targetPitch = Math.atan2(centroid[2], xyDist) * 180 / Math.PI;
          targetPitch = Math.max(-60, Math.min(60, targetPitch));

          var startOrbit = curState.rotationOrbit || 0;
          var startPitch = curState.rotationX || 20;

          // Handle orbit wraparound (take shortest rotation path)
          var orbitDiff = targetOrbit - startOrbit;
          while (orbitDiff > 180) orbitDiff -= 360;
          while (orbitDiff < -180) orbitDiff += 360;
          var adjustedOrbit = startOrbit + orbitDiff;

          var origin = [0, 0, 0];

          var startState   = makeCamState(origin, startOrbit, startPitch, startZoom);
          var closeState   = makeCamState(origin, adjustedOrbit, targetPitch, closeZoom);

          phases = [
            { name: "panZoomIn", duration: 2000, from: startState, to: closeState,
               onStart: function() {
                 addClusterCircle(cid);
               } },
            { name: "holdClose",    duration: tourStopDuration - 2500,
               onStart: function() { playClusterAudio(cid); showTourCallouts(cid); } },
            { name: "settle",       duration: 500,
               onStart: function() { fadeTourCallouts(); clearTourCircles(); rebuildLayer(); } }
          ];
        }

        // Compute total animation duration from phase list
        var totalAnimDuration = 0;
        phases.forEach(function(p) { totalAnimDuration += p.duration || 0; });
        if (tourStopDuration < totalAnimDuration + 300) tourStopDuration = totalAnimDuration + 300;

        runPhases(phases, gen);
      }

      tourIndex++;
      tourTimerId = setTimeout(visitNext, tourStopDuration);
    }

    visitNext();
  }

  // ── Ambient orbit mode with flickering ─────────────────────────────
  var ambientRunning = false;
  var ambientOrbit = 0;
  var flickerData = null;  // Stores per-point flicker phase

  function runAmbient() {
    if (ambientRunning) {
      ambientRunning = false;
      setEdgeHighlight([]);
      rebuildLayer();
      return;
    }

    // Cancel other animations first
    stopOrbit();
    stopTourMode();

    ambientRunning = true;

    // Initialize candlelight flicker: multiple frequencies + occasional flares
    flickerData = allPoints.map(function() {
      return {
        phase1: Math.random() * Math.PI * 2,
        phase2: Math.random() * Math.PI * 2,
        phase3: Math.random() * Math.PI * 2,
        speed1: 1.5 + Math.random() * 2.0,   // slow base
        speed2: 4.0 + Math.random() * 3.0,   // medium wobble
        speed3: 8.0 + Math.random() * 6.0,   // fast flutter
        flareTimer: Math.random() * 3.0,     // seconds until next flare
        flareBrightness: 0
      };
    });

    var dk = getDeck();
    if (!dk) return;

    var lastTime = performance.now();
    var orbitSpeed = 2;  // degrees per second (3 min per rotation, planetary)

    function animateAmbient() {
      if (!ambientRunning) return;
      if (userDragging) { requestAnimationFrame(animateAmbient); return; }

      var now = performance.now();
      var dt = (now - lastTime) / 1000;
      lastTime = now;

      // Update orbit angle
      ambientOrbit = (ambientOrbit + orbitSpeed * dt) % 360;

      // Simple flicker - no depth effects
      var flickerTime = now / 1000;
      var flickered = allPoints.map(function(p, i) {
        var f = flickerData[i];

        // Gentle flicker
        var base = 0.75 + 0.15 * Math.sin(flickerTime * f.speed1 * 0.5 + f.phase1);
        var shimmer = 0.1 * Math.sin(flickerTime * f.speed2 * 0.3 + f.phase2);
        var flicker = Math.min(1.0, base + shimmer);

        var alpha = Math.round(p.a * flicker);
        return {
          x: p.x, y: p.y, z: p.z,
          r: p.r, g: p.g, b: p.b, a: alpha,
          title: p.title, cluster: p.cluster
        };
      }).filter(function(p) { return isClusterVisible(p.cluster); });

      // Update deck view
      dk.setProps({
        initialViewState: {
          target: [0, 0, 0],
          rotationOrbit: ambientOrbit,
          rotationX: 15,
          zoom: defaultZoom,  // Show all points
          transitionDuration: 0
        }
      });

      // Update point layer with flickered data
      var layers = dk.props.layers;
      if (layers && layers.length > 0) {
        var newLayer = layers[0].clone({ data: flickered });
        var newLayers = [newLayer];
        if (layers.length > 1) {
          newLayers.push(layers[layers.length - 1]);
        }
        dk.setProps({ layers: newLayers });
      }

      requestAnimationFrame(animateAmbient);
    }

    animateAmbient();
  }

  // ── Highlighter annotations ────────────────────────────────────────
  var annotations = [];
  // Each: { type:"circle"|"path", points:[[x,y,z],...], color:str, width:num }

  function fitEllipse(pts, pad) {
    // Fit a bounding ellipse around 3D points (using x,y) and return
    // smooth sample points in 3D. Pad expands the radii.
    var cx = 0, cy = 0, cz = 0;
    for (var i = 0; i < pts.length; i++) {
      cx += pts[i][0]; cy += pts[i][1]; cz += (pts[i][2] || 0);
    }
    cx /= pts.length; cy /= pts.length; cz /= pts.length;

    // Covariance matrix for PCA-aligned ellipse
    var cxx = 0, cyy = 0, cxy = 0;
    for (var i = 0; i < pts.length; i++) {
      var dx = pts[i][0] - cx, dy = pts[i][1] - cy;
      cxx += dx * dx; cyy += dy * dy; cxy += dx * dy;
    }
    cxx /= pts.length; cyy /= pts.length; cxy /= pts.length;

    // Eigenvectors of 2x2 covariance → principal axes
    var trace = cxx + cyy;
    var det = cxx * cyy - cxy * cxy;
    var disc = Math.sqrt(Math.max(0, trace * trace / 4 - det));
    var lam1 = trace / 2 + disc;
    var lam2 = trace / 2 - disc;
    var angle = Math.atan2(cxy, lam1 - cyy);

    // Project points onto principal axes to find max radii
    var cosA = Math.cos(angle), sinA = Math.sin(angle);
    var maxR1 = 0, maxR2 = 0;
    for (var i = 0; i < pts.length; i++) {
      var dx = pts[i][0] - cx, dy = pts[i][1] - cy;
      var r1 = Math.abs(dx * cosA + dy * sinA);
      var r2 = Math.abs(-dx * sinA + dy * cosA);
      if (r1 > maxR1) maxR1 = r1;
      if (r2 > maxR2) maxR2 = r2;
    }
    maxR1 += pad; maxR2 += pad;
    // Ensure minimum circularity
    var minR = Math.max(maxR1, maxR2) * 0.4;
    if (maxR1 < minR) maxR1 = minR;
    if (maxR2 < minR) maxR2 = minR;

    // Sample points around the ellipse
    var nSamples = 48;
    var result = [];
    for (var i = 0; i < nSamples; i++) {
      var t = (i / nSamples) * Math.PI * 2;
      var ex = maxR1 * Math.cos(t);
      var ey = maxR2 * Math.sin(t);
      // Rotate back to data space
      var px = cx + ex * cosA - ey * sinA;
      var py = cy + ex * sinA + ey * cosA;
      result.push([px, py, cz]);
    }
    return result;
  }

  // Add a highlighter circle annotation around a cluster
  function addClusterCircle(clusterId, color, width) {
    var cPts = [];
    for (var ci = 0; ci < allPoints.length; ci++) {
      if (allPoints[ci].cluster === clusterId) {
        cPts.push([allPoints[ci].x, allPoints[ci].y, allPoints[ci].z]);
      }
    }
    if (cPts.length < 3) return;
    var xMn=1e9,xMx=-1e9,yMn=1e9,yMx=-1e9;
    for (var ci2=0;ci2<allPoints.length;ci2++) {
      if (allPoints[ci2].x<xMn) xMn=allPoints[ci2].x;
      if (allPoints[ci2].x>xMx) xMx=allPoints[ci2].x;
      if (allPoints[ci2].y<yMn) yMn=allPoints[ci2].y;
      if (allPoints[ci2].y>yMx) yMx=allPoints[ci2].y;
    }
    var extent = Math.max(xMx-xMn, yMx-yMn) || 1;
    var ellipsePts = fitEllipse(cPts, extent * 0.03);
    annotations.push({
      type: "circle",
      points: ellipsePts,
      _seed: Math.floor(Math.random() * 99999),
      _tourCircle: true,
      color: color || "rgba(255,230,0,0.35)",
      width: width || 18
    });
  }

  // Remove only tour-generated circle annotations
  function clearTourCircles() {
    annotations = annotations.filter(function(a) { return !a._tourCircle; });
  }

  function drawAnnotations(vp) {
    var canvas = document.getElementById("hl-canvas");
    if (!canvas || !vp) return;
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (annotations.length === 0) return;

    for (var i = 0; i < annotations.length; i++) {
      var ann = annotations[i];
      var screenPts = [];
      for (var j = 0; j < ann.points.length; j++) {
        try {
          var sp = vp.project(ann.points[j]);
          screenPts.push([sp[0], sp[1]]);
        } catch(e) {}
      }
      if (screenPts.length < 2) continue;

      var w = ann.width || 18;
      var color = ann.color || "rgba(255,230,0,0.35)";

      if (ann.type === "circle" && screenPts.length >= 4) {
        // Highlighter-style filled mask: build a ribbon (offset polygon)
        // along a smooth Catmull-Rom spline with taper at start/end
        var seed = ann._seed || 42;
        function wobbleRng() {
          seed = (seed * 1103515245 + 12345) & 0x7fffffff;
          return (seed / 0x7fffffff) * 2 - 1;  // -1 to 1
        }
        var n = screenPts.length;

        function catmullRom(p0, p1, p2, p3, t) {
          var t2 = t * t, t3 = t2 * t;
          return [
            0.5 * ((2*p1[0]) + (-p0[0]+p2[0])*t + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3),
            0.5 * ((2*p1[1]) + (-p0[1]+p2[1])*t + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
          ];
        }

        // Smooth closed Catmull-Rom spline, stroked as a yellow hoop
        var stepsPerSeg = 8;
        ctx.beginPath();
        var first = true;
        for (var si = 0; si < n; si++) {
          var p0 = screenPts[(si - 1 + n) % n];
          var p1 = screenPts[si];
          var p2 = screenPts[(si + 1) % n];
          var p3 = screenPts[(si + 2) % n];
          for (var st = 0; st < stepsPerSeg; st++) {
            var pt = catmullRom(p0, p1, p2, p3, st / stepsPerSeg);
            if (first) { ctx.moveTo(pt[0], pt[1]); first = false; }
            else ctx.lineTo(pt[0], pt[1]);
          }
        }
        ctx.closePath();
        ctx.lineWidth = w;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.strokeStyle = color;
        ctx.stroke();
      } else {
        // Simple path (non-circle annotations)
        ctx.strokeStyle = color;
        ctx.lineWidth = w;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.beginPath();
        ctx.moveTo(screenPts[0][0], screenPts[0][1]);
        for (var k = 1; k < screenPts.length; k++) {
          ctx.lineTo(screenPts[k][0], screenPts[k][1]);
        }
        ctx.stroke();
      }
    }
  }

  // ── Multi-level label system ─────────────────────────────────────────
  // Parse levels: keys are cluster level keys, values are label arrays
  var levelKeys = Object.keys(labelLevels).map(function(k) {
    return isNaN(Number(k)) ? k : Number(k);
  }).sort(function(a,b) {
    if (typeof a === "string") return 1;
    if (typeof b === "string") return -1;
    return a - b;
  });
  // Master copy: all levels stay available for zoom-driven label display
  var allLabelLevels = {};
  var allLevelKeys = levelKeys.slice();
  Object.keys(labelLevels).forEach(function(k) { allLabelLevels[k] = labelLevels[k]; });
  // Store z backups per level for 2D/3D toggle, then flatten for 2D init
  var zLevelsBackup = {};
  levelKeys.forEach(function(k) {
    zLevelsBackup[k] = labelLevels[k].map(function(c) { return c.z; });
    labelLevels[k].forEach(function(c) { c.z = 0; });
  });
  // Level style classes: coarsest=coarse, finest=fine, middle=mid
  function levelClass(k) {
    var idx = allLevelKeys.indexOf(k);
    if (idx === 0) return "level-coarse";
    if (idx === allLevelKeys.length - 1) return "level-fine";
    return "level-mid";
  }
  // Pre-create a pool of reusable label DOM elements inside a container
  var MAX_VISIBLE_LABELS = 40;
  var labelPool = [];
  var labelContainer = document.createElement("div");
  labelContainer.id = "cluster-label-container";
  document.body.appendChild(labelContainer);
  for (var _lp = 0; _lp < MAX_VISIBLE_LABELS; _lp++) {
    var e = document.createElement("div");
    e.className = "cl";
    e.style.opacity = "0";
    var dot = document.createElement("span");
    dot.className = "cl-dot";
    dot.style.cssText = "display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px;vertical-align:middle;";
    e.appendChild(dot);
    var txt = document.createElement("span");
    e.appendChild(txt);
    labelContainer.appendChild(e);
    labelPool.push(e);
  }

  function isClusterVisible(cid) {
    if (isolatedCluster !== null) return cid === isolatedCluster;
    return !hiddenClusters.has(cid);
  }

  // Build edge path layer data (for toggling)
  function edgeColor() {
    // Semi-transparent edges (increased alpha for visibility)
    return (currentTheme === "light") ? [30, 80, 180, 80] : [255, 255, 255, 60];
  }
  var edgePathData = [];

  function rebuildLayer() {
    var dk = getDeck();
    if (!dk || !dk.props) return;
    var is2d = (viewMode === "2d");
    var ec = edgeColor();
    var hlEc = (currentTheme === "light") ? [220, 80, 20, 200] : [255, 160, 40, 220];
    var hasHl = highlightedEdgeClusters.size > 0;

    // Update fade target based on highlight state
    edgeFadeTarget = hasHl ? 1 : 0;

    // Use 2D bundled paths in 2D mode, 3D catenary curves in 3D mode
    var edgePaths = is2d ? edgePaths2d : edgePaths3d;
    edgePathData = [];

    // Motion intensity based on velocity magnitude (for organic wobble)
    var velMag = Math.sqrt(sagVelocity[0]*sagVelocity[0] + sagVelocity[1]*sagVelocity[1] + sagVelocity[2]*sagVelocity[2]);
    var motionIntensity = Math.min(1, velMag * velMag * 0.3);

    edgePaths.forEach(function(path, idx) {
      // Skip edges connecting to hidden clusters
      if (idx < edgePairs.length) {
        var pair = edgePairs[idx];
        if (!isClusterVisible(pair[0]) || !isClusterVisible(pair[1])) return;
      }
      var w = edgeWeights[idx] || 0.5;
      var width = 0.005 + w * 0.015;  // thicker for stronger connections
      var finalPath = path;

      if (!is2d && path.length > 2) {
        var p0 = path[0], pn = path[path.length-1];
        var edgeLen = Math.sqrt(
          Math.pow(pn[0]-p0[0], 2) + Math.pow(pn[1]-p0[1], 2) + Math.pow(pn[2]-p0[2], 2)
        ) || 1;

        // Per-edge random variation
        var r = edgeRandom[idx] || { phase1: 0, phase2: 0, phase3: 0, amp: 1, freq1: 1, freq2: 0.5, freq3: 1.5, wobbleAmp: 0.2, delay: 0 };
        var t_delayed = Math.max(0, swayTime - r.delay);

        // Wobbles scaled by motion intensity
        var wobble1 = Math.sin(t_delayed * r.freq1 + r.phase1) * r.wobbleAmp * motionIntensity;
        var wobble2 = Math.sin(t_delayed * r.freq2 + r.phase2) * r.wobbleAmp * 0.7 * motionIntensity;
        var wobble3 = Math.sin(t_delayed * r.freq3 + r.phase3) * r.wobbleAmp * 0.3 * motionIntensity;
        var totalWobble = wobble1 + wobble2 + wobble3;

        // Sag direction from physics
        var sagX = currentSag[0], sagY = currentSag[1], sagZ = currentSag[2];

        // Perpendicular to sag for wobble
        var perpX = -sagY, perpY = sagX;

        finalPath = path.map(function(pt, i) {
          var t = i / (path.length - 1);
          var envelope = Math.sin(t * Math.PI);
          var curveNoise = Math.sin(t * 3.14159 * 2 + r.phase3) * 0.2 * motionIntensity;

          var sagAmount = envelope * edgeLen * 0.25 * r.amp;
          var wobbleAmount = envelope * totalWobble * edgeLen * 0.15;
          var wobble2Amount = envelope * curveNoise * wobble2 * edgeLen * 0.1;

          return [
            pt[0] + sagX * sagAmount + perpX * wobbleAmount,
            pt[1] + sagY * sagAmount + perpY * wobbleAmount,
            pt[2] + sagZ * sagAmount
          ];
        });
      }
      // Determine base color
      var baseColor = ec;
      var baseWidth = width;
      var fadeAlpha = edgeFadeAlpha;
      if (hasHl && idx < edgePairs.length) {
        var pair = edgePairs[idx];
        var isHighlighted = highlightedEdgeClusters.has(pair[0]) || highlightedEdgeClusters.has(pair[1]);
        if (isHighlighted) {
          var a = Math.round(ec[3] + (hlEc[3] - ec[3]) * fadeAlpha);
          var r = Math.round(ec[0] + (hlEc[0] - ec[0]) * fadeAlpha);
          var g = Math.round(ec[1] + (hlEc[1] - ec[1]) * fadeAlpha);
          var b = Math.round(ec[2] + (hlEc[2] - ec[2]) * fadeAlpha);
          baseColor = [r, g, b, a];
          baseWidth = width * (1 + 0.5 * fadeAlpha);
        } else {
          var dimAlpha = Math.round(ec[3] - (ec[3] - 15) * fadeAlpha);
          baseColor = [ec[0], ec[1], ec[2], dimAlpha];
        }
      }
      edgePathData.push({ path: finalPath, color: baseColor, width: baseWidth });
    });

    var hasCallouts = tourCalloutHighlightSet.size > 0;
    var visible = [];
    for (var _vi = 0; _vi < allPoints.length; _vi++) {
      var p = allPoints[_vi];
      if (!isClusterVisible(p.cluster)) continue;
      if (is2d) {
        visible.push({ x: p.x, y: p.y, z: 0, r: p.r, g: p.g, b: p.b,
                        a: 255, title: p.title, cluster: p.cluster });
      } else if (hasCallouts && tourCalloutHighlightSet.has(_vi)) {
        // Bright yellow-white for callout points
        visible.push({ x: p.x, y: p.y, z: p.z, r: 255, g: 240, b: 80,
                        a: 255, title: p.title, cluster: p.cluster });
      } else if (hasCallouts) {
        // Dim non-callout points during callout display
        visible.push({ x: p.x, y: p.y, z: p.z, r: p.r, g: p.g, b: p.b,
                        a: 60, title: p.title, cluster: p.cluster });
      } else {
        visible.push(p);
      }
    }
    var edgeData = edgePathData;
    // Use cached original layers (not current dk.props.layers which changes after setProps)
    if (!_origPointLayer) return;
    var newPointLayer = _origPointLayer.clone({ data: visible });
    var newLayers = [newPointLayer];
    // Clone pydeck's edge layer (if present) with updated data
    if (edgesVisible && _origEdgeLayer && edgeData.length > 0) {
      var newEdgeLayer = _origEdgeLayer.clone({
        data: edgeData,
        getPath: function(d) { return d.path; },
        getColor: function(d) { return d.color; },
        getWidth: function(d) { return d.width; }
      });
      newLayers.push(newEdgeLayer);
    }
    dk.setProps({ layers: newLayers });
    updateRowStyles();
  }

  function updateRowStyles() {
    rows.forEach(function(row, i) {
      var cid = rowClusterIds[i];
      if (isolatedCluster !== null) {
        row.style.opacity = (cid === isolatedCluster) ? "1" : "0.3";
        row.style.textDecoration = (cid === isolatedCluster) ? "none" : "line-through";
      } else if (hiddenClusters.has(cid)) {
        row.style.opacity = "0.3";
        row.style.textDecoration = "line-through";
      } else {
        row.style.opacity = "1";
        row.style.textDecoration = "none";
      }
    });
  }

  // Populate cluster list in panel
  var listEl = document.getElementById("cluster-list");
  var rows = [];
  var rowClusterIds = [];

  function rebuildClusterList() {
    // Clear existing rows
    listEl.innerHTML = "";
    rows = [];
    rowClusterIds = [];

    // Get unique cluster IDs from current point assignments
    var uniqueCids = [];
    var cset = {};
    allPoints.forEach(function(p) {
      if (!(p.cluster in cset)) { cset[p.cluster] = true; uniqueCids.push(p.cluster); }
    });
    uniqueCids.sort(function(a,b) { return a - b; });

    // Get color map for current clusters
    var cmap = buildColorMap(uniqueCids, currentLevelKey);

    // Get current label data
    var curLabels = labels;  // always uses the current dendrogram cut labels

    // Build name lookup from label_data
    var nameMap = {};
    curLabels.forEach(function(c) { nameMap[c.cid] = c; });

    uniqueCids.forEach(function(cid) {
      var rgb = cmap[cid];
      var info = nameMap[cid];
      var text = info ? info.text : ("Cluster " + cid);
      var size = info ? info.size : 0;
      if (!size) {
        // Count from data
        size = 0;
        allPoints.forEach(function(p) { if (p.cluster === cid) size++; });
      }

      var row = document.createElement("div");
      row.style.cursor = "pointer";
      row.style.padding = "2px 0";
      row.style.transition = "opacity 0.15s";
      row.innerHTML =
        '<span style="display:inline-block;width:10px;height:10px;border-radius:2px;' +
        'background:rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ');margin-right:6px;' +
        'vertical-align:middle;"></span>' +
        '<span style="vertical-align:middle;">' + text +
        ' <span style="color:var(--fg-muted);">(' + size + ')</span></span>';

      // Single click: toggle hide
      row.addEventListener("click", function(e) {
        e.preventDefault();
        if (isolatedCluster !== null) return;
        if (hiddenClusters.has(cid)) {
          hiddenClusters.delete(cid);
        } else {
          hiddenClusters.add(cid);
        }
        rebuildLayer();
      });

      // Double click: isolate or reset
      row.addEventListener("dblclick", function(e) {
        e.preventDefault();
        if (isolatedCluster === cid) {
          isolatedCluster = null;
          hiddenClusters.clear();
        } else {
          isolatedCluster = cid;
          hiddenClusters.clear();
        }
        rebuildLayer();
      });

      listEl.appendChild(row);
      rows.push(row);
      rowClusterIds.push(cid);
    });
  }

  // ── switchLevel: no-op for dendrogram mode (slider handles it) ──
  function switchLevel(key) {
    // Dendrogram mode: no discrete level switching
    // This function is retained for WebSocket/MCP command compatibility
    if (_dendro) return;
  }

  // ── Color mode switching (cluster vs bucket) ────────────────────────────
  // Bucket dendrogram state (built once on first switch to bucket mode)
  var _bucketDendroZ = null;     // linkage matrix for buckets
  var _bucketK = 0;              // number of natural buckets
  var _bucketIdMap = null;       // [seqIdx] → bucketId
  var _bucketIdRevMap = null;    // {bucketId → seqIdx}
  var _bucketColors = {};        // {bucketId: [r,g,b]}
  var _bucketNames = {};         // {bucketId: "name"}
  var _bucketCentroids = {};     // {bucketId: [x,y,z]} computed from UMAP
  var _bucketSizes = {};         // {bucketId: count}
  var _bucketZoomK = null;       // current zoom-driven bucket k
  var _bucketZoomTimer = null;
  var _bucketZoomPendingK = null;
  var _currentBucketMergedIds = null;  // Int32Array(N): point → merged bucket group

  function _initBucketDendrogram() {
    if (_bucketDendroZ) return;  // already built

    // Compute bucket centroids and sizes from UMAP coords
    var sums = {};
    for (var i = 0; i < _nPts; i++) {
      var bid = _lshBucketIds[i];
      if (!sums[bid]) sums[bid] = { sx: 0, sy: 0, sz: 0, count: 0 };
      sums[bid].sx += _x[i];
      sums[bid].sy += _y[i];
      sums[bid].sz += _z[i];
      sums[bid].count++;
    }

    // Build sequential index ↔ bucket ID mapping
    var bucketIds = Object.keys(sums).map(Number).sort(function(a, b) { return a - b; });
    _bucketK = bucketIds.length;
    _bucketIdMap = bucketIds;
    _bucketIdRevMap = {};
    var centroids = [];
    for (var si = 0; si < bucketIds.length; si++) {
      var bid = bucketIds[si];
      _bucketIdRevMap[bid] = si;
      var s = sums[bid];
      var cent = [s.sx / s.count, s.sy / s.count, s.sz / s.count];
      centroids.push(cent);
      _bucketCentroids[bid] = cent;
      _bucketSizes[bid] = s.count;
    }
    _bucketColors = _lshBucketColors;
    _bucketNames = _lshBucketNames;

    // Build complete-linkage dendrogram on UMAP centroids
    _bucketDendroZ = buildLinkage(centroids);
    _currentBucketMergedIds = new Int32Array(_nPts);
    console.log("[bucket] Built dendrogram for " + _bucketK + " buckets, " + _bucketDendroZ.length + " merges");
  }

  /**
   * Apply a bucket dendrogram cut: recolor points and update labels.
   * height=0 → natural k (all buckets), height=max → k=1.
   */
  function applyBucketCut(height) {
    if (!_bucketDendroZ || !_bucketK) return;

    var cut;
    if (height <= 0) {
      // Identity: each sequential index maps to itself
      cut = { mergedMap: {}, k: _bucketK };
      for (var si = 0; si < _bucketK; si++) cut.mergedMap[si] = si;
    } else {
      cut = cutLinkage(_bucketDendroZ, _bucketK, height);
      if (!cut) return;
    }

    // For each merged group, find the bucket with the largest size → inherit its color/name
    var groupBest = {};
    for (var siStr in cut.mergedMap) {
      var si = parseInt(siStr);
      var group = cut.mergedMap[siStr];
      var bid = _bucketIdMap[si];
      var sz = _bucketSizes[bid] || 0;
      if (!(group in groupBest) || sz > groupBest[group].size) {
        groupBest[group] = { bid: bid, size: sz };
      }
    }
    var mergedColors = {};
    var mergedNames = {};
    var mergedSizes = {};
    for (var gid in groupBest) {
      mergedColors[gid] = _bucketColors[String(groupBest[gid].bid)] || [128, 128, 128];
      mergedNames[gid] = _bucketNames[String(groupBest[gid].bid)] || ("Bucket " + groupBest[gid].bid);
    }
    // Compute merged sizes
    for (var siStr in cut.mergedMap) {
      var group = cut.mergedMap[siStr];
      var bid = _bucketIdMap[parseInt(siStr)];
      mergedSizes[group] = (mergedSizes[group] || 0) + (_bucketSizes[bid] || 0);
    }
    // Compute merged centroids (weighted average)
    var centSums = {};
    for (var siStr in cut.mergedMap) {
      var si = parseInt(siStr);
      var group = cut.mergedMap[siStr];
      var bid = _bucketIdMap[si];
      var cent = _bucketCentroids[bid] || [0, 0, 0];
      var sz = _bucketSizes[bid] || 0;
      if (!centSums[group]) centSums[group] = { wx: 0, wy: 0, wz: 0, total: 0 };
      centSums[group].wx += cent[0] * sz;
      centSums[group].wy += cent[1] * sz;
      centSums[group].wz += cent[2] * sz;
      centSums[group].total += sz;
    }
    var mergedCentroids = {};
    for (var gid in centSums) {
      var cs = centSums[gid];
      mergedCentroids[gid] = cs.total > 0
        ? [cs.wx / cs.total, cs.wy / cs.total, cs.wz / cs.total]
        : [0, 0, 0];
    }

    // Recolor points
    for (var i = 0; i < _nPts; i++) {
      var bid = _lshBucketIds[i];
      var si = _bucketIdRevMap[bid];
      var group = cut.mergedMap[String(si)];
      if (group === undefined) group = si;
      _currentBucketMergedIds[i] = group;
      var rgb = mergedColors[group] || [128, 128, 128];
      allPoints[i].r = rgb[0];
      allPoints[i].g = rgb[1];
      allPoints[i].b = rgb[2];
      allPoints[i].cluster = group;
    }

    // Update labels
    var newLabels = [];
    for (var gid in mergedNames) {
      var cent = mergedCentroids[gid] || [0, 0, 0];
      var rgb = mergedColors[gid] || [128, 128, 128];
      newLabels.push({
        cid: parseInt(gid),
        text: mergedNames[gid],
        x: cent[0], y: cent[1], z: cent[2] || 0,
        size: mergedSizes[gid] || 0,
        leaf_cids: [parseInt(gid)],
        color: rgb,
      });
    }
    labels = newLabels;
    labelLevels = { "bucket": newLabels };
    currentLevelKey = "bucket";
    allLabelLevels = { "bucket": newLabels };
    allLevelKeys = ["bucket"];

    // Save z backups for 3D toggle and flatten if currently in 2D
    zLevelsBackup["bucket"] = newLabels.map(function(c) { return c.z; });
    if (viewMode === "2d") {
      newLabels.forEach(function(c) { c.z = 0; });
    }

    rebuildClusterList();
    rebuildLayer();
  }

  function applyColorMode(mode) {
    if (!_lshBucketIds && mode === "bucket") return;
    _colorMode = mode;

    // Update button active state
    var btnCluster = document.getElementById("color-cluster");
    var btnBucket = document.getElementById("color-bucket");
    if (btnCluster) btnCluster.style.fontWeight = (mode === "cluster") ? "700" : "normal";
    if (btnBucket) btnBucket.style.fontWeight = (mode === "bucket") ? "700" : "normal";

    if (mode === "bucket") {
      _initBucketDendrogram();
      // Apply at natural k (no merging) — zoom will drive cuts
      _bucketZoomK = _bucketK;
      applyBucketCut(0);
    } else {
      _bucketZoomK = null;
      // Restore cluster/dendrogram coloring
      if (_dendro) {
        var sl = document.getElementById("dendro-slider");
        var height = sl ? parseFloat(sl.value) : 0;
        applyDendrogramCut(height);
      } else {
        // No dendrogram: restore grey
        for (var i = 0; i < _nPts; i++) {
          allPoints[i].r = 128;
          allPoints[i].g = 128;
          allPoints[i].b = 128;
          allPoints[i].cluster = 0;
        }
        rebuildClusterList();
        rebuildLayer();
      }
    }

    // Reset hide/isolate state
    isolatedCluster = null;
    hiddenClusters.clear();
  }

  // Wire color-by buttons
  (function() {
    var btnCluster = document.getElementById("color-cluster");
    var btnBucket = document.getElementById("color-bucket");
    if (btnCluster) btnCluster.addEventListener("click", function() { applyColorMode("cluster"); });
    if (btnBucket) btnBucket.addEventListener("click", function() { applyColorMode("bucket"); });
  })();

  // ── Initialize cluster slider UI (replaces discrete level buttons) ──
  (function() {
    if (!_dendro) return;
    var container = document.getElementById("level-selector");
    var btnContainer = document.getElementById("level-buttons");
    if (!container || !btnContainer) return;
    container.style.display = "block";

    // Clear any existing buttons
    btnContainer.innerHTML = "";

    // Compute max dendrogram height
    var maxHeight = 0;
    for (var i = 0; i < _dendroZ.length; i++) {
      if (_dendroZ[i][2] > maxHeight) maxHeight = _dendroZ[i][2];
    }

    // Create slider container
    var sliderWrap = document.createElement("div");
    sliderWrap.style.cssText = "width:100%;";

    var sliderLabel = document.createElement("div");
    sliderLabel.style.cssText = "display:flex;justify-content:space-between;margin-bottom:4px;font-size:10px;opacity:0.6;";
    var sliderLeft = document.createElement("span");
    sliderLeft.textContent = _naturalK + " clusters";
    var sliderRight = document.createElement("span");
    sliderRight.id = "dendro-k-display";
    sliderRight.textContent = "k=" + _naturalK;
    sliderLabel.appendChild(sliderLeft);
    sliderLabel.appendChild(sliderRight);

    var slider = document.createElement("input");
    slider.type = "range";
    slider.id = "dendro-slider";
    slider.min = "0";
    slider.max = String(maxHeight * 1.01);  // slightly above max to allow k=1
    slider.step = String(maxHeight / 200);
    slider.value = "0";
    slider.style.cssText = "width:100%;";

    slider.addEventListener("input", function() {
      var height = parseFloat(slider.value);
      var cut = cutDendrogram(height);
      var k = cut ? cut.k : _naturalK;
      if (height <= 0) k = _naturalK;
      sliderRight.textContent = "k=" + k;
      _dendroZoomK = k;  // sync so scroll zoom doesn't immediately override
      applyDendrogramCut(height);
    });

    sliderWrap.appendChild(sliderLabel);
    sliderWrap.appendChild(slider);
    btnContainer.appendChild(sliderWrap);
  })();

  rebuildClusterList();

  // Deck access
  var _origPointLayer = null;
  var _origEdgeLayer = null;
  function getDeck() {
    var d = window.deckInstance;
    var dk = d && d.deck ? d.deck : d || null;
    // Cache original layers on first access
    if (dk && dk.props && dk.props.layers && !_origPointLayer) {
      _origPointLayer = dk.props.layers[0];
      if (dk.props.layers.length > 1) {
        _origEdgeLayer = dk.props.layers[1];
      }
      console.log("[init] Cached original layers: point=", !!_origPointLayer, "edge=", !!_origEdgeLayer);
    }
    return dk;
  }

  // Depth-based point alpha (debounced — rebuilds layer when view settles)
  var depthTimer = null;
  var lastViewJson = "";
  function updatePointAlpha(dk, vp) {
    if (viewMode === "2d") return;
    // Skip depth alpha when callout highlighting is active (tour Phase 3)
    if (tourCalloutHighlightSet.size > 0) return;
    var depths = [];
    for (var i = 0; i < allPoints.length; i++) {
      try {
        var sp = vp.project([allPoints[i].x, allPoints[i].y, allPoints[i].z]);
        depths.push(sp[2] || 0);
      } catch(e) { depths.push(0); }
    }
    var minD = depths[0], maxD = depths[0];
    for (var i = 1; i < depths.length; i++) {
      if (depths[i] < minD) minD = depths[i];
      if (depths[i] > maxD) maxD = depths[i];
    }
    var rangeD = maxD - minD || 1;
    for (var i = 0; i < allPoints.length; i++) {
      var t = (depths[i] - minD) / rangeD;
      allPoints[i].a = Math.round(255 - t * 200);
    }
    // rebuildLayer will pick up the updated .a values
    rebuildLayer();
  }

  // ── Multi-level zoom-aware label placement ───────────────────────────
  // Zoom thresholds: map zoom level to which cluster levels to show
  // deck.gl OrbitView zoom ~5.5 default; higher = more zoomed in
  // Show one level at a time: coarse at default zoom, finer when zoomed in.
  // Separation scales inversely with zoom so more labels fit when zoomed in.
  // Zoom-driven level switching: compute zoom thresholds from level count
  // Evenly divide the zoom range [defaultZoom, defaultZoom+4] among levels.
  // Returns a single-element array with the active level key for this zoom.
  var _zoomLevelKey = null;  // track which level zoom has selected
  var _dendroZoomK = null;  // track zoom-driven dendrogram k
  var _dendroZoomTimer = null;  // debounce timer for zoom-driven cuts
  var _dendroZoomPendingK = null;  // pending k during debounce
  function _applyDendroZoomK(targetK) {
    _dendroZoomK = targetK;
    if (targetK >= _naturalK) {
      applyDendrogramCut(0);
      var sl = document.getElementById("dendro-slider");
      if (sl) sl.value = "0";
    } else {
      var lo = 0, hi = _dendroZ[_dendroZ.length - 1][2] * 1.02;
      for (var bi = 0; bi < 50; bi++) {
        var mid = (lo + hi) / 2;
        var c = cutDendrogram(mid);
        if (!c) break;
        if (c.k > targetK) lo = mid;
        else if (c.k < targetK) hi = mid;
        else { lo = mid; break; }
      }
      applyDendrogramCut(lo);
      var sl = document.getElementById("dendro-slider");
      if (sl) sl.value = String(lo);
    }
    var kd = document.getElementById("dendro-k-display");
    if (kd) kd.textContent = "k=" + targetK;
  }
  function _applyBucketZoomK(targetK) {
    _bucketZoomK = targetK;
    if (targetK >= _bucketK) {
      applyBucketCut(0);
    } else {
      var lo = 0, hi = _bucketDendroZ[_bucketDendroZ.length - 1][2] * 1.02;
      for (var bi = 0; bi < 50; bi++) {
        var mid = (lo + hi) / 2;
        var c = cutLinkage(_bucketDendroZ, _bucketK, mid);
        if (!c) break;
        if (c.k > targetK) lo = mid;
        else if (c.k < targetK) hi = mid;
        else { lo = mid; break; }
      }
      applyBucketCut(lo);
    }
  }
  function getActiveLevels(zoom) {
    // Bucket mode: zoom-driven bucket dendrogram cuts
    if (_colorMode === "bucket" && _bucketDendroZ && _bucketK > 0) {
      var zRange = 6.0;
      var minK = 2;
      var t = (zoom - defaultZoom + zRange / 2) / zRange;
      t = Math.max(0, Math.min(1, t));
      var targetK = Math.round(minK + t * (_bucketK - minK));
      targetK = Math.max(minK, Math.min(_bucketK, targetK));

      if (targetK !== _bucketZoomK && targetK !== _bucketZoomPendingK) {
        _bucketZoomPendingK = targetK;
        clearTimeout(_bucketZoomTimer);
        _bucketZoomTimer = setTimeout(function() {
          if (_bucketZoomPendingK !== null && _bucketZoomPendingK !== _bucketZoomK) {
            _applyBucketZoomK(_bucketZoomPendingK);
          }
          _bucketZoomPendingK = null;
        }, 150);
      }
      return ["bucket"];
    }
    if (_colorMode === "bucket") return ["bucket"];

    // Dendrogram mode: map scroll-zoom to dendrogram cut level
    // Centered so default zoom ≈ half of naturalK, zooming in → more clusters
    if (_dendro && _naturalK > 0 && _dendroZ) {
      var zRange = 6.0;
      var minK = 2;
      var t = (zoom - defaultZoom + zRange / 2) / zRange;
      t = Math.max(0, Math.min(1, t));
      var targetK = Math.round(minK + t * (_naturalK - minK));
      targetK = Math.max(minK, Math.min(_naturalK, targetK));

      if (targetK !== _dendroZoomK && targetK !== _dendroZoomPendingK) {
        _dendroZoomPendingK = targetK;
        clearTimeout(_dendroZoomTimer);
        _dendroZoomTimer = setTimeout(function() {
          if (_dendroZoomPendingK !== null && _dendroZoomPendingK !== _dendroZoomK) {
            _applyDendroZoomK(_dendroZoomPendingK);
          }
          _dendroZoomPendingK = null;
        }, 150);
      }
      return ["dendro"];
    }

    if (allLevelKeys.length <= 1) return allLevelKeys;
    // Spread levels across zoom range: coarsest at default, finest at +4
    var zRange = 4.0;
    var step = zRange / allLevelKeys.length;
    var idx = Math.floor((zoom - defaultZoom) / step);
    idx = Math.max(0, Math.min(allLevelKeys.length - 1, idx));
    var newKey = allLevelKeys[idx];
    // If zoom crossed a level boundary, recolor points
    if (_zoomLevelKey !== newKey) {
      _zoomLevelKey = newKey;
      currentLevelKey = String(newKey);
      rebuildLayer();
      rebuildClusterList();
    }
    return [newKey];
  }

  // Label placement: project, cull off-screen, spatial separation
  function updateLabels(vp, zoom) {
    if (!labelsVisible && !tourRunning) {
      labelContainer.style.display = "none";
      return;
    }
    labelContainer.style.display = "";
    // During tour, force finest level so individual cluster labels show as they're revealed
    var activeLevels = tourRunning ? levelKeys : getActiveLevels(zoom);
    var is2d = (viewMode === "2d");
    var w = window.innerWidth - 260;  // account for panel
    var h = window.innerHeight;

    // Minimum screen-space separation squared (pixels)
    // Scale separation inversely with zoom: tighter labels when zoomed in
    var baseSep = Math.min(w, h) * 0.06;
    var sepScale = Math.max(0.5, defaultZoom / zoom);
    var minSepSq = Math.pow(baseSep * sepScale, 2);

    var placed = [];  // {sx, sy, text, levelKey, depth}

    // In 3D mode, collect all candidates first, then sort by depth (nearest camera first)
    // so front-facing labels win the separation check over occluded back labels
    var candidates = [];

    // Process levels coarsest first
    for (var li = 0; li < activeLevels.length; li++) {
      var lk = activeLevels[li];
      var lvlLabels = allLabelLevels[lk] || labelLevels[lk];
      if (!lvlLabels) continue;
      var cls = levelClass(lk);

      for (var j = 0; j < lvlLabels.length; j++) {
        var c = lvlLabels[j];
        // Skip labels if ANY constituent BIRCH cluster is hidden (avoids floating labels
        // whose centroid sits near hidden clusters)
        if (c.leaf_cids) {
          var allVisible = true;
          for (var _lc = 0; _lc < c.leaf_cids.length; _lc++) {
            if (!isClusterVisible(c.leaf_cids[_lc])) { allVisible = false; break; }
          }
          if (!allVisible) continue;
        } else if (c.cid !== undefined && !isClusterVisible(c.cid)) continue;
        // All labels stay visible during tour — highlighting and zoom indicate the active cluster
        var lz = is2d ? 0 : c.z;
        var sp;
        try { sp = vp.project([c.x, c.y, lz]); } catch(e) { continue; }
        var sx = sp[0], sy = sp[1];

        // Cull off-screen
        if (sx < -30 || sx > w + 30 || sy < -30 || sy > h + 30) continue;

        candidates.push({ sx: sx, sy: sy, text: c.text, cls: cls, depth: sp[2] || 0, color: c.color || null });
      }
    }

    // In 3D, sort candidates by depth (nearest camera = smallest depth first)
    // so front labels claim screen space before back labels
    if (!is2d) {
      candidates.sort(function(a, b) { return a.depth - b.depth; });
    }

    // Apply separation check: first-placed labels win
    for (var ci = 0; ci < candidates.length; ci++) {
      var cand = candidates[ci];
      var tooClose = false;
      for (var p = 0; p < placed.length; p++) {
        var dx = cand.sx - placed[p].sx, dy = cand.sy - placed[p].sy;
        if (dx * dx + dy * dy < minSepSq) { tooClose = true; break; }
      }
      if (tooClose) continue;
      placed.push(cand);
      if (placed.length >= MAX_VISIBLE_LABELS) break;
    }

    // Sort by depth for z-ordering (farther = lower z-index)
    placed.sort(function(a, b) { return b.depth - a.depth; });
    var minD = placed.length ? placed[placed.length - 1].depth : 0;
    var maxD = placed.length ? placed[0].depth : 1;
    var rangeD = maxD - minD || 1;

    // Apply to pool elements
    for (var i = 0; i < MAX_VISIBLE_LABELS; i++) {
      var el = labelPool[i];
      if (i < placed.length) {
        var pl = placed[i];
        var dotEl = el.querySelector(".cl-dot");
        var txtEl = el.lastChild;
        if (txtEl && txtEl.nodeType === 3 || txtEl === dotEl) {
          // Fallback: rebuild structure if needed
          txtEl = el.childNodes[1] || el;
        }
        txtEl.textContent = pl.text;
        if (dotEl && pl.color) {
          dotEl.style.backgroundColor = "rgb(" + pl.color[0] + "," + pl.color[1] + "," + pl.color[2] + ")";
          dotEl.style.display = "inline-block";
        } else if (dotEl) {
          dotEl.style.display = "none";
        }
        el.className = "cl " + pl.cls;
        el.style.left = pl.sx + "px";
        el.style.top = pl.sy + "px";
        el.style.zIndex = 10 + i;
        if (is2d) {
          el.style.opacity = "1";
        } else {
          var t = (pl.depth - minD) / rangeD;
          el.style.opacity = (1.0 - t * 0.7).toFixed(2);
        }
      } else {
        el.style.opacity = "0";
      }
    }
  }

  // Main render loop
  function update() {
    requestAnimationFrame(update);
    var dk = getDeck();
    if (!dk || !dk.getViewports) return;
    var vps = dk.getViewports();
    if (!vps || !vps.length) return;
    var vp = vps[0];

    // Get current zoom from live tracking (set via onViewStateChange)
    var zoom = window._liveZoom != null ? window._liveZoom : defaultZoom;

    // Animate edge fade
    if (edgeFadeAlpha !== edgeFadeTarget) {
      if (edgeFadeAlpha < edgeFadeTarget) {
        edgeFadeAlpha = Math.min(edgeFadeTarget, edgeFadeAlpha + edgeFadeSpeed);
      } else {
        edgeFadeAlpha = Math.max(edgeFadeTarget, edgeFadeAlpha - edgeFadeSpeed);
      }
      rebuildLayer();  // Update edge colors during fade
    }

    // Debounced point alpha update
    var vs = dk.viewManager ? JSON.stringify(dk.viewManager.getViewState()) : "";
    if (vs !== lastViewJson) {
      lastViewJson = vs;
      clearTimeout(depthTimer);
      depthTimer = setTimeout(function() { updatePointAlpha(dk, vp); }, 150);
    }

    updateLabels(vp, zoom);
    drawAnnotations(vp);

    // Position tour label at centroid (or center of screen for intro/outro)
    if (tourRunning) {
      var tourLabelEl = document.getElementById("tour-label");
      if (tourCentroid) {
        try {
          var sp = vp.project(tourCentroid);
          tourLabelEl.style.left = sp[0] + "px";
          tourLabelEl.style.top = sp[1] + "px";
        } catch(e) {}
      } else if (tourLabelEl.style.display !== "none") {
        // Intro/outro: center label on full screen (hero transform handles centering)
        tourLabelEl.style.left = (window.innerWidth / 2) + "px";
        tourLabelEl.style.top = (window.innerHeight / 2) + "px";
      }

      // Render edge centroid labels only during wide-zoom phases, and only if
      // some of the connected cluster's points are actually visible in the viewport
      var edgeLabelsContainer = document.getElementById("tour-edge-labels");
      var showEdgeLabels = (tourPhase === "panToWide" || tourPhase === "holdWide" || tourPhase === "zoomOut");
      var html = "";
      if (showEdgeLabels) {
        var vpW = window.innerWidth, vpH = window.innerHeight;
        tourConnected.forEach(function(conn) {
          try {
            var esp = vp.project(conn.centroid);
            if (esp[0] < -50 || esp[0] > vpW + 50 || esp[1] < -50 || esp[1] > vpH + 50) return;
            // Only show label if some of this cluster's points are in the viewport
            var hasVisiblePts = false;
            for (var _si = 0; _si < conn.samplePtIndices.length; _si++) {
              var pt = allPoints[conn.samplePtIndices[_si]];
              var sp = vp.project([pt.x, pt.y, pt.z]);
              if (sp[0] >= 0 && sp[0] <= vpW && sp[1] >= 0 && sp[1] <= vpH) {
                hasVisiblePts = true;
                break;
              }
            }
            if (!hasVisiblePts) return;
            html += '<div class="tour-edge-label" style="left:' + esp[0] + 'px;top:' + esp[1] + 'px;">' + conn.name + '</div>';
          } catch(e) {}
        });
      }
      edgeLabelsContainer.innerHTML = html;

      // Position callout labels near their points with leader lines
      if (tourActiveCallouts.length > 0) {
        var calloutContainer = document.getElementById("tour-callout-labels");
        var calloutDivs = calloutContainer.querySelectorAll(".tour-callout-label");
        var hlCanvas = document.getElementById("hl-canvas");
        var hlCtx = hlCanvas ? hlCanvas.getContext("2d") : null;

        var cw = hlCanvas ? hlCanvas.clientWidth : window.innerWidth;
        var ch = hlCanvas ? hlCanvas.clientHeight : window.innerHeight;
        var margin = 50;
        var labelOffset = 25;  // px offset from point

        // Collect occupied rectangles from cluster labels to avoid overlap
        var occupied = [];
        var clEls = labelContainer.querySelectorAll(".cl");
        for (var _oi = 0; _oi < clEls.length; _oi++) {
          var el = clEls[_oi];
          if (el.style.display === "none" || !el.offsetWidth) continue;
          var r = el.getBoundingClientRect();
          occupied.push({ x: r.left, y: r.top, w: r.width, h: r.height });
        }

        // Project callout points and find placement
        tourActiveCallouts.forEach(function(co, i) {
          co._onScreen = false;
          try {
            co._screenPt = vp.project(co.pos);
            var sx = co._screenPt[0], sy = co._screenPt[1];
            co._onScreen = (sx >= -margin && sx <= cw + margin &&
                            sy >= -margin && sy <= ch + margin);
          } catch(e) {}
        });

        // Place each callout near its point, trying 8 directions
        var directions = [
          [1, -1], [1, 0], [1, 1], [0, -1],
          [0, 1], [-1, -1], [-1, 0], [-1, 1]
        ];

        tourActiveCallouts.forEach(function(co, i) {
          var div = calloutDivs[i];
          if (!div || !co._onScreen) {
            if (div) div.classList.remove("visible");
            return;
          }

          var dw = div.offsetWidth || 120;
          var dh = div.offsetHeight || 22;
          var sx = co._screenPt[0], sy = co._screenPt[1];
          var bestX = sx + labelOffset, bestY = sy - dh / 2;
          var bestScore = -Infinity;

          // Try each direction and pick least-overlapping placement
          for (var di = 0; di < directions.length; di++) {
            var dx = directions[di][0], dy = directions[di][1];
            var cx = sx + dx * (labelOffset + dw * 0.4) - (dx <= 0 ? dw : 0);
            var cy = sy + dy * (labelOffset + dh * 0.3) - dh / 2;
            // Clamp to viewport
            cx = Math.max(10, Math.min(cw - dw - 10, cx));
            cy = Math.max(10, Math.min(ch - dh - 10, cy));

            // Score: penalize overlap with occupied rects
            var score = 0;
            for (var oi = 0; oi < occupied.length; oi++) {
              var r = occupied[oi];
              var ox = Math.max(0, Math.min(cx + dw, r.x + r.w) - Math.max(cx, r.x));
              var oy = Math.max(0, Math.min(cy + dh, r.y + r.h) - Math.max(cy, r.y));
              score -= ox * oy;
            }
            // Slight preference for right/below point (natural reading direction)
            if (dx > 0) score += 5;
            if (dy > 0) score += 2;

            if (score > bestScore) {
              bestScore = score;
              bestX = cx;
              bestY = cy;
            }
          }

          co._labelX = bestX;
          co._labelY = bestY;
          div.style.left = bestX + "px";
          div.style.top = bestY + "px";
          co._lineStartX = bestX + (bestX > sx ? 0 : dw);
          co._lineStartY = bestY + dh / 2;

          // Add this callout to occupied rects for subsequent callouts
          occupied.push({ x: bestX, y: bestY, w: dw, h: dh });
        });

        // Draw leader lines from labels to points
        if (hlCtx) {
          tourActiveCallouts.forEach(function(co, i) {
            if (!co._onScreen || !co._screenPt) return;
            var div = calloutDivs[i];
            if (!div || !div.classList.contains("visible")) return;
            var lineColor = "rgba(200,200,200,0.5)";
            var dotColor = "rgba(255,255,255,0.8)";
            hlCtx.beginPath();
            hlCtx.moveTo(co._lineStartX, co._lineStartY);
            hlCtx.lineTo(co._screenPt[0], co._screenPt[1]);
            hlCtx.strokeStyle = lineColor;
            hlCtx.lineWidth = 1;
            hlCtx.stroke();
            // Small dot at point
            hlCtx.beginPath();
            hlCtx.arc(co._screenPt[0], co._screenPt[1], 3, 0, Math.PI * 2);
            hlCtx.fillStyle = dotColor;
            hlCtx.fill();
          });
        }
      }
    }
  }

  // Layer init moved to waitForDeck poll below

  // Reset view
  document.getElementById("reset-btn").addEventListener("click", function() {
    annotations.length = 0;
    setEdgeHighlight([]);
    rebuildLayer();
    var is2d = (viewMode === "2d");
    var dk = getDeck();
    if (dk && dk.setProps) {
      dk.setProps({ initialViewState: {
        target: [0,0,0],
        rotationX: is2d ? 90 : 15,
        rotationOrbit: is2d ? 0 : 30,
        zoom: defaultZoom,
        transitionDuration: 300
      } });
    }
  });

  // Toggle labels
  document.getElementById("toggle-labels").addEventListener("change", function(e) {
    labelsVisible = e.target.checked;
  });

  // Toggle bridge edges
  document.getElementById("toggle-edges").addEventListener("change", function(e) {
    edgesVisible = e.target.checked;
    rebuildLayer();
  });

  // Toggle specular sweep animation (orientation cue)
  document.getElementById("toggle-sheen").addEventListener("change", function(e) {
    if (e.target.checked) {
      startSheen();
    } else {
      stopSheen();
    }
  });

  // Toggle auto-orbit
  document.getElementById("toggle-orbit").addEventListener("change", function(e) {
    if (e.target.checked) {
      startOrbit();
    } else {
      stopOrbit();
    }
  });

  // Toggle outlier clusters visibility
  document.getElementById("toggle-outliers").addEventListener("change", function(e) {
    if (e.target.checked) {
      // Show outliers: remove them from hiddenClusters
      outlierClusterIds.forEach(function(cid) { hiddenClusters.delete(cid); });
    } else {
      // Hide outliers: add them to hiddenClusters
      outlierClusterIds.forEach(function(cid) { hiddenClusters.add(cid); });
    }
    rebuildLayer();
  });

  // Cluster tour button
  document.getElementById("tour-btn").addEventListener("click", runTour);

  // Point size slider
  document.getElementById("point-size").addEventListener("input", function(e) {
    var sz = parseFloat(e.target.value);
    document.getElementById("ps-val").textContent = sz;
    var dk = getDeck();
    if (!dk || !dk.props) return;
    var layers = dk.props.layers;
    if (!layers || !layers.length) return;
    // Clone layer props with new pointSize
    var newLayers = layers.map(function(l) {
      if (l.constructor && l.constructor.layerName === "PointCloudLayer") {
        return l.clone({ pointSize: sz });
      }
      return l;
    });
    dk.setProps({ layers: newLayers });
  });

  // ── Dark/light theme toggle ──────────────────────────────────────────
  function setTheme(theme) {
    currentTheme = theme;
    var isLight = (theme === "light");
    document.body.classList.toggle("light", isLight);
    var btn = document.getElementById("theme-btn");
    if (btn) btn.textContent = isLight ? "\u263D Dark" : "\u263C Light";
    // Update background color
    var bg = isLight ? "#f5f5f5" : "#1e1e1e";
    document.body.style.background = bg;
    // Update pydeck wrapper div (not the WebGL canvas directly)
    var deckDiv = document.getElementById("deck-container");
    if (deckDiv) deckDiv.style.background = bg;
    var deckWrapper = document.querySelector("#deckgl-wrapper");
    if (deckWrapper) deckWrapper.style.background = bg;
    // Update highlighter canvas only
    var hlCanvas = document.getElementById("hl-canvas");
    if (hlCanvas) hlCanvas.style.background = "transparent";
    // Update the deck.gl canvas CSS background (pydeck sets this)
    var deckCanvas = document.querySelector("#deck-container canvas");
    if (deckCanvas) deckCanvas.style.background = bg;
    // Update deck.gl WebGL clear color (background)
    var dk = getDeck();
    if (dk && dk.setProps) {
      // clearColor uses normalized 0-1 values: #1e1e1e = 0.118, #f5f5f5 = 0.961
      var clearColor = isLight ? [0.961, 0.961, 0.961, 1] : [0.118, 0.118, 0.118, 1];
      dk.setProps({ parameters: { clearColor: clearColor } });
    }
    // Rebuild edge layer with theme-appropriate edge color
    rebuildLayer();
  }

  document.getElementById("theme-btn").addEventListener("click", function() {
    setTheme(currentTheme === "dark" ? "light" : "dark");
  });

  // ── Fullscreen toggle ──────────────────────────────────────────────
  document.getElementById("fullscreen-btn").addEventListener("click", function() {
    var btn = document.getElementById("fullscreen-btn");
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().then(function() {
        btn.textContent = "Exit Fullscreen";
      }).catch(function() {});
    } else {
      document.exitFullscreen();
      btn.textContent = "\u26F6 Fullscreen";
    }
  });
  document.addEventListener("fullscreenchange", function() {
    var btn = document.getElementById("fullscreen-btn");
    if (btn) btn.textContent = document.fullscreenElement ? "\u26F6 Exit" : "\u26F6 Fullscreen";
  });

  // ── 2D/3D mode toggle ────────────────────────────────────────────────
  function setViewMode(mode) {
    viewMode = mode;
    var btn = document.getElementById("mode-btn");
    if (btn) btn.textContent = (mode === "2d") ? "\u29C8 3D" : "\u25A1 2D";
    var sub = document.getElementById("header-sub");
    if (sub) sub.textContent = (mode === "2d")
      ? "Scroll to zoom \u00b7 Drag to pan \u00b7 Hover for details"
      : "Scroll to zoom \u00b7 Drag to orbit \u00b7 Hover for details";
    // In dendrogram mode, colors are the same in 2D/3D — nothing to swap

    var dk = getDeck();
    if (!dk || !dk.setProps) return;
    // Reset alpha to full opacity for clean transition
    for (var i = 0; i < allPoints.length; i++) allPoints[i].a = 255;
    if (mode === "2d") {
      // Flatten Z (XY already landscape-oriented from Python)
      for (var i = 0; i < allPoints.length; i++) allPoints[i].z = 0;
      // Flatten Z in all label levels
      allLevelKeys.forEach(function(k) {
        var lvl = allLabelLevels[k] || labelLevels[k];
        if (lvl) lvl.forEach(function(c) { c.z = 0; });
      });
      // Top-down view, lock rotation, pan-only controller
      dk.setProps({
        initialViewState: {
          target: [0, 0, 0], rotationX: 90, rotationOrbit: 0, zoom: defaultZoom,
          minRotationX: 90, maxRotationX: 90,
          transitionDuration: 400
        },
        controller: { dragMode: "pan" }
      });
    } else {
      // Restore Z
      for (var i = 0; i < allPoints.length; i++) allPoints[i].z = zBackup[i];
      // Restore Z in all label levels (use backup if available, otherwise keep current z)
      allLevelKeys.forEach(function(k) {
        var backup = zLevelsBackup[k];
        var lvl = allLabelLevels[k] || labelLevels[k];
        if (lvl && backup) {
          lvl.forEach(function(c, j) { c.z = backup[j]; });
        }
      });
      // Restore orbit controls
      dk.setProps({
        initialViewState: {
          target: [0, 0, 0], rotationX: 15, rotationOrbit: 30, zoom: defaultZoom,
          minRotationX: -90, maxRotationX: 90,
          transitionDuration: 400
        },
        controller: { dragMode: "rotate" }
      });
    }
    rebuildLayer();
  }

  document.getElementById("mode-btn").addEventListener("click", function() {
    setViewMode(viewMode === "3d" ? "2d" : "3d");
  });

  // Initialize layers + 2D view as soon as deck.gl is ready
  (function waitForDeck() {
    var dk = getDeck();
    if (dk && dk.setProps) {
      // Populate the empty pydeck layer with decoded binary data
      rebuildLayer();
      // Set 2D view with computed zoom
      setViewMode("2d");
      // Update depth-based alpha
      if (dk.getViewports) {
        var vps = dk.getViewports();
        if (vps && vps.length) updatePointAlpha(dk, vps[0]);
      }
      update();
    } else {
      setTimeout(waitForDeck, 200);
    }
  })();

  // ── WebSocket bridge ──────────────────────────────────────────────────
  var mcpLogMax = 10;
  function logMCP(msg) {
    var logEl = document.getElementById("mcp-log");
    if (!logEl) return;
    var ts = new Date().toLocaleTimeString();
    var line = document.createElement("div");
    line.textContent = ts + " " + (msg.cmd || "?") + ": " + JSON.stringify(msg).slice(0, 80);
    logEl.insertBefore(line, logEl.firstChild);
    while (logEl.children.length > mcpLogMax) {
      logEl.removeChild(logEl.lastChild);
    }
  }

  (function connectWS() {
    var wsUrl = "ws://" + location.host + "/ws";
    var ws;
    try { ws = new WebSocket(wsUrl); } catch(e) { return; }
    ws.onmessage = function(e) {
      var msg;
      try { msg = JSON.parse(e.data); } catch(err) { return; }
      logMCP(msg);
      switch (msg.cmd) {
        case "hide":
          hiddenClusters.add(msg.cluster);
          isolatedCluster = null;
          rebuildLayer();
          break;
        case "show":
          hiddenClusters.delete(msg.cluster);
          isolatedCluster = null;
          rebuildLayer();
          break;
        case "isolate":
          isolatedCluster = msg.cluster;
          hiddenClusters.clear();
          rebuildLayer();
          break;
        case "show_all":
          isolatedCluster = null;
          hiddenClusters.clear();
          rebuildLayer();
          break;
        case "reset_view":
          var dk = getDeck();
          if (dk && dk.setProps) {
            if (viewMode === "2d") {
              dk.setProps({
                initialViewState: {
                  target: [0,0,0], rotationX: 90, rotationOrbit: 0, zoom: defaultZoom,
                  minRotationX: 90, maxRotationX: 90,
                  transitionDuration: 300
                },
                controller: { dragMode: "pan" }
              });
            } else {
              dk.setProps({ initialViewState: {
                target: [0,0,0], rotationX: 15, rotationOrbit: 30, zoom: defaultZoom,
                minRotationX: -90, maxRotationX: 90,
                transitionDuration: 300
              } });
            }
          }
          break;
        case "point_size":
          var dk2 = getDeck();
          if (dk2 && dk2.props && dk2.props.layers && dk2.props.layers.length) {
            var newLayers = dk2.props.layers.map(function(l) {
              if (l.constructor && l.constructor.layerName === "PointCloudLayer") {
                return l.clone({ pointSize: msg.size || 2 });
              }
              return l;
            });
            dk2.setProps({ layers: newLayers });
            var slider = document.getElementById("point-size");
            var valEl = document.getElementById("ps-val");
            if (slider) slider.value = msg.size || 2;
            if (valEl) valEl.textContent = msg.size || 2;
          }
          break;
        case "labels":
          labelsVisible = !!msg.visible;
          var cb = document.getElementById("toggle-labels");
          if (cb) cb.checked = labelsVisible;
          break;
        case "highlight":
          if (msg.indices && msg.indices.length) {
            var idxSet = new Set(msg.indices);
            var highlighted = allPoints.map(function(p, i) {
              var vis = isClusterVisible(p.cluster);
              if (!vis) return null;
              var isHl = idxSet.has(i);
              return { x: p.x, y: p.y, z: p.z,
                        r: isHl ? 255 : p.r,
                        g: isHl ? 255 : p.g,
                        b: isHl ? 0   : p.b,
                        a: isHl ? 255 : 80,
                        title: p.title, cluster: p.cluster };
            }).filter(function(p) { return p !== null; });
            var dk3 = getDeck();
            if (dk3 && dk3.props && dk3.props.layers && dk3.props.layers.length) {
              var hlLayers = [dk3.props.layers[0].clone({ data: highlighted })];
              for (var hli = 1; hli < dk3.props.layers.length; hli++) hlLayers.push(dk3.props.layers[hli]);
              dk3.setProps({ layers: hlLayers });
            }
            // Restore after 3 seconds
            setTimeout(function() { rebuildLayer(); }, 3000);
          }
          break;
        case "set_mode":
          if (msg.mode === "2d" || msg.mode === "3d") {
            setViewMode(msg.mode);
          }
          break;
        case "set_theme":
          if (msg.theme === "light" || msg.theme === "dark") {
            setTheme(msg.theme);
          }
          break;
        case "zoom_to":
          var dkZ = getDeck();
          if (dkZ && dkZ.setProps) {
            var curVS = {};
            if (dkZ.viewManager) {
              try { curVS = JSON.parse(JSON.stringify(dkZ.viewManager.getViewState())); } catch(e) {}
            }
            curVS.transitionDuration = msg.transitionDuration || 500;
            if (msg.target) curVS.target = msg.target;
            if (typeof msg.zoom === "number") curVS.zoom = msg.zoom;
            if (viewMode === "2d") {
              curVS.rotationX = 90;
              curVS.rotationOrbit = 0;
              curVS.minRotationX = 90;
              curVS.maxRotationX = 90;
            }
            curVS.transitionInterpolator = new deck.LinearInterpolator(['target', 'zoom']);
            dkZ.setProps({ initialViewState: curVS });
          }
          break;
        case "set_view":
          // Set view state directly (orbit, pitch, zoom)
          var dkV = getDeck();
          if (dkV && dkV.setProps) {
            var newVS = {
              target: msg.target || [0, 0, 0],
              rotationOrbit: typeof msg.orbit === "number" ? msg.orbit : 0,
              rotationX: typeof msg.pitch === "number" ? msg.pitch : 15,
              zoom: typeof msg.zoom === "number" ? msg.zoom : defaultZoom,
              transitionDuration: msg.transitionDuration || 0
            };
            dkV.setProps({ initialViewState: newVS });
          }
          break;
        case "clear_highlight":
          setEdgeHighlight([]);
          rebuildLayer();
          break;
        case "highlight_edges":
          if (msg.clusters && msg.clusters.length) {
            setEdgeHighlight(msg.clusters);
            // Auto-enable edges and switch to 2D if needed
            if (!edgesVisible) {
              edgesVisible = true;
              var ecbH = document.getElementById("toggle-edges");
              if (ecbH) ecbH.checked = true;
            }
            if (viewMode !== "2d") setViewMode("2d");
            else rebuildLayer();
          }
          break;
        case "clear_edge_highlight":
          setEdgeHighlight([]);
          rebuildLayer();
          break;
        case "toggle_edges":
          if (typeof msg.visible === "boolean") {
            edgesVisible = msg.visible;
            var ecbW = document.getElementById("toggle-edges");
            if (ecbW) ecbW.checked = edgesVisible;
            rebuildLayer();
          }
          break;
        case "toggle_labels":
          if (typeof msg.visible === "boolean") {
            labelsVisible = msg.visible;
            var lcbW = document.getElementById("toggle-labels");
            if (lcbW) lcbW.checked = labelsVisible;
          }
          break;
        case "set_color_mode":
          if (msg.mode === "cluster" || msg.mode === "bucket") {
            applyColorMode(msg.mode);
          }
          break;
        case "set_level":
          if (_dendro && typeof msg.level === "number") {
            var targetK = msg.level;
            _dendroZoomK = targetK;  // sync so scroll zoom doesn't override
            if (targetK >= _naturalK) {
              applyDendrogramCut(0);
              var sl = document.getElementById("dendro-slider");
              if (sl) sl.value = "0";
              var kd = document.getElementById("dendro-k-display");
              if (kd) kd.textContent = "k=" + _naturalK;
            } else if (targetK <= 1) {
              var sl = document.getElementById("dendro-slider");
              if (sl) { applyDendrogramCut(parseFloat(sl.max)); sl.value = sl.max; }
              var kd = document.getElementById("dendro-k-display");
              if (kd) kd.textContent = "k=1";
            } else {
              var lo = 0, hi = _dendroZ[_dendroZ.length - 1][2] * 1.02;
              for (var _bi = 0; _bi < 50; _bi++) {
                var mid = (lo + hi) / 2;
                var c = cutDendrogram(mid);
                if (!c) break;
                if (c.k > targetK) lo = mid;
                else if (c.k < targetK) hi = mid;
                else { lo = mid; break; }
              }
              applyDendrogramCut(lo);
              var sl = document.getElementById("dendro-slider");
              if (sl) sl.value = String(lo);
              var kd = document.getElementById("dendro-k-display");
              if (kd) kd.textContent = "k=" + targetK;
            }
          } else {
            switchLevel(String(msg.level));
          }
          break;
        case "draw_circle":
          // Fit smooth ellipse around cluster's points
          var cid = msg.cluster;
          if (typeof cid === "number") {
            var cPts = [];
            for (var ci = 0; ci < allPoints.length; ci++) {
              if (allPoints[ci].cluster === cid) {
                cPts.push([allPoints[ci].x, allPoints[ci].y, allPoints[ci].z]);
              }
            }
            if (cPts.length >= 3) {
              // Pad outward by ~5% of data extent
              var xMin=1e9,xMax=-1e9,yMin=1e9,yMax=-1e9;
              for (var ci2=0;ci2<allPoints.length;ci2++) {
                if (allPoints[ci2].x<xMin) xMin=allPoints[ci2].x;
                if (allPoints[ci2].x>xMax) xMax=allPoints[ci2].x;
                if (allPoints[ci2].y<yMin) yMin=allPoints[ci2].y;
                if (allPoints[ci2].y>yMax) yMax=allPoints[ci2].y;
              }
              var extent = Math.max(xMax-xMin, yMax-yMin) || 1;
              var ellipsePts = fitEllipse(cPts, extent * 0.03);
              annotations.push({
                type: "circle",
                points: ellipsePts,
                _seed: Math.floor(Math.random() * 99999),
                color: msg.color || "rgba(255,230,0,0.35)",
                width: msg.width || 18
              });
            }
          }
          break;
        case "draw_path":
          if (msg.points && msg.points.length >= 2) {
            annotations.push({
              type: "path",
              points: msg.points,
              color: msg.color || "rgba(255,230,0,0.35)",
              width: msg.width || 18
            });
          }
          break;
        case "draw_clear":
          annotations.length = 0;
          break;
        case "tour":
          runTour();
          break;
        case "toggle_panel":
          if (typeof window.togglePanel === "function") window.togglePanel();
          break;
        case "hide_panel":
          if (!window.panelHidden && typeof window.togglePanel === "function") window.togglePanel();
          break;
        case "show_panel":
          if (window.panelHidden && typeof window.togglePanel === "function") window.togglePanel();
          break;
        case "get_state":
          // Return current view state via WebSocket
          var dkState = getDeck();
          var state = { session: sessionId };
          if (dkState && dkState.viewManager) {
            try {
              var vs = dkState.viewManager.getViewState();
              state.zoom = vs.zoom;
              state.rotationOrbit = vs.rotationOrbit;
              state.rotationX = vs.rotationX;
              state.target = vs.target;
              // Try to get camera position from viewport
              var vp = dkState.viewManager.getViewports()[0];
              if (vp && vp.cameraPosition) {
                state.cameraPosition = vp.cameraPosition;
              }
              // Also compute estimated camera direction from orbit angles
              var orbitRad = (vs.rotationOrbit || 0) * Math.PI / 180;
              var pitchRad = (vs.rotationX || 0) * Math.PI / 180;
              // Estimate camera direction (unit vector from target toward camera)
              state.camDirEstimate = [
                Math.sin(orbitRad) * Math.cos(pitchRad),
                Math.cos(orbitRad) * Math.cos(pitchRad),
                Math.sin(pitchRad)
              ];
            } catch(e) { state.error = e.message; }
          }
          state.viewMode = viewMode;
          state.tourRunning = tourRunning;
          var _cont = document.getElementById("deckgl-wrapper") || document.getElementById("deck-container");
          state.viewport = {
            width: _cont ? _cont.clientWidth : window.innerWidth,
            height: _cont ? _cont.clientHeight : window.innerHeight,
            dpr: window.devicePixelRatio
          };
          state.defaultZoom = defaultZoom;
          state.maxExtent = maxExtent;
          try {
            state.pointStats = {
              xMin: Math.min.apply(null, xVals), xMax: Math.max.apply(null, xVals),
              yMin: Math.min.apply(null, yVals), yMax: Math.max.apply(null, yVals),
              xMean: xStat.mean, yMean: yStat.mean,
              xStd: xStat.std, yStd: yStat.std,
              count: xVals.length
            };
          } catch(e) { state.pointStats = { error: e.message }; }
          if (tourCentroid) {
            state.tourCentroid = tourCentroid;
            // Compute alignment errors to help debug the orbit formula
            var cx = tourCentroid[0], cy = tourCentroid[1];
            var centroidAngle = Math.atan2(cy, cx) * 180 / Math.PI;
            var orbit = state.rotationOrbit || 0;
            function normAngle(a) { while(a>180) a-=360; while(a<-180) a+=360; return a; }
            state.debug = {
              centroidAngle: centroidAngle,
              orbitAngle: orbit,
              // Different interpretations of how orbit maps to camera direction
              errA: normAngle(centroidAngle - (90 - orbit)),  // camera at 90-orbit from +X
              errB: normAngle(centroidAngle - (orbit - 90)),  // camera at orbit-90 from +X
              errC: normAngle(centroidAngle - (-orbit)),      // camera at -orbit from +X
              errD: normAngle(centroidAngle - orbit),         // camera at orbit from +X
              errE: normAngle(centroidAngle - (orbit + 90)),  // camera at orbit+90 from +X
              errF: normAngle(centroidAngle - (-orbit + 180)) // camera at -orbit+180 from +X
            };
          }
          ws.send(JSON.stringify({ cmd: "state_response", state: state }));
          break;
        case "reload":
          // Save state before reload
          var dkR = getDeck();
          var savedState = {
            theme: currentTheme,
            viewMode: viewMode,
            labelsVisible: labelsVisible,
            edgesVisible: edgesVisible
          };
          // Save camera viewState
          if (dkR && dkR.viewManager) {
            try {
              savedState.viewState = JSON.parse(JSON.stringify(
                dkR.viewManager.getViewState()
              ));
            } catch(e) {}
          }
          // Save point size from slider
          var psSlider = document.getElementById("point-size");
          if (psSlider) savedState.pointSize = parseFloat(psSlider.value);
          // Save annotations
          if (annotations.length > 0) savedState.annotations = annotations;
          sessionStorage.setItem("dyf_viz_state", JSON.stringify(savedState));
          setTimeout(function() { location.reload(); }, 100);
          break;
      }
    };
    ws.onclose = function() { setTimeout(connectWS, 2000); };
    ws.onerror = function() {};  // suppress console errors when server not running
  })();

  // ── Restore state after hot-reload ──────────────────────────────────
  (function restoreState() {
    var raw = sessionStorage.getItem("dyf_viz_state");
    if (!raw) return;
    sessionStorage.removeItem("dyf_viz_state");
    var s;
    try { s = JSON.parse(raw); } catch(e) { return; }

    // Restore theme immediately
    if (s.theme && s.theme !== currentTheme) {
      setTheme(s.theme);
    }

    // Restore view mode immediately
    if (s.viewMode && s.viewMode !== viewMode) {
      setViewMode(s.viewMode);
    }

    // Restore label/edge visibility
    if (typeof s.labelsVisible === "boolean") {
      labelsVisible = s.labelsVisible;
      var lcb = document.getElementById("toggle-labels");
      if (lcb) lcb.checked = labelsVisible;
    }
    if (typeof s.edgesVisible === "boolean") {
      edgesVisible = s.edgesVisible;
      var ecb = document.getElementById("toggle-edges");
      if (ecb) ecb.checked = edgesVisible;
      rebuildLayer();
    }

    // Restore point size
    if (typeof s.pointSize === "number") {
      var psEl = document.getElementById("point-size");
      var pvEl = document.getElementById("ps-val");
      if (psEl) {
        psEl.value = s.pointSize;
        psEl.dispatchEvent(new Event("input"));
      }
      if (pvEl) pvEl.textContent = s.pointSize;
    }

    // Restore annotations
    if (s.annotations && s.annotations.length) {
      annotations = s.annotations;
    }

    // Restore camera viewState after deck.gl finishes initializing
    if (s.viewState) {
      setTimeout(function() {
        var dk = getDeck();
        if (dk && dk.setProps) {
          dk.setProps({ initialViewState: s.viewState });
        }
      }, 1500);
    }
  })();
})();
