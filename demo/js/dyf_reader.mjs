/**
 * dyf_reader.mjs — Browser-based .dyf file parser
 *
 * Reads .dyf binary format: DYF1 header (16 bytes) + FlatBuffers tree + 4KB-aligned Arrow IPC batches.
 * Hand-rolls minimal FlatBuffers parsing (no external dependency).
 * Uses apache-arrow CDN for Arrow IPC batch decoding.
 */

import { tableFromIPC } from "https://cdn.jsdelivr.net/npm/apache-arrow@18.1.0/+esm";

// ── Constants ────────────────────────────────────────────────────────────────
const MAGIC = new Uint8Array([0x44, 0x59, 0x46, 0x31]); // "DYF1"
const HEADER_SIZE = 16;
const PAGE_SIZE = 4096;

// ── Minimal FlatBuffers reader ───────────────────────────────────────────────
// Reads FlatBuffers tables/vectors using DataView, no external library needed.
// vtable layout: [vtable_size:u16, table_size:u16, field0_offset:u16, field1_offset:u16, ...]

class FBTable {
  constructor(buf, pos) {
    this.buf = buf;
    this.dv = new DataView(buf);
    this.pos = pos;
    // vtable is at pos - soffset_to_vtable
    const vtableOffset = this.dv.getInt32(pos, true);
    this.vtable = pos - vtableOffset;
    this.vtableSize = this.dv.getUint16(this.vtable, true);
  }

  // Get field offset relative to table start (0 = not present)
  _fieldOffset(slotIndex) {
    const vtableEntry = 4 + slotIndex * 2; // skip vtable_size + table_size
    if (vtableEntry >= this.vtableSize) return 0;
    return this.dv.getUint16(this.vtable + vtableEntry, true);
  }

  getUint8(slot, def = 0) {
    const off = this._fieldOffset(slot);
    return off ? this.dv.getUint8(this.pos + off) : def;
  }

  getUint16(slot, def = 0) {
    const off = this._fieldOffset(slot);
    return off ? this.dv.getUint16(this.pos + off, true) : def;
  }

  getUint32(slot, def = 0) {
    const off = this._fieldOffset(slot);
    return off ? this.dv.getUint32(this.pos + off, true) : def;
  }

  getInt32(slot, def = 0) {
    const off = this._fieldOffset(slot);
    return off ? this.dv.getInt32(this.pos + off, true) : def;
  }

  getUint64(slot, def = 0) {
    const off = this._fieldOffset(slot);
    if (!off) return def;
    // Read as BigUint64 then convert to Number (safe for files < 2^53 bytes)
    return Number(this.dv.getBigUint64(this.pos + off, true));
  }

  getString(slot) {
    const off = this._fieldOffset(slot);
    if (!off) return null;
    // String: offset to string data (uoffset relative to field position)
    const strOffset = this.pos + off + this.dv.getUint32(this.pos + off, true);
    const strLen = this.dv.getUint32(strOffset, true);
    const bytes = new Uint8Array(this.buf, strOffset + 4, strLen);
    return new TextDecoder().decode(bytes);
  }

  getTable(slot) {
    const off = this._fieldOffset(slot);
    if (!off) return null;
    const tableOffset = this.pos + off + this.dv.getUint32(this.pos + off, true);
    return new FBTable(this.buf, tableOffset);
  }

  getVectorLength(slot) {
    const off = this._fieldOffset(slot);
    if (!off) return 0;
    const vecOffset = this.pos + off + this.dv.getUint32(this.pos + off, true);
    return this.dv.getUint32(vecOffset, true);
  }

  getVectorTableAt(slot, index) {
    const off = this._fieldOffset(slot);
    if (!off) return null;
    const vecOffset = this.pos + off + this.dv.getUint32(this.pos + off, true);
    // Vector: [length:u32, elem0:uoffset, elem1:uoffset, ...]
    const elemPos = vecOffset + 4 + index * 4;
    const tablePos = elemPos + this.dv.getUint32(elemPos, true);
    return new FBTable(this.buf, tablePos);
  }
}

function getRootTable(buf) {
  const dv = new DataView(buf);
  const rootOffset = dv.getUint32(0, true);
  return new FBTable(buf, rootOffset);
}

// ── .dyf Index Parsing ──────────────────────────────────────────────────────
// Schema slots for Index table (from dyf_index.fbs / Index.py):
//   0: version (string)
//   1: embedding_dim (uint16)
//   2: total_items (uint64)
//   3: num_leaves (uint32)
//   4: root (uint32)
//   5: nodes (vector of Node tables)
//   6: batches (vector of BatchDescriptor tables)
//   7: build_params (table)
//   8: metadata (vector of KeyValue tables)

// BatchDescriptor slots: 0: offset (uint64), 1: length (uint64), 2: num_rows (uint32)
// BuildParams slots: 0: max_depth (u8), 1: num_bits (u8), 2: min_leaf_size (u32),
//                    3: seed (u64), 4: quantization (string), 5: compression (string)
// KeyValue slots: 0: key (string), 1: value (string)

function parseIndex(fbBuf) {
  const root = getRootTable(fbBuf);

  const version = root.getString(0);
  const embeddingDim = root.getUint16(1);
  const totalItems = root.getUint64(2);
  const numLeaves = root.getUint32(3);
  const rootNodeId = root.getUint32(4);

  // Parse batches
  const numBatches = root.getVectorLength(6);
  const batches = [];
  for (let i = 0; i < numBatches; i++) {
    const bd = root.getVectorTableAt(6, i);
    batches.push({
      offset: bd.getUint64(0),
      length: bd.getUint64(1),
      numRows: bd.getUint32(2),
    });
  }

  // Parse build params
  let buildParams = { compression: "none", quantization: "float32" };
  const bp = root.getTable(7);
  if (bp) {
    buildParams = {
      maxDepth: bp.getUint8(0),
      numBits: bp.getUint8(1),
      minLeafSize: bp.getUint32(2),
      seed: bp.getUint64(3),
      quantization: bp.getString(4) || "float32",
      compression: bp.getString(5) || "none",
    };
  }

  // Parse metadata
  const metadata = new Map();
  const numMeta = root.getVectorLength(8);
  for (let i = 0; i < numMeta; i++) {
    const kv = root.getVectorTableAt(8, i);
    const key = kv.getString(0);
    const value = kv.getString(1);
    if (key != null && value != null) {
      metadata.set(key, value);
    }
  }

  return {
    version,
    embeddingDim,
    totalItems,
    numLeaves,
    rootNodeId,
    batches,
    buildParams,
    metadata,
  };
}

// ── Main loader ──────────────────────────────────────────────────────────────

/**
 * Load and parse a .dyf file from a URL.
 *
 * Returns: {
 *   totalItems: number,
 *   fields: { [name]: Float32Array | Int32Array | string[] },
 *   metadata: Map<string, string>,
 *   clusterLevels: number[],           // sorted available cluster levels
 *   clusterNames: { [level]: { [cid]: string } },
 *   clusterCentroids: { [level]: { [cid]: [x,y] | [x,y,z] } },
 *   buildParams: object,
 * }
 */
export async function loadDyf(url, onProgress) {
  // Fetch the file
  if (onProgress) onProgress("Downloading...");
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);

  // Read with progress if possible
  let buffer;
  if (response.body && onProgress) {
    const contentLength = +response.headers.get("Content-Length") || 0;
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (contentLength > 0) {
        const pct = ((received / contentLength) * 100).toFixed(0);
        onProgress(`Downloading... ${pct}% (${(received / 1e6).toFixed(1)} MB)`);
      }
    }
    // Concatenate chunks
    const full = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      full.set(chunk, offset);
      offset += chunk.length;
    }
    buffer = full.buffer;
  } else {
    buffer = await response.arrayBuffer();
  }

  if (onProgress) onProgress("Parsing header...");

  // Validate magic
  const magic = new Uint8Array(buffer, 0, 4);
  for (let i = 0; i < 4; i++) {
    if (magic[i] !== MAGIC[i]) {
      throw new Error(`Invalid .dyf magic: expected DYF1, got ${String.fromCharCode(...magic)}`);
    }
  }

  // Read fb_size (uint64 LE at offset 4)
  const dv = new DataView(buffer);
  const fbSize = Number(dv.getBigUint64(4, true));

  // Parse FlatBuffers section
  const fbStart = HEADER_SIZE;
  const fbBuf = buffer.slice(fbStart, fbStart + fbSize);
  const index = parseIndex(fbBuf);

  // Compute Arrow section start (4KB aligned)
  const totalHeaderFb = HEADER_SIZE + fbSize;
  const arrowStart = totalHeaderFb + ((PAGE_SIZE - (totalHeaderFb % PAGE_SIZE)) % PAGE_SIZE);

  if (onProgress) onProgress("Parsing stored fields...");

  // Parse stored field schema from metadata
  const sfJson = index.metadata.get("stored_fields");
  const storedFieldSchema = sfJson ? JSON.parse(sfJson) : {};
  const sfNames = Object.keys(storedFieldSchema);

  // Skip if no stored fields (nothing to visualize)
  if (sfNames.length === 0) {
    return {
      totalItems: index.totalItems,
      fields: {},
      metadata: index.metadata,
      clusterLevels: [],
      clusterNames: {},
      clusterCentroids: {},
      buildParams: index.buildParams,
    };
  }

  // Initialize field arrays
  const n = index.totalItems;
  const fields = {};
  for (const fname of sfNames) {
    const tname = storedFieldSchema[fname];
    if (tname === "utf8" || tname === "binary") {
      fields[fname] = new Array(n).fill(null);
    } else if (tname === "float32") {
      fields[fname] = new Float32Array(n);
    } else if (tname === "float64") {
      fields[fname] = new Float64Array(n);
    } else if (tname === "int32") {
      fields[fname] = new Int32Array(n);
    } else if (tname === "int64") {
      // Use Float64Array for int64 (JS has no Int64Array)
      fields[fname] = new Float64Array(n);
    } else {
      fields[fname] = new Array(n).fill(null);
    }
  }

  // Read all Arrow IPC batches and scatter by item_index
  const numBatches = index.batches.length;
  let zstdWarned = false;

  for (let bi = 0; bi < numBatches; bi++) {
    if (onProgress && bi % 10 === 0) {
      onProgress(`Reading batches... ${bi}/${numBatches}`);
    }
    const bd = index.batches[bi];
    const batchStart = arrowStart + bd.offset;
    const batchEnd = batchStart + bd.length;
    const batchBytes = new Uint8Array(buffer, batchStart, bd.length);

    let table;
    try {
      table = tableFromIPC(batchBytes);
    } catch (e) {
      if (!zstdWarned && index.buildParams.compression === "zstd") {
        console.warn(
          "[dyf_reader] Arrow IPC batch decode failed (likely zstd body compression). " +
          "Try apache-arrow v19+ or export an uncompressed .dyf.",
          e.message
        );
        zstdWarned = true;
      }
      throw new Error(
        `Failed to decode Arrow IPC batch ${bi}: ${e.message}. ` +
        `Compression: ${index.buildParams.compression}`
      );
    }

    // Get item_index column for scatter
    const itemIndexCol = table.getChild("item_index");
    if (!itemIndexCol) {
      console.warn(`Batch ${bi}: no item_index column, skipping`);
      continue;
    }
    const itemIndices = itemIndexCol.toArray();

    // Extract stored fields (skip embedding column — not needed for viz)
    for (const fname of sfNames) {
      const col = table.getChild(fname);
      if (!col) continue;
      const tname = storedFieldSchema[fname];

      if (tname === "utf8" || tname === "binary") {
        for (let i = 0; i < itemIndices.length; i++) {
          fields[fname][itemIndices[i]] = col.get(i);
        }
      } else {
        const values = col.toArray();
        for (let i = 0; i < itemIndices.length; i++) {
          fields[fname][itemIndices[i]] = values[i];
        }
      }
    }
  }

  if (onProgress) onProgress("Processing metadata...");

  // ── Parse dendrogram metadata (new format) ──────────────────────────────
  let dendrogram = null;
  const dendroJson = index.metadata.get("louvain_dendrogram");
  const leafCommJson = index.metadata.get("louvain_leaf_communities");
  const leafItemJson = index.metadata.get("leaf_item_map");

  if (dendroJson && leafCommJson && leafItemJson) {
    try {
      const dendroData = JSON.parse(dendroJson);
      const leafCommData = JSON.parse(leafCommJson);
      const leafItemData = JSON.parse(leafItemJson);
      dendrogram = {
        Z: dendroData.Z,                          // linkage matrix rows
        communityNames: dendroData.community_names,
        communityColors: dendroData.community_colors,
        communityCentroids: dendroData.community_centroids,
        communitySizes: dendroData.community_sizes,
        communityCohesion: dendroData.community_cohesion || null,
        leafToCommunity: leafCommData.leaf_to_community,
        naturalK: leafCommData.natural_k,
        resolution: leafCommData.resolution,
        leafItemMap: leafItemData,                 // {leaf_idx: [item_idx, ...]}
        communityIds: fields.community_id || null, // per-point stored field (post-reassignment)
      };
    } catch (e) {
      console.warn("[dyf_reader] Error parsing dendrogram metadata:", e);
    }
  }

  if (onProgress) onProgress("Done");

  return {
    totalItems: index.totalItems,
    fields,
    metadata: index.metadata,
    dendrogram,
    buildParams: index.buildParams,
  };
}
