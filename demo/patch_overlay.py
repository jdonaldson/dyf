#!/usr/bin/env python3
"""Re-render the JS overlay template in existing dyfviz HTML files.

Extracts embedded data blobs from the HTML, re-evaluates the overlay
template from dyfviz.py with those same values, and writes the patched file.
This avoids the full rebuild (UMAP, clustering, labeling, TTS).

Usage:
    python demo/patch_overlay.py demo/rog_3d_birch_clusters.html
    python demo/patch_overlay.py demo/rog_3d_*.html   # patch multiple
"""

import json
import re
import sys
from pathlib import Path


def extract_between(html, start_marker, end_marker):
    """Extract text between two markers."""
    i = html.find(start_marker)
    j = html.find(end_marker)
    if i < 0 or j < 0:
        return None
    return html[i + len(start_marker):j]


def extract_b64_blob(overlay, prefix):
    """Extract a base64 blob following a known prefix string."""
    idx = overlay.find(prefix)
    if idx < 0:
        return ""
    start = idx + len(prefix)
    # Find the closing quote
    quote_char = overlay[start - 1] if start > 0 else '"'
    # The blob is inside b64toBytes("...") — find the closing ")
    end = overlay.find('"', start)
    if end < 0:
        end = overlay.find("'", start)
    if end < 0:
        return ""
    return overlay[start:end]


def extract_json_var(overlay, var_name):
    """Extract a JSON assignment like 'var foo = {...};' or 'var foo = [...]'."""
    pattern = rf'var {var_name}\s*=\s*'
    m = re.search(pattern, overlay)
    if not m:
        return "{}"
    start = m.end()
    # Find the balanced JSON object/array
    depth = 0
    i = start
    in_string = False
    escape = False
    opener = overlay[i] if i < len(overlay) else '{'
    while i < len(overlay):
        ch = overlay[i]
        if escape:
            escape = False
            i += 1
            continue
        if ch == '\\':
            escape = True
            i += 1
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch in '{[':
                depth += 1
            elif ch in '}]':
                depth -= 1
                if depth == 0:
                    return overlay[start:i + 1]
        i += 1
    return "{}"


def patch_file(html_path):
    html = Path(html_path).read_text()

    overlay = extract_between(html, "<!-- DYF_OVERLAY_START -->", "<!-- DYF_OVERLAY_END -->")
    if overlay is None:
        print(f"  No overlay markers found in {html_path}, skipping")
        return False

    # Extract data blobs
    points_ipc_b64 = extract_b64_blob(overlay, 'b64toBytes("')
    # Find 2d and 3d edge blobs - they appear in loadEdges calls
    edge_calls = list(re.finditer(r'loadEdges\("([^"]*?)"\)', overlay))
    edges_2d_ipc_b64 = edge_calls[0].group(1) if len(edge_calls) > 0 else ""
    edges_3d_ipc_b64 = edge_calls[1].group(1) if len(edge_calls) > 1 else ""

    label_json = extract_json_var(overlay, "labels")
    levels_json = extract_json_var(overlay, "labelLevels")
    edge_pairs_json = extract_json_var(overlay, "edgePairs")
    narration_json = extract_json_var(overlay, "tourNarration")
    callouts_json = extract_json_var(overlay, "tourCallouts")
    audio_json = extract_json_var(overlay, "tourAudio")

    # Extract title from header div
    title_m = re.search(r'font-weight:700[^"]*">(.*?)</div>', html)
    title_str = title_m.group(1) if title_m else ""

    # Extract logo HTML
    logo_m = re.search(r'(<img src="data:image/png;base64,[^"]*"[^>]*>)', html)
    client_logo_html = logo_m.group(1) if logo_m else ""

    # Extract tour_title from the intro textContent assignment
    tour_title_m = re.search(r'tourLabelEl\.textContent\s*=\s*"([^"]*)"', overlay)
    tour_title = tour_title_m.group(1) if tour_title_m else None
    if tour_title == "Welcome":
        tour_title = None

    # Now import and render the overlay template
    sys.path.insert(0, str(Path(__file__).parent))
    import dyfviz
    # Build the overlay by calling the template section of build_pydeck
    # We need to render the f-string with extracted values
    # Import the function and call it with the extracted data
    from dyfviz import build_pydeck_overlay
    new_overlay = build_pydeck_overlay(
        points_ipc_b64=points_ipc_b64,
        edges_2d_ipc_b64=edges_2d_ipc_b64,
        edges_3d_ipc_b64=edges_3d_ipc_b64,
        label_json=label_json,
        levels_json=levels_json,
        edge_pairs_json=edge_pairs_json,
        narration_json=narration_json,
        callouts_json=callouts_json,
        audio_json=audio_json,
        title_str=title_str,
        client_logo_html=client_logo_html,
        tour_title=tour_title,
    )

    # Replace overlay in HTML
    start_marker = "<!-- DYF_OVERLAY_START -->"
    end_marker = "<!-- DYF_OVERLAY_END -->"
    i = html.find(start_marker)
    j = html.find(end_marker) + len(end_marker)
    new_html = html[:i] + new_overlay + html[j:]

    Path(html_path).write_text(new_html)
    print(f"  Patched {html_path}")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo/patch_overlay.py <html_file> [<html_file> ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        patch_file(path)
