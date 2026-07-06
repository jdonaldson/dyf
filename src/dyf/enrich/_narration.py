"""Tour narration generation for the enrich pipeline."""

import re
from collections import defaultdict

import numpy as np

from ._labeling import _sample_spatial
from ._ollama import _call_ollama_chat, _make_domain_context


def _number_to_words(n):
    """Convert a small integer to English words."""
    words = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
        20: "twenty",
        25: "twenty-five",
        30: "thirty",
        40: "forty",
        50: "fifty",
    }
    return words.get(n, str(n))


def _approx_number_words(n):
    """Convert a count to approximate spoken form."""
    if n < 100:
        return f"{n}"
    elif n < 1000:
        hundreds = round(n / 100) * 100
        return f"about {hundreds}"
    elif n < 10000:
        thousands = round(n / 100) * 100
        return f"about {thousands:,}"
    else:
        thousands = round(n / 1000) * 1000
        return f"about {thousands:,}"


def _build_narration_prompts(cluster_names, cluster_points, titles, coords, sorted_cids, model, domain):
    """Build per-cluster LLM prompts for narration.

    Returns list of (cid, prompt, sample_titles) task tuples.
    """
    tasks = []
    for cid in sorted_cids:
        name = cluster_names[cid]
        pts = cluster_points.get(cid, [])
        n_pts = len(pts)
        n_approx = _approx_number_words(n_pts)

        sample_idx = _sample_spatial(pts, coords, 20)
        seen = set()
        sample_titles = []
        for idx in sample_idx:
            t = titles[idx] if hasattr(titles, "__getitem__") else str(idx)
            if t not in seen:
                seen.add(t)
                sample_titles.append(t)
                if len(sample_titles) >= 12:
                    break

        dc = _make_domain_context(domain)
        items_str = "\n".join(f"  - {t}" for t in sample_titles)
        prompt = (
            f"You are narrating a guided tour of a "
            f"{dc['landscape']} for a general audience.\n\n"
            f'Cluster name: "{name}"\n'
            f"Size: {n_approx} {dc['items']}\n\n"
            f"Sample {dc['items']}:\n{items_str}\n\n"
            f"Write 2-3 sentences that:\n"
            f'1. Start with "{name}."\n'
            f"2. Explain in plain language what this category of "
            f"{dc['items']} represents\n"
            f"3. Say roughly how many {dc['items']} are in this "
            f'group (use "{n_approx}")\n\n'
            f"Style: calm British documentary narrator. "
            f"Written for text-to-speech — spell out all numbers, "
            f"no abbreviations, no special characters, no quotes. "
            f"Do NOT list raw codes or model numbers.\n"
        )
        tasks.append((cid, prompt, sample_titles))

    return tasks


def _build_intro_outro(cluster_names, sorted_cids, total_pts, title):
    """Generate intro and outro narration text.

    Returns dict with 'intro' and 'outro' keys.
    """
    n_clusters = len(cluster_names)
    n_words = _number_to_words(n_clusters)
    top3 = [cluster_names[c] for c in sorted_cids[:3]]
    total_words = _approx_number_words(total_pts)
    display_title = title or "Embedding Landscape"
    intro = (
        f"{display_title}. {total_words} items organized into "
        f"{n_words} clusters. The largest regions are {top3[0]}"
        + (f", {top3[1]}" if len(top3) > 1 else "")
        + (f", and {top3[2]}" if len(top3) > 2 else "")
        + ". Let's take a look."
    )
    outro = (
        "That completes our tour. Clusters nearby share deeper "
        "similarities, and the bridges between them trace where one "
        "category shades into the next."
    )
    return {"intro": intro, "outro": outro}


def _generate_narration(cluster_names, titles, labels, coords, model="gpt-oss:20b", title=None, domain=None):
    """Generate tour narration using Ollama, with sample-title fallback."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    label_arr = np.asarray(labels)
    cluster_points = defaultdict(list)
    for i, cid in enumerate(label_arr):
        cluster_points[int(cid)].append(i)

    total_pts = sum(len(v) for v in cluster_points.values())
    sorted_cids = sorted(cluster_names.keys(), key=lambda c: len(cluster_points.get(c, [])), reverse=True)

    # Check Ollama availability
    ollama_ok = _call_ollama_chat("Say OK.", model=model, timeout=30) is not None
    if ollama_ok:
        print(f"\n  Generating narration via Ollama ({model})...")
    else:
        print("\n  Ollama not available — using sample-title narration")

    narration = {}

    if ollama_ok:
        tasks = _build_narration_prompts(cluster_names, cluster_points, titles, coords, sorted_cids, model, domain)
    else:
        tasks = []
        for cid in sorted_cids:
            name = cluster_names[cid]
            pts = cluster_points.get(cid, [])
            n_approx = _approx_number_words(len(pts))
            dc = _make_domain_context(domain)
            narration[cid] = f"{name}. {n_approx} {dc['items']} in this category."

    if ollama_ok and tasks:
        completed = 0

        def _do_one(task):
            cid, prompt, samples = task
            text = _call_ollama_chat(prompt, model=model, timeout=120)
            if text:
                text = re.sub(r"\s+", " ", text).strip().strip("\"'")
                return cid, text
            name = cluster_names[cid]
            dc = _make_domain_context(domain)
            n_approx = _approx_number_words(len(cluster_points.get(cid, [])))
            return cid, f"{name}. {n_approx} {dc['items']} in this category."

        n_workers = min(2, len(tasks))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_do_one, t): t for t in tasks}
            for future in as_completed(futures):
                cid, text = future.result()
                narration[cid] = text
                completed += 1
                if completed % 5 == 0 or completed == len(tasks):
                    print(f"    Narrated {completed}/{len(tasks)} clusters...", flush=True)

    # Intro and outro
    bookends = _build_intro_outro(cluster_names, sorted_cids, total_pts, title)
    narration["intro"] = bookends["intro"]
    narration["outro"] = bookends["outro"]

    # Preview
    for cid in sorted_cids[:3]:
        print(f"    [{cid:2d}] {narration[cid][:80]}...")

    return narration
