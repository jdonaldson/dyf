"""Shared Ollama LLM helpers for the enrich pipeline."""

import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def _make_domain_context(domain):
    """Build prompt template variables from a domain string."""
    if not domain:
        domain = "items"
    return {
        "domain": domain,
        "items": domain,
        "landscape": f"{domain} landscape",
    }


def _call_ollama(model, prompt, timeout=300):
    """Call Ollama HTTP API and return text response."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", "").strip()
    except Exception as e:
        logger.warning("Ollama error: %s", e)
        return ""


def _call_ollama_chat(prompt, model="gpt-oss:20b", timeout=30):
    """Call Ollama via HTTP API. Returns response text or None."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
            return body.get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.warning("Ollama chat error: %s", e)
        return None
