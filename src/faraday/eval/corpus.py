"""Fetch the themed eval corpus from Wikipedia (dev-time, network) into committed
plain-text files + a SOURCES.md attribution. Uses the MediaWiki action API's
plaintext extracts (no HTML/markup cleaning needed). Pure helpers are unit-tested;
fetch_all() is a one-shot live script.
"""
from __future__ import annotations

import re
from pathlib import Path

import httpx

from faraday.eval import config

API = "https://en.wikipedia.org/w/api.php"


def slugify(title: str) -> str:
    s = title.lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", s)


def parse_pages(payload: dict) -> tuple[str, str]:
    """Pull (plaintext extract, canonical url) out of an action-API response."""
    pages = payload["query"]["pages"]
    page = next(iter(pages.values()))
    if "extract" not in page or not page["extract"].strip():
        raise ValueError(f"no extract for page: {page.get('title')!r}")
    return page["extract"], page.get("fullurl", "")


def fetch_extract(client: httpx.Client, title: str) -> tuple[str, str]:
    resp = client.get(API, params={
        "action": "query", "format": "json", "prop": "extracts|info",
        "titles": title, "explaintext": 1, "redirects": 1, "inprop": "url",
    })
    resp.raise_for_status()
    return parse_pages(resp.json())


def fetch_all(out_dir: Path | None = None) -> Path:
    """Fetch every CORPUS_TITLES article -> out_dir/<slug>.txt + SOURCES.md."""
    out_dir = out_dir or config.CORPUS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = ["# Eval corpus sources",
               "", "Wikipedia articles, text licensed CC BY-SA 4.0. "
               "Fetched via the MediaWiki action API (plaintext extracts).", ""]
    with httpx.Client(timeout=30.0, headers={"User-Agent": "faraday-eval/0.1"}) as client:
        for title in config.CORPUS_TITLES:
            text, url = fetch_extract(client, title)
            (out_dir / f"{slugify(title)}.txt").write_text(text, encoding="utf-8")
            sources.append(f"- **{title}** — {url}")
            print(f"  {slugify(title)}.txt  ({len(text)} chars)", flush=True)
    (out_dir / "SOURCES.md").write_text("\n".join(sources) + "\n", encoding="utf-8")
    return out_dir


if __name__ == "__main__":
    fetch_all()
