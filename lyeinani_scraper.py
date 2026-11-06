#!/usr/bin/env python3
"""
Lyeinani site scraper -> one PDF per page (knowledge base for a RAG chatbot).

How it finds pages:
  1. Reads the Shopify sitemap (/sitemap.xml and its child sitemaps:
     products, collections, pages, blogs).
  2. Crawls internal links from every rendered page to catch anything the
     sitemap misses.

Each page is rendered in headless Chromium, so JavaScript-loaded content
(prices, variants, product details, app widgets) is captured, then saved
as a PDF. A manifest.json records URL, title and file for every PDF. That
metadata is useful later for citations in the RAG step.

Setup:
    pip install playwright requests
    playwright install chromium

Run:
    python lyeinani_scraper.py
    python lyeinani_scraper.py --max-pages 50 --delay 2 --out site_pdfs
    python lyeinani_scraper.py --no-crawl          # sitemap pages only
"""

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

DEFAULT_BASE = "https://www.lyeinani.com"
USER_AGENT = "LyeinaniKB-Scraper/1.0 (site owner; RAG knowledge base)"
SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

# Shopify paths that are private, transactional, or have no knowledge-base value.
SKIP_PATH_PREFIXES = (
    "/cart", "/checkout", "/checkouts", "/account", "/orders", "/search",
    "/password", "/admin", "/cdn/", "/apps/", "/services/", "/tools/",
    "/challenge", "/recommendations", "/wpm@",
)
SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".pdf", ".zip",
    ".mp4", ".mov", ".webm", ".mp3", ".css", ".js", ".json", ".xml", ".txt",
)

# Hide cookie banners / newsletter popups so they don't cover the PDF content.
HIDE_OVERLAYS_CSS = """
#shopify-pc__banner,
[class*="klaviyo-form"],
[id*="klaviyo"],
div[role="dialog"][aria-modal="true"] { display: none !important; }
"""


# ----------------------------------------------------------------- helpers
def normalize(url, base_netloc):
    """Canonicalize a URL. Returns None if it's off-site or should be skipped."""
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return None
    if p.netloc.lower().removeprefix("www.") != base_netloc.removeprefix("www."):
        return None

    path = p.path or "/"
    # Shopify serves one product at /collections/<c>/products/<p> AND /products/<p>.
    m = re.match(r"^/collections/[^/]+(/products/.+)$", path)
    if m:
        path = m.group(1)
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    low = path.lower()
    if low.startswith(SKIP_PATH_PREFIXES) or low.endswith(SKIP_EXTENSIONS):
        return None
    # Query strings (?variant=, ?page=, sort params) are dropped on purpose:
    # they create near-duplicate pages. Products are still found via the sitemap.
    return urlunparse(("https", base_netloc, path, "", "", ""))


def sitemap_urls(session, base, base_netloc):
    """Walk the sitemap index and return all page URLs it lists."""
    found, seen = [], set()
    todo = deque([urljoin(base, "/sitemap.xml")])
    while todo:
        sm = todo.popleft()
        if sm in seen:
            continue
        seen.add(sm)
        try:
            r = session.get(sm, timeout=20)
            r.raise_for_status()
            root = ET.fromstring(r.content)
        except Exception as e:
            print(f"  [sitemap] could not read {sm}: {e}")
            continue

        if root.tag.endswith("sitemapindex"):
            for loc in root.iter(f"{SITEMAP_NS}loc"):
                if loc.text:
                    todo.append(loc.text.strip())
        else:
            for url_el in root.iter(f"{SITEMAP_NS}url"):
                loc = url_el.find(f"{SITEMAP_NS}loc")  # ignores image:loc entries
                if loc is not None and loc.text:
                    u = normalize(loc.text.strip(), base_netloc)
                    if u:
                        found.append(u)
    return list(dict.fromkeys(found))  # de-dupe, keep order


def load_robots(session, base):
    rp = robotparser.RobotFileParser()
    try:
        r = session.get(urljoin(base, "/robots.txt"), timeout=20)
        rp.parse(r.text.splitlines() if r.ok else [])
    except Exception:
        rp.parse([])
    return rp


def pdf_name(url, used):
    """Readable, unique filename from the URL path, e.g. products__nana-tee.pdf"""
    path = urlparse(url).path.strip("/") or "home"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", path.replace("/", "__"))[:150]
    name, n = slug, 2
    while name in used:
        name = f"{slug}_{n}"
        n += 1
    used.add(name)
    return name + ".pdf"


def render_page(page, url, out_path, timeout_ms):
    """Load a page fully, save it as PDF, return (status, final_url, title, links)."""
    resp = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except PWTimeout:
        pass  # Shopify apps often keep connections open; content is loaded by now

    # Scroll down in steps so lazy-loaded images and sections render.
    page.evaluate(
        """async () => {
            for (let y = 0, i = 0; y < document.body.scrollHeight && i < 60; y += 600, i++) {
                window.scrollTo(0, y);
                await new Promise(r => setTimeout(r, 150));
            }
            window.scrollTo(0, 0);
        }"""
    )
    page.add_style_tag(content=HIDE_OVERLAYS_CSS)
    page.wait_for_timeout(800)

    final_url = page.url
    title = page.title()
    links = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")

    page.emulate_media(media="screen")  # PDF looks like the live site, not print CSS
    page.pdf(
        path=str(out_path),
        format="Letter",
        print_background=True,
        margin={"top": "0.4in", "bottom": "0.4in", "left": "0.4in", "right": "0.4in"},
    )
    return (resp.status if resp else None), final_url, title, links


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Scrape a site into one PDF per page.")
    ap.add_argument("--base", default=DEFAULT_BASE, help="Site root URL")
    ap.add_argument("--out", default="lyeinani_pdfs", help="Output folder")
    ap.add_argument("--max-pages", type=int, default=500, help="Stop after N PDFs")
    ap.add_argument("--delay", type=float, default=1.5, help="Seconds between pages")
    ap.add_argument("--timeout", type=int, default=45, help="Page load timeout (s)")
    ap.add_argument("--no-crawl", action="store_true", help="Only use sitemap URLs")
    ap.add_argument("--ignore-robots", action="store_true",
                    help="Skip robots.txt checks (fine for your own site)")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    base_netloc = urlparse(base).netloc.lower()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    rp = load_robots(session, base)

    print("Reading sitemap...")
    seeds = sitemap_urls(session, base, base_netloc)
    print(f"  {len(seeds)} URLs found in sitemap")

    home = normalize(base + "/", base_netloc)
    queue = deque(dict.fromkeys([home] + seeds))
    queued = set(queue)
    saved_urls = set()
    used_names = set()
    manifest, failures = [], []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=USER_AGENT, viewport={"width": 1280, "height": 1600}
        )
        page = context.new_page()

        while queue and len(manifest) < args.max_pages:
            url = queue.popleft()
            if not args.ignore_robots and not rp.can_fetch(USER_AGENT, url):
                print(f"  [robots] skip {url}")
                continue

            out_path = out_dir / pdf_name(url, used_names)
            print(f"[{len(manifest) + 1}] {url}")
            try:
                status, final_url, title, links = render_page(
                    page, url, out_path, args.timeout * 1000
                )
            except Exception as e:
                print(f"    FAILED: {e}")
                failures.append({"url": url, "error": str(e)})
                out_path.unlink(missing_ok=True)
                continue

            if urlparse(final_url).path.startswith("/password"):
                out_path.unlink(missing_ok=True)
                print("\nThe store is password-protected (redirected to /password).")
                print("Disable the storefront password or scrape after launch.")
                break

            if status and status >= 400:
                print(f"    HTTP {status}, skipped")
                failures.append({"url": url, "error": f"HTTP {status}"})
                out_path.unlink(missing_ok=True)
                continue

            final_norm = normalize(final_url, base_netloc) or url
            if final_norm in saved_urls:  # redirected to a page we already have
                out_path.unlink(missing_ok=True)
                continue
            saved_urls.update({url, final_norm})

            manifest.append({
                "url": final_norm,
                "title": title,
                "file": out_path.name,
                "page_type": (urlparse(final_norm).path.strip("/").split("/") or ["home"])[0] or "home",
                "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })

            if not args.no_crawl:
                for link in links:
                    n = normalize(link, base_netloc)
                    if n and n not in queued:
                        queued.add(n)
                        queue.append(n)

            time.sleep(args.delay)

        browser.close()

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if failures:
        (out_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    print(f"\nDone: {len(manifest)} PDFs saved to {out_dir.resolve()}")
    print(f"Manifest: {out_dir / 'manifest.json'}")
    if failures:
        print(f"{len(failures)} pages failed, see {out_dir / 'failures.json'}")


if __name__ == "__main__":
    main()
