#!/usr/bin/env python3
"""
Scrape a web page to export CSV data.

Supports two patterns:

1) Table-like rows (legacy):

   <tr>
     <td>
       <a href="...">Paper Title</a>
       Poster Session 3<br>
       <div class="indented"><i>Author A · Author B · ...</i></div>
     </td>
   </tr>

   Exports columns: title, authors, session

2) Card-style events (e.g., conference schedule):

   <div class="displaycards touchup-date" id="event-2843">
     <div class="virtual-card">
       <a class="small-title text-underline-hover" href="...">The 4th DataCV Workshop and Challenge</a>
     </div>
     <div class="author-str">Author 1, Author 2, ...</div>
     <div class="text-muted touchup-date-div">Sun 19 Oct 06:00 PM UTC</div>
     <details>
       <summary>Abstract</summary>
       <div class="text-start p-4">Abstract text here...</div>
     </details>
   </div>

   Exports columns: title, authors, time/date, abstract

Usage:
  python utility/scrape_to_csv.py "https://example.com/page" -o output.csv

Optional:
  --row-selector to narrow down which rows are scraped (default: "tr")
"""

from __future__ import annotations

import argparse
import csv
import sys
from typing import List, Dict
import re

import requests
from bs4 import BeautifulSoup


def fetch_html(url: str, timeout: int = 30) -> str:
    """Fetch the HTML content of a URL with a reasonable User-Agent."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or resp.encoding
    return resp.text


def _soup_from_html(html: str) -> BeautifulSoup:
    """Create a BeautifulSoup instance with a preferred parser order."""
    for parser in ("lxml", "html5lib", "html.parser"):
        try:
            return BeautifulSoup(html, parser)
        except Exception:
            continue
    # Fallback to the built-in parser if everything else fails
    return BeautifulSoup(html, "html.parser")


def _parse_cards(html: str) -> List[Dict[str, str]]:
    """
    Parse card-style event blocks and extract title, authors, time/date, abstract.

    This is tailored to structures like:
      <div class="displaycards touchup-date" id="event-2753"> ...
        <div class="virtual-card"> <a class="small-title">Title</a> </div>
        <div class="author-str">Author 1, Author 2</div>
        <div class="text-muted touchup-date-div" id="touchup-date-event-2753">Sun 19 Oct 06:00 PM UTC</div>
        <details> <summary>Abstract</summary> <div class="text-start p-4"> ... </div> </details>
      </div>
    """
    soup = _soup_from_html(html)
    results: List[Dict[str, str]] = []

    # Prefer a strict selector; if none match, fallback to class checks
    cards = soup.select('div.displaycards.touchup-date[id^="event-"]')
    if not cards:
        cards = [
            div for div in soup.find_all("div", id=re.compile(r"^event-"))
            if div.get("class") and "displaycards" in div.get("class") and "touchup-date" in div.get("class")
        ]

    for card in cards:
        # Title
        title_el = (
            card.select_one("div.virtual-card a.small-title")
            or card.select_one("a.small-title")
        )
        title = title_el.get_text(strip=True) if title_el else ""
        if not title:
            # Skip cards with no title
            continue

        # Authors: concatenate all non-empty author-str blocks
        author_list: List[str] = []
        for blk in card.find_all("div", class_=lambda c: c and "author-str" in c.split()):
            text = blk.get_text(" ", strip=True)
            if text:
                author_list.append(text)
        authors = "; ".join(author_list)

        # Time/Date: be permissive — grab any descendant whose class contains 'touchup-date-div'
        # and return its full text as-is (no post-processing)
        time_el = (
            card.select_one('[class*="touchup-date-div"]')
            or card.find(attrs={"id": re.compile(r"^touchup-date-")})
        )
        time_date = time_el.get_text(" ", strip=True) if time_el else ""

        # Abstract: take details content without the <summary>
        abstract = ""
        details_el = card.find("details")
        if details_el:
            # Clone via new soup from inner HTML to avoid mutating overall soup
            # But here, we can safely remove summary from this element
            summ = details_el.find("summary")
            if summ:
                summ.extract()
            # Prefer a specific content div, fallback to entire details text
            content = details_el.select_one(".text-start") or details_el
            abstract = content.get_text(" ", strip=True)

        results.append({
            "title": title,
            "authors": authors,
            "time/date": time_date,
            "abstract": abstract,
        })

    return results


def parse_rows(html: str, row_selector: str = "tr") -> List[Dict[str, str]]:
    """
    Parse rows and extract title, authors, session from the first <td> in each row.

    Heuristics:
    - The title is taken from the first <a> text inside the <td>.
    - The authors are taken from an <i> tag inside the same <td>, if present.
    - The session is the text string immediately following the title within the <td>,
      commonly containing the word "Session" (e.g., "Poster Session 3").
    """
    # First, try to parse card-style blocks if present
    card_rows = _parse_cards(html)
    if card_rows:
        return card_rows

    # Prefer lxml if available per requirements.txt; fallback to html5lib, then html.parser
    soup = _soup_from_html(html)

    rows = soup.select(row_selector) if row_selector else soup.find_all("tr")
    results: List[Dict[str, str]] = []

    for tr in rows:
        td = tr.find("td")
        if not td:
            continue

        a = td.find("a")
        title = a.get_text(strip=True) if a else ""
        if not title:
            # Skip rows without a link-based title
            continue

        # Collect the strings in the order they appear within the <td>
        pieces = list(td.stripped_strings)

        # Authors usually live inside <i> under a div.indented
        i_tag = td.find("i")
        authors = i_tag.get_text(" ", strip=True) if i_tag else ""

        # Try to find a session-like string after the title
        session = ""
        # Identify the index of title in the sequence of strings (first occurrence)
        try:
            title_idx = pieces.index(title)
        except ValueError:
            title_idx = 0

        # Prefer a string containing "session" right after the title
        for s in pieces[title_idx + 1 :]:
            if authors and s == authors:
                # Stop when we reach authors block
                break
            if "session" in s.lower():
                session = s
                break

        # Fallback: pick the first non-empty string after title that isn't authors
        if not session:
            for s in pieces[title_idx + 1 :]:
                if authors and s == authors:
                    break
                if s and s != title:
                    session = s
                    break

        results.append({"title": title, "authors": authors, "session": session})

    return results


def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    # Prefer new card-style columns if present; otherwise fall back to legacy
    card_cols = ["title", "authors", "time/date", "abstract"]
    legacy_cols = ["title", "authors", "session"]

    if rows and any("abstract" in r or "time/date" in r for r in rows):
        fieldnames = card_cols
    else:
        fieldnames = legacy_cols
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Scrape conference cards (title/authors/time/date/abstract) or "
            "fallback table rows (title/authors/session) to CSV"
        ),
    )
    p.add_argument("url", help="Source page URL to scrape")
    p.add_argument(
        "-o",
        "--output",
        default="output.csv",
        help="Output CSV file path (default: output.csv)",
    )
    p.add_argument(
        "--row-selector",
        default="tr",
        help="CSS selector for rows to parse (default: 'tr')",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        html = fetch_html(args.url)
    except Exception as e:
        print(f"Failed to fetch URL: {e}", file=sys.stderr)
        return 1

    try:
        rows = parse_rows(html, row_selector=args.row_selector)
    except Exception as e:
        print(f"Failed to parse content: {e}", file=sys.stderr)
        return 2

    if not rows:
        print("No matching rows found. Try adjusting --row-selector.", file=sys.stderr)
        # Still write an empty CSV header for convenience
        try:
            write_csv(args.output, rows)
        except Exception:
            pass
        return 3

    try:
        write_csv(args.output, rows)
    except Exception as e:
        print(f"Failed to write CSV: {e}", file=sys.stderr)
        return 4

    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
