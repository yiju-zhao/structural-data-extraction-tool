#!/usr/bin/env python3
"""
Extract sponsor URLs using web search as fallback
Use this when agent-browser method doesn't work for some companies

Usage:
    cd projects/gtc-2026
    python3 scripts/extract_urls_search.py [company_names...]

Examples:
    # Search for specific companies
    python3 scripts/extract_urls_search.py "Netris" "nVent" "Sanmina"

    # Search for all missing URLs
    python3 scripts/extract_urls_search.py --all
"""

import json
import sys
import re
import subprocess
from typing import List, Dict, Optional


def load_sponsors() -> list:
    """Load sponsors from JSON file"""
    with open("sponsors.json", "r", encoding="utf-8") as f:
        return json.load(f)


def save_sponsors(sponsors: list) -> None:
    """Save sponsors to JSON file"""
    with open("sponsors.json", "w", encoding="utf-8") as f:
        json.dump(sponsors, f, indent=2, ensure_ascii=False)


def find_missing_urls(sponsors: list) -> List[tuple]:
    """Find sponsors with empty URLs"""
    return [
        (i, s["name"], s["tier"])
        for i, s in enumerate(sponsors)
        if not s.get("company_url")
    ]


def search_company_url(company_name: str) -> Optional[str]:
    """
    Search for company URL using web search
    Returns best guess URL or None
    """
    # Clean company name for search
    search_name = company_name.replace('"', "").replace("'", "")

    # Try web search
    try:
        result = subprocess.run(
            [
                "python3",
                "-c",
                f"""
import sys
sys.path.insert(0, '/Users/eason/.local/share/opencode/skills/websearch')
try:
    from websearch import web_search
    results = web_search('{search_name} official website', num_results=5)
    for r in results[:3]:
        url = r.get('url', '')
        if url and 'http' in url:
            print(url)
            break
except Exception as e:
    print(f'Error: {{e}}', file=sys.stderr)
""",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Parse result
        url = result.stdout.strip().split("\n")[0]
        if url and url.startswith("http"):
            return url
    except Exception as e:
        print(f"  Search error: {e}")

    return None


def generate_url_from_name(company_name: str) -> Optional[str]:
    """
    Generate likely URL from company name
    This is a last resort heuristic
    """
    # Clean up company name
    name = company_name.lower()

    # Remove common suffixes
    suffixes = [
        " inc.",
        " inc",
        " llc",
        " ltd.",
        " ltd",
        " corp.",
        " corp",
        " corporation",
        " co.",
        " co",
        " company",
        " limited",
        " gmbh",
        " ag",
        " se",
        " s.a.",
        " bv",
        ".com",
        ".ai",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()

    # Remove parentheses and their contents
    name = re.sub(r"\([^)]*\)", "", name)

    # Remove special characters
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = name.strip().replace(" ", "")

    if name:
        # Try common domains
        domains = [".com", ".ai", ".io", ".tech", ".co"]
        for domain in domains:
            return f"https://www.{name}{domain}/"

    return None


def main():
    # Load sponsors
    sponsors = load_sponsors()
    missing = find_missing_urls(sponsors)

    # Check command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            # Search for all missing
            targets = missing
        else:
            # Search for specific companies
            target_names = sys.argv[1:]
            targets = [(i, n, t) for i, n, t in missing if n in target_names]
    else:
        print("Usage: python3 extract_urls_search.py [company_names...]")
        print("       python3 extract_urls_search.py --all")
        print("\nMissing URLs:")
        for idx, name, tier in missing[:20]:
            print(f"  - {name} ({tier})")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        return

    print(f"Searching URLs for {len(targets)} companies...\n")

    extracted = {}

    for idx, name, tier in targets:
        print(f"Searching: {name} ({tier})")

        # Try web search first
        url = search_company_url(name)

        if url:
            print(f"  ✓ Found: {url}")
            extracted[name] = url
        else:
            # Fallback to generated URL
            url = generate_url_from_name(name)
            if url:
                print(f"  ? Generated: {url}")
                extracted[name] = url
            else:
                print(f"  ✗ Could not determine URL")

    # Update sponsors
    print(f"\n\nUpdating sponsors.json...")
    updated = 0
    for sponsor in sponsors:
        if sponsor["name"] in extracted and not sponsor.get("company_url"):
            sponsor["company_url"] = extracted[sponsor["name"]]
            updated += 1

    save_sponsors(sponsors)
    print(f"✓ Updated {updated} sponsors")

    # Show remaining
    remaining = find_missing_urls(sponsors)
    print(f"Remaining missing URLs: {len(remaining)}")


if __name__ == "__main__":
    main()
