#!/usr/bin/env python3
"""
Extract sponsor URLs from NVIDIA GTC website using agent-browser
Run this script to automatically collect all missing sponsor URLs

Usage:
    cd projects/gtc-2026
    python3 scripts/extract_sponsor_urls.py

Requirements:
    - agent-browser must be installed and available in PATH
    - sponsors.json must exist in the current directory
"""

import json
import subprocess
import re
import time
import sys
from typing import Dict, Optional, Tuple


def load_sponsors() -> list:
    """Load sponsors from JSON file"""
    with open("sponsors.json", "r", encoding="utf-8") as f:
        return json.load(f)


def save_sponsors(sponsors: list) -> None:
    """Save sponsors to JSON file"""
    with open("sponsors.json", "w", encoding="utf-8") as f:
        json.dump(sponsors, f, indent=2, ensure_ascii=False)


def find_missing_urls(sponsors: list) -> list:
    """Find sponsors with empty URLs, returns list of (index, name, tier)"""
    return [
        (i, s["name"], s["tier"])
        for i, s in enumerate(sponsors)
        if not s.get("company_url")
    ]


def get_page_refs() -> Dict[str, str]:
    """
    Get all sponsor references from the current page snapshot
    Returns: {company_name: ref_id}
    """
    try:
        result = subprocess.run(
            ["agent-browser", "snapshot", "-i"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = result.stdout
        refs = {}

        # Parse output to find company links with refs
        # Pattern: - link "Company Name" [ref=e123]
        for match in re.finditer(r'- link "([^"]+)" \[ref=([^\]]+)\]', output):
            company_name = match.group(1)
            ref_id = match.group(2)
            refs[company_name] = ref_id

        return refs
    except Exception as e:
        print(f"Error getting page refs: {e}")
        return {}


def extract_url_from_dialog() -> Optional[str]:
    """
    Extract URL from currently open dialog
    Returns URL or None if not found
    """
    try:
        result = subprocess.run(
            ["agent-browser", "snapshot"], capture_output=True, text=True, timeout=10
        )

        output = result.stdout

        # Look for URL in link element
        # Pattern 1: - link "https://..." [ref=e1]:
        match = re.search(r'- link "(https?://[^"]+)" \[ref=[^\]]+\]:', output)
        if match:
            return match.group(1)

        # Pattern 2: /url: https://...
        match = re.search(r"/url:\s*(https?://\S+)", output)
        if match:
            return match.group(1)

        return None
    except Exception as e:
        print(f"Error extracting URL from dialog: {e}")
        return None


def click_sponsor_and_extract(ref_id: str) -> Optional[str]:
    """
    Click on sponsor link and extract URL from dialog
    Returns URL or None
    """
    try:
        # Click on sponsor
        subprocess.run(
            ["agent-browser", "click", f"@{ref_id}"], capture_output=True, timeout=10
        )
        time.sleep(1.5)  # Wait for dialog to open

        # Extract URL
        url = extract_url_from_dialog()

        # Close dialog
        subprocess.run(
            ["agent-browser", "key", "Escape"], capture_output=True, timeout=5
        )
        time.sleep(0.5)

        return url
    except Exception as e:
        print(f"Error clicking sponsor {ref_id}: {e}")
        return None


def open_sponsors_page() -> bool:
    """Open the NVIDIA GTC sponsors page"""
    try:
        subprocess.run(
            ["agent-browser", "open", "https://www.nvidia.com/gtc/sponsors/#/"],
            capture_output=True,
            timeout=30,
        )
        time.sleep(3)  # Wait for page to load
        return True
    except Exception as e:
        print(f"Error opening sponsors page: {e}")
        return False


def extract_batch_urls(
    sponsors: list, missing: list, batch_size: int = 20
) -> Dict[str, str]:
    """
    Extract URLs for a batch of missing sponsors
    Returns: {company_name: url}
    """
    extracted = {}

    # Get current page refs
    refs = get_page_refs()

    for idx, name, tier in missing[:batch_size]:
        if name in refs:
            ref_id = refs[name]
            print(f"  Processing: {name} ({tier}) - ref: {ref_id}")

            url = click_sponsor_and_extract(ref_id)
            if url:
                extracted[name] = url
                print(f"    ✓ Found: {url}")
            else:
                print(f"    ✗ URL not found")
        else:
            print(f"  Not found on page: {name} ({tier})")

    return extracted


def update_sponsors_with_urls(sponsors: list, urls: Dict[str, str]) -> int:
    """
    Update sponsors list with extracted URLs
    Returns number of updates made
    """
    updated = 0
    for sponsor in sponsors:
        if sponsor["name"] in urls and not sponsor.get("company_url"):
            sponsor["company_url"] = urls[sponsor["name"]]
            updated += 1
    return updated


def main():
    print("=" * 70)
    print("NVIDIA GTC Sponsor URL Extractor")
    print("=" * 70)

    # Load sponsors
    print("\n1. Loading sponsors.json...")
    sponsors = load_sponsors()
    print(f"   Total sponsors: {len(sponsors)}")

    # Find missing URLs
    missing = find_missing_urls(sponsors)
    print(f"   Missing URLs: {len(missing)}")

    if not missing:
        print("\n✓ All sponsors already have URLs!")
        return

    # Group by tier for reporting
    by_tier = {}
    for _, name, tier in missing:
        by_tier[tier] = by_tier.get(tier, 0) + 1
    print("\n   Missing by tier:")
    for tier, count in sorted(by_tier.items(), key=lambda x: -x[1]):
        print(f"     - {tier}: {count}")

    # Ask for confirmation
    print(f"\n2. Ready to extract URLs for {len(missing)} sponsors")
    response = input("   Continue? (y/n): ")
    if response.lower() != "y":
        print("   Aborted.")
        return

    # Open sponsors page
    print("\n3. Opening NVIDIA GTC sponsors page...")
    if not open_sponsors_page():
        print("   ✗ Failed to open page")
        return
    print("   ✓ Page loaded")

    # Process in batches
    batch_size = 20
    total_extracted = 0

    while missing:
        batch = missing[:batch_size]
        print(f"\n4. Processing batch of {len(batch)} sponsors...")

        urls = extract_batch_urls(sponsors, batch, batch_size)

        if urls:
            updated = update_sponsors_with_urls(sponsors, urls)
            total_extracted += updated
            print(f"   ✓ Extracted {updated} URLs in this batch")

            # Save progress after each batch
            save_sponsors(sponsors)
            print(f"   ✓ Saved progress to sponsors.json")

        # Remove processed items from missing list
        processed_names = {name for _, name, _ in batch}
        missing = [(i, n, t) for i, n, t in missing if n not in processed_names]

        if missing:
            response = input(
                f"\n   {len(missing)} sponsors remaining. Continue? (y/n): "
            )
            if response.lower() != "y":
                break

    # Final report
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Total URLs extracted: {total_extracted}")

    remaining = find_missing_urls(sponsors)
    print(f"Remaining missing URLs: {len(remaining)}")

    if remaining:
        print("\nRemaining sponsors:")
        for _, name, tier in remaining[:10]:
            print(f"  - {name} ({tier})")
        if len(remaining) > 10:
            print(f"  ... and {len(remaining) - 10} more")

    print("\n✓ Results saved to sponsors.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)
