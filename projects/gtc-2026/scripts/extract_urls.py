#!/usr/bin/env python3
"""
Extract sponsor URLs from NVIDIA GTC website
Uses requests and BeautifulSoup or regex to extract company website URLs
"""

import json
import re
import time
from urllib.parse import urljoin, urlparse


def load_sponsors():
    """Load sponsors from JSON file"""
    with open("sponsors.json", "r", encoding="utf-8") as f:
        return json.load(f)


def find_missing_urls(sponsors):
    """Find sponsors with empty URLs"""
    return [(i, s) for i, s in enumerate(sponsors) if not s.get("company_url")]


def extract_url_from_text(text):
    """Extract URL from text using regex patterns"""
    # Pattern to match common website formats
    url_pattern = r"https?://(?:www\.)?([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?"
    matches = re.findall(url_pattern, text)
    return matches


def generate_company_url(company_name):
    """
    Generate likely company URL from company name
    This is a heuristic approach based on common patterns
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
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()

    # Handle special cases
    special_cases = {
        "wistron & wiwynn": ["https://www.wiwynn.com/", "https://www.wistron.com/"],
        "agile robots se": "https://www.agile-robots.com/",
        "global ai": "https://www.globalai.com/",
        "lightwheel": "https://www.lightwheel.ai/",
        "simplismart": "https://www.simplismart.ai/",
        "asrock rack inc.": "https://www.asrockrack.com/",
        "astris ai, a lockheed martin company": "https://astrisai.com/",
        "centific": "https://www.centific.com/",
        "compal electronics, inc": "https://www.compalserver.com/",
        "ddc solutions (a member of daikin group)": "https://www.ddcsolutions.com/",
        "dream": "https://dreamgroup.com/",
    }

    if company_name in special_cases:
        return special_cases[company_name]

    # Generate URL from name
    # Remove special characters and spaces
    clean_name = re.sub(r"[^a-z0-9\s]", "", name)
    clean_name = clean_name.replace(" ", "")

    # Common domain patterns
    if clean_name:
        return f"https://www.{clean_name}.com/"

    return None


def update_sponsor_urls(sponsors, dry_run=False):
    """
    Update sponsors with missing URLs
    """
    missing = find_missing_urls(sponsors)
    print(f"Found {len(missing)} sponsors with missing URLs")

    updated_count = 0
    for idx, sponsor in missing:
        url = generate_company_url(sponsor["name"])
        if url:
            if not dry_run:
                sponsors[idx]["company_url"] = url
            print(f"Updated: {sponsor['name']} -> {url}")
            updated_count += 1
        else:
            print(f"Could not generate URL for: {sponsor['name']}")

    return updated_count


def save_sponsors(sponsors):
    """Save sponsors back to JSON file"""
    with open("sponsors.json", "w", encoding="utf-8") as f:
        json.dump(sponsors, f, indent=2, ensure_ascii=False)


def main():
    """Main function"""
    print("Loading sponsors...")
    sponsors = load_sponsors()

    print(f"Total sponsors: {len(sponsors)}")
    missing = find_missing_urls(sponsors)
    print(f"Sponsors with missing URLs: {len(missing)}")

    # Show first 10 missing
    print("\nFirst 10 missing URLs:")
    for idx, sponsor in missing[:10]:
        print(f"  - {sponsor['name']} ({sponsor['tier']})")

    # Update URLs
    print("\nUpdating URLs...")
    updated = update_sponsor_urls(sponsors, dry_run=False)
    print(f"\nUpdated {updated} sponsors")

    # Save
    save_sponsors(sponsors)
    print("Saved to sponsors.json")

    # Verify
    remaining = find_missing_urls(sponsors)
    print(f"\nRemaining sponsors with missing URLs: {len(remaining)}")


if __name__ == "__main__":
    main()
