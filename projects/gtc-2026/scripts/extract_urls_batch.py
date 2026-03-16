#!/usr/bin/env python3
"""
Extract sponsor URLs from NVIDIA GTC website using agent-browser
"""

import subprocess
import json
import re
import time

# List of sponsors with missing URLs (index, name, ref)
missing_sponsors = [
    (25, "Wistron & Wiwynn", "e35"),
    (50, "Agile Robots SE", "e60"),
    (60, "Global AI", "e70"),
    (67, "Lightwheel", "e77"),
    (76, "Simplismart", "e86"),
    (84, "ASROCK RACK INC.", "e94"),
    (86, "Astris AI, A Lockheed Martin Company", "e96"),
    (88, "Centific", "e98"),
    (92, "Compal Electronics, Inc", "e102"),
    (95, "DDC Solutions (a member of Daikin group)", "e105"),
    (96, "Dream", "e106"),
]


def extract_url_from_sponsor(ref_id):
    """Click on sponsor and extract URL"""
    try:
        # Click on sponsor
        result = subprocess.run(
            ["agent-browser", "click", f"@{ref_id}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        time.sleep(1)

        # Get snapshot
        result = subprocess.run(
            ["agent-browser", "snapshot"], capture_output=True, text=True, timeout=10
        )
        output = result.stdout

        # Close dialog
        subprocess.run(
            ["agent-browser", "key", "Escape"], capture_output=True, timeout=5
        )
        time.sleep(0.5)

        # Extract URL from output
        url_match = re.search(r'link "(https?://[^"]+)"', output)
        if url_match:
            return url_match.group(1)

        # Try alternative pattern
        url_match = re.search(r"/url:\s*(https?://\S+)", output)
        if url_match:
            return url_match.group(1)

        return None
    except Exception as e:
        print(f"Error extracting URL: {e}")
        return None


def main():
    # Load current sponsors
    with open("sponsors.json", "r") as f:
        sponsors = json.load(f)

    # Extract URLs for each missing sponsor
    extracted_urls = {}
    for idx, name, ref in missing_sponsors[:10]:  # Process first 10
        print(f"\nExtracting URL for: {name} (ref: {ref})")
        url = extract_url_from_sponsor(ref)
        if url:
            extracted_urls[name] = url
            sponsors[idx]["company_url"] = url
            print(f"  ✓ Found: {url}")
        else:
            print(f"  ✗ Not found")

    # Save updated sponsors
    with open("sponsors.json", "w") as f:
        json.dump(sponsors, f, indent=2, ensure_ascii=False)

    print(f"\n\nExtracted {len(extracted_urls)} URLs")
    print("Updated sponsors.json")


if __name__ == "__main__":
    main()
