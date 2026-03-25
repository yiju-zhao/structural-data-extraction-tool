#!/usr/bin/env python3
"""
Web Data Extraction Script with Error Recovery

Extracts structured data from websites using agent-browser commands.
Supports single-page and multi-page extraction with automatic error detection
and recovery capabilities.

Usage:
    python extract_data.py --url <url> --output <output_dir> [--pattern <pattern>]
    python extract_data.py --links-file <json_file> --output <output_dir>
"""

import argparse
import base64
import csv
import json
import os
import re
import subprocess
import time
import unicodedata
from pathlib import Path
from typing import Optional, List, Dict, Any


# Unicode replacement mappings for cleaning ambiguous characters
UNICODE_REPLACEMENTS = {
    '\u2018': "'",   # LEFT SINGLE QUOTATION MARK (curly quote)
    '\u2019': "'",   # RIGHT SINGLE QUOTATION MARK (curly quote)
    '\u201c': '"',   # LEFT DOUBLE QUOTATION MARK (curly quote)
    '\u201d': '"',   # RIGHT DOUBLE QUOTATION MARK (curly quote)
    '\u201a': ',',   # SINGLE LOW-9 QUOTATION MARK
    '\u201e': ',,',  # DOUBLE LOW-9 QUOTATION MARK
    '\u2013': '-',   # EN DASH
    '\u2014': '-',   # EM DASH
    '\u2015': '-',   # HORIZONTAL BAR
    '\u2026': '...', # HORIZONTAL ELLIPSIS
    '\u00a0': ' ',   # NON-BREAKING SPACE
    '\u200b': '',    # ZERO WIDTH SPACE
    '\u200c': '',    # ZERO WIDTH NON-JOINER
    '\u200d': '',    # ZERO WIDTH JOINER
    '\u2028': ' ',   # LINE SEPARATOR
    '\u2029': ' ',   # PARAGRAPH SEPARATOR
    '\u202f': ' ',   # NARROW NO-BREAK SPACE
    '\u205f': ' ',   # MEDIUM MATHEMATICAL SPACE
    '\u3000': ' ',   # IDEOGRAPHIC SPACE
    '\u00b7': '.',   # MIDDLE DOT
    '\u2022': '*',   # BULLET
    '\u2023': '>',   # TRIANGULAR BULLET
    '\u2032': "'",   # PRIME
    '\u2033': '"',   # DOUBLE PRIME
}


def clean_unicode(text: str) -> str:
    """Replace ambiguous unicode characters with ASCII equivalents."""
    if not isinstance(text, str):
        return text

    # Apply direct replacements
    for old, new in UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)

    # Normalize unicode (NFKC form)
    text = unicodedata.normalize('NFKC', text)

    # Remove control characters except newlines and tabs
    text = ''.join(char for char in text if not (
        unicodedata.category(char) == 'Cc' and char not in '\n\t'
    ))

    return text


def clean_for_csv(text: str) -> str:
    """Clean text for CSV - remove line breaks within cells."""
    if not isinstance(text, str):
        return text

    # First clean unicode
    text = clean_unicode(text)

    # Replace multiple whitespace/newlines with single space
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def clean_data(data: Any) -> Any:
    """Recursively clean all string values in data structure."""
    if isinstance(data, str):
        return clean_unicode(data)
    elif isinstance(data, list):
        return [clean_data(item) for item in data]
    elif isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items()}
    return data


def run_browser_cmd(cmd: List[str], timeout: int = 60) -> tuple[bool, str]:
    """Execute an agent-browser command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, (result.stdout.strip() + result.stderr.strip())
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def eval_js(script: str, timeout: int = 30) -> tuple[bool, Any]:
    """Evaluate JavaScript in browser and return decoded result."""
    escaped = script.replace('"', '\\"').replace("`", "\\`")
    success, output = run_browser_cmd(["eval", escaped], timeout=timeout)

    if not success or not output.strip():
        return False, None

    output = output.strip()

    # Try to decode as base64 (preferred method)
    try:
        decoded_bytes = base64.b64decode(output)
        decoded_str = decoded_bytes.decode('utf-8')
        result = json.loads(decoded_str)
        return True, result
    except (base64.binascii.Error, UnicodeDecodeError, json.JSONDecodeError):
        pass

    # Fallback: try parsing as plain JSON
    try:
        if output.startswith('"'):
            output = output[1:-1].replace('\\"', '"').replace('\\\\', '\\')
        result = json.loads(output)
        return True, result
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"


def open_page(url: str, wait: int = 3) -> bool:
    """Open browser and navigate to URL."""
    success, output = run_browser_cmd(["open", url], timeout=90)
    time.sleep(wait)
    return success


def close_browser():
    """Close the browser session."""
    run_browser_cmd(["close"])
    time.sleep(1)


def scroll_to_load_all(scrolls: int = 5, delay: float = 1.5):
    """Scroll page to trigger lazy-loaded content."""
    for _ in range(scrolls):
        run_browser_cmd(["scroll", "down", "5000"])
        time.sleep(delay)


def click_load_more(max_clicks: int = 20, delay: float = 2) -> int:
    """Click 'Load More' button repeatedly until no more content."""
    clicks = 0
    for _ in range(max_clicks):
        success, result = eval_js("""
        (() => {
            const btns = Array.from(document.querySelectorAll('button'));
            const loadMore = btns.find(b => /Load More|Show More|Show All/i.test(b.textContent));
            if (loadMore && !loadMore.disabled) {
                loadMore.click();
                return 'clicked';
            }
            return 'not found';
        })()
        """)
        if success and result == 'clicked':
            clicks += 1
            time.sleep(delay)
        else:
            break
    return clicks


def extract_from_page(extraction_js: str) -> List[Dict]:
    """Extract data from page using JavaScript and parse as JSON."""
    success, result = eval_js(extraction_js)

    if not success:
        print(f"  Extraction error: {result}")
        return []

    if isinstance(result, list):
        return result
    elif isinstance(result, dict):
        return [result]
    return []


def investigate_failure(url: str, output_dir: Path) -> Dict:
    """Investigate a failing extraction and return debug info."""
    debug_file = output_dir / "debug_info.json"
    snapshot_file = output_dir / "failure_snapshot.txt"

    debug_info = {'url': url, 'timestamp': time.time()}

    # Try to open the page and get snapshot
    if open_page(url, wait=2):
        success, snapshot = run_browser_cmd(["snapshot"], timeout=30)
        if success:
            debug_info['snapshot'] = snapshot[:5000]  # Truncate to avoid huge files
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                f.write(snapshot)
            debug_info['snapshot_file'] = str(snapshot_file)

    # Save debug info
    with open(debug_file, 'w', encoding='utf-8') as f:
        json.dump(debug_info, f, indent=2)

    return debug_info


def validate_data(data: Dict, required_fields: List[str]) -> bool:
    """Check if extracted data has required fields with non-empty values."""
    for field in required_fields:
        if not data.get(field):
            return False
    return True


def extract_with_recovery(
    url: str,
    extraction_js: str,
    output_dir: Path,
    max_retries: int = 2,
    required_fields: Optional[List[str]] = None
) -> Dict:
    """Extract data with automatic error recovery."""
    required_fields = required_fields or []

    for attempt in range(max_retries):
        data = extract_detail_page(url, extraction_js)

        # Check if extraction succeeded
        if data:
            if not required_fields or validate_data(data, required_fields):
                return data

        # If we failed and this is not the last attempt
        if attempt < max_retries - 1:
            print(f"    Attempt {attempt + 1} failed, investigating...")
            debug_info = investigate_failure(url, output_dir)
            print(f"    Debug info saved to {output_dir / 'debug_info.json'}")
            print(f"    Update the extraction script to handle this case.")
            return {'error': 'Extraction failed', 'url': url, 'debug_info': debug_info}

    # All attempts failed
    return {'error': 'Failed after retries', 'url': url}


def extract_detail_page(url: str, extraction_js: str) -> Dict:
    """Extract data from a single detail page."""
    if not open_page(url, wait=2):
        return {}

    time.sleep(1)
    data_list = extract_from_page(extraction_js)
    return data_list[0] if data_list else {}


def extract_links(link_pattern: str, base_url: str = "") -> List[Dict]:
    """Extract links matching a pattern from the page."""
    js = f"""
    (() => {{
      const data = Array.from(document.querySelectorAll('a')).filter(a => {{
        return {link_pattern};
      }}).map(a => ({{
        text: a.textContent.trim(),
        url: a.href
      }})).filter((v, i, a) => a.findIndex(t => t.url === v.url) === i);
      return btoa(unescape(encodeURIComponent(JSON.stringify(data))));
    }})()
    """
    return extract_from_page(js)


def save_json(data: List[Dict], filepath: Path):
    """Save data to JSON file with UTF-8 encoding and cleaned unicode."""
    # Clean data before saving
    cleaned_data = clean_data(data)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(cleaned_data)} records to {filepath.name}")


def save_csv(data: List[Dict], filepath: Path):
    """Save data to CSV file with UTF-8 encoding and cleaned unicode."""
    if not data:
        print("  No data to save")
        return

    # Clean data first
    cleaned_data = clean_data(data)

    # Get fieldnames from first item
    fieldnames = list(cleaned_data[0].keys())

    # Use csv.writer with QUOTE_ALL to properly handle all fields
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(fieldnames)

        for item in cleaned_data:
            row = []
            for field in fieldnames:
                value = item.get(field, '')
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                # Clean for CSV - remove line breaks within cells
                value = clean_for_csv(str(value)) if value else ''
                row.append(value)
            writer.writerow(row)

    print(f"  Saved {len(cleaned_data)} records to {filepath.name}")


def save_failures(failures: List[Dict], filepath: Path):
    """Save failed extractions for later review."""
    if failures:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(failures)} failed items to {filepath.name}")


def main():
    parser = argparse.ArgumentParser(description='Extract structured data from websites with error recovery')
    parser.add_argument('--url', help='URL to extract data from')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--pattern', help='CSS selector or JS pattern for extraction')
    parser.add_argument('--links-file', help='JSON file containing links to detail pages')
    parser.add_argument('--name', default='extracted', help='Base name for output files')
    parser.add_argument('--extraction-js', help='JavaScript code for extraction')
    parser.add_argument('--required-fields', help='Comma-separated list of required fields (e.g., "title,url")')
    parser.add_argument('--scroll', type=int, default=5, help='Number of scroll iterations')
    parser.add_argument('--load-more', type=int, default=20, help='Max load more clicks')
    parser.add_argument('--max-retries', type=int, default=2, help='Max retries for failed extractions')
    parser.add_argument('--no-recovery', action='store_true', help='Disable automatic error recovery')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_fields = args.required_fields.split(',') if args.required_fields else []
    use_recovery = not args.no_recovery

    if args.links_file:
        # Multi-page extraction: read links from file
        with open(args.links_file, 'r', encoding='utf-8') as f:
            links = json.load(f)

        print(f"Extracting {len(links)} detail pages...")

        results = []
        failures = []

        for i, link in enumerate(links):
            url = link.get('url', link) if isinstance(link, dict) else link
            print(f"  [{i+1}/{len(links)}] {url[:80]}...")

            if use_recovery:
                data = extract_with_recovery(
                    url,
                    args.extraction_js or '(() => { return btoa(unescape(encodeURIComponent(JSON.stringify({})))) })()',
                    output_dir,
                    max_retries=args.max_retries,
                    required_fields=required_fields
                )
            else:
                data = extract_detail_page(
                    url,
                    args.extraction_js or '(() => { return btoa(unescape(encodeURIComponent(JSON.stringify({})))) })()'
                )

            if data and not data.get('error'):
                data['_source_url'] = url
                results.append(data)
            else:
                failures.append(data)
                print(f"    Failed: {data.get('error', 'Unknown error')}")

        save_json(results, output_dir / f"{args.name}.json")
        save_csv(results, output_dir / f"{args.name}.csv")
        save_failures(failures, output_dir / f"{args.name}_failures.json")

        print(f"\nSummary: {len(results)} succeeded, {len(failures)} failed")
        if failures:
            print(f"Review failures in {output_dir / f'{args.name}_failures.json'}")

    elif args.url:
        # Single-page extraction
        print(f"Opening {args.url}...")
        if not open_page(args.url):
            print("Failed to open page")
            return

        print("Loading all content...")
        scroll_to_load_all(args.scroll)
        clicks = click_load_more(args.load_more)
        if clicks:
            print(f"  Clicked 'Load More' {clicks} times")

        print("Extracting data...")
        extraction_js = args.extraction_js or """
        (() => {
          const data = Array.from(document.querySelectorAll('a')).map(a => ({
            text: a.textContent.trim(),
            url: a.href
          }));
          return btoa(unescape(encodeURIComponent(JSON.stringify(data))));
        })()
        """
        data = extract_from_page(extraction_js)

        save_json(data, output_dir / f"{args.name}.json")
        save_csv(data, output_dir / f"{args.name}.csv")

        close_browser()

    print("\nExtraction complete!")


if __name__ == "__main__":
    main()
