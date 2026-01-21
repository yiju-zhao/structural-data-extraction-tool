#!/usr/bin/env python3
"""
Batch extract session details from multiple GTC session URLs.
Uses the extract skill's tools to scrape each session page.
"""

import argparse
import csv
import re
import time
import yaml
from pathlib import Path
from playwright.sync_api import sync_playwright


def load_extraction_config():
    """Load extraction configuration from YAML file."""
    config_path = Path(__file__).parent.parent / 'configs' / 'session_details_general.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # Return the custom section which has our detailed extraction config
    return config['custom']


def extract_session_data(url, browser, config):
    """Extract data from a single session page using config."""
    page = browser.new_page()

    try:
        page.goto(url, wait_until='domcontentloaded', timeout=60000)
        page.wait_for_timeout(3000)

        # Get body text for general extraction
        body = page.query_selector('body')
        if not body:
            return None

        text = body.inner_text()
        html = page.content()

        # Extract speakers from HTML structure using config selector
        speakers = []

        # Get speaker selector from config
        speaker_selector = config['selectors']['speakers']['html']
        speaker_spans = page.query_selector_all(speaker_selector)
        for span in speaker_spans:
            span_text = span.inner_text()
            if '|' in span_text:
                parts = [p.strip() for p in span_text.split('|')]
                if len(parts) >= 2:
                    name = parts[0]
                    title = parts[1] if len(parts) > 1 else ''
                    company = parts[2] if len(parts) > 2 else 'NVIDIA'

                    if title:
                        speakers.append(f"{name} ({title}, {company})")
                    else:
                        speakers.append(f"{name} ({company})")

        # Method 2: Fallback to text-based extraction if no speakers found
        if not speakers:
            lines = text.split('\n')
            for line in lines:
                if '|' in line and 'NVIDIA' in line:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) == 3:
                        name, title, company = parts
                        if title:
                            speakers.append(f"{name} ({title}, {company})")
                        else:
                            speakers.append(f"{name} ({company})")

        # Extract abstract from HTML using config selectors
        abstract = ''

        # Get abstract selectors from config
        abstract_config = config['selectors']['abstract']
        container_selector = abstract_config['container']
        paragraph_selector = abstract_config['paragraphs']
        min_length = abstract_config['min_length']
        important_marker = config['patterns'].get('important_marker')

        # Look for container divs
        divs = page.query_selector_all(container_selector)
        for div in divs:
            div_html = div.inner_html()

            # Skip if this div doesn't contain paragraph tags
            if '<p>' not in div_html.lower():
                continue

            # Extract all paragraph text from this div
            paragraphs = div.query_selector_all(paragraph_selector)
            p_texts = []
            for p in paragraphs:
                p_text = p.inner_text().strip()
                # Skip speaker paragraphs (contain |)
                if '|' in p_text:
                    continue
                # Split on "Prerequisite" to remove prerequisites section
                if 'Prerequisite' in p_text:
                    parts = p_text.split('Prerequisite')
                    p_text = parts[0].strip()
                # Split on "Certificate:" to remove certificate section
                if 'Certificate:' in p_text:
                    parts = p_text.split('Certificate:')
                    p_text = parts[0].strip()
                # Split on "Important:" marker (for Connect With Experts sessions)
                if important_marker and important_marker in p_text:
                    parts = p_text.split(important_marker)
                    p_text = parts[0].strip()
                if p_text:
                    p_texts.append(p_text)

            if p_texts:
                combined = ' '.join(p_texts)
                # Clean up multiple newlines and whitespace
                combined = ' '.join(combined.split())
                # Only use if substantial length
                if len(combined) >= min_length:
                    abstract = combined
                    break

        # Extract topics/industry using config pattern
        topics = []
        industry_pattern = config['selectors']['topics']['industry_pattern']
        industry_match = re.search(industry_pattern, text)
        if industry_match:
            topics.append(industry_match.group(1).strip())

        # Extract NVIDIA technologies from HTML "NVIDIA Technology:" labels
        tech_set = set()

        html_config = config['technologies']['html_selector']
        # Find all divs with "NVIDIA Technology:" label
        tech_containers = page.query_selector_all(html_config['container'])
        for container in tech_containers:
            container_text = container.inner_text()
            # Check if this container has the "NVIDIA Technology:" label
            if 'NVIDIA Technology:' in container_text or 'NVIDIA Technology' in container_text:
                # Extract the technology name from the span after the label
                spans = container.query_selector_all(html_config['value'])
                for span in spans:
                    span_text = span.inner_text().strip()
                    # Skip if it's the label itself
                    if span_text and 'NVIDIA Technology' not in span_text and span_text != ':':
                        tech_set.add(span_text)

        return {
            'speakers': '; '.join(speakers),
            'abstract': abstract,
            'topics': ', '.join(topics),
            'nvidia_technologies': ', '.join(sorted(tech_set))
        }

    except Exception as e:
        print(f"    Error: {e}")
        return None
    finally:
        page.close()


def main():
    # Load extraction configuration from YAML
    print("Loading extraction configuration from session_details.yaml...")
    config = load_extraction_config()
    print(f"  Using custom extraction rules:")
    print(f"    - Speakers: {config['selectors']['speakers']['html']}")
    print(f"    - Abstract: {config['selectors']['abstract']['container']}")
    print(f"    - Technologies: HTML selector (NVIDIA Technology: label)")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Extract GTC session details')
    parser.add_argument('--session-id', type=str, help='Extract only a specific session (e.g., S81595)')
    parser.add_argument('--input', type=str, help='Input CSV file (default: gtc-2026-all-sessions.csv)')
    args = parser.parse_args()

    # Paths
    if args.input:
        input_csv = Path(__file__).parent.parent / 'output' / args.input
    else:
        input_csv = Path(__file__).parent.parent / 'output' / 'gtc-2026-all-sessions.csv'

    # Determine output filename based on input
    input_name = input_csv.stem  # e.g., 'gtc-2026-all-sessions'
    output_csv = Path(__file__).parent.parent / 'output' / f'{input_name}-detailed.csv'

    print(f"Reading: {input_csv}")

    # Read input CSV
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_sessions = list(reader)

    # Initialize existing_data with all sessions from input CSV
    existing_data = {}
    for row in all_sessions:
        existing_data[row['session_code']] = {
            'session_code': row['session_code'],
            'title': row['title'],
            'session_type': row['session_type'],
            'url': row['url'],
            'speakers': '',
            'abstract': '',
            'topics': '',
            'nvidia_technologies': ''
        }

    # Load existing detailed CSV if it exists and merge with input data
    if output_csv.exists():
        print(f"Loading existing detailed data from: {output_csv}")
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                session_code = row['session_code']
                # Only update if this session exists in current input CSV
                if session_code in existing_data:
                    # Merge existing extracted data (speakers, abstract, etc.)
                    existing_data[session_code].update({
                        'speakers': row.get('speakers', ''),
                        'abstract': row.get('abstract', ''),
                        'topics': row.get('topics', ''),
                        'nvidia_technologies': row.get('nvidia_technologies', '')
                    })
        print(f"Merged {len(existing_data)} sessions\n")
    else:
        print(f"Creating new detailed CSV\n")

    # Filter by session ID if provided
    if args.session_id:
        rows_to_extract = [row for row in all_sessions if row.get('session_code', '').strip() == args.session_id]
        if not rows_to_extract:
            print(f"Error: Session ID '{args.session_id}' not found in CSV")
            return
        print(f"Extracting single session: {args.session_id}\n")
        rows = rows_to_extract
    else:
        print(f"Found {len(all_sessions)} sessions to extract\n")
        rows = all_sessions

    # Open output CSV for writing row by row
    print(f"\nWriting to: {output_csv}")
    fieldnames = ['session_code', 'title', 'session_type', 'url', 'speakers', 'abstract', 'topics', 'nvidia_technologies']

    # Process each session and write immediately
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, row in enumerate(rows, 1):
                url = row.get('url', '').strip()
                session_code = row['session_code']

                # Check if we should extract this session
                should_extract = True
                if args.session_id:
                    # Only extract if this is the specified session
                    should_extract = (session_code == args.session_id)

                if should_extract and url:
                    print(f"[{i}/{len(rows)}] {session_code}: {row['title'][:50]}...")

                    data = extract_session_data(url, browser, config)

                    if data:
                        # Update the existing_data entry with extracted data
                        existing_data[session_code].update(data)
                        print(f"    ✓ Speakers: {len(data['speakers'].split(';') if data['speakers'] else [])} | "
                              f"Abstract: {len(data['abstract'])} chars | "
                              f"Tech: {len(data['nvidia_technologies'].split(',') if data['nvidia_technologies'] else [])}")
                    else:
                        # Mark as failed but keep existing data
                        print(f"    ✗ Failed to extract")

                    # Delay between requests
                    if i < len(rows):
                        time.sleep(1)

                # Write the row immediately (either newly extracted or existing data)
                writer.writerow(existing_data[session_code])
                f.flush()  # Ensure data is written to disk immediately

        browser.close()

    print(f"\n✓ Complete! Total sessions in file: {len(existing_data)}")
    print(f"✓ Detailed data saved to: {output_csv.name}")


if __name__ == '__main__':
    main()
