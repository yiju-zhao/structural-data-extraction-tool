#!/usr/bin/env python3
"""
Batch extract session details for Training Lab, Full-Day Workshop, and Q&A sessions.
Uses YAML config for extraction rules.
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
    config_path = Path(__file__).parent.parent / 'configs' / 'session_details_training_workshop_qa.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['custom']


def extract_session_data(url, browser, config):
    """Extract data from a single session page using config."""
    page = browser.new_page()

    try:
        page.goto(url, wait_until='domcontentloaded', timeout=60000)
        page.wait_for_timeout(3000)

        body = page.query_selector('body')
        if not body:
            return None

        text = body.inner_text()

        # Extract speakers from HTML structure
        speakers = []
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

        # Extract abstract from HTML - skip speaker paragraphs and prerequisites
        abstract = ''
        abstract_config = config['selectors']['abstract']
        container_selector = abstract_config['container']
        paragraph_selector = abstract_config['paragraphs']
        min_length = abstract_config['min_length']
        exclude_pattern = abstract_config.get('exclude_pattern', '|')

        divs = page.query_selector_all(container_selector)
        for div in divs:
            div_html = div.inner_html()

            # Skip if no paragraph tags
            if '<p>' not in div_html.lower():
                continue

            # Get all paragraphs
            paragraphs = div.query_selector_all(paragraph_selector)
            p_texts = []

            for p in paragraphs:
                p_text = p.inner_text().strip()
                # Skip speaker paragraphs (contain |)
                if exclude_pattern in p_text:
                    continue
                # Split on "Prerequisite" to remove prerequisites section
                if 'Prerequisite' in p_text:
                    parts = p_text.split('Prerequisite')
                    p_text = parts[0].strip()
                # Split on "Certificate:" to remove certificate section
                if 'Certificate:' in p_text:
                    parts = p_text.split('Certificate:')
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

        # Extract topics/industry
        topics = []
        industry_pattern = config['selectors']['topics']['industry_pattern']
        industry_match = re.search(industry_pattern, text)
        if industry_match:
            topics.append(industry_match.group(1).strip())

        # Extract NVIDIA technologies
        tech_set = set()
        html_config = config['technologies']['html_selector']
        tech_containers = page.query_selector_all(html_config['container'])

        for container in tech_containers:
            container_text = container.inner_text()
            if 'NVIDIA Technology' in container_text:
                spans = container.query_selector_all(html_config['value'])
                for span in spans:
                    span_text = span.inner_text().strip()
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
    parser = argparse.ArgumentParser(description='Extract GTC Training Lab/Workshop/Q&A session details')
    parser.add_argument('--session-id', type=str, help='Extract only a specific session')
    parser.add_argument('--limit', type=int, help='Limit number of sessions to extract')
    args = parser.parse_args()

    # Load config
    print("Loading extraction configuration...")
    config = load_extraction_config()
    print(f"  Speakers selector: {config['selectors']['speakers']['html']}")
    print(f"  Abstract container: {config['selectors']['abstract']['container']}")

    # Paths
    input_csv = Path(__file__).parent.parent / 'output' / 'gtc-2026-traning-lab-workshop-qa.csv'
    output_csv = Path(__file__).parent.parent / 'output' / 'gtc-2026-traning-lab-workshop-qa-detailed.csv'

    print(f"\nReading: {input_csv}")

    # Read input CSV (no header in source file)
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        all_sessions = []
        for row in reader:
            if len(row) >= 4:
                all_sessions.append({
                    'session_code': row[0],
                    'title': row[1],
                    'session_type': row[2],
                    'url': row[3]
                })

    print(f"Found {len(all_sessions)} sessions")

    # Filter by session ID if provided
    if args.session_id:
        all_sessions = [s for s in all_sessions if s['session_code'] == args.session_id]
        if not all_sessions:
            print(f"Error: Session ID '{args.session_id}' not found")
            return
        print(f"Extracting single session: {args.session_id}")

    # Apply limit if provided
    if args.limit:
        all_sessions = all_sessions[:args.limit]
        print(f"Limited to {args.limit} sessions")

    # Process sessions
    fieldnames = ['session_code', 'title', 'session_type', 'url', 'speakers', 'abstract', 'topics', 'nvidia_technologies']

    print(f"\nWriting to: {output_csv}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, session in enumerate(all_sessions, 1):
                url = session['url'].strip()
                session_code = session['session_code']

                print(f"[{i}/{len(all_sessions)}] {session_code}: {session['title'][:50]}...")

                if url:
                    data = extract_session_data(url, browser, config)

                    row = {
                        'session_code': session_code,
                        'title': session['title'],
                        'session_type': session['session_type'],
                        'url': url,
                        'speakers': data['speakers'] if data else '',
                        'abstract': data['abstract'] if data else '',
                        'topics': data['topics'] if data else '',
                        'nvidia_technologies': data['nvidia_technologies'] if data else ''
                    }

                    if data:
                        print(f"    ✓ Speakers: {len(data['speakers'].split(';')) if data['speakers'] else 0} | "
                              f"Abstract: {len(data['abstract'])} chars | "
                              f"Tech: {len(data['nvidia_technologies'].split(',')) if data['nvidia_technologies'] else 0}")
                    else:
                        print(f"    ✗ Failed to extract")

                    writer.writerow(row)
                    f.flush()

                    # Delay between requests
                    if i < len(all_sessions):
                        time.sleep(1)

        browser.close()

    print(f"\n✓ Complete! Extracted {len(all_sessions)} sessions")
    print(f"✓ Output saved to: {output_csv.name}")


if __name__ == '__main__':
    main()
