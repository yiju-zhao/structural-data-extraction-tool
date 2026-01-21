#!/usr/bin/env python3
"""
Extract all sessions from GTC 2026 session catalog page.
Extracts each session type separately using URL parameters.
"""

import csv
import time
from pathlib import Path
from urllib.parse import quote
from playwright.sync_api import sync_playwright


def load_all_sessions_for_url(page, url):
    """Navigate to URL and click Load More until all sessions are loaded."""
    page.goto(url, wait_until='domcontentloaded', timeout=60000)
    page.wait_for_timeout(5000)

    iteration = 0
    max_iterations = 50

    while iteration < max_iterations:
        iteration += 1

        # Scroll to bottom
        page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
        time.sleep(1)

        # Get current session count
        cards = page.query_selector_all('[class*="session-card"]')
        current_count = len(cards)

        print(f"    Loaded {current_count} sessions...")

        # Try to find and click "Load More" button
        load_more = page.query_selector('button:has-text("Load More")')

        if load_more and load_more.is_visible():
            try:
                load_more.click()
                time.sleep(2)
            except:
                break
        else:
            break

    return current_count


def extract_sessions(page):
    """Extract session data from loaded page."""
    sessions = []

    cards = page.query_selector_all('[class*="session-card"]')

    for card in cards:
        try:
            # Extract session code from class attribute
            class_attr = card.get_attribute('class') or ''
            session_code = ''
            if 'session-code-' in class_attr:
                parts = class_attr.split('session-code-')
                if len(parts) > 1:
                    code_part = parts[1].split()[0]
                    session_code = code_part.upper()

            # Extract title from h3 link
            title_link = card.query_selector('h3 a')
            title = ''
            url_path = ''
            if title_link:
                title = title_link.inner_text().strip()
                url_path = title_link.get_attribute('href') or ''

            # Extract session type from badge
            session_type = ''
            type_badges = card.query_selector_all('[class*="badge-"]')
            for badge in type_badges:
                badge_text = badge.inner_text().strip()
                # Skip "In-Person" and "Virtual" badges
                if badge_text not in ['In-Person', 'Virtual']:
                    session_type = badge_text
                    break

            # Convert relative URL to absolute
            if url_path:
                if url_path.startswith('sessions/'):
                    full_url = f'https://www.nvidia.com/gtc/session-catalog/{url_path}'
                elif url_path.startswith('/'):
                    full_url = f'https://www.nvidia.com{url_path}'
                else:
                    full_url = url_path
            else:
                full_url = ''

            # Only add if we have at least session code and title
            if session_code and title:
                sessions.append({
                    'session_code': session_code,
                    'title': title,
                    'session_type': session_type,
                    'url': full_url
                })

        except Exception as e:
            print(f"    Warning: Failed to extract session: {e}")
            continue

    return sessions


def main():
    output_path = Path(__file__).parent.parent / 'output' / 'gtc-2026-all-sessions.csv'

    print("Starting GTC 2026 session catalog extraction...")
    print("Extracting sessions by session type using URL parameters\n")

    # Define all session types (skip "Connect With the Experts" as it's already extracted)
    session_types = [
        "Keynote",
        "Talk",
        "Talks or Panels",
        "Lightning Talk",
        "Theater Talk",
        "Fireside Chat",
        "Tutorial",
        "Training Lab",
        "Full-Day Workshop",
        "Q&A with NVIDIA"
    ]

    all_sessions = []
    session_codes_seen = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for session_type in session_types:
            print("=" * 60)
            print(f"EXTRACTING: {session_type}")
            print("=" * 60)

            # Build URL with session type filter
            encoded_type = quote(session_type)
            url = f'https://www.nvidia.com/gtc/session-catalog/?sessionTypes={encoded_type}'

            print(f"  URL: {url}")

            # Load all sessions for this type
            load_all_sessions_for_url(page, url)

            # Extract sessions
            sessions = extract_sessions(page)
            print(f"  Extracted {len(sessions)} sessions")

            # Add unique sessions
            new_count = 0
            for session in sessions:
                if session['session_code'] not in session_codes_seen:
                    all_sessions.append(session)
                    session_codes_seen.add(session['session_code'])
                    new_count += 1

            print(f"  New unique sessions: {new_count}\n")

        browser.close()

    # Write to CSV
    print("=" * 60)
    print(f"Writing {len(all_sessions)} total sessions to CSV...")
    print(f"Output: {output_path}")

    fieldnames = ['session_code', 'title', 'session_type', 'url']

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_sessions)

    print(f"\n✓ Complete! Extracted {len(all_sessions)} unique sessions")
    print(f"✓ Saved to: {output_path.name}")

    # Show session type breakdown
    type_counts = {}
    for session in all_sessions:
        session_type = session['session_type'] or 'Unknown'
        type_counts[session_type] = type_counts.get(session_type, 0) + 1

    print("\nSession Type Breakdown:")
    for session_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {session_type}: {count}")


if __name__ == '__main__':
    main()
