#!/usr/bin/env python3
"""Extract session titles from GTC Fireside Chat page."""

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re
import csv
import json
from pathlib import Path

TARGET_URL = 'https://www.nvidia.com/gtc/session-catalog/?sessionTypes=Fireside%20Chat'
OUTPUT_DIR = Path('projects/gtc-2026/output')

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(TARGET_URL, wait_until='networkidle')

    # Scroll to load all sessions
    prev_count = 0
    for _ in range(10):
        page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
        page.wait_for_timeout(1000)

        html = page.content()
        soup = BeautifulSoup(html, 'lxml')
        h3_elements = soup.find_all('h3', class_='font-semibold')

        if len(h3_elements) == prev_count:
            break
        prev_count = len(h3_elements)

    print(f'\n=== Found {len(h3_elements)} Fireside Chat Sessions ===\n')

    sessions = []
    for h3 in h3_elements:
        title = h3.get_text(strip=True)

        # Extract session code from title (e.g., [S81537])
        code_match = re.search(r'\[S(\d+)\]', title)
        session_code = f'S{code_match.group(1)}' if code_match else ''

        # Clean title (remove session code)
        clean_title = re.sub(r'\s*\[S\d+\]$', '', title)

        sessions.append({
            'session_code': session_code,
            'title': clean_title
        })

        print(f'{session_code}: {clean_title}')

    # Save to CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / 'gtc-2026-fireside-chats.csv'

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['session_code', 'title'])
        writer.writeheader()
        writer.writerows(sessions)

    print(f'\n✓ Saved {len(sessions)} sessions to {csv_path}')

    browser.close()
