#!/usr/bin/env python3
"""Analyze the GTC Fireside Chat session catalog page structure."""

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from collections import Counter

TARGET_URL = 'https://www.nvidia.com/gtc/session-catalog/?sessionTypes=Fireside%20Chat'

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(TARGET_URL, wait_until='networkidle')
    page.wait_for_timeout(3000)  # Wait for JS to load sessions

    html = page.content()
    soup = BeautifulSoup(html, 'lxml')

    # Look for session cards/items
    print('=== Looking for session containers ===')

    # Common patterns for session listings
    patterns = [
        'session', 'card', 'item', 'talk', 'event', 'result'
    ]

    for pattern in patterns:
        matches = soup.find_all(class_=lambda x: x and pattern in x.lower())
        if matches:
            print(f'\nElements with "{pattern}" in class: {len(matches)}')
            classes = set()
            for m in matches[:5]:
                classes.update(m.get('class', []))
            print(f'  Sample classes: {list(classes)[:5]}')

    # Look for h2, h3, h4 elements that might be titles
    print('\n=== Heading elements ===')
    for tag in ['h1', 'h2', 'h3', 'h4']:
        headings = soup.find_all(tag)
        if headings:
            print(f'\n{tag}: {len(headings)} found')
            for h in headings[:3]:
                text = h.get_text(strip=True)[:80]
                classes = h.get('class', [])
                print(f'  "{text}" - classes: {classes}')

    # Look for links that might be session links
    print('\n=== Session links ===')
    session_links = soup.find_all('a', href=lambda x: x and 'session' in x.lower())
    print(f'Links with "session" in href: {len(session_links)}')
    for link in session_links[:5]:
        text = link.get_text(strip=True)[:60]
        href = link.get('href', '')[:80]
        print(f'  "{text}" -> {href}')

    browser.close()
