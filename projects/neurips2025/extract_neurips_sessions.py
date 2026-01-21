#!/usr/bin/env python3
"""Extract NeurIPS 2025 San Diego sessions from calendar page."""

import asyncio
import json
import re
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

BASE_URL = "https://neurips.cc"

async def fetch_page(url):
    """Fetch the page HTML."""
    config = BrowserConfig(headless=True)
    async with AsyncWebCrawler(config=config) as crawler:
        result = await crawler.arun(url)
        return result.html

def parse_sessions(html):
    """Parse all sessions from the calendar page."""
    soup = BeautifulSoup(html, 'html.parser')
    sessions = []

    # Track current day for context
    current_day = None
    current_time = None

    # Find all day containers
    day_containers = soup.find_all('div', class_=re.compile(r'container2.*day-'))

    for day_container in day_containers:
        # Get day header
        day_header = day_container.find('div', class_='hdrbox')
        if day_header:
            current_day = day_header.get_text(strip=True)

        # Find all timeboxes within this day
        timeboxes = day_container.find_all('div', class_='timebox')

        for timebox in timeboxes:
            # Get time
            time_div = timebox.find('div', class_='time')
            if time_div:
                current_time = time_div.get_text(strip=True)

            # Find all event sessions (direct eventsession divs)
            event_sessions = timebox.find_all('div', class_='eventsession', recursive=False)
            for event in event_sessions:
                session = extract_event_session(event, current_day, current_time)
                if session:
                    sessions.append(session)

            # Find oral/poster session blocks (including san-diego-* and regular poster-session)
            oral_poster_blocks = timebox.find_all('div', class_=re.compile(r'(san-diego-)?(oral|poster)-session'))
            for block in oral_poster_blocks:
                # Extract session header info
                block_sessions = extract_session_block(block, current_day, current_time)
                sessions.extend(block_sessions)

            # Find exhibitor spot talks
            exhibitor_sessions = timebox.find_all('div', class_='exhibitor-spot-talks')
            for exhibit in exhibitor_sessions:
                exhibit_sessions = extract_exhibitor_talks(exhibit, current_day, current_time)
                sessions.extend(exhibit_sessions)

    return sessions


def extract_session_block(block, day, time):
    """Extract all oral/poster sessions from a session block."""
    sessions = []

    # Determine session type from class
    classes = ' '.join(block.get('class', []))
    if 'oral-session' in classes:
        default_type = 'Oral'
    elif 'poster-session' in classes:
        default_type = 'Poster'
    else:
        default_type = 'Session'

    # Find eventblocks which contain the actual sessions
    eventblocks = block.find_all('div', class_='eventblock')

    for eventblock in eventblocks:
        # Check subevent header for time range and type
        subevent_header = eventblock.find('div', class_='subevent-header')
        time_range = ''
        if subevent_header:
            header_text = subevent_header.get_text(strip=True)
            # Extract time range like "11:00-2:00"
            time_match = re.search(r'(\d+:\d+-\d+:\d+)', header_text)
            if time_match:
                time_range = time_match.group(1)

            # Determine type from class
            if 'oral' in subevent_header.get('class', []):
                session_type = 'Oral'
            elif 'poster' in subevent_header.get('class', []):
                session_type = 'Poster'
            else:
                session_type = default_type
        else:
            session_type = default_type

        # Find all content items (oral or poster)
        content_items = eventblock.find_all('div', class_=re.compile(r'content\s+(oral|poster)'))

        for item in content_items:
            # Determine specific type from class
            item_classes = item.get('class', [])
            if 'oral' in item_classes:
                item_type = 'Oral'
            elif 'poster' in item_classes:
                item_type = 'Poster'
            else:
                item_type = session_type

            # Get time if available (orals have [HH:MM] format)
            item_time = time
            item_text = item.get_text()
            time_match = re.search(r'\[(\d+:\d+)\]', item_text)
            if time_match:
                item_time = time_match.group(1)
            elif time_range:
                item_time = time_range.split('-')[0]  # Use start time

            # Get title and URL
            link = item.find('a')
            if link:
                title = link.get_text(strip=True)
                url = BASE_URL + link.get('href', '')
            else:
                # For items without links, get text and clean it
                title = item.get_text(strip=True)
                # Remove time prefix
                title = re.sub(r'^\[\d+:\d+\]\s*', '', title)
                url = ''

            if title:
                sessions.append({
                    'title': title,
                    'date': day,
                    'time': item_time,
                    'end_time': time_range.split('-')[1] if '-' in time_range else '',
                    'speaker': '',
                    'session_type': item_type,
                    'url': url,
                    'abstract': ''
                })

    return sessions

def extract_event_session(event, day, time):
    """Extract a single event session."""
    # Get session type from class
    classes = event.get('class', [])
    session_type = None
    for cls in classes:
        if cls not in ['eventsession', 'pad'] and not cls.startswith('room-'):
            session_type = cls.replace('-', ' ').title()
            break

    # Check for header style (e.g., "Affinity Event:", "Expo Talk Panel:")
    hdr = event.find('div', class_='hdr-style')
    if hdr:
        hdr_text = hdr.get_text(strip=True).rstrip(':')
        if hdr_text:
            session_type = hdr_text

    # Get title and URL
    title_div = event.find('div', class_='title-style')
    if not title_div:
        return None

    title_link = title_div.find('a')
    if title_link:
        title = title_link.get_text(strip=True)
        url = BASE_URL + title_link.get('href', '')
    else:
        title = title_div.get_text(strip=True)
        url = ''

    # Get end time and clean it
    end_time_span = event.find('span', class_='end-time')
    end_time = ''
    if end_time_span:
        end_text = end_time_span.get_text(strip=True)
        # Extract time from "(ends X:XX PM)" format
        time_match = re.search(r'(\d+:\d+\s*(?:AM|PM)?)', end_text, re.IGNORECASE)
        if time_match:
            end_time = time_match.group(1)

    return {
        'title': title,
        'date': day,
        'time': time,
        'end_time': end_time,
        'speaker': '',  # Not available on calendar page
        'session_type': session_type or 'Event',
        'url': url,
        'abstract': ''  # Not available on calendar page
    }


def extract_exhibitor_talks(exhibit, day, time):
    """Extract exhibitor/sponsor talks."""
    sessions = []

    # Find all talk content items
    talk_items = exhibit.find_all('div', class_='content talk')

    for item in talk_items:
        link = item.find('a')
        if link:
            title = link.get_text(strip=True)
            url = BASE_URL + link.get('href', '')

            if title:
                sessions.append({
                    'title': title,
                    'date': day,
                    'time': time,
                    'end_time': '',
                    'speaker': '',
                    'session_type': 'Exhibitor Talk',
                    'url': url,
                    'abstract': ''
                })

    return sessions

async def main():
    """Main extraction function."""
    print("Fetching NeurIPS 2025 San Diego calendar...")
    url = "https://neurips.cc/virtual/2025/loc/san-diego/calendar"

    html = await fetch_page(url)
    print(f"Fetched {len(html)} bytes")

    print("Parsing sessions...")
    sessions = parse_sessions(html)

    print(f"Found {len(sessions)} sessions")

    # Save to JSON
    output_file = "neurips_2025_sandiego_sessions.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_file}")

    # Convert to CSV
    import csv
    csv_file = "neurips_2025_sandiego_sessions.csv"
    if sessions:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sessions[0].keys())
            writer.writeheader()
            writer.writerows(sessions)
        print(f"Saved to {csv_file}")

    # Print summary by type
    print("\n=== Sessions by Type ===")
    type_counts = {}
    for s in sessions:
        t = s['session_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    return sessions

if __name__ == "__main__":
    asyncio.run(main())
