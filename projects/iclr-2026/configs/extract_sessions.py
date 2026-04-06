#!/usr/bin/env python3
"""Extract ICLR 2025 oral and poster sessions from calendar page."""

import json
import csv
from playwright.sync_api import sync_playwright

BASE_URL = "https://iclr.cc"
CALENDAR_URL = f"{BASE_URL}/virtual/2025/calendar"

def extract_sessions():
    """Extract oral and poster sessions with date, time, and end_time."""
    sessions = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(CALENDAR_URL, wait_until="networkidle")
        page.wait_for_timeout(3000)

        # Process each day container
        day_containers = page.query_selector_all("[class*='container2'][class*='day-']")

        for day in day_containers:
            # Extract date from first line of day container
            date_text = day.evaluate(
                "el => el.innerText.trim().split(String.fromCharCode(10))[0]"
            )

            # Find all timeboxes within this day
            timeboxes = day.query_selector_all(".timebox")

            for timebox in timeboxes:
                # Get time from timebox
                time_el = timebox.query_selector(".time")
                time_text = time_el.inner_text().strip() if time_el else ""

                # Find sessions in this timebox
                session_els = timebox.query_selector_all(".eventsession")

                for session_el in session_els:
                    # Get title and URL
                    title_link = session_el.query_selector(".title-style a, a")
                    if not title_link:
                        continue

                    title = title_link.inner_text().strip()
                    url = title_link.get_attribute("href") or ""

                    # Filter: only oral and poster sessions
                    title_lower = title.lower()
                    if "oral" not in title_lower and "poster" not in title_lower:
                        continue

                    # Determine type from title
                    if "oral" in title_lower:
                        session_type = "oral"
                    else:
                        session_type = "poster"

                    # Get end time
                    end_time_el = session_el.query_selector(".end-time")
                    end_time = ""
                    if end_time_el:
                        end_time_text = end_time_el.inner_text().strip()
                        # Extract time from "(ends 11:00 AM)" format
                        if "ends" in end_time_text.lower():
                            end_time = end_time_text.replace("(", "").replace(")", "")
                            end_time = end_time.replace("ends", "").strip()

                    # Make URL absolute
                    if url and not url.startswith("http"):
                        url = BASE_URL + url

                    sessions.append({
                        "title": title,
                        "type": session_type,
                        "date": date_text,
                        "time": time_text,
                        "end_time": end_time,
                        "url": url
                    })

        browser.close()

    return sessions

def save_json(sessions, filepath):
    """Save sessions to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)

def save_csv(sessions, filepath):
    """Save sessions to CSV file."""
    if not sessions:
        return

    fieldnames = ["title", "type", "date", "time", "end_time", "url"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sessions)

if __name__ == "__main__":
    print("Extracting ICLR 2025 oral and poster sessions...")
    sessions = extract_sessions()

    print(f"Found {len(sessions)} sessions")

    # Save outputs
    save_json(sessions, "output/sessions.json")
    save_csv(sessions, "output/sessions.csv")

    print("Saved to output/sessions.json and output/sessions.csv")
