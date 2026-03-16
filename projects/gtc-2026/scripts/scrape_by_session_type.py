#!/usr/bin/env python3
"""
GTC 2026 Session Scraper - by Session Type
Scrapes all sessions by iterating through each session type filter.
Output: gtc-2026-all-sessions-new.csv + comparison report
"""

import csv
import json
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

BASE_DIR = Path(__file__).parent.parent
OLD_CSV = BASE_DIR / "gtc-2026-all-sessions-backup.csv"
NEW_CSV = BASE_DIR / "gtc-2026-all-sessions-new.csv"
REPORT = BASE_DIR / "session_update_report.txt"

SESSION_TYPES = [
    "Certification",
    "Connect With the Experts",
    "Fireside Chat",
    "Full-Day Workshop",
    "Hackathon",
    "Keynote",
    "Lightning Talk",
    "Panel",
    "Posters",
    "Pregame",
    "Q&A With NVIDIA Experts",
    "Self-Paced Training",
    "Talk",
    "Theater Talk",
    "Training Lab",
    "Tutorial",
    "Watch Party",
]


def run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def js(script):
    # Collapse multiline to single line before escaping
    single = " ".join(script.split())
    escaped = single.replace('"', '\\"').replace("`", "\\`")
    return run(f'agent-browser eval "{escaped}"', 30)


def get_sessions_on_page():
    # Return array directly (not JSON.stringify) so agent-browser doesn't double-wrap
    result = js("""
    Array.from(document.querySelectorAll('a')).filter(a => /\\[[A-Z]{1,6}\\d{4,5}\\]/.test(a.textContent)).map(a => {
        const match = a.textContent.match(/\\[([A-Z]{1,6}\\d{4,5})\\]/);
        return { session_id: match ? match[1] : null, title: a.textContent.replace(/\\[[A-Z]{1,6}\\d+\\]/, '').trim(), url: a.href };
    }).filter(s => s.session_id)
    """)
    try:
        return json.loads(result) if result and result.startswith("[") else []
    except json.JSONDecodeError:
        return []


def get_see_all_urls():
    """Extract the URLs that each 'See All' button navigates to."""
    # Each See All button navigates to a slot-filtered view=all URL.
    # We simulate a click on each and capture the resulting URL, then go back.
    result = js("""
    Array.from(document.querySelectorAll('button')).filter(b => b.textContent.trim() === 'See All').map((b, i) => i)
    """)
    try:
        indices = json.loads(result) if result and result.startswith("[") else []
    except json.JSONDecodeError:
        indices = []

    urls = []
    for idx in indices:
        # Click the button at this index
        js(f"""
        Array.from(document.querySelectorAll('button')).filter(b => b.textContent.trim() === 'See All')[{idx}]?.click()
        """)
        time.sleep(2)
        url = run("agent-browser get url", 10)
        if url and "view=all" in url:
            urls.append(url)
        run("agent-browser back", 30)
        time.sleep(2)

    return urls


def scrape_page_all_sessions(seen_ids: set) -> list[dict]:
    """Scrape all sessions from current page, handling Load More."""
    sessions = []
    no_new_streak = 0

    for i in range(50):
        current = get_sessions_on_page()
        new_count = 0
        for s in current:
            if s["session_id"] not in seen_ids:
                seen_ids.add(s["session_id"])
                sessions.append(s)
                new_count += 1

        if new_count > 0:
            no_new_streak = 0
        else:
            no_new_streak += 1

        if no_new_streak >= 4:
            break

        # Try Load More
        status = js("""(() => {
            const btn = Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Load More');
            if (btn && !btn.disabled) { btn.click(); return 'clicked'; }
            return btn ? 'disabled' : 'notfound';
        })()""")

        if status == "clicked":
            time.sleep(3)
        elif status == "disabled":
            for s in get_sessions_on_page():
                if s["session_id"] not in seen_ids:
                    seen_ids.add(s["session_id"])
                    sessions.append(s)
            break
        else:
            run("agent-browser scroll down 3000")
            time.sleep(2)

    return sessions


def scrape_session_type(session_type: str) -> list[dict]:
    base_url = f"https://www.nvidia.com/gtc/session-catalog/?sessionTypes={quote(session_type)}"
    print(f"\n[{session_type}]")

    run(f'agent-browser open "{base_url}"', 90)
    time.sleep(4)
    run("agent-browser eval \"document.querySelector('button[class*=Accept]')?.click()\"")
    time.sleep(1)

    seen_ids: set = set()
    all_sessions: list[dict] = []

    # Step 1: collect sessions visible on the main page
    main_sessions = scrape_page_all_sessions(seen_ids)
    all_sessions.extend(main_sessions)
    print(f"  main page: {len(main_sessions)} sessions")

    # Step 2: find all "See All" slot URLs
    slot_urls = get_see_all_urls()
    print(f"  found {len(slot_urls)} slot pages")

    # Step 3: visit each slot's view=all page
    for slot_url in slot_urls:
        run(f'agent-browser open "{slot_url}"', 60)
        time.sleep(3)
        slot_sessions = scrape_page_all_sessions(seen_ids)
        all_sessions.extend(slot_sessions)
        if slot_sessions:
            print(f"  slot: +{len(slot_sessions)} ({len(all_sessions)} total) — {slot_url.split('slot=')[-1][:50]}")

    print(f"  TOTAL for [{session_type}]: {len(all_sessions)}")
    return all_sessions


def load_old_csv() -> dict[str, dict]:
    sessions = {}
    with open(OLD_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sessions[row["session_id"]] = row
    return sessions


def main():
    print("=" * 60)
    print("GTC 2026 Session Scraper - By Session Type")
    print("=" * 60)

    all_sessions = {}  # session_id -> dict

    for st in SESSION_TYPES:
        sessions = scrape_session_type(st)
        for s in sessions:
            if s["session_id"] not in all_sessions:
                all_sessions[s["session_id"]] = s
        run("agent-browser close")
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"Total unique sessions scraped: {len(all_sessions)}")

    # Save new CSV
    with open(NEW_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "title", "url"])
        writer.writeheader()
        for s in all_sessions.values():
            writer.writerow({k: s.get(k, "") for k in ["session_id", "title", "url"]})
    print(f"Saved: {NEW_CSV.name}")

    # Compare with old
    old = load_old_csv()
    new_ids = set(all_sessions.keys())
    old_ids = set(old.keys())

    added = new_ids - old_ids
    removed = old_ids - new_ids

    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"旧 CSV:  {len(old_ids)} sessions")
    print(f"新抓取:  {len(new_ids)} sessions")
    print(f"净变化:  {len(new_ids)-len(old_ids):+d}")
    print(f"新增:    {len(added)}")
    print(f"移除:    {len(removed)}")

    if added:
        print(f"\n--- 新增 ({len(added)}) ---")
        for sid in sorted(added):
            s = all_sessions[sid]
            print(f"  + [{sid}] {s['title'][:70]}")

    if removed:
        print(f"\n--- 移除 ({len(removed)}) ---")
        for sid in sorted(removed):
            s = old[sid]
            print(f"  - [{sid}] {s['title'][:70]}")

    # Save report
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(f"GTC 2026 Session Update Report\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"旧: {len(old_ids)} sessions\n新: {len(new_ids)} sessions\n净变化: {len(new_ids)-len(old_ids):+d}\n")
        f.write(f"新增: {len(added)}  移除: {len(removed)}\n\n")
        if added:
            f.write("=== 新增 ===\n")
            for sid in sorted(added):
                s = all_sessions[sid]
                f.write(f"[{sid}] {s['title']}\n{s['url']}\n\n")
        if removed:
            f.write("=== 移除 ===\n")
            for sid in sorted(removed):
                s = old[sid]
                f.write(f"[{sid}] {s['title']}\n{s.get('url','')}\n\n")
    print(f"\nReport: {REPORT.name}")


if __name__ == "__main__":
    main()
