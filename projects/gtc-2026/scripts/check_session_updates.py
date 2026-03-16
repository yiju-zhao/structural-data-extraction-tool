#!/usr/bin/env python3
"""
GTC 2026 Session Update Checker

Re-scrapes the session catalog and compares with existing CSV to find:
- New sessions added
- Sessions removed

Usage:
    python check_session_updates.py
"""

import csv
import json
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
EXISTING_CSV = BASE_DIR / "gtc-2026-all-sessions.csv"
FRESH_CSV = BASE_DIR / "gtc-2026-sessions-fresh.csv"

CATALOG_URL = "https://www.nvidia.com/gtc/session-catalog/"


def run_browser_command(cmd: str, timeout: int = 60) -> str:
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"  Timeout: {cmd[:60]}...")
        return ""
    except Exception as e:
        print(f"  Error: {e}")
        return ""


def open_browser(url: str) -> bool:
    run_browser_command(f'agent-browser open "{url}"', timeout=90)
    time.sleep(3)
    return True


def close_browser():
    run_browser_command("agent-browser close")
    time.sleep(1)


def eval_js(script: str) -> str:
    escaped = script.replace('"', '\\"').replace("`", "\\`")
    return run_browser_command(f'agent-browser eval "{escaped}"', timeout=30)


def accept_cookies():
    run_browser_command("agent-browser eval \"document.querySelector('button[class*=Accept]')?.click()\"")
    time.sleep(1)


def scrape_all_sessions() -> list[dict]:
    """Scrape the full catalog (no date filter) using Load More."""
    print(f"Opening full catalog: {CATALOG_URL}")
    open_browser(CATALOG_URL)
    accept_cookies()
    time.sleep(2)

    sessions = []
    seen_ids = set()
    no_new_count = 0

    extract_script = """
    Array.from(document.querySelectorAll('a')).filter(a => /\\[[A-Z]{1,6}\\d{4,5}\\]/.test(a.textContent)).map(a => {
        const match = a.textContent.match(/\\[([A-Z]{1,6}\\d{4,5})\\]/);
        return {
            session_id: match ? match[1] : null,
            title: a.textContent.replace(/\\[[A-Z]{1,6}\\d+\\]/, '').trim(),
            url: a.href
        };
    }).filter(s => s.session_id)
    """

    for i in range(60):  # Up to 60 rounds to handle 900+ sessions
        result = eval_js(extract_script)
        try:
            current = json.loads(result) if result.startswith("[") else []
            new_count = 0
            for s in current:
                if s["session_id"] not in seen_ids:
                    seen_ids.add(s["session_id"])
                    sessions.append(s)
                    new_count += 1
            if new_count == 0:
                no_new_count += 1
            else:
                no_new_count = 0
                print(f"  Round {i+1}: +{new_count} new ({len(sessions)} total)")
        except json.JSONDecodeError:
            time.sleep(1)
            continue

        clicked = eval_js(
            "(() => { const btn = Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Load More'); if (btn && !btn.disabled) { btn.click(); return true; } return false; })()"
        )
        if clicked == "true":
            time.sleep(2)
        else:
            run_browser_command("agent-browser scroll down 3000")
            time.sleep(1.5)

        if no_new_count >= 5:
            print(f"  No new sessions for 5 rounds, done. Total: {len(sessions)}")
            break

    return sessions


def load_existing_sessions() -> dict[str, dict]:
    sessions = {}
    with open(EXISTING_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sessions[row["session_id"]] = row
    return sessions


def main():
    print("=" * 60)
    print("GTC 2026 Session Update Checker")
    print("=" * 60)

    # Load existing
    existing = load_existing_sessions()
    print(f"\nExisting CSV: {len(existing)} sessions")

    # Scrape fresh (full catalog, no date filter)
    print("\nScraping full catalog (no date filter)...")
    all_fresh = scrape_all_sessions()
    close_browser()

    print(f"\nFresh scrape: {len(all_fresh)} unique sessions")

    # Save fresh CSV
    with open(FRESH_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "title", "url"])
        writer.writeheader()
        for s in all_fresh:
            writer.writerow({k: s.get(k, "") for k in ["session_id", "title", "url"]})
    print(f"Saved fresh CSV: {FRESH_CSV.name}")

    # Compare
    fresh_ids = {s["session_id"] for s in all_fresh}
    fresh_map = {s["session_id"]: s for s in all_fresh}
    existing_ids = set(existing.keys())

    added = fresh_ids - existing_ids
    removed = existing_ids - fresh_ids
    common = fresh_ids & existing_ids

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Previously: {len(existing_ids)} sessions")
    print(f"Now:        {len(fresh_ids)} sessions")
    print(f"Net change: {len(fresh_ids) - len(existing_ids):+d}")
    print(f"\nNew sessions added:   {len(added)}")
    print(f"Sessions removed:     {len(removed)}")
    print(f"Sessions unchanged:   {len(common)}")

    if added:
        print(f"\n--- NEW SESSIONS ({len(added)}) ---")
        for sid in sorted(added):
            s = fresh_map[sid]
            print(f"  + [{sid}] {s['title'][:70]}")

    if removed:
        print(f"\n--- REMOVED SESSIONS ({len(removed)}) ---")
        for sid in sorted(removed):
            s = existing[sid]
            print(f"  - [{sid}] {s['title'][:70]}")

    # Save comparison report
    report_path = BASE_DIR / "session_update_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"GTC 2026 Session Update Report\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Previously: {len(existing_ids)} sessions\n")
        f.write(f"Now:        {len(fresh_ids)} sessions\n")
        f.write(f"Net change: {len(fresh_ids) - len(existing_ids):+d}\n")
        f.write(f"New sessions added: {len(added)}\n")
        f.write(f"Sessions removed:   {len(removed)}\n\n")
        if added:
            f.write(f"NEW SESSIONS:\n")
            for sid in sorted(added):
                s = fresh_map[sid]
                f.write(f"  [{sid}] {s['title']}\n  URL: {s['url']}\n\n")
        if removed:
            f.write(f"REMOVED SESSIONS:\n")
            for sid in sorted(removed):
                s = existing[sid]
                f.write(f"  [{sid}] {s['title']}\n  URL: {s.get('url','')}\n\n")
    print(f"\nReport saved: {report_path.name}")


if __name__ == "__main__":
    main()
