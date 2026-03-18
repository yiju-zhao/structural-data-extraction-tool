#!/usr/bin/env python3
"""
Patch GTC 2026 session JSON/CSV with missing date, time, and location info.
Reads existing detailed JSON, opens each session page, extracts schedule data,
and writes updated JSON + CSV.
"""

import csv
import json
import subprocess
import time
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
INPUT_JSON = PROJECT_DIR / "gtc-2026-sessions-detailed.json"
OUTPUT_JSON = PROJECT_DIR / "gtc-2026-sessions-detailed.json"
OUTPUT_CSV = PROJECT_DIR / "gtc-2026-sessions-detailed.csv"
PROGRESS_FILE = PROJECT_DIR / ".patch_schedule_progress.json"

EXTRACT_JS = r"""
(function() {
  const text = document.body.innerText;
  const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);

  // Find schedule line: "DayName, Month DD  |  H:MM a.m./p.m. - H:MM a.m./p.m. PDT"
  const schedulePattern = /(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*(March|April)\s*\d+\s*\|\s*[\d:]+.+?(?:PDT|PST)/;
  const schedIdx = lines.findIndex(l => schedulePattern.test(l));
  const schedLine = schedIdx >= 0 ? lines[schedIdx] : '';

  // Location is the line immediately after the schedule line
  const location = (schedIdx >= 0 && schedIdx + 1 < lines.length) ? lines[schedIdx + 1] : '';

  // Parse out date and time from schedule line
  // e.g. "Monday, March 16  |  11:00 a.m. - 1:00 p.m. PDT"
  let date = '';
  let timeSlot = '';
  if (schedLine) {
    const parts = schedLine.split('|').map(p => p.trim());
    if (parts.length >= 2) {
      date = parts[0].trim();
      timeSlot = parts.slice(1).join('|').trim();
    }
  }

  // Sanity check: skip nav items that appear after schedule (e.g. "Explore", "Conference Schedule")
  const navKeywords = ['explore', 'conference', 'session catalog', 'speakers', 'networking',
                       'exhibit', 'workshops', 'startups', 'poster', 'pricing', 'sponsors',
                       'travel', 'code of conduct', 'contact', 'faq', 'privacy'];
  const isNavLine = navKeywords.some(k => location.toLowerCase().includes(k));
  const cleanLocation = isNavLine ? '' : location;

  const result = {
    date: date,
    time: timeSlot,
    location: cleanLocation,
    raw_schedule: schedLine
  };
  return btoa(unescape(encodeURIComponent(JSON.stringify(result))));
})();
"""


def run_browser(args: list, timeout: int = 60) -> tuple[bool, str]:
    try:
        r = subprocess.run(["agent-browser"] + args, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def extract_schedule(url: str) -> dict | None:
    # Open page
    ok, _ = run_browser(["open", url], timeout=90)
    if not ok:
        for _ in range(2):
            time.sleep(5)
            ok, _ = run_browser(["open", url], timeout=90)
            if ok:
                break
    if not ok:
        return None

    # Dismiss cookie banner
    run_browser(["eval", "document.getElementById('onetrust-accept-btn-handler')?.click()"], timeout=10)
    time.sleep(2)

    # Run extraction JS
    ok, output = run_browser(["eval", EXTRACT_JS], timeout=30)
    if not ok or not output.strip():
        return None

    try:
        import base64
        decoded = base64.b64decode(output.strip()).decode("utf-8")
        return json.loads(decoded)
    except Exception:
        try:
            return json.loads(output.strip())
        except Exception:
            return None


def load_progress() -> set:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return set(json.load(f).get("patched_ids", []))
    return set()


def save_progress(patched_ids: set):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"patched_ids": list(patched_ids)}, f)


def save_results(sessions: list):
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(sessions)} sessions to JSON")

    fieldnames = [
        "session_id", "title", "session_type", "format", "recording",
        "abstract", "industry", "topic", "technical_level", "intended_audience",
        "nvidia_technology", "key_takeaways", "speakers", "url",
        "date", "time", "location", "key_themes"
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for s in sessions:
            row = dict(s)
            row["key_takeaways"] = " | ".join(s.get("key_takeaways", []))
            row["speakers"] = json.dumps(s.get("speakers", []), ensure_ascii=False)
            row["key_themes"] = " | ".join(s.get("key_themes", []))
            writer.writerow(row)
    print(f"  Saved {len(sessions)} sessions to CSV")


def main():
    print("GTC 2026 Schedule Info Patcher")
    print("=" * 50)

    with open(INPUT_JSON, encoding="utf-8") as f:
        sessions = json.load(f)
    print(f"Loaded {len(sessions)} sessions")

    patched_ids = load_progress()
    session_map = {s["session_id"]: s for s in sessions}

    # Find sessions missing schedule info (not already patched)
    needs_patch = [
        s for s in sessions
        if s.get("session_id") and s.get("session_id") not in patched_ids
        and (not s.get("date") or not s.get("time") or not s.get("location"))
    ]
    print(f"Already patched: {len(patched_ids)}, Need patching: {len(needs_patch)}")

    if not needs_patch:
        print("All sessions already have schedule info!")
        return

    consecutive_failures = 0
    MAX_FAILURES = 5

    try:
        for i, session in enumerate(needs_patch):
            sid = session["session_id"]
            url = session.get("url", "")
            print(f"\n[{i+1}/{len(needs_patch)}] {sid}: {session.get('title','')[:50]}...")

            if not url:
                print("  No URL, skipping")
                patched_ids.add(sid)
                continue

            data = extract_schedule(url)

            if data:
                session_map[sid]["date"] = data.get("date", "")
                session_map[sid]["time"] = data.get("time", "")
                session_map[sid]["location"] = data.get("location", "")
                print(f"  OK - {data.get('date')} | {data.get('time')} | {data.get('location')}")
                patched_ids.add(sid)
                consecutive_failures = 0
            else:
                print(f"  Failed to extract schedule")
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print(f"\nToo many consecutive failures. Saving and stopping.")
                    save_results(list(session_map.values()))
                    save_progress(patched_ids)
                    return

            if (i + 1) % 20 == 0:
                save_results(list(session_map.values()))
                save_progress(patched_ids)
                print(f"  Progress saved ({i+1}/{len(needs_patch)})")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Saving progress...")

    save_results(list(session_map.values()))
    save_progress(patched_ids)
    print(f"\nDone! Patched sessions total: {len(patched_ids)}")


if __name__ == "__main__":
    main()
