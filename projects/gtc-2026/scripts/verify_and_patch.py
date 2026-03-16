#!/usr/bin/env python3
"""
Verify CSV against each session type page count, patch missing sessions.
"""
import csv, json, subprocess, time
from pathlib import Path
from urllib.parse import quote

BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "gtc-2026-all-sessions-new.csv"

SESSION_TYPES = [
    "Certification", "Connect With the Experts", "Fireside Chat",
    "Full-Day Workshop", "Hackathon", "Keynote", "Lightning Talk",
    "Panel", "Posters", "Pregame", "Q&A With NVIDIA Experts",
    "Self-Paced Training", "Talk", "Theater Talk", "Training Lab",
    "Tutorial", "Watch Party",
]

EXPECTED = {
    "Certification": 5, "Connect With the Experts": 71, "Fireside Chat": 10,
    "Full-Day Workshop": 9, "Hackathon": 2, "Keynote": 1, "Lightning Talk": 17,
    "Panel": 65, "Posters": 158, "Pregame": 1, "Q&A With NVIDIA Experts": 3,
    "Self-Paced Training": 2, "Talk": 409, "Theater Talk": 82,
    "Training Lab": 66, "Tutorial": 18, "Watch Party": 35,
}


def run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def get_sessions_on_page():
    result = run('agent-browser eval "Array.from(document.querySelectorAll(\'a\')).filter(a => /\\[[A-Z]{1,6}\\d{4,5}[A-Z]?\\]/.test(a.textContent)).map(a => { const m = a.textContent.match(/\\[([A-Z]{1,6}\\d{4,5}[A-Z]?)\\]/); return { session_id: m?.[1], title: a.textContent.replace(/\\[[A-Z]{1,6}\\d{4,5}[A-Z]?\\]/,\'\').trim(), url: a.href }; }).filter(s=>s.session_id)"', 30)
    try:
        return json.loads(result) if result and result.startswith("[") else []
    except:
        return []


def scrape_all_for_type(session_type):
    """Scrape main page + all See All slot pages."""
    url = f"https://www.nvidia.com/gtc/session-catalog/?sessionTypes={quote(session_type)}"
    run(f'agent-browser open "{url}"', 90)
    time.sleep(4)
    run("agent-browser eval \"document.querySelector('button[class*=Accept]')?.click()\"")
    time.sleep(1)
    # Click "All" to include both In-Person and Virtual sessions
    run("agent-browser eval \"Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'All')?.click()\"")
    time.sleep(2)

    seen, sessions = set(), []

    def collect():
        for s in get_sessions_on_page():
            if s["session_id"] not in seen:
                seen.add(s["session_id"])
                sessions.append(s)

    collect()

    # Click each See All button one at a time, go back, collect
    n_btns_result = run('agent-browser eval "Array.from(document.querySelectorAll(\'button\')).filter(b=>b.textContent.trim()===\'See All\').length"', 15)
    try:
        n_btns = int(n_btns_result.strip('"'))
    except:
        n_btns = 0

    for idx in range(n_btns):
        run(f'agent-browser eval "Array.from(document.querySelectorAll(\'button\')).filter(b=>b.textContent.trim()===\'See All\')[{idx}]?.click()"', 10)
        time.sleep(2)
        collect()
        # Handle Load More on slot page
        for _ in range(20):
            before = len(seen)
            status = run('agent-browser eval "(() => { const b = Array.from(document.querySelectorAll(\'button\')).find(b=>b.textContent.trim()===\'Load More\'); if(b&&!b.disabled){b.click();return \'clicked\';} return b?\'disabled\':\'notfound\'; })()"', 15)
            if status == '"clicked"':
                time.sleep(3)
                collect()
            else:
                break
            if len(seen) == before:
                break
        run("agent-browser back", 30)
        time.sleep(2)

    return sessions


def main():
    # session_id -> {session_id, title, url, session_types: list}
    all_sessions: dict = {}

    for st in SESSION_TYPES:
        expected = EXPECTED.get(st, 0)
        if expected == 0:
            print(f"[{st}] expected 0, skip")
            continue

        print(f"\n[{st}] expected={expected}")
        scraped = scrape_all_for_type(st)
        run("agent-browser close")
        time.sleep(0.5)

        print(f"  scraped={len(scraped)}", end="")
        if len(scraped) < expected:
            print(f"  ⚠ < expected {expected}", end="")
        print()

        for s in scraped:
            sid = s["session_id"]
            if sid not in all_sessions:
                all_sessions[sid] = {
                    "session_id": sid,
                    "title": s["title"],
                    "url": s["url"],
                    "session_types": [st],
                }
            else:
                if st not in all_sessions[sid]["session_types"]:
                    all_sessions[sid]["session_types"].append(st)
                    print(f"  dup [{sid}] also in: {st}")

    # Save CSV with session_type column (comma-joined if multiple)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "title", "url", "session_type"])
        writer.writeheader()
        for s in all_sessions.values():
            writer.writerow({
                "session_id": s["session_id"],
                "title": s["title"],
                "url": s["url"],
                "session_type": ", ".join(s["session_types"]),
            })

    print(f"\n{'='*60}")
    print(f"Total unique sessions: {len(all_sessions)}")
    print(f"Saved to: {CSV_PATH.name}")

    # Show duplicates summary
    dups = {sid: s for sid, s in all_sessions.items() if len(s["session_types"]) > 1}
    if dups:
        print(f"\nSessions appearing in multiple types ({len(dups)}):")
        for sid, s in dups.items():
            print(f"  [{sid}] {', '.join(s['session_types'])}")


if __name__ == "__main__":
    main()
