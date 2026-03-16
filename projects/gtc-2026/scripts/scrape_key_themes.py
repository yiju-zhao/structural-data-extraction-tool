#!/usr/bin/env python3
"""
GTC 2026 Key Themes Scraper
Scrapes key theme filter labels from the sessions page, collects session IDs
per theme, then augments gtc-2026-sessions-detailed.json and .csv with a
new `key_themes` field.
"""

import csv
import json
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
JSON_PATH = BASE_DIR / "gtc-2026-sessions-detailed.json"
CSV_PATH = BASE_DIR / "gtc-2026-sessions-detailed.csv"
PROGRESS_FILE = BASE_DIR / ".key_themes_progress.json"

SESSIONS_URL = "https://www.nvidia.com/gtc/sessions/"


# ---------------------------------------------------------------------------
# Browser helpers (same pattern as scrape_by_session_type.py)
# ---------------------------------------------------------------------------

def run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def js(script):
    """Evaluate JS in current browser tab."""
    single = " ".join(script.split())
    escaped = single.replace('"', '\\"').replace("`", "\\`")
    return run(f'agent-browser eval "{escaped}"', 30)


def dismiss_cookie():
    js("document.getElementById('onetrust-accept-btn-handler')?.click()")
    time.sleep(1)


# ---------------------------------------------------------------------------
# Session ID collection helpers
# ---------------------------------------------------------------------------

def get_session_ids_on_page() -> list[str]:
    """Return all session IDs visible on the current page."""
    result = js(r"""
    Array.from(document.querySelectorAll('a'))
      .map(a => { const m = a.textContent.match(/\[([A-Z]{1,6}\d{4,5}[A-Z]?)\]/); return m ? m[1] : null; })
      .filter(Boolean)
    """)
    try:
        return json.loads(result) if result and result.startswith("[") else []
    except json.JSONDecodeError:
        return []


def collect_with_load_more(seen: set) -> list[str]:
    """Collect IDs on current page, paginating through Load More. Returns new IDs."""
    collected = []
    no_new_streak = 0

    for _ in range(50):
        ids = get_session_ids_on_page()
        new_count = 0
        for sid in ids:
            if sid not in seen:
                seen.add(sid)
                collected.append(sid)
                new_count += 1

        if new_count > 0:
            no_new_streak = 0
        else:
            no_new_streak += 1

        if no_new_streak >= 3:
            break

        status = js("""(() => {
            const btn = Array.from(document.querySelectorAll('button')).find(b => b.textContent.trim() === 'Load More');
            if (btn && !btn.disabled) { btn.click(); return 'clicked'; }
            return btn ? 'disabled' : 'notfound';
        })()""")

        if status == "clicked":
            time.sleep(3)
        elif status == "disabled":
            # Final harvest after disabled
            for sid in get_session_ids_on_page():
                if sid not in seen:
                    seen.add(sid)
                    collected.append(sid)
            break
        else:
            run("agent-browser scroll down 3000")
            time.sleep(2)

    return collected


# ---------------------------------------------------------------------------
# Progress checkpoint helpers
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"completed": {}}  # {theme_name: [session_ids]}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main scraping logic
# ---------------------------------------------------------------------------

def discover_themes() -> list[str]:
    """Open the sessions page and extract all key theme names."""
    print(f"Opening {SESSIONS_URL} ...")
    run(f'agent-browser open "{SESSIONS_URL}"', 90)
    time.sleep(4)
    dismiss_cookie()
    time.sleep(1)

    result = js(r"""
    Array.from(document.querySelectorAll('label[for^="keytheme-desktop-"]'))
      .map(el => el.getAttribute('for').replace('keytheme-desktop-', '').trim())
      .filter(n => n.length > 0)
    """)
    try:
        themes = json.loads(result) if result and result.startswith("[") else []
    except json.JSONDecodeError:
        themes = []

    print(f"Discovered {len(themes)} key themes: {themes}")
    return themes


def scrape_theme(theme: str) -> list[str]:
    """Activate a theme filter, collect all session IDs, then deactivate. Returns list of IDs."""
    print(f"\n  [{theme}]")
    seen: set = set()

    # Activate filter
    escaped_theme = theme.replace('"', '\\"').replace("'", "\\'")
    js(f'document.querySelector(\'label[for="keytheme-desktop-{escaped_theme}"]\')?.click()')
    time.sleep(2)

    # Collect IDs on main filtered page
    main_ids = collect_with_load_more(seen)
    print(f"    main page: {len(main_ids)} session IDs")

    # Count "See All" buttons (integer-parse trick from verify_and_patch.py)
    n_btns_result = run(
        'agent-browser eval "Array.from(document.querySelectorAll(\'button\')).filter(b=>b.textContent.trim()===\'See All\').length"',
        15
    )
    try:
        n_btns = int(n_btns_result.strip('"'))
    except (ValueError, AttributeError):
        n_btns = 0
    print(f"    See All buttons: {n_btns}")

    # For each slot's "See All"
    for idx in range(n_btns):
        run(
            f'agent-browser eval "Array.from(document.querySelectorAll(\'button\')).filter(b=>b.textContent.trim()===\'See All\')[{idx}]?.click()"',
            10
        )
        time.sleep(2)

        slot_ids = collect_with_load_more(seen)
        if slot_ids:
            print(f"    slot {idx}: +{len(slot_ids)} IDs")

        run("agent-browser back", 30)
        time.sleep(2)
        dismiss_cookie()

    # Deactivate filter (click label again only if still checked)
    is_checked = js(f'document.querySelector(\'input#keytheme-desktop-{escaped_theme}\')?.checked')
    if is_checked == "true":
        js(f'document.querySelector(\'label[for="keytheme-desktop-{escaped_theme}"]\')?.click()')
        time.sleep(2)

    all_ids = list(seen)
    print(f"    TOTAL for [{theme}]: {len(all_ids)} sessions")
    return all_ids


def main():
    print("=" * 60)
    print("GTC 2026 Key Themes Scraper")
    print("=" * 60)

    progress = load_progress()
    completed: dict[str, list[str]] = progress.get("completed", {})

    # Step 1: Discover themes (or use cached if all already done)
    themes = discover_themes()
    if not themes:
        print("ERROR: No themes discovered. Exiting.")
        return

    # Step 2: Scrape each theme (with resume support)
    for theme in themes:
        if theme in completed:
            print(f"  [{theme}] already done ({len(completed[theme])} IDs), skipping")
            continue

        ids = scrape_theme(theme)
        completed[theme] = ids
        progress["completed"] = completed
        save_progress(progress)

    run("agent-browser close")
    time.sleep(0.5)

    # Step 3: Build inverted map {session_id -> [themes]}
    print(f"\n{'='*60}")
    print("Building inverted session→themes map ...")
    session_to_themes: dict[str, list[str]] = {}
    for theme, ids in completed.items():
        for sid in ids:
            session_to_themes.setdefault(sid, []).append(theme)

    print(f"Sessions with at least one theme: {len(session_to_themes)}")

    # Step 4: Augment JSON
    print(f"Augmenting {JSON_PATH.name} ...")
    sessions = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    for session in sessions:
        sid = session.get("session_id", "")
        session["key_themes"] = session_to_themes.get(sid, [])
    JSON_PATH.write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8")
    matched = sum(1 for s in sessions if s.get("key_themes"))
    print(f"  {matched}/{len(sessions)} sessions have key_themes")

    # Step 5: Augment CSV
    print(f"Augmenting {CSV_PATH.name} ...")
    rows = []
    fieldnames = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "key_themes" not in fieldnames:
        fieldnames.append("key_themes")

    for row in rows:
        sid = row.get("session_id", "")
        themes_list = session_to_themes.get(sid, [])
        row["key_themes"] = " | ".join(themes_list)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  CSV updated with key_themes column (last column).")

    print(f"\n{'='*60}")
    print("Done! Theme counts:")
    for theme, ids in sorted(completed.items(), key=lambda x: -len(x[1])):
        print(f"  {len(ids):4d}  {theme}")
    print(f"\nDelete {PROGRESS_FILE.name} after verifying results.")


if __name__ == "__main__":
    main()
