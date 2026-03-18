#!/usr/bin/env python3
"""
Verify GTC 2026 session titles against live pages, then re-scrape mismatched sessions.

Phase 1: Open each session page, extract live title, compare to JSON title.
Phase 2: Re-scrape full details for mismatched sessions; preserve date/time/location.
"""

import base64
import csv
import json
import subprocess
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
INPUT_JSON = PROJECT_DIR / "gtc-2026-sessions-detailed.json"
OUTPUT_JSON = PROJECT_DIR / "gtc-2026-sessions-detailed.json"
OUTPUT_CSV = PROJECT_DIR / "gtc-2026-sessions-detailed.csv"
MISMATCH_REPORT = PROJECT_DIR / ".mismatch_report.json"
PROGRESS_FILE = PROJECT_DIR / ".verify_rescrape_progress.json"

# Minimal JS: extract just title + session_id for fast verification
VERIFY_JS = r"""
(function() {
  const h1 = document.querySelector('h1')?.textContent?.trim() || '';
  const match = h1.match(/(.+?)\s*\[([A-Za-z0-9]+)\]/);
  const result = { title: match ? match[1].trim() : h1, session_id: match ? match[2] : '' };
  return btoa(unescape(encodeURIComponent(JSON.stringify(result))));
})();
"""

# Full extraction JS (reused from extract_session_details.py)
EXTRACT_JS = r"""
(function() {
  const bodyText = document.body.innerText;
  const heading = document.querySelector('h1')?.textContent?.trim() || '';
  const titleMatch = heading.match(/(.+?)\s*\[([A-Za-z0-9]+)\]/);
  const title = titleMatch ? titleMatch[1].trim() : heading;
  const sessionId = titleMatch ? titleMatch[2] : '';

  const badges = Array.from(document.querySelectorAll('.badge')).map(b => b.textContent.trim());
  const sessionType = badges[0] || '';

  const hasInPerson = badges.includes('In-Person');
  const hasVirtual = badges.includes('Virtual');
  let format = '';
  if (hasInPerson && hasVirtual) format = 'Both';
  else if (hasInPerson) format = 'In-Person';
  else if (hasVirtual) format = 'Virtual';

  const hasNotRecorded = badges.some(b => b.toLowerCase().includes('not record'));
  const recording = hasNotRecorded ? 'No' : 'Yes';

  const abstractEl = document.querySelector('div.abstract');
  const abstract = abstractEl ? abstractEl.textContent.trim() : '';

  const speakerLines = [];
  document.querySelectorAll('span.p--medium').forEach(span => {
    const text = span.textContent.trim();
    if (text.includes(' | ')) {
      const parts = text.split(' | ').map(p => p.trim());
      if (parts.length >= 2) {
        speakerLines.push({
          name: parts[0],
          title: parts.length >= 3 ? parts[1] : '',
          company: parts[parts.length - 1]
        });
      }
    }
  });

  const industryMatch = bodyText.match(/Industry:\s*([^\n]+)/);
  const industry = industryMatch ? industryMatch[1].trim() : '';

  const topicMatch = bodyText.match(/Topic:\s*([^\n]+)/);
  const topic = topicMatch ? topicMatch[1].trim() : '';

  const techLevelMatch = bodyText.match(/Technical Level:\s*([^\n]+)/);
  const technicalLevel = techLevelMatch ? techLevelMatch[1].trim() : '';

  const audienceMatch = bodyText.match(/Intended Audience:\s*([^\n]+)/);
  const intendedAudience = audienceMatch ? audienceMatch[1].trim() : '';

  const techMatch = bodyText.match(/NVIDIA Technology:\s*([^\n]+)/);
  const nvidiaTechnology = techMatch ? techMatch[1].trim() : '';

  const keyTakeaways = [];
  const takeawayLabel = Array.from(document.querySelectorAll('span, b')).find(el =>
    el.textContent.includes('Key Takeaways:')
  );
  if (takeawayLabel) {
    const parent = takeawayLabel.parentElement;
    if (parent) {
      const ul = parent.querySelector('ul');
      if (ul) {
        Array.from(ul.querySelectorAll('li')).forEach(li => {
          const text = li.textContent.trim();
          if (text) keyTakeaways.push(text);
        });
      }
    }
  }
  if (keyTakeaways.length === 0) {
    const takeawaysMatch = bodyText.match(/Key Takeaways:\n([\s\S]+?)(?:\n(?:Add to Schedule|Register|Share)|$)/);
    if (takeawaysMatch) {
      const items = takeawaysMatch[1].split(/\n[•*\-]?\s*/).filter(t => t.trim());
      items.forEach(t => {
        const trimmed = t.trim();
        if (trimmed && !trimmed.includes('Add to Schedule') && !trimmed.includes('Register')) {
          keyTakeaways.push(trimmed);
        }
      });
    }
  }

  const data = {
    session_id: sessionId,
    title: title,
    session_type: sessionType,
    format: format,
    recording: recording,
    abstract: abstract,
    industry: industry,
    topic: topic,
    technical_level: technicalLevel,
    intended_audience: intendedAudience,
    nvidia_technology: nvidiaTechnology,
    key_takeaways: keyTakeaways,
    speakers: speakerLines,
    url: window.location.href
  };
  const jsonString = JSON.stringify(data);
  return btoa(unescape(encodeURIComponent(jsonString)));
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


def decode_b64_json(output: str) -> dict | None:
    output = output.strip()
    try:
        decoded = base64.b64decode(output).decode("utf-8")
        return json.loads(decoded)
    except Exception:
        try:
            return json.loads(output)
        except Exception:
            return None


def open_page(url: str) -> bool:
    for attempt in range(3):
        ok, _ = run_browser(["open", url], timeout=90)
        if ok:
            return True
        if attempt < 2:
            time.sleep(5)
    return False


def dismiss_cookie_banner():
    run_browser(["eval", "document.getElementById('onetrust-accept-btn-handler')?.click()"], timeout=10)
    time.sleep(1)


def eval_js(js: str) -> dict | None:
    dismiss_cookie_banner()
    ok, output = run_browser(["eval", js], timeout=30)
    if not ok or not output.strip():
        return None
    return decode_b64_json(output)


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"verified_ids": [], "rescrape_ids": [], "completed_ids": []}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


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


def phase1_verify(sessions: list, progress: dict) -> list:
    """Check live page titles against JSON titles; return list of mismatched session_ids."""
    already_verified = set(progress["verified_ids"])
    rescrape_ids = set(progress["rescrape_ids"])

    pending = [s for s in sessions if s.get("session_id") and s["session_id"] not in already_verified]
    print(f"\nPhase 1 — Title Verification")
    print(f"Already verified: {len(already_verified)}, Pending: {len(pending)}")

    consecutive_failures = 0
    MAX_FAILURES = 5

    try:
        for i, session in enumerate(pending):
            sid = session["session_id"]
            url = session.get("url", "")
            json_title = session.get("title", "").strip().lower()

            print(f"  [{i+1}/{len(pending)}] {sid}: {session.get('title','')[:50]}...", end=" ", flush=True)

            if not url:
                print("skip (no URL)")
                already_verified.add(sid)
                progress["verified_ids"] = list(already_verified)
                progress["rescrape_ids"] = list(rescrape_ids)
                continue

            if not open_page(url):
                print("FAIL (open)")
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print(f"\nToo many consecutive failures. Saving progress.")
                    break
                continue

            consecutive_failures = 0
            time.sleep(2)

            data = eval_js(VERIFY_JS)
            if not data:
                print("FAIL (eval)")
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print(f"\nToo many consecutive failures. Saving progress.")
                    break
                continue

            live_title = data.get("title", "").strip().lower()
            already_verified.add(sid)

            if live_title and live_title != json_title:
                rescrape_ids.add(sid)
                print(f"MISMATCH\n    JSON: {json_title[:60]}\n    Live: {live_title[:60]}")
            else:
                print("OK")

            progress["verified_ids"] = list(already_verified)
            progress["rescrape_ids"] = list(rescrape_ids)

            if (i + 1) % 50 == 0:
                save_progress(progress)
                print(f"  [Progress saved: {i+1}/{len(pending)}]")

    except KeyboardInterrupt:
        print("\nInterrupted during Phase 1.")

    save_progress(progress)
    return list(rescrape_ids)


def phase2_rescrape(sessions: list, rescrape_ids: list, progress: dict) -> list:
    """Re-scrape full details for mismatched sessions; preserve date/time/location/key_themes."""
    completed_ids = set(progress["completed_ids"])
    session_map = {s["session_id"]: s for s in sessions}

    pending = [sid for sid in rescrape_ids if sid not in completed_ids]
    print(f"\nPhase 2 — Re-scraping {len(pending)} mismatched sessions")
    print(f"Already re-scraped: {len(completed_ids)}")

    if not pending:
        print("Nothing to re-scrape.")
        return sessions

    consecutive_failures = 0
    MAX_FAILURES = 5

    try:
        for i, sid in enumerate(pending):
            session = session_map.get(sid)
            if not session:
                print(f"  [{i+1}/{len(pending)}] {sid}: not found in session_map, skipping")
                completed_ids.add(sid)
                continue

            url = session.get("url", "")
            print(f"  [{i+1}/{len(pending)}] {sid}: {session.get('title','')[:50]}...", end=" ", flush=True)

            if not url:
                print("skip (no URL)")
                completed_ids.add(sid)
                continue

            if not open_page(url):
                print("FAIL (open)")
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print(f"\nToo many consecutive failures. Saving progress.")
                    break
                continue

            consecutive_failures = 0
            time.sleep(3)

            data = eval_js(EXTRACT_JS)
            if not data or not data.get("session_id"):
                print("FAIL (extract)")
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print(f"\nToo many consecutive failures. Saving progress.")
                    break
                continue

            # Verify extracted session_id matches expected
            if data["session_id"] != sid:
                print(f"FAIL (id mismatch: got {data['session_id']})")
                consecutive_failures += 1
                continue

            # Preserve already-patched schedule fields and key_themes
            data["date"] = session.get("date", "")
            data["time"] = session.get("time", "")
            data["location"] = session.get("location", "")
            data["key_themes"] = session.get("key_themes", [])

            session_map[sid] = data
            completed_ids.add(sid)
            print(f"OK - {data.get('title','')[:50]}")

            progress["completed_ids"] = list(completed_ids)

            if (i + 1) % 10 == 0:
                save_results(list(session_map.values()))
                save_progress(progress)
                print(f"  [Progress saved: {i+1}/{len(pending)}]")

    except KeyboardInterrupt:
        print("\nInterrupted during Phase 2.")

    save_results(list(session_map.values()))
    save_progress(progress)
    return list(session_map.values())


def main():
    print("GTC 2026 Session Title Verifier & Re-scraper")
    print("=" * 50)

    with open(INPUT_JSON, encoding="utf-8") as f:
        sessions = json.load(f)
    print(f"Loaded {len(sessions)} sessions from JSON")

    progress = load_progress()

    # Phase 1: verify titles
    rescrape_ids = phase1_verify(sessions, progress)

    # Reload sessions in case JSON was modified externally between runs
    with open(INPUT_JSON, encoding="utf-8") as f:
        sessions = json.load(f)

    # Save mismatch report
    mismatch_entries = [
        {"session_id": sid, "json_title": next((s["title"] for s in sessions if s["session_id"] == sid), ""),
         "url": next((s.get("url", "") for s in sessions if s["session_id"] == sid), "")}
        for sid in rescrape_ids
    ]
    with open(MISMATCH_REPORT, "w", encoding="utf-8") as f:
        json.dump(mismatch_entries, f, ensure_ascii=False, indent=2)

    print(f"\nVerification complete.")
    print(f"  Total checked: {len(progress['verified_ids'])}")
    print(f"  Mismatches:    {len(rescrape_ids)}")
    if rescrape_ids[:5]:
        print(f"  Sample IDs:    {rescrape_ids[:5]}")
    print(f"  Report saved:  {MISMATCH_REPORT}")

    if not rescrape_ids:
        print("\nNo mismatches found. Nothing to re-scrape.")
        PROGRESS_FILE.unlink(missing_ok=True)
        return

    # Phase 2: re-scrape mismatched sessions
    sessions = phase2_rescrape(sessions, rescrape_ids, progress)

    print(f"\nAll done!")
    print(f"  Re-scraped: {len(progress['completed_ids'])} sessions")
    print(f"  JSON: {OUTPUT_JSON}")
    print(f"  CSV:  {OUTPUT_CSV}")

    # Clean up progress file on success
    all_done = set(progress["rescrape_ids"]) == set(progress["completed_ids"])
    if all_done:
        PROGRESS_FILE.unlink(missing_ok=True)
        print("  Progress file cleaned up.")


if __name__ == "__main__":
    main()
