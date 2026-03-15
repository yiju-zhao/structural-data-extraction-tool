#!/usr/bin/env python3
"""
Extract detailed information from GTC 2026 session pages.
This script reads session URLs from a CSV file and extracts detailed information from each page.
"""

import csv
import json
import subprocess
import time
import os
import re
import sys
from pathlib import Path

# Configuration - paths are relative to parent directory (gtc-2026)
PROJECT_DIR = Path(__file__).parent.parent
INPUT_CSV = PROJECT_DIR / "gtc-2026-all-sessions-new.csv"
OUTPUT_JSON = PROJECT_DIR / "gtc-2026-sessions-detailed.json"
OUTPUT_CSV = PROJECT_DIR / "gtc-2026-sessions-detailed.csv"
PROGRESS_FILE = PROJECT_DIR / ".extraction_progress.json"

# JavaScript to extract session details
EXTRACT_JS_FILE = PROJECT_DIR / ".extract_session.js"
EXTRACT_OUTPUT_FILE = PROJECT_DIR / ".extract_output.json"

EXTRACT_JS = '''
(function() {
  const bodyText = document.body.innerText;
  const heading = document.querySelector('h1')?.textContent?.trim() || '';
  const titleMatch = heading.match(/(.+?)\\s*\\[([A-Za-z0-9]+)\\]/);
  const title = titleMatch ? titleMatch[1].trim() : heading;
  const sessionId = titleMatch ? titleMatch[2] : '';

  // Extract badges from DOM
  const badges = Array.from(document.querySelectorAll('.badge')).map(b => b.textContent.trim());

  // Session type is the first badge (Keynote, Talk, Panel, etc.)
  const sessionType = badges[0] || '';

  // Format: check for In-Person and/or Virtual badges
  const hasInPerson = badges.includes('In-Person');
  const hasVirtual = badges.includes('Virtual');
  let format = '';
  if (hasInPerson && hasVirtual) format = 'Both';
  else if (hasInPerson) format = 'In-Person';
  else if (hasVirtual) format = 'Virtual';

  // Recording: Not Recorded badge → No, else Yes
  const hasNotRecorded = badges.some(b => b.toLowerCase().includes('not record'));
  const recording = hasNotRecorded ? 'No' : 'Yes';

  // Extract abstract from div.abstract element
  const abstractEl = document.querySelector('div.abstract');
  const abstract = abstractEl ? abstractEl.textContent.trim() : '';

  // Extract speakers from DOM (span.p--medium elements containing " | ")
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

  // Extract other fields from text
  const industryMatch = bodyText.match(/Industry:\\s*([^\\n]+)/);
  const industry = industryMatch ? industryMatch[1].trim() : '';

  const topicMatch = bodyText.match(/Topic:\\s*([^\\n]+)/);
  const topic = topicMatch ? topicMatch[1].trim() : '';

  const techLevelMatch = bodyText.match(/Technical Level:\\s*([^\\n]+)/);
  const technicalLevel = techLevelMatch ? techLevelMatch[1].trim() : '';

  const audienceMatch = bodyText.match(/Intended Audience:\\s*([^\\n]+)/);
  const intendedAudience = audienceMatch ? audienceMatch[1].trim() : '';

  const techMatch = bodyText.match(/NVIDIA Technology:\\s*([^\\n]+)/);
  const nvidiaTechnology = techMatch ? techMatch[1].trim() : '';

  // Extract key takeaways from DOM (find "Key Takeaways:" label, then adjacent ul/li)
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
  // Fallback to text parsing if DOM method fails
  if (keyTakeaways.length === 0) {
    const takeawaysMatch = bodyText.match(/Key Takeaways:\\n([\\s\\S]+?)(?:\\n(?:Add to Schedule|Register|Share)|$)/);
    if (takeawaysMatch) {
      const items = takeawaysMatch[1].split(/\\n[•*\\-]?\\s*/).filter(t => t.trim());
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
'''


def run_browser_cmd(cmd_args: list, timeout: int = 60) -> tuple[bool, str]:
    """Run an agent-browser command and return (success, output)."""
    try:
        result = subprocess.run(
            ["agent-browser"] + cmd_args,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout.strip() + result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def open_page(url: str) -> bool:
    """Open a URL in the browser."""
    success, output = run_browser_cmd(["open", url], timeout=90)
    if not success:
        print(f"  Browser open error: {output[:100]}")
    return success


def dismiss_cookie_banner():
    """Dismiss the NVIDIA cookie consent banner if present."""
    run_browser_cmd(["eval", "document.getElementById('onetrust-accept-btn-handler')?.click()"], timeout=10)
    time.sleep(1)


def extract_session_data() -> dict | None:
    """Extract session data from the current page using JS file."""
    try:
        dismiss_cookie_banner()

        with open(EXTRACT_JS_FILE, 'r') as f:
            js_code = f.read()

        success, output = run_browser_cmd(["eval", js_code], timeout=30)

        if success and output.strip():
            output = output.strip()
            # Decode base64 encoded JSON
            import base64
            try:
                decoded_bytes = base64.b64decode(output)
                decoded_str = decoded_bytes.decode('utf-8')
                result = json.loads(decoded_str)
                return result
            except (base64.binascii.Error, UnicodeDecodeError) as e:
                # Fallback: try parsing as plain JSON (for backward compatibility)
                try:
                    result = json.loads(output)
                    if isinstance(result, str):
                        result = json.loads(result)
                    return result
                except json.JSONDecodeError:
                    print(f"  Base64 decode error: {e}, raw output: {output[:100]}")
                    return None
        else:
            print(f"  Eval error: {output[:100] if output else 'unknown'}")
            return None
    except json.JSONDecodeError as e:
        print(f"  JSON error: {e}")
        return None
    except Exception as e:
        print(f"  Extract error: {e}")
        return None


def load_progress() -> int:
    """Load the last processed index from progress file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            data = json.load(f)
            return data.get('last_index', 0)
    return 0


def save_progress(index: int):
    """Save the current progress."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({'last_index': index}, f)


def load_existing_results() -> list:
    """Load existing results if resuming."""
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_results(results: list):
    """Save results to JSON and CSV files."""
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(results)} sessions to JSON")

    csv_rows = []
    for session in results:
        row = {
            'session_id': session.get('session_id', ''),
            'title': session.get('title', ''),
            'session_type': session.get('session_type', ''),
            'format': session.get('format', ''),
            'recording': session.get('recording', ''),
            'abstract': session.get('abstract', ''),
            'industry': session.get('industry', ''),
            'topic': session.get('topic', ''),
            'technical_level': session.get('technical_level', ''),
            'intended_audience': session.get('intended_audience', ''),
            'nvidia_technology': session.get('nvidia_technology', ''),
            'key_takeaways': ' | '.join(session.get('key_takeaways', [])),
            'speakers': json.dumps(session.get('speakers', []), ensure_ascii=False),
            'url': session.get('url', '')
        }
        csv_rows.append(row)

    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  Saved {len(csv_rows)} sessions to CSV")


def read_session_urls() -> list:
    """Read session URLs from the input CSV."""
    sessions = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sessions.append({
                'session_id': row.get('session_id', ''),
                'title': row.get('title', ''),
                'url': row.get('url', ''),
                'session_type': row.get('session_type', '')
            })
    return sessions


def main():
    print("GTC 2026 Session Details Extractor")
    print("=" * 50)

    with open(EXTRACT_JS_FILE, 'w') as f:
        f.write(EXTRACT_JS)
    print(f"JS extraction code saved to {EXTRACT_JS_FILE}")

    sessions = read_session_urls()
    total = len(sessions)
    print(f"Total sessions to process: {total}")

    results = load_existing_results()
    processed_ids = {r.get('session_id') for r in results if r.get('session_id')}
    pending = [s for s in sessions if s['session_id'] not in processed_ids]

    print(f"Already processed: {len(processed_ids)}, Remaining: {len(pending)}")

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5

    try:
        for i, session in enumerate(pending):
            session_id = session['session_id']
            url = session['url']

            print(f"\n[{i + 1}/{len(pending)}] {session_id}: {session['title'][:40]}...")

            opened = False
            for retry in range(3):
                if open_page(url):
                    opened = True
                    break
                print(f"  Retry {retry + 1}/3...")
                time.sleep(5)

            if not opened:
                print(f"  Failed to open page after 3 retries")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"\nToo many consecutive failures. Saving progress and stopping.")
                    save_results(results)
                    return
                continue

            consecutive_failures = 0
            time.sleep(3)

            data = extract_session_data()
            if data and data.get('session_id'):
                # Use session_type from CSV if available
                csv_session_type = session.get('session_type', '')
                if csv_session_type:
                    data['session_type'] = csv_session_type
                results.append(data)
                print(f"  OK - Format: {data.get('format')}, Recording: {data.get('recording')}, Speakers: {len(data.get('speakers', []))}")
            else:
                print(f"  Failed to extract data")
                results.append({
                    'session_id': session_id,
                    'title': session['title'],
                    'url': url,
                    'error': 'Failed to extract'
                })

            if (i + 1) % 10 == 0:
                save_results(results)
                print(f"  Progress saved ({i + 1}/{len(pending)})")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...")
        save_results(results)
        save_progress(len(results))
        print("Progress saved. Run again to resume.")
        return

    save_results(results)
    save_progress(len(results))
    print(f"\nExtraction complete! Total sessions: {len(results)}")

    run_browser_cmd(["close"])


if __name__ == "__main__":
    main()
