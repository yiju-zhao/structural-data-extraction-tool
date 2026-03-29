#!/usr/bin/env python3
"""
Extract full session details for one day of ICSE 2026.
Usage: python3 extract_day_details.py <session_name> <day_label> <day_file> <output_file>
  session_name: browser session name (e.g., icse-sun-detail)
  day_label: day header text (e.g., "Sun 12 Apr")
  day_file: input JSON with title + data_event_modal from Phase 1
  output_file: output JSON path for enriched data
"""
import subprocess
import json
import base64
import unicodedata
import sys
import time
import re
import os

SESSION = sys.argv[1]
DAY_LABEL = sys.argv[2]
DAY_FILE = sys.argv[3]
OUTPUT_FILE = sys.argv[4]

URL = "https://conf.researchr.org/program/icse-2026/program-icse-2026/?date=Sun%2012%20Apr%202026"

UNICODE_REPLACEMENTS = {
    '\u2018': "'", '\u2019': "'",
    '\u201c': '"', '\u201d': '"',
    '\u2013': '-', '\u2014': '-',
    '\u2026': '...', '\u00a0': ' ', '\u200b': ''
}

def clean_unicode(text):
    if not isinstance(text, str):
        return text
    for old, new in UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)
    return unicodedata.normalize('NFKC', text)

def clean_data(data):
    if isinstance(data, str):
        return clean_unicode(data)
    elif isinstance(data, list):
        return [clean_data(item) for item in data]
    elif isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items()}
    return data

def run_browser(cmd):
    """Run an agent-browser command and return output."""
    full_cmd = f"agent-browser --session {SESSION} {cmd}"
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=60)
    return result.stdout.strip()

def run_eval(js):
    """Run JS eval in browser and return result."""
    # Use subprocess list form to avoid shell quoting issues with multi-line JS
    result = subprocess.run(
        ['agent-browser', '--session', SESSION, 'eval', js],
        capture_output=True, text=True, timeout=30
    )
    output = result.stdout.strip()
    # Remove any ANSI escape codes
    output = re.sub(r'\x1b\[[0-9;]*m', '', output)
    return output

def decode_b64_json(b64_str):
    """Decode base64-encoded JSON string."""
    b64_str = b64_str.strip().strip('"')
    raw = base64.b64decode(b64_str).decode('utf-8')
    return json.loads(raw)

def extract_table_row_data():
    """Extract data from table rows for the target day."""
    js = """(() => {
  const dayWrappers = document.querySelectorAll('div.day-wrapper');
  let targetWrapper = null;
  for (const w of dayWrappers) {
    const header = w.querySelector('h4.day-header');
    if (header && header.textContent.trim().startsWith('DAY_LABEL')) {
      targetWrapper = w;
      break;
    }
  }
  if (!targetWrapper) return btoa(unescape(encodeURIComponent(JSON.stringify({error: 'day not found'}))));

  const rows = targetWrapper.querySelectorAll('tr[data-slot-id]');
  const data = [];
  const seen = new Set();

  rows.forEach(row => {
    const link = row.querySelector('a[data-event-modal]');
    if (!link) return;
    const uuid = link.getAttribute('data-event-modal');
    if (seen.has(uuid)) return;
    seen.add(uuid);

    const startTime = row.querySelector('.start-time')?.textContent?.trim() || '';
    const duration = row.querySelector('.text-muted strong')?.textContent?.trim() || '';
    const eventType = row.querySelector('.event-type')?.textContent?.trim() || '';
    const track = row.querySelector('.prog-track')?.textContent?.trim() || '';
    const title = link.textContent?.trim() || '';

    // Extract authors from performers div
    const performersDiv = row.querySelector('.performers');
    const authors = [];
    if (performersDiv) {
      const authorLinks = performersDiv.querySelectorAll('a.navigate');
      authorLinks.forEach(a => {
        const name = a.textContent?.trim() || '';
        const profileUrl = a.href || '';
        const affSpan = a.nextElementSibling;
        const affiliation = (affSpan && affSpan.classList.contains('prog-aff')) ? affSpan.textContent?.trim() : '';
        if (name) authors.push({name, affiliation, profile_url: profileUrl});
      });
    }

    data.push({
      data_event_modal: uuid,
      title,
      start_time: startTime,
      duration,
      event_type: eventType,
      track,
      authors
    });
  });

  return btoa(unescape(encodeURIComponent(JSON.stringify(data))));
})()""".replace('DAY_LABEL', DAY_LABEL)

    output = run_eval(js)
    return decode_b64_json(output)

def extract_modal_data_batch(uuids):
    """Click a batch of session links and extract modal data.
    Returns dict of uuid -> modal data.
    """
    # Build JS that clicks one link, waits, extracts, closes
    results = {}
    for uuid in uuids:
        try:
            # Click the link
            click_js = "(() => { const l = document.querySelector('a[data-event-modal=\\'" + uuid + "\\']'); if(l){l.click();return 'ok';} return 'not found'; })()"
            click_result = run_eval(click_js)

            if 'not found' in click_result:
                results[uuid] = {"abstract": "", "track_full_name": "", "room": "", "session_name": ""}
                continue

            # Wait for modal to appear
            time.sleep(1.5)

            # Extract modal data
            extract_js_template = """(() => {
  const m = document.querySelector('#modal-__UUID__');
  if (!m) return btoa(unescape(encodeURIComponent(JSON.stringify({error: 'no modal'}))));

  const trackFull = m.querySelector('.modal-header .text-muted')?.textContent?.trim() || '';
  const dateTimeStr = m.querySelector('.modal-header strong')?.textContent?.trim() || '';
  const sessionLink = m.querySelector('.modal-header a.navigate');
  const sessionName = sessionLink ? sessionLink.textContent?.trim() : '';
  const roomLink = m.querySelector('.modal-header a.room-link');
  const room = roomLink ? roomLink.textContent?.trim() : '';

  // Extract abstract - get all p tags in event-description
  const descDiv = m.querySelector('.event-description');
  let abstract = '';
  if (descDiv) {
    const paragraphs = descDiv.querySelectorAll('p');
    abstract = Array.from(paragraphs).map(p => p.textContent?.trim()).filter(t => t).join('\\n\\n');
  }

  // Extract authors from modal (more detailed than table row)
  const authorBlocks = m.querySelectorAll('.media');
  const authors = [];
  authorBlocks.forEach(block => {
    const nameEl = block.querySelector('.media-heading');
    const name = nameEl ? nameEl.childNodes[0]?.textContent?.trim() : '';
    const affEl = block.querySelector('.text-black');
    const affiliation = affEl ? affEl.textContent?.trim() : '';
    const countryEl = block.querySelector('small');
    const country = countryEl ? countryEl.textContent?.trim() : '';
    const linkEl = block.closest('a');
    const profileUrl = linkEl ? linkEl.href : '';
    if (name) authors.push({name, affiliation, country, profile_url: profileUrl});
  });

  return btoa(unescape(encodeURIComponent(JSON.stringify({
    track_full_name: trackFull,
    datetime_str: dateTimeStr,
    room,
    session_name: sessionName,
    abstract,
    modal_authors: authors
  }))));
})()"""
            extract_js = extract_js_template.replace('__UUID__', uuid)

            modal_output = run_eval(extract_js)
            modal_data = decode_b64_json(modal_output)

            if 'error' in modal_data:
                results[uuid] = {"abstract": "", "track_full_name": "", "room": "", "session_name": ""}
            else:
                results[uuid] = modal_data

            # Close modal
            close_js = "(() => { const c = document.querySelector('#modal-" + uuid + " .close'); if(c){c.click();return 'closed';} return 'no close btn'; })()"
            run_eval(close_js)
            time.sleep(0.3)

        except Exception as e:
            print(f"  Error extracting {uuid}: {e}", file=sys.stderr)
            results[uuid] = {"abstract": "", "track_full_name": "", "room": "", "session_name": "", "error": str(e)}

    return results

def main():
    print(f"Starting extraction for {DAY_LABEL}...")

    # Step 1: Open page
    print("Opening page...")
    run_browser(f'open "{URL}"')
    time.sleep(2)
    run_browser('wait --load networkidle')

    # Step 2: Extract table row data
    print("Extracting table row data...")
    sessions = extract_table_row_data()
    print(f"Found {len(sessions)} sessions for {DAY_LABEL}")

    # Step 3: Loop through sessions to get modal data
    print("Extracting modal details...")
    uuids = [s['data_event_modal'] for s in sessions]

    batch_size = 10
    all_modal_data = {}

    for i in range(0, len(uuids), batch_size):
        batch = uuids[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(uuids) + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} ({i+1}-{min(i+batch_size, len(uuids))} of {len(uuids)})...")

        batch_results = extract_modal_data_batch(batch)
        all_modal_data.update(batch_results)

    # Step 4: Merge table row + modal data
    print("Merging data...")
    enriched = []
    for session in sessions:
        uuid = session['data_event_modal']
        modal = all_modal_data.get(uuid, {})

        # Use modal authors if available (more detailed), else table-row authors
        authors = modal.get('modal_authors', session.get('authors', []))
        if not authors:
            authors = session.get('authors', [])

        merged = {
            "title": session['title'],
            "data_event_modal": uuid,
            "start_time": session['start_time'],
            "duration": session['duration'],
            "event_type": session['event_type'],
            "track": session['track'],
            "track_full_name": modal.get('track_full_name', ''),
            "datetime_str": modal.get('datetime_str', ''),
            "room": modal.get('room', ''),
            "session_name": modal.get('session_name', ''),
            "abstract": modal.get('abstract', ''),
            "authors": authors
        }
        enriched.append(merged)

    # Step 5: Clean and save
    print("Cleaning and saving...")
    enriched = clean_data(enriched)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    # Stats
    with_abstract = sum(1 for s in enriched if s['abstract'])
    with_authors = sum(1 for s in enriched if s['authors'])
    print(f"\nDone! Saved {len(enriched)} sessions to {OUTPUT_FILE}")
    print(f"  With abstract: {with_abstract}")
    print(f"  With authors: {with_authors}")

    # Step 6: Close browser
    run_browser('close')

if __name__ == '__main__':
    # Force unbuffered output for progress visibility
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    main()
