#!/usr/bin/env python3
"""Extract MLSys 2026 sessions + invited talks via agent-browser."""

import base64
import csv
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

CALENDAR_URL = (
    "https://mlsys.org/virtual/2026/calendar"
    "?filter_events=Invited+Talk%2CSession&filter_rooms="
)

HERE = Path(__file__).parent
RAW_DIR = HERE / "raw"

# ----- Unicode cleanup ------------------------------------------------------

UNICODE_REPLACEMENTS = {
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "–": "-", "—": "-",
    "…": "...", " ": " ",
    "​": "", " ": " ", " ": " ",
    "⋅": " * ",  # dot operator (used as separator in author lists)
}


def clean_unicode(text):
    if not isinstance(text, str):
        return text
    for old, new in UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = unicodedata.normalize("NFKC", text)
    # Page renders &#x000D; (carriage return) as the literal text 'x000D' inside
    # an <em> tag. Strip that artifact and any standalone CR characters.
    text = re.sub(r"\bx000D\b", "", text)
    text = text.replace("\r", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_data(obj):
    if isinstance(obj, str):
        return clean_unicode(obj)
    if isinstance(obj, list):
        return [clean_data(x) for x in obj]
    if isinstance(obj, dict):
        return {k: clean_data(v) for k, v in obj.items()}
    return obj


# ----- agent-browser wrapper -----------------------------------------------

def ab_open(url):
    subprocess.run(
        ["agent-browser", "open", url],
        check=True,
        capture_output=True,
        text=True,
    )


def ab_eval(js):
    """Run a JS expression that returns base64-encoded JSON. Decode to Python."""
    res = subprocess.run(
        ["agent-browser", "eval", js],
        check=True,
        capture_output=True,
        text=True,
    )
    out = res.stdout.strip()
    # output is wrapped in quotes
    if out.startswith('"') and out.endswith('"'):
        out = out[1:-1]
    if not out:
        return None
    return json.loads(base64.b64decode(out))


def ab_close():
    try:
        subprocess.run(["agent-browser", "close"], check=False, capture_output=True)
    except Exception:
        pass


# ----- Time parsing --------------------------------------------------------

DATE_MAP = {
    "MON 18 MAY": "2026-05-18",
    "TUE 19 MAY": "2026-05-19",
    "WED 20 MAY": "2026-05-20",
    "THU 21 MAY": "2026-05-21",
    "FRI 22 MAY": "2026-05-22",
}


def parse_time(raw):
    """Convert '8:45 a.m.' / '1 p.m.' / '8:45 AM' / '10 AM' to 'HH:MM'."""
    if not raw:
        return ""
    s = raw.strip().lower().replace(".", "").replace(" ", "")
    m = re.match(r"^(\d+)(?::(\d+))?(am|pm|noon|midnight)?$", s)
    if not m:
        return ""
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    suffix = m.group(3) or ""
    if suffix == "pm" and hour != 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    elif suffix == "noon":
        hour = 12
    elif suffix == "midnight":
        hour = 0
    return f"{hour:02d}:{minute:02d}"


# ----- Phase A: crawl calendar --------------------------------------------

PHASE_A_JS = r"""
(() => {
  const text = (n) => (n && n.textContent || '').trim().replace(/\s+/g, ' ');
  const visible = (el) => el && el.offsetParent !== null;

  // Day labels are rendered as <h4 class="day-name"> (or similar) at the top of each gcol.days.
  // Fall back to the day index 0..4 -> Mon..Fri if we can't find one.
  const dayCols = Array.from(document.querySelectorAll('div.gcol.days'));
  const dayLabels = ['MON 18 MAY', 'TUE 19 MAY', 'WED 20 MAY', 'THU 21 MAY', 'FRI 22 MAY'];

  const entries = [];
  dayCols.forEach((col, dayIdx) => {
    const blocks = Array.from(col.querySelectorAll('div.eventsession.pad, div.session.pad')).filter(visible);
    blocks.forEach(b => {
      const a = b.querySelector('div.title-style > a, div.sessiontitle > a');
      if (!a) return;
      const href = a.href.split('#')[0];
      const m = href.match(/\/virtual\/2026\/([a-z-]+)\/(\d+)/);
      if (!m) return;
      const kind = m[1];

      // Time: the .time element is in the enclosing .timebox
      const timebox = b.closest('.timebox');
      const timeEl = timebox && timebox.querySelector('.time');

      // Header label like 'Invited Talk:' or 'Keynote Talk:'
      const hdrEl = b.querySelector('.hdr-style');
      const speakerEl = b.querySelector('.speaker-style');
      const endEl = b.querySelector('.end-time');

      // Room from class name 'room-grand-ballroom-1'
      const roomCls = Array.from(b.classList).find(c => c.startsWith('room-'));
      const room = roomCls ? roomCls.replace('room-', '').split('-').map(w => w.charAt(0).toUpperCase()+w.slice(1)).join(' ') : '';

      // Nested papers: oral sessions list them in <div class="content">, poster sessions
      // use the same content divs but inside <details>. Use the union.
      const papers = Array.from(b.querySelectorAll('div.content a')).map(pa => {
        const ph = pa.href.split('#')[0];
        return { title: text(pa), url: ph };
      });

      entries.push({
        kind: kind,
        id: m[2],
        url: href,
        title: text(a),
        day: dayLabels[dayIdx] || '',
        startRaw: timeEl ? text(timeEl) : '',
        endRaw: endEl ? text(endEl).replace(/^\(ends\s*/, '').replace(/\)$/, '') : '',
        kindLabel: hdrEl ? text(hdrEl).replace(/:$/, '').trim() : (kind === 'invited-talk' ? 'Invited Talk' : 'Session'),
        speakerOnCalendar: speakerEl ? text(speakerEl) : '',
        roomFromCalendar: room,
        nestedPapers: papers,
      });
    });
  });

  return btoa(unescape(encodeURIComponent(JSON.stringify(entries))));
})()
"""


def crawl_calendar():
    print("[Phase A] Opening calendar...", flush=True)
    ab_open(CALENDAR_URL)
    time.sleep(1)
    entries = ab_eval(PHASE_A_JS)
    if not entries:
        raise RuntimeError("Phase A returned no entries")
    print(f"[Phase A] Captured {len(entries)} top-level entries", flush=True)
    return entries


# ----- Phase B: enrich -----------------------------------------------------

INVITED_DETAIL_JS = r"""
(() => {
  const main = document.querySelector('main');
  if (!main) return btoa('{}');
  const text = (n) => (n && n.textContent || '').trim().replace(/\s+/g, ' ');

  const h1 = main.querySelector('h1');
  const title = h1 ? text(h1) : '';

  // Header: 'INVITED TALK', date string, room, h1, speaker, abstract
  const staticTexts = [];
  const walker = document.createTreeWalker(main, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = walker.nextNode())) {
    const t = (n.textContent || '').trim();
    if (t) staticTexts.push(t);
  }

  // Find date line and room: between '(INVITED|KEYNOTE) TALK' label and h1 title.
  let dateLine = '', room = '', speaker = '';
  const idxLabel = staticTexts.findIndex(s => /^(INVITED|KEYNOTE)\s+TALK$/i.test(s));
  if (idxLabel >= 0) {
    dateLine = staticTexts[idxLabel + 1] || '';
    room = staticTexts[idxLabel + 2] || '';
  }
  // Speaker: the line right after the title.
  if (title) {
    const idxTitle = staticTexts.findIndex(s => s === title);
    if (idxTitle >= 0) speaker = staticTexts[idxTitle + 1] || '';
  }

  // Abstract: element following the h3 'Abstract' (could be <div> or <p>).
  let abstract = '';
  const h3s = Array.from(main.querySelectorAll('h3'));
  for (const h3 of h3s) {
    if (/^abstract$/i.test(text(h3))) {
      const nx = h3.nextElementSibling;
      if (nx) abstract = text(nx);
      break;
    }
  }

  const out = { kind: 'invited-talk', title, dateLine, room, speaker, abstract };
  return btoa(unescape(encodeURIComponent(JSON.stringify(out))));
})()
"""

SESSION_DETAIL_JS = r"""
(() => {
  const main = document.querySelector('main');
  if (!main) return btoa('{}');
  const text = (n) => (n && n.textContent || '').trim().replace(/\s+/g, ' ');

  const h2 = main.querySelector('h2');
  const title = h2 ? text(h2) : '';
  const h5 = main.querySelector('h5');
  const room = h5 ? text(h5) : '';

  // Time line: a StaticText that matches '... — ...' pattern
  let timeLine = '';
  const walker = document.createTreeWalker(main, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = walker.nextNode())) {
    const t = (n.textContent || '').trim();
    if (/^(Mon|Tue|Wed|Thu|Fri)\s+\d+\s+\w+\s+.+(—|--).+/.test(t)) {
      timeLine = t.replace(/\s+/g, ' ');
      break;
    }
  }

  // Papers: each is wrapped in <div class="track-schedule-card"> with the h5
  // title + authors <p> in one inner div, and a sibling <div class="abstract">.
  const papers = [];
  const cards = Array.from(main.querySelectorAll('div.track-schedule-card'));
  for (const card of cards) {
    const h = card.querySelector('h5');
    if (!h) continue;
    const a = h.querySelector('a');
    const pTitle = text(h);
    const pUrl = a ? a.href.split('#')[0] : '';
    const authorsEl = card.querySelector('p.text-muted, p');
    const abstractEl = card.querySelector('.abstract');
    const authors = authorsEl ? text(authorsEl) : '';
    const abstract = abstractEl ? text(abstractEl) : '';
    if (pTitle) papers.push({ title: pTitle, url: pUrl, authors, abstract });
  }

  const out = { kind: 'session', title, room, timeLine, papers };
  return btoa(unescape(encodeURIComponent(JSON.stringify(out))));
})()
"""


def enrich_one(entry, attempt=1):
    url = entry["url"]
    js = INVITED_DETAIL_JS if entry["kind"] == "invited-talk" else SESSION_DETAIL_JS
    ab_open(url)
    time.sleep(0.4)
    detail = ab_eval(js)
    if not detail or not detail.get("title"):
        if attempt < 3:
            time.sleep(1.5 * attempt)
            return enrich_one(entry, attempt + 1)
        print(f"  ! empty detail after retries: {url}", flush=True)
        detail = {"title": entry["title"], "room": "", "abstract": "", "papers": [], "timeLine": "", "dateLine": "", "speaker": ""}
    return detail


def enrich_all(entries):
    print(f"[Phase B] Enriching {len(entries)} entries...", flush=True)
    for i, e in enumerate(entries, 1):
        print(f"  [{i}/{len(entries)}] {e['kind']} {e['id']}", flush=True)
        e["detail"] = enrich_one(e)
    return entries


# ----- Phase C: normalize --------------------------------------------------

AUTHOR_SEP = re.compile(r"\s*[*⋅]\s*|\s*·\s*")


def split_authors(s):
    if not s:
        return []
    parts = re.split(r"\s+\*\s+", s)  # using cleaned ' * ' separator
    parts = [p.strip().strip(",") for p in parts if p and p.strip().strip(",")]
    return parts


def parse_time_range(line):
    """Parse 'Tue 19 May 1 p.m. PDT — 2:30 p.m. PDT' -> (start, end) in HH:MM."""
    if not line:
        return "", ""
    # Normalize all dashes
    line = re.sub(r"\s+(--|—|-)\s+", " - ", line)
    m = re.search(r"\b(\d{1,2}(?::\d{2})?\s*(?:a\.m\.|p\.m\.))\b.*?-\s*(\d{1,2}(?::\d{2})?\s*(?:a\.m\.|p\.m\.))\b", line, re.IGNORECASE)
    if not m:
        return "", ""
    return parse_time(m.group(1)), parse_time(m.group(2))


def parse_invited_date_line(line):
    """Parse 'Mon, May 18, 2026 * 8:45 AM - 9:05 AM PDT' (after unicode cleanup)."""
    if not line:
        return "", "", ""
    # Find date
    m = re.search(r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s*(\w+)\s+(\d{1,2}),\s*(\d{4})", line)
    date = ""
    if m:
        from datetime import datetime
        try:
            date = datetime.strptime(f"{m.group(2)} {m.group(3)} {m.group(4)}", "%B %d %Y").strftime("%Y-%m-%d")
        except Exception:
            date = ""
    # Find time range
    tm = re.search(r"(\d{1,2}(?::\d{2})?\s*(?:AM|PM))\s*[-–—]\s*(\d{1,2}(?::\d{2})?\s*(?:AM|PM))", line, re.IGNORECASE)
    if tm:
        start = parse_time(tm.group(1))
        end = parse_time(tm.group(2))
        return date, start, end
    return date, "", ""


def determine_type(entry):
    # Calendar's hdr-style label is the most reliable signal, since some
    # 'Keynote Talk:' entries reuse the /invited-talk/ URL prefix.
    label = entry.get("kindLabel", "")
    if label == "Keynote Talk":
        return "Keynote"
    if label == "Invited Talk":
        return "Invited Talk"
    if entry["kind"] == "invited-talk":
        return "Invited Talk"
    title = entry.get("title", "")
    if re.search(r"Oral Presentation", title):
        return "Paper Session"
    return "Session"


def derive_topic(entry, clean_title):
    """Topic is the part after 'Oral Presentation:' for oral sessions, else the
    cleaned session title (no trailing time bracket)."""
    m = re.match(r".*Oral Presentation:\s*(.+)$", clean_title)
    if m:
        return m.group(1).strip()
    return clean_title


def derive_track(entry):
    """Pull 'Research-Track' / 'Industry-Track' from an oral session group title."""
    title = entry.get("title", "")
    if "Research-Track" in title:
        return "Research-Track"
    if "Industry-Track" in title:
        return "Industry-Track"
    return ""


def aggregate_authors(papers):
    seen = []
    for p in papers:
        for a in split_authors(p.get("authors", "")):
            if a and a not in seen:
                seen.append(a)
    return seen


def normalize(entries):
    print("[Phase C] Normalizing...", flush=True)
    sessions = []
    for e in entries:
        detail = e.get("detail") or {}
        type_ = determine_type(e)

        # Date
        date = DATE_MAP.get(e.get("day", ""), "")

        # Times
        start = parse_time(e.get("startRaw", ""))
        end = parse_time(e.get("endRaw", ""))
        # Prefer detail times if available
        if e["kind"] == "invited-talk":
            d2, s2, en2 = parse_invited_date_line(detail.get("dateLine", ""))
            if d2:
                date = d2
            if s2:
                start = s2
            if en2:
                end = en2
        else:
            s2, en2 = parse_time_range(detail.get("timeLine", ""))
            if s2:
                start = s2
            if en2:
                end = en2

        # Room
        location = detail.get("room", "")

        # Title: prefer the detail-page title (cleaner) and strip the trailing
        # '[8:30-10:00]' time bracket the calendar appends to some titles.
        title = (detail.get("title") or e.get("title") or "").strip()
        title = re.sub(r"\s*\[\d{1,2}:\d{2}\s*[-–—]\s*\d{1,2}:\d{2}\]\s*$", "", title)
        title = re.sub(r"\s+", " ", title).strip()

        # Topic (uses cleaned title)
        topic = derive_topic(e, title)

        # Oral session groups: expand into one record per paper using /oral/<id>
        # URLs. Each paper inherits the parent group's date/time/room and
        # carries the track (Research-Track / Industry-Track) as metadata.
        if type_ == "Paper Session" and detail.get("papers"):
            track = derive_track(e)
            for p in detail["papers"]:
                authors = split_authors(p.get("authors", ""))
                sess = {
                    "title": p["title"],
                    "type": "Paper Session",
                    "date": date,
                    "startTime": start,
                    "endTime": end,
                    "location": location,
                    "sessionUrl": p["url"],
                    "sessionFormat": "IN_PERSON",
                    "hasRecording": False,
                    "topic": [topic] if topic else [],
                    "speaker": authors,
                }
                if track:
                    sess["track"] = track
                if p.get("abstract"):
                    sess["abstract"] = p["abstract"]
                sessions.append(sess)
            continue

        # Non-oral entries: emit as single session record.
        if e["kind"] == "invited-talk":
            speakers = [detail.get("speaker", "").strip()] if detail.get("speaker") else []
            if not speakers and e.get("speakerOnCalendar"):
                speakers = [e["speakerOnCalendar"]]
        else:
            calendar_speaker = e.get("speakerOnCalendar", "").strip()
            papers = detail.get("papers") or []
            if papers:
                speakers = aggregate_authors(papers)
            elif calendar_speaker:
                speakers = [calendar_speaker]
            else:
                speakers = []

        abstract = detail.get("abstract", "") if e["kind"] == "invited-talk" else ""
        publication_titles = [p["title"] for p in (detail.get("papers") or [])]

        sess = {
            "title": title,
            "type": type_,
            "date": date,
            "startTime": start,
            "endTime": end,
            "location": location,
            "sessionUrl": e["url"],
            "sessionFormat": "IN_PERSON",
            "hasRecording": False,
            "topic": [topic] if topic else [],
            "speaker": speakers,
        }
        if abstract:
            sess["abstract"] = abstract
        if publication_titles:
            sess["publicationTitles"] = publication_titles

        sessions.append(sess)

    sessions.sort(key=lambda s: (s["date"], s["startTime"], s["title"]))
    return sessions


def to_conferenceflow(sessions):
    out = []
    for s in sessions:
        rec = {
            "title": s["title"],
            "date": s["date"],
            "start": s["startTime"],
            "end": s["endTime"],
            "session_type": s["type"],
            "topic": (s["topic"][0] if s.get("topic") else ""),
            "url": s["sessionUrl"],
            "format": "In-Person",
            "speakers": [{"name": n} for n in s.get("speaker", [])],
        }
        if s.get("location"):
            rec["room"] = s["location"]
        themes = list(s.get("topic", []))
        if s.get("track") and s["track"] not in themes:
            themes.insert(0, s["track"])
        if themes:
            rec["key_themes"] = themes
        out.append(rec)
    return out


def write_outputs(raw_entries, sessions):
    raw_path = RAW_DIR / "calendar-extract.json"
    sparkflow_path = HERE / "mlsys26-sessions-sparkflow.json"
    cflow_path = HERE / "mlsys26-sessions-conferenceflow.json"

    raw_payload = clean_data({
        "source_url": CALENDAR_URL,
        "venue": "MLSys",
        "year": 2026,
        "entries": raw_entries,
    })
    raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2))
    print(f"[Phase C] wrote {raw_path}", flush=True)

    spark = clean_data({"venue": "MLSys", "year": 2026, "sessions": sessions})
    sparkflow_path.write_text(json.dumps(spark, ensure_ascii=False, indent=2))
    print(f"[Phase C] wrote {sparkflow_path}", flush=True)

    cflow = clean_data(to_conferenceflow(sessions))
    cflow_path.write_text(json.dumps(cflow, ensure_ascii=False, indent=2))
    print(f"[Phase C] wrote {cflow_path}", flush=True)


def main():
    try:
        entries = crawl_calendar()
        # Save raw Phase A as a checkpoint
        (RAW_DIR / "phase-a.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2))
        entries = enrich_all(entries)
        # Clean unicode at the source
        entries = clean_data(entries)
        sessions = normalize(entries)
        write_outputs(entries, sessions)
        print(f"\n[Done] {len(sessions)} sessions written.")
    finally:
        ab_close()


if __name__ == "__main__":
    main()
