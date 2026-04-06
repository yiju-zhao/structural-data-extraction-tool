#!/usr/bin/env python3
"""Extract oral session details from ICLR 2026 virtual site."""
import json
import base64
import subprocess
import sys
import time
import unicodedata

UNICODE_REPLACEMENTS = {
    '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
    '\u2013': '-', '\u2014': '-', '\u2026': '...', '\u00a0': ' ',
    '\u200b': '', '\u2028': ' ', '\u2029': ' ',
}

def clean_unicode(text):
    if not isinstance(text, str):
        return text
    for old, new in UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)
    return unicodedata.normalize('NFKC', text)

def run_browser(cmd):
    result = subprocess.run(
        ['npx', 'agent-browser'] + cmd,
        capture_output=True, text=True, timeout=30
    )
    return result.stdout.strip()

JS_CLICK_ABSTRACT = r'''(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const absBtn = buttons.find(b => b.textContent.includes('Abstract'));
    if (absBtn && absBtn.getAttribute('aria-expanded') !== 'true') absBtn.click();
    return absBtn ? 'clicked' : 'not found';
})()'''

JS_EXTRACT = r'''(() => {
    const d = {};

    // Title from h2
    const h2s = document.querySelectorAll('h2');
    for (const h of h2s) {
        const t = h.textContent.trim();
        if (t && t !== 'Main Navigation' && !t.includes('ICLR')) { d.title = t; break; }
    }

    // Authors
    const h3s = document.querySelectorAll('h3');
    for (const h of h3s) {
        if (h.textContent.includes('\u00b7')) { d.authors = h.textContent.trim(); break; }
    }

    // Session
    const sessionLink = document.querySelector('a[href*="/session/"]');
    d.session_name = sessionLink ? sessionLink.textContent.trim() : '';
    d.session_url = sessionLink ? 'https://iclr.cc' + sessionLink.getAttribute('href') : '';

    // OpenReview link - find the one with /forum
    const allLinks = Array.from(document.querySelectorAll('a'));
    const orLink = allLinks.find(a => a.href && a.href.includes('openreview.net/forum'));
    d.openreview_url = orLink ? orLink.href : '';

    // Slides link
    const slidesLink = allLinks.find(a => a.textContent.includes('Slides') && (a.href.includes('/attachment') || a.href.includes('slides')));
    d.slides_url = slidesLink ? slidesLink.href : '';

    // Date/time
    const text = document.body.innerText;
    const timeMatch = text.match(/(\w{3}\s+\d+\s+\w+)\s+([\d:]+\s*[ap]\.m\.)\s*\w+\s*[-\u2014]\s*([\d:]+\s*[ap]\.m\.)\s*\w+/i);
    if (timeMatch) {
        d.date = timeMatch[1].trim();
        d.start_time = timeMatch[2].trim();
        d.end_time = timeMatch[3].trim();
    }

    // Abstract - find between 'Abstract:' and 'Chat'/'Directory'
    const absIdx = text.indexOf('Abstract:');
    if (absIdx > -1) {
        const afterAbs = text.substring(absIdx + 9).trim();
        const chatIdx = afterAbs.indexOf('\nChat\n');
        const dirIdx = afterAbs.indexOf('\nDirectory');
        let endIdx = afterAbs.length;
        if (chatIdx > 0) endIdx = Math.min(endIdx, chatIdx);
        if (dirIdx > 0) endIdx = Math.min(endIdx, dirIdx);
        d.abstract = afterAbs.substring(0, endIdx).trim();
    } else {
        d.abstract = '';
    }

    d.url = window.location.href;
    return btoa(unescape(encodeURIComponent(JSON.stringify(d))));
})()'''

def extract_one(url):
    """Extract metadata from a single oral session page."""
    run_browser(['open', url])
    time.sleep(1.5)

    # Click abstract button first, then wait for DOM update
    run_browser(['find', 'text', 'Abstract', 'click'])
    time.sleep(1)

    raw = run_browser(['eval', JS_EXTRACT])
    raw = raw.strip().strip('"')
    try:
        data = json.loads(base64.b64decode(raw).decode('utf-8'))
        for k, v in data.items():
            if isinstance(v, str):
                data[k] = clean_unicode(v)
        return data
    except Exception as e:
        return {'error': str(e), 'url': url}

def main():
    day = sys.argv[1]  # e.g., "4_23"
    outdir = '/Users/eason/Documents/HW-Project/Agent/structural-data-extraction-tool/projects/iclr-2026/output'

    links_file = f'{outdir}/oral_sessions_day_{day}.json'
    with open(links_file) as f:
        links = json.load(f)

    if not links:
        print(f'No oral links for day {day}')
        return

    print(f'Extracting {len(links)} oral sessions for day {day}...')
    results = []

    for i, item in enumerate(links):
        url = item['url']
        print(f'  [{i+1}/{len(links)}] {item["title"][:60]}...', flush=True)
        try:
            data = extract_one(url)
            results.append(data)
            if 'error' in data:
                print(f'    ERROR: {data["error"]}')
        except Exception as e:
            print(f'    ERROR: {e}')
            results.append({'error': str(e), 'url': url, 'title': item['title']})

    outfile = f'{outdir}/oral_details_day_{day}.json'
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results if 'error' not in r)
    print(f'\nSaved {ok}/{len(results)} to {outfile}')

if __name__ == '__main__':
    main()
