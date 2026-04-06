#!/usr/bin/env python3
"""Extract workshop details from ICLR 2026 virtual site."""
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

JS_EXTRACT = r'''(() => {
    const d = {};
    const text = document.body.innerText;

    // Title from h1
    const h1 = document.querySelector('h1');
    d.title = h1 ? h1.textContent.trim() : '';

    // Date/time - "Sun, Apr 26, 2026 • 5:00 AM – 1:00 PM PDT"
    const dtMatch = text.match(/(\w{3}, \w{3} \d+, \d{4})\s*[•·]\s*([\d:]+\s*[AP]M)\s*[-–]\s*([\d:]+\s*[AP]M)\s*(\w+)/);
    if (dtMatch) {
        d.date = dtMatch[1].trim();
        d.start_time = dtMatch[2].trim();
        d.end_time = dtMatch[3].trim();
        d.timezone = dtMatch[4].trim();
    }

    // Organizers - look for the line with · separators after the title
    const h3s = document.querySelectorAll('h3');
    for (const h of h3s) {
        if (h.textContent.includes('\u00b7')) {
            d.organizers = h.textContent.trim();
            break;
        }
    }
    // Fallback: find organizers from text (line with multiple · )
    if (!d.organizers) {
        const lines = text.split('\n');
        for (const line of lines) {
            if ((line.includes('\u00b7') || line.includes('·')) && line.split('·').length >= 3) {
                d.organizers = line.trim();
                break;
            }
        }
    }

    // Abstract
    const absIdx = text.indexOf('Abstract\n');
    if (absIdx > -1) {
        const afterAbs = text.substring(absIdx + 9).trim();
        const endMarkers = ['\nLog in', '\nSchedule\n', '\nChat\n', '\nICLR uses'];
        let endIdx = afterAbs.length;
        for (const marker of endMarkers) {
            const idx = afterAbs.indexOf(marker);
            if (idx > 0) endIdx = Math.min(endIdx, idx);
        }
        d.abstract = afterAbs.substring(0, endIdx).trim();
    } else {
        d.abstract = '';
    }

    // Project page link
    const allLinks = Array.from(document.querySelectorAll('a'));
    const projLink = allLinks.find(a => a.textContent.includes('Project Page'));
    d.project_url = projLink ? projLink.href : '';

    d.url = window.location.href;
    return btoa(unescape(encodeURIComponent(JSON.stringify(d))));
})()'''

def extract_one(url):
    """Extract metadata from a single workshop page."""
    run_browser(['open', url])
    time.sleep(1.5)

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
    day = sys.argv[1]  # e.g., "4_26"
    outdir = '/Users/eason/Documents/HW-Project/Agent/structural-data-extraction-tool/projects/iclr-2026/output'

    links_file = f'{outdir}/workshops_day_{day}.json'
    with open(links_file) as f:
        links = json.load(f)

    if not links:
        print(f'No workshops for day {day}')
        return

    print(f'Extracting {len(links)} workshops for day {day}...')
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

    outfile = f'{outdir}/workshop_details_day_{day}.json'
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results if 'error' not in r)
    print(f'\nSaved {ok}/{len(results)} to {outfile}')

if __name__ == '__main__':
    main()
