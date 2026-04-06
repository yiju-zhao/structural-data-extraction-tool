#!/usr/bin/env python3
"""Save base64-encoded JSON data from stdin to a day-specific JSON file."""
import sys
import base64
import json
import re
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

def clean_data(data):
    if isinstance(data, str):
        return clean_unicode(data)
    elif isinstance(data, list):
        return [clean_data(item) for item in data]
    elif isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items()}
    return data

if __name__ == '__main__':
    day = sys.argv[1]  # e.g., "4_23"
    b64_data = sys.stdin.read().strip().strip('"')

    data = json.loads(base64.b64decode(b64_data).decode('utf-8'))
    cleaned = clean_data(data)

    outdir = '/Users/eason/Documents/HW-Project/Agent/structural-data-extraction-tool/projects/iclr-2025/output'
    outfile = f'{outdir}/oral_sessions_day_{day}.json'

    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f'Saved {len(cleaned)} oral links to {outfile}')
