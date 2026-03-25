---
name: web-data-extractor
description: Extract structured data from websites and save to CSV/JSON with UTF-8 encoding. Use when the user wants to (1) scrape or extract data from a website, (2) convert web content to CSV or JSON format, (3) extract lists, tables, cards, or product listings from web pages, (4) collect data from multiple detail pages linked from a main page, or (5) harvest structured information from conference catalogs, e-commerce sites, or directory listings. Triggers on phrases like "extract data from website", "scrape this page", "get all items from URL", "download data from website to CSV", or "collect information from web page".
---

# Web Data Extractor

Extract structured data from websites using the agent-browser skill. Supports single-page and multi-page extraction patterns with automatic error recovery.

## Workflow

### Step 1: Analyze the Target

1. Open the URL using agent-browser
2. Take a snapshot to understand page structure
3. Identify the data structure (cards, tables, lists, detail pages, etc.)

### Step 2: Confirm with User

Ask the user to confirm:
- What data fields to extract?
- Is all data on one page or across multiple pages?
- Any filters needed?
- Output filename and location?
- Enable auto-recovery for extraction errors?

### Step 3: Extract Data

Create extraction JavaScript that returns data encoded with base64 for safety:

```javascript
(() => {
  const data = Array.from(document.querySelectorAll('YOUR_SELECTOR')).map(el => ({
    field1: el.querySelector('selector1')?.textContent?.trim() || '',
    field2: el.querySelector('selector2')?.href || ''
  }));
  return btoa(unescape(encodeURIComponent(JSON.stringify(data))));
})()
```

**IMPORTANT:** Always use `btoa(unescape(encodeURIComponent(JSON.stringify(data))))` to safely pass JSON through subprocess. This prevents issues with quotes, apostrophes, newlines, and other special characters.

### Step 4: Error Recovery & Feedback Loop

When extraction fails, use this systematic approach:

1. **Detect failures:** Check for empty required fields, JSON errors, timeouts
2. **Investigate:** Open the failing URL, take snapshot, compare structure
3. **Identify root cause:** Selectors changed? Layout varies? Missing elements? Special characters?
4. **Update script:** Add fallback selectors, optional chaining, DOM-based queries
5. **Retry:** Re-run extraction with updated script

**Recovery pattern:**

```python
def extract_with_recovery(url: str, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            data = extract_data(url)
            if validate_required_fields(data):
                return data
            debug_info = investigate_failure(url)
            update_extraction_script(debug_info)
        except Exception as e:
            log_error(url, e)
    return {'error': 'Failed after retries', 'url': url}
```

### Step 5: Clean Data

**IMPORTANT:** Always clean extracted data before saving to avoid issues with:

1. **Ambiguous Unicode Characters** - Replace curly quotes, em dashes, etc. with ASCII equivalents
2. **Line Breaks in CSV** - Remove embedded newlines within cell values

```python
import re
import unicodedata

# Unicode replacement mappings
UNICODE_REPLACEMENTS = {
    '\u2018': "'",   # LEFT SINGLE QUOTATION MARK
    '\u2019': "'",   # RIGHT SINGLE QUOTATION MARK
    '\u201c': '"',   # LEFT DOUBLE QUOTATION MARK
    '\u201d': '"',   # RIGHT DOUBLE QUOTATION MARK
    '\u2013': '-',   # EN DASH
    '\u2014': '-',   # EM DASH
    '\u2026': '...', # HORIZONTAL ELLIPSIS
    '\u00a0': ' ',   # NON-BREAKING SPACE
    '\u200b': '',    # ZERO WIDTH SPACE
    '\u2028': ' ',   # LINE SEPARATOR
    '\u2029': ' ',   # PARAGRAPH SEPARATOR
}

def clean_unicode(text):
    """Replace ambiguous unicode characters with ASCII equivalents."""
    if not isinstance(text, str):
        return text
    for old, new in UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = unicodedata.normalize('NFKC', text)
    return text

def clean_for_csv(text):
    """Clean text for CSV - remove line breaks within cells."""
    if not isinstance(text, str):
        return text
    text = clean_unicode(text)
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
    return text.strip()

def clean_data(data):
    """Recursively clean all string values in data structure."""
    if isinstance(data, str):
        return clean_unicode(data)
    elif isinstance(data, list):
        return [clean_data(item) for item in data]
    elif isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items()}
    return data
```

### Step 6: Save Output

Save to JSON and CSV with UTF-8 encoding:

```python
import json
import csv

# Clean data first
data = clean_data(data)

# JSON
with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# CSV - use csv.writer with QUOTE_ALL and clean line breaks
fieldnames = list(data[0].keys())
with open('output.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(fieldnames)
    for item in data:
        row = []
        for field in fieldnames:
            value = item.get(field, '')
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            row.append(clean_for_csv(str(value)))
        writer.writerow(row)
```

### Step 7: Close Browser

```bash
agent-browser close
```

## Best Practices

1. **Always use base64 encoding** for JS output
2. **Use optional chaining** (`?.`) for safe element access
3. **Add fallback selectors** for varying layouts
4. **Validate data** before saving
5. **Log errors with context** (URL, error details)
6. **Prefer DOM queries** over text/regex parsing
7. **Handle special characters** via base64 encoding
8. **Add retry logic** for transient failures
9. **Clean unicode characters** before saving (curly quotes, em dashes, etc.)
10. **Remove line breaks in CSV cells** - each record should be on a single row

## Resources

- **[extraction-patterns.md](references/extraction-patterns.md)**: Common JS patterns
- **[extract_data.py](scripts/extract_data.py)**: Python script with error recovery
