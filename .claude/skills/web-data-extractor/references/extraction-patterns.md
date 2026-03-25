# Extraction Patterns Reference

Common JavaScript patterns for extracting structured data from different page layouts.

## Data Cleaning (IMPORTANT)

**Always clean extracted data before saving to avoid these common issues:**

### 1. Ambiguous Unicode Characters

Web content often contains curly quotes, em dashes, and other unicode characters that cause issues:

| Character | Code | Replace With | Name |
|-----------|------|--------------|------|
| ' | \u2018 | ' | Left single quote |
| ' | \u2019 | ' | Right single quote |
| " | \u201c | " | Left double quote |
| " | \u201d | " | Right double quote |
| – | \u2013 | - | En dash |
| — | \u2014 | - | Em dash |
| … | \u2026 | ... | Ellipsis |
|   | \u00a0 | space | Non-breaking space |

### 2. Line Breaks in CSV Cells

Multi-line text from web pages breaks CSV structure. Always normalize whitespace:

```python
import re

def clean_for_csv(text):
    """Replace multiple whitespace/newlines with single space."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
```

### Python Cleaning Functions

```python
import re
import unicodedata

UNICODE_REPLACEMENTS = {
    '\u2018': "'", '\u2019': "'",  # curly quotes
    '\u201c': '"', '\u201d': '"',  # curly double quotes
    '\u2013': '-', '\u2014': '-',  # dashes
    '\u2026': '...',               # ellipsis
    '\u00a0': ' ',                 # non-breaking space
    '\u200b': '',                  # zero-width space
    '\u2028': ' ', '\u2029': ' ',  # line/paragraph separators
}

def clean_unicode(text):
    for old, new in UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)
    return unicodedata.normalize('NFKC', text)

def clean_for_csv(text):
    text = clean_unicode(text)
    return re.sub(r'\s+', ' ', text).strip()
```

---

## Base64 Encoding Pattern (ALWAYS USE THIS)

**Always wrap your extraction with base64 encoding** to safely pass JSON through subprocess:

```javascript
(() => {
  const data = /* your extraction logic here */;
  return btoa(unescape(encodeURIComponent(JSON.stringify(data))));
})()
```

This prevents errors from special characters like quotes, apostrophes, newlines, etc.

---

## Common Data Structures

### Card Grids

For product cards, event cards, profile cards:

```javascript
(() => {
  const data = Array.from(document.querySelectorAll('.card, [class*="card"], .item, [class*="item"]')).map(card => ({
    title: card.querySelector('h1, h2, h3, h4, [class*="title"]')?.textContent?.trim() || '',
    description: card.querySelector('p, [class*="desc"], [class*="summary"]')?.textContent?.trim() || '',
    link: card.querySelector('a')?.href || '',
    image: card.querySelector('img')?.src || ''
  })).filter(c => c.title);
  return btoa(unescape(encodeURIComponent(JSON.stringify(data))));
})()
```

### List Items

For unordered/ordered lists or list-like containers:

```javascript
(() => {
  const data = Array.from(document.querySelectorAll('li, .list-item, [role="listitem"]')).map(item => ({
    text: item.textContent.trim(),
    link: item.querySelector('a')?.href || ''
  }));
  return btoa(unescape(encodeURIComponent(JSON.stringify(data))));
})()
```

### HTML Tables

```javascript
(() => {
  const rows = document.querySelectorAll('table tr');
  const headers = Array.from(rows[0].querySelectorAll('th')).map(th => th.textContent.trim());
  const data = Array.from(rows.slice(1)).map(row => {
    const cells = row.querySelectorAll('td');
    const obj = {};
    headers.forEach((h, i) => obj[h] = cells[i]?.textContent?.trim() || '');
    return obj;
  });
  return btoa(unescape(encodeURIComponent(JSON.stringify(data))));
})()
```

### Generic Links Collection

Extract all links matching a pattern:

```javascript
(() => {
  const data = Array.from(document.querySelectorAll('a'))
    .filter(a => /YOUR_PATTERN_HERE/.test(a.href))
    .map(a => ({
      text: a.textContent.trim(),
      url: a.href
    }))
    .filter((v, i, a) => a.findIndex(t => t.url === v.url) === i);
  return btoa(unescape(encodeURIComponent(JSON.stringify(data))));
})()
```

---

## Helper Functions

### Wait for Element

```javascript
((selector, timeout = 5000) => {
    return new Promise((resolve) => {
        const start = Date.now();
        const check = () => {
            if (document.querySelector(selector)) {
                resolve(true);
            } else if (Date.now() - start > timeout) {
                resolve(false);
            } else {
                setTimeout(check, 100);
            }
        };
        check();
    });
})('YOUR_SELECTOR')
```

### Click Load More Button

```javascript
(() => {
    const btns = Array.from(document.querySelectorAll('button'));
    const loadMore = btns.find(b => /Load More|Show More|Show All/i.test(b.textContent));
    if (loadMore && !loadMore.disabled) {
        loadMore.click();
        return 'clicked';
    }
    return 'not found';
})()
```

### Click Filter/Selection Button

```javascript
(() => {
    const btns = Array.from(document.querySelectorAll('button'));
    const targetBtn = btns.find(b => b.textContent.trim() === 'All' || b.textContent.trim() === 'TARGET_TEXT');
    if (targetBtn) {
        targetBtn.click();
        return 'clicked';
    }
    return 'not found';
})()
```

---

## Error Recovery Patterns

### Robust Selector with Fallbacks

```javascript
// Try multiple selectors in order
title = document.querySelector('.title')?.textContent?.trim()
  || document.querySelector('h1')?.textContent?.trim()
  || document.querySelector('[class*="title"]')?.textContent?.trim()
  || '';
```

### Extract List Items from DOM

```javascript
(() => {
  const items = [];
  const labelEl = Array.from(document.querySelectorAll('span, b, strong')).find(el =>
    el.textContent.includes('YOUR_LABEL:')
  );
  if (labelEl) {
    const parent = labelEl.parentElement;
    if (parent) {
      const ul = parent.querySelector('ul, ol');
      if (ul) {
        Array.from(ul.querySelectorAll('li')).forEach(li => {
          const text = li.textContent.trim();
          if (text) items.push(text);
        });
      }
    }
  }
  return btoa(unescape(encodeURIComponent(JSON.stringify(items))));
})()
```

### Extract Text with " | " Separator

```javascript
(() => {
  const items = [];
  document.querySelectorAll('span.p--medium, .YOUR_CONTAINER_SELECTOR').forEach(el => {
    const text = el.textContent.trim();
    if (text.includes(' | ')) {
      const parts = text.split(' | ').map(p => p.trim());
      if (parts.length >= 2) {
        items.push({
          field1: parts[0],
          field2: parts.length >= 3 ? parts[1] : '',
          field3: parts[parts.length - 1]
        });
      }
    }
  });
  return btoa(unescape(encodeURIComponent(JSON.stringify(items))));
})()
```

### Match Pattern in Text with Regex

```javascript
(() => {
  const bodyText = document.body.innerText;
  const match = bodyText.match(/YOUR_PATTERN:?\s*([^\n]+)/);
  const value = match ? match[1].trim() : '';
  const data = { field_name: value };
  return btoa(unescape(encodeURIComponent(JSON.stringify(data))));
})()
```

---

## Common Failure Patterns & Solutions

| Pattern | Issue | Solution |
|---------|-------|----------|
| Single quotes in text | JSON parse error | Use base64 encoding |
| Multi-word values | Regex doesn't match | Use DOM query instead of text parsing |
| Missing elements | Selector fails | Use optional chaining `?.` and fallback selectors |
| Special characters | Encoding issues | Use base64 encoding |
| Lazy loading | Content not loaded | Add wait/scroll before extraction |
| Layout variations | Fixed selector fails | Add multiple fallback selectors |
| Nested lists | Wrong structure | Use DOM traversal (`parent.querySelector('ul')`) |
| Curly quotes, em dashes | Ambiguous unicode | Use `clean_unicode()` before saving |
| Multi-line text | CSV row breaks | Use `clean_for_csv()` to normalize whitespace |

---

## Debugging Commands

```bash
# Take snapshot to inspect structure
agent-browser snapshot -i

# Screenshot for visual debugging
agent-browser screenshot -o debug.png

# Evaluate and see raw output (for debugging)
agent-browser eval "document.querySelector('selector')?.textContent"

# Get all elements matching a pattern
agent-browser eval "Array.from(document.querySelectorAll('selector')).map(e => e.textContent)"
```
