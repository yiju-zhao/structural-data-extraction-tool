---
name: extract
description: Extract structured data from websites. Use when the user wants to scrape a webpage, extract data from a URL, or get structured information from a website.
allowed-tools: Read, Bash, Write, Glob, Grep, WebFetch
---

# Structured Data Extraction (DSL)

A declarative YAML-based extraction system that handles everything from simple flat lists to complex hierarchical structures with context inheritance.

## Prerequisites

**IMPORTANT**: Always activate the virtual environment before running the extraction script:

```bash
source .venv/bin/activate
```

Or use the venv Python directly:
```bash
.venv/bin/python .claude/skills/extract/scripts/extract.py ...
```

## Quick Start

### 1. Analyze Page Structure
```bash
.venv/bin/python .claude/skills/extract/scripts/extract.py --analyze <url>
```

### 2. Create Extraction Config
Choose a template based on your page structure:
- `configs/examples/simple_list.yaml` - Flat lists (products, articles)
- `configs/examples/hierarchical.yaml` - Nested structures (schedules, categories)
- `configs/examples/multi_pattern.yaml` - Multiple item types

### 3. Run Extraction
```bash
.venv/bin/python .claude/skills/extract/scripts/extract.py <config.yaml>
```

### 4. Convert JSON to CSV (optional)
```bash
.venv/bin/python .claude/skills/extract/scripts/extract.py --json2csv output.json
```

---

## Choosing the Right Template

```
Is the page structure flat with uniform items?
├─ YES → Use simple_list.yaml
│        Examples: product listings, article lists, search results
│
└─ NO → Does it have nested context (day→time→sessions)?
        ├─ YES → Use hierarchical.yaml
        │        Examples: conference schedules, nested categories
        │
        └─ NO → Does it have multiple item types?
                ├─ YES → Use multi_pattern.yaml
                │        Examples: mixed events, different content types
                │
                └─ NO → Combine hierarchical + multi_pattern features
```

---

## Config Structure Reference

### Basic Config

```yaml
version: "1.0"
name: "my_extraction"
description: "What this extracts"

source:
  url: "https://example.com/page"
  # OR
  file: "cached_page.html"

  options:
    headless: true
    wait_for: ".loaded"  # Wait for JS rendering

output:
  format: [json, csv]
  path: "output/data"

items:
  - name: item_type
    selector: "div.item"
    fields:
      title:
        selector: "h2"
        extract: text
      url:
        selector: "a"
        extract: href
        transform: absolute_url
```

### With Context Hierarchy

```yaml
# Contexts define nested containers that pass fields to children
contexts:
  - name: day
    selector: "div.day-container"
    fields:
      date: {selector: ".day-header", extract: text}

  - name: time_block
    selector: "div.timebox"
    parent: day  # Inherits 'date' field
    fields:
      time: {selector: ".time-label", extract: text}

items:
  - name: session
    selector: "div.session"
    context: time_block  # Inherits date + time
    fields:
      title: {selector: ".title", extract: text}
```

### With Multiple Patterns

```yaml
items:
  # Different HTML structures → different patterns
  - name: talk
    selector: "div.talk-session"
    fields:
      title: {selector: "h3.talk-title", extract: text}
      speaker: {selector: ".speaker", extract: text}

  - name: workshop
    selector: "div.workshop-item"
    fields:
      title: {selector: ".workshop-name", extract: text}
      instructor: {selector: ".instructor", extract: text}

  - name: break
    selector: "div.break-slot"
    fields:
      title: {selector: ".break-name", extract: text}

# Unified schema ensures all items have same fields
schema:
  fields:
    - name: title
      required: true
    - name: speaker
    - name: instructor
```

---

## Field Configuration

### Short Form
```yaml
fields:
  title: "h2.title::text"        # selector::attribute
  url: "a::href"
  image: "img::src"
```

### Full Form
```yaml
fields:
  title:
    selector: "h2.title"
    extract: text
  url:
    selector: "a"
    extract: href
    transform: absolute_url
  type:
    source: class  # Extract from element's class attribute
    transform:
      regex: '(talk|workshop|break)'
```

### Extraction Methods

| Value | Description |
|-------|-------------|
| `text` | Element text content (default) |
| `html` | Inner HTML |
| `href` | href attribute |
| `src` | src attribute |
| `class` | class attribute |
| Any attribute | e.g., `data-id`, `title` |

### Source Options

| Value | Description |
|-------|-------------|
| `selector` | Use CSS selector (default) |
| `class` | Element's class attribute |
| `id` | Element's id attribute |
| `tag` | Element's tag name |

---

## Transforms

### Built-in Transforms

```yaml
transform: strip          # Remove whitespace
transform: absolute_url   # Convert relative URLs
transform: lowercase      # Convert to lowercase
transform: uppercase      # Convert to uppercase
```

### Regex Transform

```yaml
transform:
  regex: '(\d+:\d+\s*(?:AM|PM)?)'
  group: 1  # Capture group (default: 1)
```

### Replace Transform

```yaml
transform:
  replace: ['\n', ' ']  # [old, new]
```

### URL Join Transform

```yaml
transform:
  type: url_join
  base: "https://example.com"
```

### Named Transforms (reusable)

```yaml
transforms:
  clean_time:
    type: regex
    pattern: '(\d{1,2}:\d{2})'
    group: 1

  site_url:
    type: url_join
    base: "https://mysite.com"

items:
  - name: session
    fields:
      time:
        selector: ".time"
        extract: text
        transform: clean_time  # Reference named transform
```

---

## Output Schema

Ensure consistent output fields across all item types:

```yaml
schema:
  fields:
    - name: title
      required: true
    - name: date
      required: true
    - name: time
    - name: speaker
    - name: url
```

Items missing fields will have `null` values. The `_type` field is automatically added to indicate which pattern matched.

---

## CLI Reference

```bash
# Analyze page structure (overview)
python extract.py --analyze <url>
python extract.py -a <url>

# Inspect specific selector (detailed exploration)
python extract.py --inspect <url> --selector "div.item"
python extract.py -i <url> -s "div.item"

# Inspect with field testing
python extract.py --inspect <url> --selector "div.item" \
    --fields "title:h2::text,url:a::href"

# Inspect with parent hierarchy
python extract.py --inspect <url> --selector "div.item" --parents

# More samples (default is 3)
python extract.py --inspect <url> --selector "div.item" --sample 10

# Run extraction from config
python extract.py <config.yaml>

# Convert JSON to CSV
python extract.py --json2csv data.json

# Show browser (not headless)
python extract.py --no-headless --analyze <url>
```

---

## Inspect Mode Details

The `--inspect` mode replaces ad-hoc Playwright scripts for DOM exploration. It provides:

### 1. Selector Testing
```bash
python extract.py -i <url> -s "div.poster-session"
```
Output:
- Element count
- Sample HTML for each match
- Suggestions if no matches (similar class names)

### 2. Parent Hierarchy Discovery
```bash
python extract.py -i <url> -s "div.session" --parents
```
Shows parent chain to help identify context containers:
```
Parent chain:
  └─ div.timebox
    └─ div.day-container
      └─ body
```

### 3. Field Extraction Testing
```bash
python extract.py -i <url> -s "div.session" \
    --fields "title:.title::text,url:a::href,speaker:.author::text"
```
Tests field definitions before writing config:
```
--- Sample extraction ---
  title: Machine Learning Advances
  url: /session/123
  speaker: Dr. Jane Smith
```

### 4. Context Discovery
Automatically shows common parent containers that could be used as contexts:
```
=== Potential Context Elements ===
Common parent containers:
  div.timebox: 42
  div.day-container: 5
```

---

## Workflow Example: Conference Schedule

### Step 1: Analyze (Overview)
```bash
python extract.py --analyze https://conf.example.com/schedule
```

Output reveals:
- `div.day-container` (5 children)
- `div.timebox` (many children)
- Classes: `session`, `talk`, `poster`, `break`

### Step 2: Inspect (Detailed Exploration)
```bash
# Test main selector
python extract.py -i https://conf.example.com/schedule -s "div.session" --parents

# Test field extraction
python extract.py -i https://conf.example.com/schedule -s "div.session" \
    --fields "title:.title::text,speaker:.speaker::text,url:a::href"

# Check poster sessions (different structure?)
python extract.py -i https://conf.example.com/schedule -s ".content.poster" --parents
```

This reveals:
- `div.session` has 87 matches with parent chain: `timebox → day-container`
- `.content.poster` has 3320 matches with different parent structure
- Field extraction works with `.title::text`

### Step 3: Create Config

```yaml
version: "1.0"
name: "conference_2025"

source:
  url: "https://conf.example.com/schedule"

output:
  format: [json, csv]
  path: "output/sessions"

contexts:
  - name: day
    selector: "div.day-container"
    fields:
      date: {selector: ".day-header", extract: text}

  - name: time_block
    selector: "div.timebox"
    parent: day
    fields:
      time: {selector: ".time-label", extract: text}

items:
  - name: session
    selector: "div.session"
    context: time_block
    fields:
      title: {selector: ".title", extract: text}
      speaker: {selector: ".speaker", extract: text}
      url:
        selector: "a"
        extract: href
        transform: absolute_url

schema:
  fields:
    - {name: title, required: true}
    - {name: date, required: true}
    - {name: time, required: true}
    - name: speaker
    - name: url
```

### Step 3: Run
```bash
python extract.py configs/conference_2025.yaml
```

Output: `output/sessions.json` and `output/sessions.csv`

---

## Tips

1. **Always analyze first** - Run `--analyze` to understand page structure before writing config

2. **Start simple** - Begin with basic selectors, add complexity as needed

3. **Test incrementally** - Extract a few items first (`limit` in items), then expand

4. **Use context inheritance** - For nested data, define contexts to avoid repeating selectors

5. **Unified schema** - When extracting multiple item types, define a schema for consistent output

6. **Named transforms** - Create reusable transforms for common patterns (URLs, times, etc.)

7. **Check browser rendering** - If data is missing, the page may need JS. Try `wait_for` option

---

## Files Reference

```
.claude/skills/extract/
├── SKILL.md                        # This documentation
├── scripts/
│   └── extract.py                  # Unified extraction tool
├── configs/
│   └── examples/
│       ├── simple_list.yaml        # Flat list template
│       ├── hierarchical.yaml       # Nested structure template
│       └── multi_pattern.yaml      # Multiple item types template
└── schema/
    └── extraction_schema.json      # JSON schema for validation
```
