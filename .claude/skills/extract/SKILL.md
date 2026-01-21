---
name: extract
description: Extract structured data from websites. Use when the user wants to scrape a webpage, extract data from a URL, or get structured information from a website.
allowed-tools: Read, Bash, Write, Glob, Grep, WebFetch
---

# Structured Data Extraction

Workflow guide for extracting structured web data using YAML configs and custom Python scripts.

## Core Principle

**Atomic capability**: `URL → Config → Script → Output`

This skill provides:
- YAML config format for declarative extraction
- Config examples for common patterns
- Workflow guidance for the agent

Everything else (batch processing, custom logic) is project-specific and built on top of this atomic operation.

---

## Project Structure

Each extraction project follows this structure:

```
projects/<project_name>/
├── configs/          # YAML configs (declarative extraction rules)
├── scripts/          # Custom Python scripts (agent writes as needed)
├── output/           # Extraction results (JSON, CSV)
└── docs/             # Project notes (optional)
```

**Simple rule**: configs/ = YAML, scripts/ = Python, output/ = results, docs/ = notes

---

## Agent Workflow

### Phase 1: Explore

1. **Analyze page structure**
   - Use `.claude/skills/extract/examples/scripts/analyze_page.py` (replace `URL_HERE`)
   - Identifies element counts, CSS classes, container patterns

2. **Identify extraction pattern**
   - Flat list → `simple_list.yaml`
   - Hierarchical (day→time→session) → `hierarchical.yaml`
   - Multiple item types → `multi_pattern.yaml`

3. **Ask user for schema confirmation** ⚠️ REQUIRED
   - NEVER assume fields
   - Use AskUserQuestion to confirm exact fields to extract
   - Example: "What fields? Title only / Title+Type / Title+Type+URL / Other"

### Phase 2: Configure

1. **Create project directory**
   ```bash
   mkdir -p projects/<name>/{configs,scripts,output,docs}
   ```

2. **Write YAML config** in `projects/<name>/configs/`
   - Copy from `.claude/skills/extract/examples/configs/`
   - Customize selectors, fields, transforms

### Phase 3: Extract

1. **Write extraction script** in `projects/<name>/scripts/`
   - Copy `.claude/skills/extract/examples/scripts/extract_template.py`
   - Customize `extract_items()` function
   - Script should: load config → fetch HTML → extract → save output

2. **Run extraction**
   ```bash
   cd projects/<name>
   ../../.venv/bin/python scripts/extract.py
   ```

### Phase 4: Iterate

If extraction fails:
- Copy `.claude/skills/extract/examples/scripts/extract_with_debug.py`
- Run with `--debug` to see selector matches and sample values
- Run with `--validate` to count null/missing fields
- Update config or add custom logic to script

---

## YAML Config Format

### Basic Structure

See: `.claude/skills/extract/examples/configs/simple_list.yaml`

Every config has:
- **version**: Config format version
- **name**: Descriptive name
- **source**: URL or file + options (headless, wait_for)
- **output**: Format (json/csv) and file path
- **items**: Selector patterns and field definitions

### Field Extraction

**Short form**: `"selector::attribute"` (e.g., `"h2.title::text"`, `"a::href"`)

**Full form**: Object with `selector`, `extract` keys

**Extract methods**: `text`, `html`, `href`, `src`, `class`, or any HTML attribute

Note: Example configs may reference `transform` but the basic template doesn't implement it - agent can add transform logic to custom scripts if needed.

### Hierarchical Data

See: `.claude/skills/extract/examples/configs/hierarchical.yaml`

For nested structures (day → time → sessions):
- Define **contexts** for parent containers
- Use **parent** to inherit fields from ancestors
- Items reference **context** to get inherited fields

### Multiple Patterns

See: `.claude/skills/extract/examples/configs/multi_pattern.yaml`

For pages with different item types:
- Define multiple **items** patterns with different selectors
- Use **schema** to enforce consistent output fields
- Items get `_type` field to identify which pattern matched

---

## When to Use YAML vs Custom Scripts

### Use YAML when:
- Single-page extraction
- Standard CSS selectors work
- Simple field transformations

### Write custom scripts when:
- Batch processing multiple URLs
- Complex pagination
- Custom data transformations
- Need to combine multiple extractions

**Example**: `projects/gtc-2026/` uses custom batch scripts that build on YAML configs

---

## Real-World Example: GTC 2026

```
projects/gtc-2026/
├── configs/
│   ├── session_list.yaml         # Extract session IDs from listing
│   └── session_details.yaml      # Extract details per session
├── scripts/
│   ├── extract_session_list.py   # Fetch list, save IDs
│   └── batch_extract.py          # Loop IDs, extract details, combine
└── output/
    ├── session_list.json
    ├── session_details.json
    └── sessions.csv
```

**Workflow**: Agent creates YAML configs → writes batch script → extracts all sessions

---

## Important Rules for Agent

### Schema Confirmation (REQUIRED)

**ALWAYS use AskUserQuestion to confirm exact fields before extraction.**

Example:
```
User: "Extract conference sessions"

Agent: [Uses AskUserQuestion]
  "What fields do you want to extract?"
  - Title only
  - Title + Type
  - Title + Type + Speaker + URL
  - Other (I'll specify)
```

**Rules**:
1. NEVER assume fields
2. NEVER add extra fields "to be helpful"
3. NEVER invent fields that don't exist on page
4. ALWAYS confirm schema before creating config

### Script Guidelines

1. **Start with YAML** - use configs for declarative extraction
2. **Add scripts when needed** - batch processing, custom logic
3. **Keep scripts project-specific** - no shared scripts across projects
4. **Use virtual environment** - `.venv/bin/python`
5. **NEVER modify** `.claude/skills/extract/` - it's read-only

---

## Reference: Example Files

**Config templates**: `.claude/skills/extract/examples/configs/`
- `simple_list.yaml`, `hierarchical.yaml`, `multi_pattern.yaml`

**Script templates**: `.claude/skills/extract/examples/scripts/`
- `analyze_page.py`, `extract_template.py`, `extract_with_debug.py`
