#!/usr/bin/env python3
"""
Unified Web Data Extraction Tool

A declarative YAML-based extraction system that handles everything from
simple flat lists to complex hierarchical structures with context inheritance.

Usage:
    # Analyze page structure
    python extract.py --analyze <url>

    # Run extraction from config
    python extract.py <config.yaml>

    # Convert JSON to CSV
    python extract.py --json2csv data.json
"""

import argparse
import asyncio
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

import yaml
from bs4 import BeautifulSoup, Tag
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode


# =============================================================================
# HTML Fetching
# =============================================================================

async def fetch_html(url: str, headless: bool = True, wait_for: str = None) -> str:
    """Fetch HTML content from URL using crawl4ai."""
    browser_cfg = BrowserConfig(headless=headless)
    crawl_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_for=wait_for
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=crawl_cfg)
        if result.success:
            return result.html
        else:
            raise Exception(f"Failed to fetch {url}: {result.error_message}")


# =============================================================================
# HTML Analysis (for --analyze mode)
# =============================================================================

def analyze_html(html: str) -> dict:
    """Analyze HTML structure and return statistics."""
    soup = BeautifulSoup(html, "lxml")

    elements = Counter()
    classes = Counter()
    ids = Counter()

    for tag in soup.find_all(True):
        elements[tag.name] += 1
        if tag.get("class"):
            for cls in tag.get("class"):
                classes[cls] += 1
        if tag.get("id"):
            ids[tag.get("id")] += 1

    # Find potential list containers (elements with multiple similar children)
    containers = []
    for tag in soup.find_all(["div", "ul", "ol", "section", "article"]):
        children = tag.find_all(recursive=False)
        if len(children) >= 3:
            child_tags = [c.name for c in children]
            if len(set(child_tags)) <= 2:
                selector = tag.name
                if tag.get("class"):
                    selector += "." + ".".join(tag.get("class"))
                elif tag.get("id"):
                    selector += "#" + tag.get("id")
                containers.append((selector, len(children)))

    return {
        "elements": dict(elements.most_common(20)),
        "classes": dict(classes.most_common(30)),
        "ids": dict(ids.most_common(10)),
        "potential_containers": sorted(containers, key=lambda x: x[1], reverse=True)[:10]
    }


def print_analysis(stats: dict):
    """Print analysis results in a readable format."""
    print("\n=== Top Elements ===")
    for elem, count in list(stats["elements"].items())[:15]:
        print(f"  {elem}: {count}")

    print("\n=== Top Classes ===")
    for cls, count in list(stats["classes"].items())[:20]:
        print(f"  .{cls}: {count}")

    print("\n=== Potential List Containers ===")
    for selector, count in stats["potential_containers"]:
        print(f"  {selector} ({count} children)")


# =============================================================================
# Transforms
# =============================================================================

class Transforms:
    """Collection of field transformation functions."""

    @staticmethod
    def apply(value: str, transform: Any, base_url: str = "") -> str:
        """Apply a transform to a value."""
        if value is None:
            return None

        if isinstance(transform, str):
            # Named transform
            if transform == "strip":
                return value.strip()
            elif transform == "absolute_url":
                return urljoin(base_url, value) if base_url else value
            elif transform == "lowercase":
                return value.lower()
            elif transform == "uppercase":
                return value.upper()
            else:
                return value

        elif isinstance(transform, dict):
            # Transform with parameters
            if "regex" in transform:
                pattern = transform["regex"]
                group = transform.get("group", 1)
                match = re.search(pattern, value)
                if match:
                    try:
                        return match.group(group)
                    except IndexError:
                        return match.group(0)
                return None

            elif "replace" in transform:
                old, new = transform["replace"]
                return value.replace(old, new)

            elif "strip" in transform and transform["strip"]:
                return value.strip()

            elif "type" in transform:
                t_type = transform["type"]
                if t_type == "url_join":
                    base = transform.get("base", base_url)
                    return urljoin(base, value) if base else value
                elif t_type == "regex":
                    pattern = transform["pattern"]
                    group = transform.get("group", 1)
                    match = re.search(pattern, value)
                    if match:
                        try:
                            return match.group(group)
                        except IndexError:
                            return match.group(0)
                    return None

        return value


# =============================================================================
# Config Extractor
# =============================================================================

class ConfigExtractor:
    """Interprets YAML extraction configs and executes extraction."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.base_url = ""
        self.transforms = {}
        self.soup = None  # Shared soup object to maintain element references

    def _load_config(self) -> dict:
        """Load and validate YAML config."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Basic validation
        if "items" not in config:
            raise ValueError("Config must have 'items' section")

        return config

    async def run(self) -> list[dict]:
        """Execute the extraction pipeline."""
        # 1. Fetch HTML
        html = await self._fetch_source()

        # 2. Parse HTML once and reuse (critical for context matching)
        self.soup = BeautifulSoup(html, "lxml")

        # 3. Parse transforms
        self.transforms = self.config.get("transforms", {})

        # 4. Build context tree if contexts defined
        context_data = self._build_context_tree()

        # 5. Extract items from each pattern
        results = []
        for pattern in self.config["items"]:
            items = self._extract_pattern(pattern, context_data)
            results.extend(items)

        # 6. Apply output schema if defined
        if "schema" in self.config:
            results = self._apply_schema(results)

        # 7. Save outputs
        self._save_outputs(results)

        return results

    async def _fetch_source(self) -> str:
        """Fetch HTML from URL or file."""
        source = self.config.get("source", {})

        if "file" in source:
            file_path = source["file"]
            # Resolve relative to config file
            if not Path(file_path).is_absolute():
                file_path = self.config_path.parent / file_path
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        elif "url" in source:
            url = source["url"]
            self.base_url = url
            options = source.get("options", {})
            headless = options.get("headless", True)
            wait_for = options.get("wait_for")

            print(f"Fetching {url}...")
            return await fetch_html(url, headless=headless, wait_for=wait_for)

        else:
            raise ValueError("Config source must have 'url' or 'file'")

    def _build_context_tree(self) -> dict:
        """Build context hierarchy from HTML.

        Returns a dict mapping element -> context data
        """
        contexts_config = self.config.get("contexts", [])
        if not contexts_config:
            return {}

        context_data = {}

        # Build parent chain
        parent_map = {}
        for ctx in contexts_config:
            if "parent" in ctx:
                parent_map[ctx["name"]] = ctx["parent"]

        # Process contexts in order (parents before children)
        for ctx in contexts_config:
            selector = ctx["selector"]
            ctx_name = ctx["name"]
            fields = ctx.get("fields", {})
            parent_name = ctx.get("parent")

            # Find all elements matching this context
            elements = self.soup.select(selector)

            for elem in elements:
                # Extract fields for this context element
                elem_data = {}
                for field_name, field_config in fields.items():
                    value = self._extract_field_value(elem, field_config)
                    elem_data[field_name] = value

                # Inherit from parent context if applicable
                if parent_name:
                    parent_elem = self._find_parent_context(elem, parent_name, context_data)
                    if parent_elem:
                        # Merge parent data
                        elem_data = {**context_data.get(id(parent_elem), {}), **elem_data}

                context_data[id(elem)] = elem_data

                # Also store reference for child lookup
                elements_key = f"_elements_{ctx_name}"
                if elements_key not in context_data:
                    context_data[elements_key] = []
                context_data[elements_key].append((elem, elem_data))

        return context_data

    def _find_parent_context(self, elem: Tag, parent_name: str, context_data: dict) -> Optional[Tag]:
        """Find the parent context element for a given element."""
        elements_key = f"_elements_{parent_name}"
        if elements_key not in context_data:
            return None

        # Walk up the DOM to find which parent context element contains this element
        for parent_elem, _ in context_data[elements_key]:
            if elem in parent_elem.descendants:
                return parent_elem

        return None

    def _extract_pattern(self, pattern: dict, context_data: dict) -> list[dict]:
        """Extract items matching a pattern."""
        selector = pattern["selector"]
        selector_type = pattern.get("selector_type", "css")
        context_name = pattern.get("context")
        fields = pattern.get("fields", {})
        pattern_name = pattern.get("name", "item")

        # Find matching elements using shared soup object
        if selector_type == "regex":
            # Regex matching on class names
            elements = []
            for sel in selector.split(","):
                sel = sel.strip()
                if "[class*=" in sel or "[class~=" in sel:
                    # CSS attribute selector
                    elements.extend(self.soup.select(sel))
                else:
                    # Treat as regex pattern on class
                    pattern_re = re.compile(sel.replace("div", "").strip("."))
                    for tag in self.soup.find_all("div"):
                        classes = " ".join(tag.get("class", []))
                        if pattern_re.search(classes):
                            elements.append(tag)
        else:
            elements = self.soup.select(selector)

        results = []
        for elem in elements:
            item = {"_type": pattern_name}

            # Get context data if applicable
            if context_name and context_data:
                ctx_elem = self._find_context_for_element(elem, context_name, context_data)
                if ctx_elem:
                    inherited = context_data.get(id(ctx_elem), {})
                    item.update(inherited)

            # Extract fields
            for field_name, field_config in fields.items():
                value = self._extract_field_value(elem, field_config)
                item[field_name] = value

            results.append(item)

        return results

    def _find_context_for_element(self, elem: Tag, context_name: str, context_data: dict) -> Optional[Tag]:
        """Find which context element contains the given element."""
        elements_key = f"_elements_{context_name}"
        if elements_key not in context_data:
            return None

        for ctx_elem, _ in context_data[elements_key]:
            if elem in ctx_elem.descendants or elem == ctx_elem:
                return ctx_elem

        return None

    def _extract_field_value(self, elem: Tag, field_config: Any) -> Optional[str]:
        """Extract a field value from an element based on config."""
        if isinstance(field_config, str):
            # Short form: "selector::attr" or just "selector"
            if "::" in field_config:
                selector, attr = field_config.rsplit("::", 1)
            else:
                selector, attr = field_config, "text"

            target = elem.select_one(selector.strip()) if selector.strip() else elem
            return self._get_element_value(target, attr)

        elif isinstance(field_config, dict):
            # Full form with explicit keys
            source = field_config.get("source", "selector")

            if source == "class":
                # Extract from element's class
                classes = " ".join(elem.get("class", []))
                value = classes
            elif source == "id":
                value = elem.get("id", "")
            elif source == "tag":
                value = elem.name
            else:
                # CSS selector
                selector = field_config.get("selector", "")
                extract = field_config.get("extract", "text")
                target = elem.select_one(selector) if selector else elem
                value = self._get_element_value(target, extract)

            # Apply transform
            if "transform" in field_config and value:
                transform = field_config["transform"]
                # Check if it's a named transform from config
                if isinstance(transform, str) and transform in self.transforms:
                    transform = self.transforms[transform]
                value = Transforms.apply(value, transform, self.base_url)

            return value

        return None

    def _get_element_value(self, elem: Optional[Tag], attr: str) -> Optional[str]:
        """Get value from element by attribute type."""
        if elem is None:
            return None

        if attr == "text":
            return elem.get_text(strip=True)
        elif attr == "html":
            return str(elem)
        elif attr == "class":
            return " ".join(elem.get("class", []))
        else:
            # Attribute value
            value = elem.get(attr)
            # Auto-resolve URLs
            if value and attr in ("href", "src") and self.base_url:
                value = urljoin(self.base_url, value)
            return value

    def _apply_schema(self, results: list[dict]) -> list[dict]:
        """Apply output schema to ensure consistent fields."""
        schema = self.config["schema"]
        fields = schema.get("fields", [])

        if not fields:
            return results

        # Build field list
        field_names = [f["name"] if isinstance(f, dict) else f for f in fields]

        normalized = []
        for item in results:
            new_item = {}
            for field_name in field_names:
                new_item[field_name] = item.get(field_name)

            # Remove internal fields like _type unless in schema
            if "_type" in item and "_type" not in field_names:
                pass  # Don't include
            elif "_type" in field_names:
                new_item["_type"] = item.get("_type")

            normalized.append(new_item)

        return normalized

    def _save_outputs(self, results: list[dict]):
        """Save extraction results to configured output formats."""
        output_config = self.config.get("output", {})

        formats = output_config.get("format", ["json"])
        if isinstance(formats, str):
            formats = [formats]

        base_path = output_config.get("path", "output")
        base_path = Path(base_path)

        # Ensure parent directory exists
        base_path.parent.mkdir(parents=True, exist_ok=True)

        for fmt in formats:
            if fmt == "json":
                json_path = base_path.with_suffix(".json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Saved {len(results)} items to {json_path}")

            elif fmt == "csv":
                csv_path = base_path.with_suffix(".csv")
                if results:
                    fieldnames = list(results[0].keys())
                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in results:
                            writer.writerow({k: (v if v is not None else "") for k, v in row.items()})
                    print(f"Saved {len(results)} items to {csv_path}")


# =============================================================================
# JSON to CSV Utility
# =============================================================================

def json_to_csv(json_path: str):
    """Convert JSON file to CSV."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("No data to convert")
        return

    csv_path = Path(json_path).with_suffix(".csv")
    fieldnames = list(data[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow({k: (v if v is not None else "") for k, v in row.items()})

    print(f"Converted to {csv_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Unified Web Data Extraction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze page structure
    python extract.py --analyze https://example.com/products

    # Run extraction from config
    python extract.py configs/my_extraction.yaml

    # Convert JSON to CSV
    python extract.py --json2csv output.json
        """
    )
    parser.add_argument("config", nargs="?", help="YAML config file or URL (with --analyze)")
    parser.add_argument("--analyze", "-a", action="store_true", help="Analyze page structure")
    parser.add_argument("--json2csv", help="Convert JSON file to CSV")
    parser.add_argument("--no-headless", action="store_true", help="Show browser window")

    args = parser.parse_args()

    # JSON to CSV conversion
    if args.json2csv:
        json_to_csv(args.json2csv)
        return

    if not args.config:
        parser.print_help()
        return

    # Analyze mode
    if args.analyze:
        url = args.config
        print(f"Fetching {url}...")
        html = await fetch_html(url, headless=not args.no_headless)
        print("\nAnalyzing page structure...")
        stats = analyze_html(html)
        print_analysis(stats)
        return

    # Config-based extraction
    config_path = args.config
    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        return

    extractor = ConfigExtractor(config_path)
    results = await extractor.run()
    print(f"\nExtraction complete: {len(results)} items")


if __name__ == "__main__":
    asyncio.run(main())
