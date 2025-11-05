"""
Generic Web Extractor Base Class

This module provides an abstract base class for extracting structured data from web pages
using Crawl4AI's JsonCssExtractionStrategy or JsonXPathExtractionStrategy.
It can be extended for specific conference or website extraction needs.
"""

import json
import csv
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
from crawl4ai import JsonCssExtractionStrategy, JsonXPathExtractionStrategy


class WebExtractor(ABC):
    """
    Abstract base class for web content extraction using Crawl4AI strategies.

    Subclasses must implement:
    - get_schemas(): Return extraction schema(s) for CSS or XPath
    - post_process(): Transform extracted data as needed

    Provides common utilities:
    - URL conversion (relative to absolute)
    - CSV/JSON export
    - Statistics reporting
    """

    def __init__(self, strategy_type: str = "css", use_browser: bool = True):
        """
        Initialize the web extractor.

        Args:
            strategy_type: Type of extraction strategy ("css" or "xpath")
            use_browser: If True, use headless browser for JavaScript rendering (default: True)
        """
        self.strategy_type = strategy_type.lower()
        self.use_browser = use_browser
        if self.strategy_type not in ("css", "xpath"):
            raise ValueError("strategy_type must be 'css' or 'xpath'")

    @abstractmethod
    def get_schemas(self) -> Union[Dict, List[Dict]]:
        """
        Define the extraction schema(s) for this extractor.

        Returns:
            Single schema dict or list of schema dicts for multiple extraction passes

        Example schema:
            {
                "name": "Events",
                "baseSelector": "div.event",
                "fields": [
                    {"name": "title", "selector": "h2.title", "type": "text"},
                    {"name": "url", "selector": "a", "type": "attribute", "attribute": "href"}
                ]
            }
        """
        pass

    @abstractmethod
    def post_process(self, raw_data: List[Dict], base_url: str = "") -> Dict:
        """
        Post-process the raw extracted data.

        Args:
            raw_data: Raw data extracted by crawl4ai strategy
            base_url: Base URL for converting relative URLs to absolute

        Returns:
            Dictionary with 'total_events' and 'events' keys
        """
        pass

    @staticmethod
    def make_absolute_url(url: str, base_url: str) -> str:
        """
        Convert relative URL to absolute URL.

        Args:
            url: The URL (could be relative or absolute)
            base_url: The base URL to prefix (e.g., 'https://neurips.cc')

        Returns:
            Absolute URL
        """
        if not url or not base_url:
            return url

        # If already absolute, return as-is
        if url.startswith(("http://", "https://")):
            return url

        # Remove trailing slash from base_url if present
        base_url = base_url.rstrip("/")

        # Ensure relative URL starts with /
        if not url.startswith("/"):
            url = "/" + url

        return base_url + url

    async def extract(
        self, url_or_html: str, use_raw_html: bool = False, base_url: str = ""
    ) -> Dict:
        """
        Extract structured data from a URL or raw HTML using crawl4ai strategies.

        Args:
            url_or_html: URL to crawl or raw HTML string
            use_raw_html: If True, treats url_or_html as raw HTML
            base_url: Base URL to prefix for relative URLs

        Returns:
            Dictionary with 'total_events' and 'events' keys
        """
        crawl_url = f"raw://{url_or_html}" if use_raw_html else url_or_html

        # Get schema(s) from subclass
        schemas = self.get_schemas()
        if not isinstance(schemas, list):
            schemas = [schemas]

        all_extracted_data = []

        # Configure browser if needed for JavaScript rendering
        browser_config = BrowserConfig(headless=True) if self.use_browser else None

        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Run extraction for each schema
            for schema in schemas:
                # Create appropriate extraction strategy
                if self.strategy_type == "css":
                    strategy = JsonCssExtractionStrategy(schema, verbose=False)
                else:
                    strategy = JsonXPathExtractionStrategy(schema, verbose=False)

                # Configure crawler
                config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS, extraction_strategy=strategy
                )

                # Run extraction
                result = await crawler.arun(url=crawl_url, config=config)

                if result.success and result.extracted_content:
                    try:
                        data = json.loads(result.extracted_content)
                        if isinstance(data, list):
                            all_extracted_data.extend(data)
                        else:
                            all_extracted_data.append(data)
                    except json.JSONDecodeError:
                        print(
                            f"Warning: Failed to parse extracted content for schema '{schema.get('name', 'unknown')}'"
                        )

        # Post-process the data
        return self.post_process(all_extracted_data, base_url)

    @staticmethod
    def save_to_csv(events: List[Dict], output_file: str, fieldnames: List[str] = None):
        """
        Save events to CSV file.

        Args:
            events: List of event dictionaries
            output_file: Output CSV file path
            fieldnames: List of field names to include (defaults to all keys from first event)
        """
        if not events:
            print("No events to save!")
            return

        # Auto-detect fieldnames if not provided
        if fieldnames is None:
            fieldnames = list(events[0].keys())

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for event in events:
                # Only write fields that exist in fieldnames
                row = {k: event.get(k, "") for k in fieldnames}
                writer.writerow(row)

        print(f"Saved {len(events)} events to {output_file}")

    @staticmethod
    def print_statistics(events: List[Dict], group_by: str = "type"):
        """
        Print statistics about events grouped by a field.

        Args:
            events: List of event dictionaries
            group_by: Field name to group statistics by
        """
        counts = {}
        for event in events:
            key = event.get(group_by, "Unknown")
            counts[key] = counts.get(key, 0) + 1

        print(f"\n{group_by.title()} Statistics:")
        print("-" * 40)
        for key, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{str(key):20s}: {count:4d}")
        print("-" * 40)
        print(f"{'Total':20s}: {len(events):4d}\n")
