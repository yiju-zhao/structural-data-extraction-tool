import asyncio
import json
import argparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai import JsonCssExtractionStrategy


async def extract_conference_events(url_or_html: str, use_raw_html: bool = False):
    """
    Extract conference events with type, title, date, time, and URL.

    Args:
        url_or_html: URL to crawl or raw HTML string
        use_raw_html: If True, treats url_or_html as raw HTML with raw:// prefix
    """

    # Schema for extracting individual event sessions
    events_schema = {
        "name": "Conference Events",
        "baseSelector": ".eventsession",
        "fields": [
            {"name": "type", "selector": ".hdr-style", "type": "text", "default": ""},
            {"name": "title", "selector": ".title-style a", "type": "text"},
            {
                "name": "url",
                "selector": ".title-style a",
                "type": "attribute",
                "attribute": "href",
            },
            {
                "name": "end_time",
                "selector": ".end-time",
                "type": "text",
                "default": "",
            },
        ],
    }

    # Prepare URL
    crawl_url = f"raw://{url_or_html}" if use_raw_html else url_or_html

    async with AsyncWebCrawler() as crawler:
        # Extract events
        result = await crawler.arun(
            url=crawl_url,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                extraction_strategy=JsonCssExtractionStrategy(events_schema),
            ),
        )

        events_data = json.loads(result.extracted_content)

        # Post-process to add date and time information
        # Note: Since CSS selectors can't easily traverse up to parent/ancestor elements,
        # we need to extract this information separately from the HTML
        processed_events = []

        for event in events_data:
            # Clean up type field (remove trailing colon and whitespace)
            event_type = event.get("type", "").strip().rstrip(":")

            # Clean up end_time (remove parentheses)
            end_time = event.get("end_time", "").strip("()").strip()

            processed_event = {
                "type": event_type if event_type else "General",
                "title": event.get("title", "").strip(),
                "url": event.get("url", "").strip(),
                "end_time": end_time,
                # These fields need to be extracted via custom logic or additional schemas
                "date": "",  # Will be populated below
                "time": "",  # Will be populated below
            }

            processed_events.append(processed_event)

        return {
            "total_events": len(processed_events),
            "events": processed_events,
            "raw_html_available": result.html is not None,
        }


async def extract_with_context(url_or_html: str, use_raw_html: bool = False):
    """
    Enhanced extraction that captures date and time context using custom processing.
    This version extracts the full HTML and processes it to maintain hierarchical context.
    """
    from bs4 import BeautifulSoup

    crawl_url = f"raw://{url_or_html}" if use_raw_html else url_or_html

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=crawl_url, config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        )

        # Parse HTML with BeautifulSoup for better context extraction
        soup = BeautifulSoup(result.html, "html.parser")

        all_events = []

        # Find all day containers
        day_containers = soup.find_all("div", class_=lambda x: x and "container2" in x)

        for container in day_containers:
            # Extract date from header box
            hdrbox = container.find("div", class_="hdrbox")
            date = hdrbox.get_text(strip=True) if hdrbox else "Unknown Date"

            # Find all timeboxes in this day
            timeboxes = container.find_all("div", class_="timebox")

            for timebox in timeboxes:
                # Extract time
                time_elem = timebox.find("div", class_="time")
                time = time_elem.get_text(strip=True) if time_elem else "Unknown Time"

                # Find all event sessions in this timebox
                events = timebox.find_all("div", class_="eventsession")

                for event in events:
                    # Extract type
                    hdr_style = event.find("div", class_="hdr-style")
                    event_type = (
                        hdr_style.get_text(strip=True).rstrip(":")
                        if hdr_style
                        else "General"
                    )

                    # Extract title and URL
                    title_link = (
                        event.find("div", class_="title-style").find("a")
                        if event.find("div", class_="title-style")
                        else None
                    )
                    title = (
                        title_link.get_text(strip=True) if title_link else "No Title"
                    )
                    url = title_link.get("href", "") if title_link else ""

                    # Extract end time
                    end_time_elem = event.find("span", class_="end-time")
                    end_time = (
                        end_time_elem.get_text(strip=True).strip("()")
                        if end_time_elem
                        else ""
                    )

                    all_events.append(
                        {
                            "date": date,
                            "time": time,
                            "type": event_type,
                            "title": title,
                            "url": url,
                            "end_time": end_time,
                        }
                    )

        return {"total_events": len(all_events), "events": all_events}


async def main(url: str, output_file: str = None):
    """
    Main function to extract conference events from a URL.

    Args:
        url: The URL to crawl and extract events from
        output_file: Optional file path to save JSON output
    """
    print(f"Extracting events from: {url}")
    print("=" * 80)

    result = await extract_with_context(url, use_raw_html=False)

    # Print to console
    print(json.dumps(result, indent=2))

    # Save to file if specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n{'=' * 80}")
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract conference events (type, title, date, time, URL) from a webpage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python crawl_structure_output.py https://neurips.cc/virtual/2025/events/schedule
  python crawl_structure_output.py https://neurips.cc/virtual/2025/events/schedule -o events.json
        """,
    )
    parser.add_argument("url", type=str, help="URL of the webpage to crawl")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (optional)",
    )

    args = parser.parse_args()
    asyncio.run(main(args.url, args.output))
