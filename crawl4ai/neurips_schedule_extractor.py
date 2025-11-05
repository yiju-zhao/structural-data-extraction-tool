#!/usr/bin/env python3
"""
NeurIPS Conference Schedule Extractor

Extracts conference events (workshops, tutorials, invited talks, oral sessions, etc.)
from NeurIPS virtual conference schedule pages using crawl4ai extraction strategies.
"""

import asyncio
import json
import argparse
from typing import Dict, List, Set
from bs4 import BeautifulSoup
from crawl_web_extractor import WebExtractor


# ============================================
# NEURIPS-SPECIFIC EVENT TYPES TO INCLUDE
# ============================================
NEURIPS_INCLUDED_TYPES = {
    "Workshop",
    "Tutorial",
    "Invited Talk",
    "Oral",
    "Oral Session",
    "Expo Talk Panel",
    "Expo Demonstration",
    "Expo Workshop",
}

# Location prefixes used in NeurIPS event types
NEURIPS_LOCATION_PREFIXES = ["Mexico City", "San Diego", "Vancouver", "Vienna"]


class NeurIPSScheduleExtractor(WebExtractor):
    """
    Extractor for NeurIPS conference schedule pages.

    Handles three types of event structures:
    1. Timebox events - Events inside <div class="timebox">
    2. Oral sessions - <div class="oral-session"> with nested papers
    3. Standalone events - Direct <div class="eventsession"> elements
    """

    def __init__(
        self,
        included_types: Set[str] = None,
        location_prefixes: List[str] = None,
        strategy_type: str = "css",
        use_browser: bool = True,
    ):
        """
        Initialize NeurIPS schedule extractor.

        Args:
            included_types: Set of event types to filter for
            location_prefixes: List of location prefixes to normalize
            strategy_type: Extraction strategy type ("css" or "xpath")
            use_browser: If True, use headless browser for JavaScript rendering (default: True)
        """
        super().__init__(strategy_type=strategy_type, use_browser=use_browser)
        self.included_types = included_types or NEURIPS_INCLUDED_TYPES
        self.location_prefixes = location_prefixes or NEURIPS_LOCATION_PREFIXES

    def get_schemas(self) -> Dict:
        """
        Return the extraction schema for NeurIPS schedule pages.

        Since NeurIPS has complex nested structures with parent-child relationships,
        we extract day containers as HTML blocks for post-processing with BeautifulSoup.
        """
        if self.strategy_type == "css":
            return {
                "name": "NeurIPS Day Containers",
                "baseSelector": "div[class*='container2']",
                "fields": [
                    {
                        "name": "date_header",
                        "selector": "div.hdrbox",
                        "type": "text",
                        "default": "Unknown Date",
                    },
                    {
                        "name": "container_html",
                        "selector": "",  # Empty selector = get current element's HTML
                        "type": "html",
                    },
                ],
            }
        else:  # xpath
            return {
                "name": "NeurIPS Day Containers",
                "baseSelector": "//div[contains(@class, 'container2')]",
                "fields": [
                    {
                        "name": "date_header",
                        "selector": ".//div[@class='hdrbox']",
                        "type": "text",
                        "default": "Unknown Date",
                    },
                    {
                        "name": "container_html",
                        "selector": ".",
                        "type": "html",
                    },
                ],
            }

    def post_process(self, raw_data: List[Dict], base_url: str = "") -> Dict:
        """
        Post-process extracted day containers to extract individual events.

        Args:
            raw_data: List of day container dicts with 'date_header' and 'container_html'
            base_url: Base URL for converting relative URLs to absolute

        Returns:
            Dictionary with 'total_events' and 'events' keys
        """
        all_events = []

        for container_data in raw_data:
            date = container_data.get("date_header", "Unknown Date").strip()
            container_html = container_data.get("container_html", "")

            if not container_html:
                continue

            # Parse the container HTML with BeautifulSoup
            soup = BeautifulSoup(container_html, "html.parser")

            # Extract different event types
            all_events.extend(self._extract_timebox_events(soup, date, base_url))
            all_events.extend(self._extract_standalone_events(soup, date, base_url))
            all_events.extend(self._extract_oral_sessions(soup, date, base_url))

        return {"total_events": len(all_events), "events": all_events}

    def _extract_timebox_events(
        self, container_soup: BeautifulSoup, date: str, base_url: str
    ) -> List[Dict]:
        """Extract events from timebox structures."""
        events = []
        timeboxes = container_soup.find_all("div", class_="timebox")

        for timebox in timeboxes:
            # Extract time
            time_elem = timebox.find("div", class_="time")
            time = time_elem.get_text(strip=True) if time_elem else "Unknown Time"

            # Find all event sessions in this timebox
            event_sessions = timebox.find_all("div", class_="eventsession")

            for event in event_sessions:
                event_data = self._extract_event_details(event, date, time, base_url)
                if event_data:
                    events.append(event_data)

        return events

    def _extract_standalone_events(
        self, container_soup: BeautifulSoup, date: str, base_url: str
    ) -> List[Dict]:
        """Extract standalone event sessions (not inside timeboxes)."""
        events = []

        # Get all eventsessions, excluding those in timeboxes
        all_eventsessions = container_soup.find_all("div", class_="eventsession")

        for event in all_eventsessions:
            # Check if this event is inside a timebox
            is_in_timebox = any(
                "timebox" in (parent.get("class") or []) for parent in event.parents
            )

            if not is_in_timebox:
                event_data = self._extract_event_details(
                    event, date, "", base_url
                )  # No time for standalone
                if event_data:
                    events.append(event_data)

        return events

    def _extract_oral_sessions(
        self, container_soup: BeautifulSoup, date: str, base_url: str
    ) -> List[Dict]:
        """Extract oral session events with nested papers."""
        events = []

        # Find all oral-session events
        oral_sessions = container_soup.find_all(
            "div", class_=lambda x: x and "oral-session" in x
        )

        for session in oral_sessions:
            session_title_elem = session.find("div", class_="sessiontitle")
            if not session_title_elem:
                continue

            session_link = session_title_elem.find("a")
            if not session_link:
                continue

            # Extract session title and time
            session_time_span = session_link.find("span", class_="sessiontime")
            session_time = ""
            if session_time_span:
                session_time = session_time_span.get_text(strip=True).strip("[]")
                session_title = (
                    session_link.get_text(strip=True)
                    .replace(session_time_span.get_text(strip=True), "")
                    .strip()
                )
            else:
                session_title = session_link.get_text(strip=True)

            session_url = self.make_absolute_url(
                session_link.get("href", ""), base_url
            )

            # Extract end time
            end_time_elem = session.find("span", class_="end-time")
            end_time = (
                end_time_elem.get_text(strip=True).strip("()")
                if end_time_elem
                else ""
            )

            # Extract papers
            papers = []
            content_items = session.find_all("div", class_="content")
            for content in content_items:
                content_link = content.find("a")
                if content_link:
                    paper_url = self.make_absolute_url(
                        content_link.get("href", ""), base_url
                    )
                    papers.append(
                        {
                            "title": content_link.get_text(strip=True),
                            "url": paper_url,
                        }
                    )

            events.append(
                {
                    "date": date,
                    "time": session_time,
                    "type": "Oral Session",
                    "title": session_title,
                    "url": session_url,
                    "speaker": "",
                    "end_time": end_time,
                    "papers": papers,
                }
            )

        return events

    def _extract_event_details(
        self, event_elem: BeautifulSoup, date: str, time: str, base_url: str
    ) -> Dict:
        """Extract details from a single event element."""
        # Extract type
        hdr_style = event_elem.find("div", class_="hdr-style")
        event_type = (
            hdr_style.get_text(strip=True).rstrip(":") if hdr_style else "General"
        )

        # Extract title and URL
        title_link = (
            event_elem.find("div", class_="title-style").find("a")
            if event_elem.find("div", class_="title-style")
            else None
        )
        title = title_link.get_text(strip=True) if title_link else "No Title"
        url = self.make_absolute_url(
            title_link.get("href", "") if title_link else "", base_url
        )

        # Extract speaker
        speaker_elem = event_elem.find("div", class_="speaker-style")
        speaker = speaker_elem.get_text(strip=True) if speaker_elem else ""

        # Extract end time
        end_time_elem = event_elem.find("span", class_="end-time")
        end_time = (
            end_time_elem.get_text(strip=True).strip("()") if end_time_elem else ""
        )

        return {
            "date": date,
            "time": time,
            "type": event_type,
            "title": title,
            "url": url,
            "speaker": speaker,
            "end_time": end_time,
        }

    def filter_events(self, events: List[Dict]) -> List[Dict]:
        """
        Filter events by type and normalize location prefixes.

        Args:
            events: List of event dictionaries

        Returns:
            Filtered list of events
        """
        filtered = []
        for event in events:
            event_type = event.get("type", "").strip()

            # Normalize location prefixes
            for prefix in self.location_prefixes:
                if event_type.startswith(prefix):
                    event_type = event_type.replace(prefix, "").strip()
                    break

            # Check if normalized type is in included types
            if event_type in self.included_types:
                event["type"] = event_type
                filtered.append(event)

        return filtered

    @staticmethod
    def flatten_oral_sessions(events: List[Dict]) -> List[Dict]:
        """
        Flatten oral sessions with papers into separate CSV rows.

        Args:
            events: List of event dictionaries

        Returns:
            Flattened list with papers as separate entries
        """
        flattened = []
        for event in events:
            if "papers" in event and event["papers"]:
                # Add the session itself
                session_event = {k: v for k, v in event.items() if k != "papers"}
                flattened.append(session_event)

                # Add each paper as a separate event
                for paper in event["papers"]:
                    paper_event = session_event.copy()
                    paper_event["title"] = f"  → {paper['title']}"
                    paper_event["url"] = paper["url"]
                    paper_event["type"] = "Oral Paper"
                    flattened.append(paper_event)
            else:
                flattened.append(event)
        return flattened


async def main(
    url: str,
    output_json: str = None,
    output_csv: str = None,
    filter_types: bool = True,
    no_flatten: bool = False,
    base_url: str = "",
):
    """
    Main function to extract NeurIPS conference events from a URL.

    Args:
        url: The URL to crawl and extract events from
        output_json: Optional file path to save JSON output
        output_csv: Optional file path to save CSV output
        filter_types: Whether to filter events by NEURIPS_INCLUDED_TYPES
        no_flatten: If True, don't flatten oral session papers in CSV
        base_url: Base URL to prefix for relative URLs (e.g., 'https://neurips.cc')
    """
    print(f"Extracting events from: {url}")
    print("=" * 80)

    # Initialize extractor
    extractor = NeurIPSScheduleExtractor()

    # Extract events
    result = await extractor.extract(url, use_raw_html=False, base_url=base_url)

    # Print statistics for all events
    print(f"\nTotal events extracted: {result['total_events']}")
    WebExtractor.print_statistics(result["events"])

    # Filter events if requested
    events_to_save = result["events"]
    if filter_types:
        print(f"Filtering for types: {', '.join(NEURIPS_INCLUDED_TYPES)}")
        events_to_save = extractor.filter_events(result["events"])
        print(f"Filtered to {len(events_to_save)} events")
        WebExtractor.print_statistics(events_to_save)

    # Save to JSON if specified
    if output_json:
        output_data = {"total_events": len(events_to_save), "events": events_to_save}
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"JSON saved to: {output_json}")

    # Save to CSV if specified
    if output_csv:
        # Flatten if needed
        csv_events = events_to_save
        if not no_flatten:
            csv_events = extractor.flatten_oral_sessions(events_to_save)

        WebExtractor.save_to_csv(
            csv_events,
            output_csv,
            fieldnames=["date", "time", "type", "title", "url", "speaker", "end_time"],
        )

    # Print to console if no output files specified
    if not output_json and not output_csv:
        print("\n" + "=" * 80)
        print("JSON Output:")
        print("=" * 80)
        print(
            json.dumps(
                {"total_events": len(events_to_save), "events": events_to_save},
                indent=2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Extract NeurIPS conference events and export to JSON/CSV. Filter types: {', '.join(NEURIPS_INCLUDED_TYPES)}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract to CSV with absolute URLs
  python neurips_schedule_extractor.py https://neurips.cc/virtual/2025/events/schedule -c events.csv -b https://neurips.cc

  # Extract to JSON only
  python neurips_schedule_extractor.py https://neurips.cc/virtual/2025/events/schedule -j events.json -b https://neurips.cc

  # Extract to both JSON and CSV
  python neurips_schedule_extractor.py https://neurips.cc/virtual/2025/events/schedule -j events.json -c events.csv -b https://neurips.cc

  # Extract all events without filtering
  python neurips_schedule_extractor.py https://neurips.cc/virtual/2025/events/schedule -c events.csv --no-filter -b https://neurips.cc

  # Don't flatten oral session papers
  python neurips_schedule_extractor.py https://neurips.cc/virtual/2025/events/schedule -c events.csv --no-flatten -b https://neurips.cc
        """,
    )
    parser.add_argument("url", type=str, help="URL of the NeurIPS schedule page")
    parser.add_argument(
        "-j",
        "--json",
        type=str,
        default=None,
        help="Output JSON file path (optional)",
    )
    parser.add_argument(
        "-c",
        "--csv",
        type=str,
        default=None,
        help="Output CSV file path (optional)",
    )
    parser.add_argument(
        "-b",
        "--base-url",
        type=str,
        default="",
        help="Base URL to prefix for relative URLs (e.g., https://neurips.cc)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Don't filter events by type (include all events)",
    )
    parser.add_argument(
        "--no-flatten",
        action="store_true",
        help="Don't flatten oral session papers into separate CSV rows",
    )

    args = parser.parse_args()
    asyncio.run(
        main(
            args.url,
            output_json=args.json,
            output_csv=args.csv,
            filter_types=not args.no_filter,
            no_flatten=args.no_flatten,
            base_url=args.base_url,
        )
    )
