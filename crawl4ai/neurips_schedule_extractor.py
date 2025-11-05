#!/usr/bin/env python3
"""
NeurIPS Conference Schedule Extractor (LLM-based)

Uses the reusable LLM extractor (crawl_llm_extractor.py) with a Pydantic schema
to extract conference events (workshops, tutorials, invited talks, oral sessions, etc.)
from NeurIPS virtual conference schedule pages.
"""

import asyncio
import json
import argparse
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field

from crawl_web_extractor import WebExtractor  # for utilities: save_to_csv, print_statistics
from crawl_llm_extractor import LLMExtractor


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


# ============================================
# Pydantic Schemas for LLM Extraction
# ============================================
class PaperItem(BaseModel):
    """Nested paper item for oral sessions."""

    title: str = Field(description="Paper title as shown on the page")
    url: Optional[str] = Field(
        default=None, description="Absolute or relative URL to the paper/session"
    )


class NeurIPSEvent(BaseModel):
    """Schema for an extracted conference event."""

    date: Optional[str] = Field(
        default=None,
        description="Event date as indicated on the page (e.g., by day header)",
    )
    time: Optional[str] = Field(
        default=None, description="Start time of the event as displayed (e.g., 10:00)"
    )
    end_time: Optional[str] = Field(
        default=None, description="End time if available (e.g., 11:30)"
    )
    type: Optional[str] = Field(
        default=None,
        description="Event type (e.g., Workshop, Tutorial, Invited Talk, Oral Session)",
    )
    title: str = Field(description="Event title as shown on the page")
    url: Optional[str] = Field(
        default=None, description="Absolute or relative URL to the event details"
    )
    speaker: Optional[str] = Field(
        default=None, description="Speaker name(s) if displayed for the event"
    )
    papers: Optional[List[PaperItem]] = Field(
        default=None,
        description="For oral sessions: list of papers (title and url)",
    )


class NeurIPSScheduleExtractor:
    """
    LLM-powered extractor for NeurIPS conference schedule pages.
    """

    def __init__(
        self,
        included_types: Optional[Set[str]] = None,
        location_prefixes: Optional[List[str]] = None,
        model: str = "gpt-4o",
        headless: bool = True,
        verbose: bool = False,
    ):
        self.included_types = included_types or NEURIPS_INCLUDED_TYPES
        self.location_prefixes = location_prefixes or NEURIPS_LOCATION_PREFIXES
        self.llm = LLMExtractor(model=model, headless=headless, verbose=verbose)

    def _instruction(self) -> str:
        """LLM instruction to extract events from a NeurIPS schedule page."""
        included = ", ".join(sorted(self.included_types))
        prefixes = ", ".join(self.location_prefixes)
        return f"""
You are extracting a conference schedule from a NeurIPS virtual conference page.

Task: Return a JSON array of events matching the provided schema. Do not wrap inside an object. One JSON array only.

For each event, capture:
- date: As shown by the day header or section (e.g., 'Monday, Dec 1, 2025'). If not explicitly shown, leave null.
- time: Start time (e.g., '10:00') as displayed near the event.
- end_time: End time when shown (e.g., '11:30'); otherwise null.
- type: The event type string exactly as shown, but remove any city/location prefix such as: {prefixes}.
- title: The event title text.
- url: The hyperlink to the event details.
- speaker: Speaker(s) if displayed nearby; otherwise null.

Special handling:
- Oral sessions: Set type to 'Oral Session' and include a 'papers' array with objects having 'title' and 'url' for each listed paper in the session.
- If event entries include nested items, keep the parent as the main event and list nested items under 'papers'.
- Preserve the order as they appear on the page.

Constraints:
- Output must be a JSON array of objects matching the schema. Do not include extra fields.
- Do not invent missing data; use nulls instead when a field is not shown.
- Preserve relative URLs as-is (we will normalize later).

Focus on these types (keep others too if clearly marked, but these are most important): {included}.
        """

    @staticmethod
    def _make_absolute_url(url: Optional[str], base_url: str) -> Optional[str]:
        if not url:
            return url
        return WebExtractor.make_absolute_url(url, base_url)

    def _normalize_type(self, event_type: Optional[str]) -> Optional[str]:
        if not event_type:
            return event_type
        et = event_type.strip()
        for prefix in self.location_prefixes:
            if et.startswith(prefix):
                et = et.replace(prefix, "").strip()
                break
        return et

    def filter_events(self, events: List[Dict]) -> List[Dict]:
        filtered = []
        for e in events:
            et = self._normalize_type(e.get("type")) or ""
            if et in self.included_types:
                e["type"] = et
                filtered.append(e)
        return filtered

    @staticmethod
    def flatten_oral_sessions(events: List[Dict]) -> List[Dict]:
        flattened = []
        for event in events:
            if event.get("papers"):
                session_event = {k: v for k, v in event.items() if k != "papers"}
                flattened.append(session_event)
                for p in event["papers"]:
                    paper_event = session_event.copy()
                    paper_event["title"] = f"  → {p.get('title','')}"
                    paper_event["url"] = p.get("url", "")
                    paper_event["type"] = "Oral Paper"
                    flattened.append(paper_event)
            else:
                flattened.append(event)
        return flattened

    async def extract(self, url: str, base_url: str = "") -> Dict:
        """Run LLM-powered extraction and post-processing."""
        instruction = self._instruction()

        data, strategy = await self.llm.extract(
            url=url,
            schema=NeurIPSEvent,
            instruction=instruction,
            apply_chunking=True,
            chunk_token_threshold=5000,
            overlap_rate=0.15,
            input_format="markdown",
            temperature=0.0,
            max_tokens=15000,
        )

        # data can be a single item or list; normalize to list of dicts
        events: List[Dict] = []
        if data:
            if isinstance(data, list):
                events = [item.model_dump() for item in data]
            else:
                events = [data.model_dump()]

        # Normalize URLs and types
        for e in events:
            if e.get("url"):
                e["url"] = self._make_absolute_url(e["url"], base_url)
            if e.get("papers"):
                for p in e["papers"]:
                    if p.get("url"):
                        p["url"] = self._make_absolute_url(p["url"], base_url)

        return {"total_events": len(events), "events": events}


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

    # Initialize LLM extractor
    extractor = NeurIPSScheduleExtractor(verbose=True)

    # Extract events via LLM
    result = await extractor.extract(url, base_url=base_url)

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
