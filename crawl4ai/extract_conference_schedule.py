#!/usr/bin/env python3
"""
Extract conference schedule from website using LLM extraction.
"""

import asyncio
import argparse
import csv
import sys
import os
from typing import List
from pydantic import BaseModel, Field
from crawl_llm_extractor import LLMExtractor


class ScheduleEntry(BaseModel):
    """Conference schedule entry model."""

    title: str = Field(default="", description="Session or event title")
    type: str = Field(
        default="",
        description="Session type (e.g., Workshop, Tutorial, Talk, Poster, etc.)",
    )
    date: str = Field(
        default="", description="Date (e.g., 'SUN 30 NOV' or 'MON 1 DEC')"
    )
    start_time: str = Field(
        default="", description="Start time (e.g., '9:30 a.m.' or '1 p.m.')"
    )
    end_time: str = Field(
        default="", description="End time (e.g., '5:00 p.m.' or '8:00 PM')"
    )
    url: str = Field(default="", description="URL link to the session page")

    def to_dict(self):
        """Convert to dictionary for CSV writing."""
        return {
            "title": self.title,
            "type": self.type,
            "date": self.date,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "url": self.url,
        }


async def extract_schedule(
    extractor: LLMExtractor, url: str, verbose: bool = False
) -> tuple[List[ScheduleEntry], object]:
    """
    Extract conference schedule from a URL using LLM extractor.

    Args:
        extractor: LLMExtractor instance
        url: URL of the schedule page
        verbose: Print detailed logs

    Returns:
        Tuple of (list of ScheduleEntry, LLMExtractionStrategy)
    """

    instruction = """
    Extract ALL conference schedule entries from this page. For each session/event, extract:

    - title: The session or event title
    - type: The session type (e.g., Workshop, Tutorial, Talk, Poster, Panel, Keynote, etc.)
    - start_time: The start time (preserve format like "9:30 a.m." or "1 p.m.")
    - end_time: The end time (preserve format like "5:00 p.m." or "8:00 PM")
    - url: The full URL link to the session detail page (if available)

    Return a list of objects, one for each schedule entry. If a field is not found, use empty string.
    Extract ALL entries you can find on the page.
    """

    data, llm_strategy = await extractor.extract(
        url=url,
        schema=ScheduleEntry,
        instruction=instruction,
        apply_chunking=True,
        chunk_token_threshold=6000,
        max_tokens=4000,
    )

    # Ensure we return a list
    if isinstance(data, list):
        schedule_entries = data
    elif isinstance(data, ScheduleEntry):
        schedule_entries = [data]
    else:
        schedule_entries = []

    if verbose:
        print(f"  📋 Extracted {len(schedule_entries)} schedule entries")

    return schedule_entries, llm_strategy


async def main():
    parser = argparse.ArgumentParser(
        description="Extract conference schedule from website using LLM extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract schedule from a conference website
  python extract_conference_schedule.py https://example.com/schedule schedule.csv

  # Use GPT-4 for better extraction quality
  python extract_conference_schedule.py https://example.com/schedule schedule.csv --model gpt-4o

  # Verbose output
  python extract_conference_schedule.py https://example.com/schedule schedule.csv -v
        """,
    )

    parser.add_argument("url", help="URL of the conference schedule page")
    parser.add_argument(
        "output",
        help="Output CSV file (default: schedule.csv)",
        nargs="?",
        default="schedule.csv",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Validate API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return 1

    # Create LLM extractor instance
    try:
        extractor = LLMExtractor(model=args.model, verbose=args.verbose)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    print(f"🌐 Extracting schedule from: {args.url}")
    print(f"🤖 Using LLM extraction ({args.model})")
    print()

    try:
        # Extract schedule
        entries, llm_strategy = await extract_schedule(
            extractor, args.url, verbose=args.verbose
        )

        if not entries:
            print("⚠️  No schedule entries extracted")
            return 1

        # Write to CSV
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["title", "type", "start_time", "end_time", "url"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in entries:
                writer.writerow(entry.to_dict())

        # Print summary
        print()
        print("=" * 60)
        print(f"✅ Successfully extracted {len(entries)} schedule entries")
        print(f"💾 Results saved to: {args.output}")

        # Show sample entries
        if entries and args.verbose:
            print()
            print("📋 Sample entries:")
            for entry in entries[:3]:
                print(f"  • {entry.title}")
                print(f"    Type: {entry.type} | Time: {entry.start_time} - {entry.end_time}")
                if entry.url:
                    print(f"    URL: {entry.url}")

        # Show token usage
        if args.verbose:
            print()
            print("📊 Token Usage:")
            extractor.show_usage(llm_strategy)

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code if exit_code else 0)
