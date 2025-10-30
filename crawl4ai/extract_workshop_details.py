import asyncio
import argparse
import csv
import sys
import os
from pydantic import BaseModel, Field
from crawl_llm_extractor import LLMExtractor


class SessionInfo(BaseModel):
    """Session information model."""

    title: str = Field(default="", description="Session title")
    authors: str = Field(default="", description="Authors/Organizers (comma separated)")
    date: str = Field(default="", description="Session date (e.g., Sun 7 Dec)")
    time: str = Field(
        default="", description="Session time (e.g., 8 a.m. PST — 5 p.m. PST)"
    )
    location: str = Field(default="", description="Session location/room")
    abstract: str = Field(default="", description="Session abstract or description")
    website: str = Field(default="", description="Official session website URL")

    def to_dict(self):
        """Convert to dictionary for CSV writing."""
        return {
            "title": self.title,
            "authors": self.authors,
            "date": self.date,
            "time": self.time,
            "location": self.location,
            "abstract": self.abstract,
            "website": self.website,
        }


async def extract_session_info(
    extractor: LLMExtractor, url: str, session_type: str = "workshop"
):
    """
    Extract session details using LLM extractor.

    Args:
        extractor: LLMExtractor instance
        url: URL of the session page
        session_type: Type of session (e.g., "workshop", "tutorial", "conference")

    Returns:
        Tuple of (SessionInfo, LLMExtractionStrategy)
    """

    instruction = f"""
    Extract {session_type} information from this page. You must extract these exact fields:

    - title: The {session_type} title (usually in a large heading)
    - authors: The {session_type} organizers/authors (comma separated names)
    - date: The {session_type} date (e.g., "Sun 7 Dec" or "MON 1 DEC")
    - time: The {session_type} time slot (e.g., "8 a.m. PST — 5 p.m. PST")
    - location: The {session_type} location or room (e.g., "Upper Level Room 27AB")
    - abstract: The {session_type} abstract or description (the detailed text about the {session_type})
    - website: The official {session_type} website URL (look for links labeled "Website", "Project Page", or external URLs)

    Return a single object with these fields. If a field is not found, use empty string.
    """

    data, llm_strategy = await extractor.extract(
        url=url,
        schema=SessionInfo,
        instruction=instruction,
        apply_chunking=False,
        max_tokens=2000,
    )

    # Handle extraction result
    if isinstance(data, list):
        # If we got a list, take the first item or return empty
        session_info = data[0] if len(data) > 0 else SessionInfo()
    elif isinstance(data, SessionInfo):
        # Single SessionInfo object
        session_info = data
    else:
        # Empty result
        session_info = SessionInfo()

    return session_info, llm_strategy


async def process_sessions(
    input_csv: str,
    output_csv: str,
    filter_type: str = "Workshop",
    session_type: str = "workshop",
    model: str = "gpt-4o-mini",
    verbose: bool = False,
):
    """
    Process sessions from CSV and extract details using LLM extraction.

    Args:
        input_csv: Input CSV file with session URLs
        output_csv: Output CSV file for results
        filter_type: Filter by type column (e.g., 'Workshop', 'Tutorial')
        session_type: Type of session for instruction (e.g., 'workshop', 'tutorial')
        model: OpenAI model to use
        verbose: Print detailed logs
    """

    # Validate API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return 1

    # Create LLM extractor instance
    try:
        extractor = LLMExtractor(model=model, verbose=verbose)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    # Read input CSV
    sessions = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["type"] == filter_type:
                sessions.append(row)

    print(f"📋 Found {len(sessions)} items with type '{filter_type}'")
    print(f"🤖 Using LLM extraction ({model})")
    print()

    # Track overall statistics
    total_strategies = []
    successful_extractions = 0
    failed_extractions = 0

    # Write output CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "title",
            "authors",
            "date",
            "time",
            "location",
            "abstract",
            "website",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, session in enumerate(sessions, 1):
            print(f"[{idx}/{len(sessions)}] {session['title']}")

            try:
                info, llm_strategy = await extract_session_info(
                    extractor, session["url"], session_type=session_type
                )
                total_strategies.append(llm_strategy)

                # Check if we got meaningful data
                if info.title or info.abstract:
                    successful_extractions += 1
                    writer.writerow(info.to_dict())
                    f.flush()

                    if verbose:
                        print(f"  📝 Title: {info.title[:60]}...")
                        print(f"  👥 Authors: {info.authors[:60]}...")
                        print(f"  🔗 Website: {info.website}")
                else:
                    failed_extractions += 1
                    # Still write empty data
                    writer.writerow(info.to_dict())
                    f.flush()
                    print(f"  ⚠️  No data extracted")

            except Exception as e:
                failed_extractions += 1
                print(f"  ❌ Error: {e}")
                error_info = SessionInfo(
                    title=session["title"],
                    abstract=f"ERROR: {e}",
                    website=session["url"],
                )
                writer.writerow(error_info.to_dict())
                f.flush()

            # Rate limiting
            await asyncio.sleep(2)

    # Print summary
    print()
    print("=" * 60)
    print(f"💾 Results saved to: {output_csv}")
    print(f"✅ Successful extractions: {successful_extractions}")
    print(f"❌ Failed extractions: {failed_extractions}")

    # Show token usage statistics
    if total_strategies and verbose:
        print()
        print("📊 Token Usage Summary:")
        extractor.show_usage(total_strategies[0])

    return 0


async def main():
    parser = argparse.ArgumentParser(
        description="Extract session details using LLM extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_workshop_details.py neurips_2025_schedule.csv workshops.csv
  python extract_workshop_details.py neurips_2025_schedule.csv tutorials.csv -t Tutorial --session-type tutorial -v
  python extract_workshop_details.py neurips_2025_schedule.csv workshops.csv --model gpt-4o -v
        """,
    )

    parser.add_argument("input", help="Input CSV file with session URLs")
    parser.add_argument(
        "output",
        help="Output CSV file (default: workshops.csv)",
        nargs="?",
        default="workshops.csv",
    )
    parser.add_argument(
        "-t", "--type", default="Workshop", help="Filter by type (default: Workshop)"
    )
    parser.add_argument(
        "--session-type",
        default="workshop",
        help="Session type for instruction (default: workshop)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    return await process_sessions(
        args.input, args.output, args.type, args.session_type, args.model, args.verbose
    )


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code if exit_code else 0)
