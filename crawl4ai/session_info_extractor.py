#!/usr/bin/env python3
"""
Extract additional session information from URLs in a CSV file.
For each session, extracts:
1. Abstract from the session page
2. Project website URL and generates overview using LLM
"""

import asyncio
import csv
import argparse
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl_llm_extractor import LLMExtractor


def parse_cookie_txt(cookie_file_path: str) -> List[Dict]:
    """
    Parse Netscape format cookie.txt file into Crawl4AI cookie format.

    Args:
        cookie_file_path: Path to cookie.txt file in Netscape format

    Returns:
        List of cookie dictionaries in Crawl4AI format
    """
    cookies = []
    with open(cookie_file_path, "r", encoding="utf-8") as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split("\t")
            if len(parts) >= 7:
                cookies.append(
                    {
                        "name": parts[5],
                        "value": parts[6],
                        "domain": parts[0],
                        "path": parts[2],
                        "expires": float(parts[4]) if parts[4] != "0" else -1,
                        "httpOnly": False,
                        "secure": parts[3] == "TRUE",
                        "sameSite": "None",
                    }
                )
    return cookies


class ProjectOverview(BaseModel):
    """Schema for project website overview extraction."""

    overview: str = Field(
        description="A comprehensive overview of the project, research, or work presented"
    )
    research_interests: List[str] = Field(
        description="List of key research interests, topics, or themes"
    )
    key_findings: Optional[str] = Field(
        default=None, description="Key findings or contributions if mentioned"
    )


async def extract_abstract_from_url(url: str, crawler: AsyncWebCrawler) -> str:
    """
    Extract abstract from a session URL using CSS selectors.

    Args:
        url: The session URL
        crawler: Shared AsyncWebCrawler instance

    Returns:
        Abstract text or empty string if not found
    """
    try:
        result = await crawler.arun(
            url=url, config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        )

        if not result.success or not result.html:
            return ""

        soup = BeautifulSoup(result.html, "html.parser")

        # Look for abstract in div#abstractExample
        abstract_div = soup.find("div", id="abstractExample")
        if abstract_div:
            # Try to find paragraphs first
            paragraphs = abstract_div.find_all("p")
            if paragraphs:
                # Get text from all paragraphs, join with newlines
                abstract_text = "\n\n".join(
                    p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
                )
            else:
                # No paragraphs found, extract text directly from div
                abstract_text = abstract_div.get_text(separator=" ", strip=True)

                # Remove "Abstract:" label if present at the beginning
                if abstract_text.startswith("Abstract:"):
                    abstract_text = abstract_text[len("Abstract:") :].strip()

            return abstract_text if abstract_text else ""

        return ""

    except Exception as e:
        print(f"  ⚠️  Error extracting abstract from {url}: {e}")
        return ""


async def find_project_website(url: str, crawler: AsyncWebCrawler) -> Optional[str]:
    """
    Find project website URL from a session page.

    Looks for two specific patterns:
    1. <a class="card-link" ... href="...">Workshop Website</a> (with "website" in text)
    2. <a href="...">Project Page</a> (with "project page" in text)

    Logic:
    - Find both if they exist
    - If both found and same URL: return that URL
    - If both found and different URLs: return the first one (card-link)
    - If only one found: return that one
    - If none found: return None

    Args:
        url: The session URL
        crawler: Shared AsyncWebCrawler instance

    Returns:
        Project website URL or None if not found
    """
    try:
        result = await crawler.arun(
            url=url, config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        )

        if not result.success or not result.html:
            return None

        soup = BeautifulSoup(result.html, "html.parser")

        card_link_url = None
        project_page_url = None

        # Pattern 1: Look for <a class="card-link"> with "website" in text
        card_links = soup.find_all("a", class_="card-link", href=True)
        for link in card_links:
            link_text = link.get_text(strip=True).lower()
            if "website" in link_text:
                href = link.get("href", "")
                if href.startswith("http") and "neurips.cc" not in href:
                    card_link_url = href
                    break

        # Pattern 2: Look for links with "Project Page" text
        all_links = soup.find_all("a", href=True)
        for link in all_links:
            link_text = link.get_text(strip=True).lower()
            if "project page" in link_text:
                href = link.get("href", "")
                if href.startswith("http") and "neurips.cc" not in href:
                    project_page_url = href
                    break

        # Apply logic: both found, only one found, or none found
        if card_link_url and project_page_url:
            # Both found: if same return it, if different return first (card-link)
            return card_link_url
        elif card_link_url:
            # Only card-link found
            return card_link_url
        elif project_page_url:
            # Only project page found
            return project_page_url
        else:
            # None found
            return None

    except Exception as e:
        print(f"  ⚠️  Error finding project website from {url}: {e}")
        return None


async def extract_overview_from_website(
    project_url: str, llm_extractor: LLMExtractor
) -> str:
    """
    Extract overview from project website using LLM.

    Args:
        project_url: The project website URL
        llm_extractor: LLMExtractor instance

    Returns:
        Formatted overview text
    """
    try:
        instruction = """
        Analyze this project/research webpage and extract:
        1. A comprehensive overview of the work, project, or research presented
        2. Key research interests, topics, or themes
        3. Any key findings or contributions mentioned

        Be concise but thorough. Focus on the main ideas and contributions.
        """

        data, strategy = await llm_extractor.extract(
            url=project_url,
            schema=ProjectOverview,
            instruction=instruction,
            temperature=0.0,
            max_tokens=1000,
        )

        if data:
            # If it's a list, take the first item
            if isinstance(data, list):
                data = data[0] if data else None

            if data:
                # Format the overview
                overview_parts = [f"Overview: {data.overview}"]

                if data.research_interests:
                    interests = ", ".join(data.research_interests)
                    overview_parts.append(f"Research Interests: {interests}")

                if data.key_findings:
                    overview_parts.append(f"Key Findings: {data.key_findings}")

                return " | ".join(overview_parts)

        return ""

    except Exception as e:
        print(f"  ⚠️  Error extracting overview from {project_url}: {e}")
        return ""


async def process_session(
    session: Dict,
    crawler: AsyncWebCrawler,
    llm_extractor: LLMExtractor,
    progress: int,
    total: int,
    skip_existing: bool = False,
) -> Dict:
    """
    Process a single session to extract abstract and overview.

    Args:
        session: Dictionary containing session data
        crawler: Shared AsyncWebCrawler instance
        llm_extractor: LLMExtractor instance
        progress: Current progress number
        total: Total number of sessions
        skip_existing: Skip extraction if data already exists

    Returns:
        Updated session dictionary with abstract and overview
    """
    url = session.get("url", "").strip()

    print(
        f"\n[{progress}/{total}] Processing: {session.get('title', 'Unknown')[:60]}..."
    )
    print(f"  URL: {url}")

    # Check existing data
    existing_abstract = session.get("abstract", "").strip()
    existing_overview = session.get("overview", "").strip()

    # Initialize fields if they don't exist
    if "abstract" not in session:
        session["abstract"] = ""
    if "overview" not in session:
        session["overview"] = ""

    if not url:
        print("  ⏭️  No URL, skipping")
        return session

    # Step 1: Extract abstract
    if skip_existing and existing_abstract:
        print(
            f"  ⏭️  Abstract already exists ({len(existing_abstract)} chars), skipping extraction"
        )
    else:
        print("  📄 Extracting abstract...")
        abstract = await extract_abstract_from_url(url, crawler)
        if abstract:
            session["abstract"] = abstract
            print(f"  ✅ Abstract extracted ({len(abstract)} chars)")
        else:
            print("  ⚠️  No abstract found")

    # Step 2: Find and process project website
    if skip_existing and existing_overview:
        print(
            f"  ⏭️  Overview already exists ({len(existing_overview)} chars), skipping extraction"
        )
    else:
        print("  🔍 Looking for project website...")
        project_url = await find_project_website(url, crawler)

        if project_url:
            print(f"  🌐 Found project website: {project_url}")
            print("  🤖 Generating overview using LLM...")
            overview = await extract_overview_from_website(project_url, llm_extractor)
            if overview:
                session["overview"] = overview
                print(f"  ✅ Overview generated ({len(overview)} chars)")
            else:
                print("  ⚠️  Failed to generate overview")
        else:
            print("  ⚠️  No project website found")

    return session


async def process_csv(
    input_csv: str,
    output_csv: str,
    model: str = "gpt-4o-mini",
    limit: Optional[int] = None,
    start_from: int = 0,
    cookie_file: Optional[str] = None,
    type_filter: Optional[str] = None,
    skip_existing: bool = False,
):
    """
    Process CSV file and enrich with abstracts and overviews.

    Args:
        input_csv: Input CSV file path
        output_csv: Output CSV file path
        model: OpenAI model to use for LLM extraction
        limit: Optional limit on number of rows to process (for testing)
        start_from: Row index to start processing from (0-indexed, for resuming)
        cookie_file: Optional path to cookie.txt file for authenticated sessions
        type_filter: Optional session type filter (e.g., 'Oral', 'Workshop')
        skip_existing: Skip extraction if abstract/overview already exists
    """
    # Read input CSV
    print(f"Reading input CSV: {input_csv}")
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        sessions = list(reader)

    total_sessions = len(sessions)
    print(f"Found {total_sessions} sessions")

    # Apply type filter if specified
    if type_filter:
        sessions = [s for s in sessions if s.get("type") == type_filter]
        print(f"Filtering to {len(sessions)} sessions of type '{type_filter}'")

    # Apply start_from slicing
    if start_from > 0:
        if start_from >= total_sessions:
            print(
                f"⚠️  start_from ({start_from}) >= total sessions ({total_sessions}), nothing to process"
            )
            return
        sessions = sessions[start_from:]
        print(f"Starting from row {start_from} ({len(sessions)} sessions remaining)")

    # Apply limit if specified
    if limit:
        sessions = sessions[:limit]
        print(f"Limiting to first {limit} sessions")

    # Initialize LLM extractor
    print(f"\nInitializing LLM extractor (model: {model})...")
    llm_extractor = LLMExtractor(model=model, headless=True, verbose=False)

    # Initialize web crawler (single instance for all sessions)
    print("Initializing web crawler...")
    if cookie_file:
        print(f"Loading cookies from: {cookie_file}")
        cookies = parse_cookie_txt(cookie_file)
        print(f"Loaded {len(cookies)} cookies")
        browser_config = BrowserConfig(cookies=cookies)
        crawler = AsyncWebCrawler(config=browser_config)
    else:
        crawler = AsyncWebCrawler()
    await crawler.__aenter__()

    # Open output CSV file for writing
    # If start_from == 0, create new file and write header
    # If start_from > 0, append to existing file
    file_mode = "w" if start_from == 0 else "a"
    print(f"\nOpening output CSV: {output_csv} (mode: {file_mode})")

    output_file = open(output_csv, file_mode, newline="", encoding="utf-8")

    # We'll initialize the writer after processing the first session
    # so we know all the fieldnames
    writer = None
    fieldnames = None

    # Statistics tracking
    processed_count = 0
    with_abstract_count = 0
    with_overview_count = 0

    try:
        # Process each session
        for i, session in enumerate(sessions, 1):
            # Actual row number in original CSV
            actual_row = start_from + i

            try:
                enriched_session = await process_session(
                    session,
                    crawler,
                    llm_extractor,
                    actual_row,
                    total_sessions,
                    skip_existing,
                )

                # Initialize writer on first session
                if writer is None:
                    fieldnames = list(enriched_session.keys())
                    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                    # Write header only if creating new file
                    if start_from == 0:
                        writer.writeheader()

                # Write this session immediately
                writer.writerow(enriched_session)
                output_file.flush()  # Ensure data is written to disk

                # Update statistics
                processed_count += 1
                if enriched_session.get("abstract"):
                    with_abstract_count += 1
                if enriched_session.get("overview"):
                    with_overview_count += 1

            except Exception as e:
                print(f"  ❌ Error processing session: {e}")
                # Add empty fields and write anyway
                session["abstract"] = ""
                session["overview"] = ""

                # Initialize writer if needed
                if writer is None:
                    fieldnames = list(session.keys())
                    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                    if start_from == 0:
                        writer.writeheader()

                writer.writerow(session)
                output_file.flush()
                processed_count += 1

            # Small delay to avoid rate limiting
            await asyncio.sleep(1)

    finally:
        # Close the output file
        output_file.close()
        # Close the web crawler
        await crawler.__aexit__(None, None, None)

    # Print final statistics
    print(f"\n{'=' * 80}")
    print(f"✅ Successfully processed {processed_count} sessions")
    print(f"\nStatistics:")
    print(f"  Sessions with abstract: {with_abstract_count}/{processed_count}")
    print(f"  Sessions with overview: {with_overview_count}/{processed_count}")
    print(f"\nResults written to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract abstracts and project overviews from session URLs in a CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire CSV
  python session_info_extractor.py events.csv -o enriched_events.csv

  # Process with GPT-4 for better quality
  python session_info_extractor.py events.csv -o enriched_events.csv -m gpt-4o

  # Process only first 5 sessions (for testing)
  python session_info_extractor.py events.csv -o enriched_events.csv --limit 5

  # Process only Oral sessions with authentication
  python session_info_extractor.py events.csv -o enriched_events.csv --cookie-file cookies.txt --type Oral

  # Process only Workshop sessions (no authentication needed)
  python session_info_extractor.py events.csv -o enriched_events.csv --type Workshop

  # Resume from row 50 (0-indexed)
  python session_info_extractor.py events.csv -o enriched_events.csv --start-from 50

  # Skip extraction if abstract/overview already exists in input CSV
  python session_info_extractor.py events.csv -o enriched_events.csv --skip-existing

  # Re-run on same CSV, only filling in missing data
  python session_info_extractor.py enriched_events.csv -o enriched_events.csv --skip-existing

  # Resume from row 50 and process 10 more Oral sessions with cookies, skip existing
  python session_info_extractor.py events.csv -o enriched_events.csv --cookie-file cookies.txt --type Oral --start-from 50 --limit 10 --skip-existing

Note: Requires OPENAI_API_KEY environment variable to be set.
Data is saved line-by-line, so you can resume from any row if interrupted.
Cookie file should be in Netscape format (export from browser extension).
Use --skip-existing to avoid re-extracting data that already exists in the input CSV.
        """,
    )

    parser.add_argument("input_csv", type=str, help="Input CSV file with session URLs")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output CSV file path",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of sessions to process (for testing)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start processing from this row index (0-indexed, for resuming)",
    )
    parser.add_argument(
        "--cookie-file",
        type=str,
        default=None,
        help="Path to cookie.txt file (Netscape format) for authenticated sessions",
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        help="Filter by session type (e.g., 'Oral', 'Workshop')",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip extraction if abstract/overview already exists in the input CSV",
    )

    args = parser.parse_args()

    asyncio.run(
        process_csv(
            args.input_csv,
            args.output,
            model=args.model,
            limit=args.limit,
            start_from=args.start_from,
            cookie_file=args.cookie_file,
            type_filter=args.type,
            skip_existing=args.skip_existing,
        )
    )


if __name__ == "__main__":
    main()
