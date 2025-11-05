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
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl_llm_extractor import LLMExtractor


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


async def extract_abstract_from_url(url: str) -> str:
    """
    Extract abstract from a session URL using CSS selectors.

    Args:
        url: The session URL

    Returns:
        Abstract text or empty string if not found
    """
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url=url, config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
            )

            if not result.success or not result.html:
                return ""

            soup = BeautifulSoup(result.html, "html.parser")

            # Look for abstract in div#abstractExample
            abstract_div = soup.find("div", id="abstractExample")
            if abstract_div:
                # Find the paragraph after the "Abstract:" label
                paragraphs = abstract_div.find_all("p")
                if paragraphs:
                    # Get text from all paragraphs, join with newlines
                    abstract_text = "\n\n".join(
                        p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)
                    )
                    return abstract_text

            return ""

    except Exception as e:
        print(f"  ⚠️  Error extracting abstract from {url}: {e}")
        return ""


async def find_project_website(url: str) -> Optional[str]:
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

    Returns:
        Project website URL or None if not found
    """
    try:
        async with AsyncWebCrawler() as crawler:
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


async def extract_overview_from_website(project_url: str, llm_extractor: LLMExtractor) -> str:
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
    session: Dict, llm_extractor: LLMExtractor, progress: int, total: int
) -> Dict:
    """
    Process a single session to extract abstract and overview.

    Args:
        session: Dictionary containing session data
        llm_extractor: LLMExtractor instance
        progress: Current progress number
        total: Total number of sessions

    Returns:
        Updated session dictionary with abstract and overview
    """
    url = session.get("url", "").strip()

    print(f"\n[{progress}/{total}] Processing: {session.get('title', 'Unknown')[:60]}...")
    print(f"  URL: {url}")

    # Initialize new fields
    session["abstract"] = ""
    session["overview"] = ""

    if not url:
        print("  ⏭️  No URL, skipping")
        return session

    # Step 1: Extract abstract
    print("  📄 Extracting abstract...")
    abstract = await extract_abstract_from_url(url)
    if abstract:
        session["abstract"] = abstract
        print(f"  ✅ Abstract extracted ({len(abstract)} chars)")
    else:
        print("  ⚠️  No abstract found")

    # Step 2: Find and process project website
    print("  🔍 Looking for project website...")
    project_url = await find_project_website(url)

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
):
    """
    Process CSV file and enrich with abstracts and overviews.

    Args:
        input_csv: Input CSV file path
        output_csv: Output CSV file path
        model: OpenAI model to use for LLM extraction
        limit: Optional limit on number of rows to process (for testing)
    """
    # Read input CSV
    print(f"Reading input CSV: {input_csv}")
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        sessions = list(reader)

    total_sessions = len(sessions)
    print(f"Found {total_sessions} sessions")

    if limit:
        sessions = sessions[:limit]
        print(f"Limiting to first {limit} sessions")

    # Initialize LLM extractor
    print(f"\nInitializing LLM extractor (model: {model})...")
    llm_extractor = LLMExtractor(model=model, headless=True, verbose=False)

    # Process each session
    enriched_sessions = []
    for i, session in enumerate(sessions, 1):
        try:
            enriched_session = await process_session(
                session, llm_extractor, i, len(sessions)
            )
            enriched_sessions.append(enriched_session)
        except Exception as e:
            print(f"  ❌ Error processing session: {e}")
            # Add empty fields and continue
            session["abstract"] = ""
            session["overview"] = ""
            enriched_sessions.append(session)

        # Small delay to avoid rate limiting
        await asyncio.sleep(1)

    # Write output CSV
    print(f"\n{'=' * 80}")
    print(f"Writing results to: {output_csv}")

    if enriched_sessions:
        # Get all fieldnames (original + new)
        fieldnames = list(enriched_sessions[0].keys())

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enriched_sessions)

        print(f"✅ Successfully processed {len(enriched_sessions)} sessions")

        # Print statistics
        with_abstract = sum(1 for s in enriched_sessions if s.get("abstract"))
        with_overview = sum(1 for s in enriched_sessions if s.get("overview"))
        print(f"\nStatistics:")
        print(f"  Sessions with abstract: {with_abstract}/{len(enriched_sessions)}")
        print(f"  Sessions with overview: {with_overview}/{len(enriched_sessions)}")
    else:
        print("⚠️  No sessions to write")


def main():
    parser = argparse.ArgumentParser(
        description="Extract abstracts and project overviews from session URLs in a CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire CSV
  python extract_session_info.py events.csv -o enriched_events.csv

  # Process with GPT-4 for better quality
  python extract_session_info.py events.csv -o enriched_events.csv -m gpt-4o

  # Process only first 5 sessions (for testing)
  python extract_session_info.py events.csv -o enriched_events.csv --limit 5

Note: Requires OPENAI_API_KEY environment variable to be set.
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
        default="gpt-5-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of sessions to process (for testing)",
    )

    args = parser.parse_args()

    asyncio.run(
        process_csv(
            args.input_csv,
            args.output,
            model=args.model,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
