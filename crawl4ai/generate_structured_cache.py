#!/usr/bin/env python3
"""
Generate structured team_translations_cache.json with a single English challenge summary per BU
and a rich_profile for agentic retrieval.
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Dict
from dotenv import load_dotenv
import time

load_dotenv()

# --- Configuration ---
RESEARCH_INTEREST_FILE = "research_interest_v3.md"
OUTPUT_CACHE_FILE = "team_translations_cache_v3.json"
MODEL_NAME = os.getenv("OPENAI_RICH_PROFILE_MODEL", "gpt-5")


# --- Pydantic Models ---
class BURichProfile(BaseModel):
    """Rich profile for a BU to power agentic retrieval and reranking"""

    bu_name: str = Field(..., description="BU name as-is")
    # Per requirement, keep the field name as 'challenge_sumary' (intentional spelling)
    challenge_sumary: str = Field(
        ..., description="Concise English summary paragraph (80-150 words)"
    )
    key_points_en: List[str] = Field(
        ...,
        description="3-7 bullet points capturing core technical challenges in English",
    )
    keywords_en: List[str] = Field(..., description="10-20 concise English keywords")
    candidate_queries_en: List[str] = Field(
        ...,
        description="3-5 short English sub-queries (<= 15 words each) covering method/problem/application views",
    )


def parse_research_interests(file_path: str) -> List[dict]:
    """Parse research_interest.md and extract BU information"""
    teams_map: Dict[str, Dict[str, str]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_started = False
    for line in lines:
        line = line.strip()
        if "|:----" in line or "|----" in line:
            data_started = True
            continue
        if not data_started or not line.startswith("|"):
            continue

        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) >= 3:
            bu, focus, challenges = parts[0], parts[1], parts[2]
            if not bu or not focus:
                continue

            if bu in teams_map:
                teams_map[bu]["focus"] += f"; {focus}"
                teams_map[bu]["challenges"] += f"\n{challenges}"
            else:
                teams_map[bu] = {"bu": bu, "focus": focus, "challenges": challenges}

    return list(teams_map.values())


def extract_rich_profile(
    bu_name: str, focus: str, challenges: str, client: OpenAI
) -> BURichProfile:
    """
    Extract a consolidated English challenge summary and a rich profile for a BU using LLM.

    Returns:
        BURichProfile object containing:
        - challenge_sumary
        - key_points_en
        - keywords_en
        - candidate_queries_en
    """
    prompt = f"""You are a senior AI research lead and technical writing expert. Based on the BU's Chinese "focus" and "challenges",
produce a high-quality English rich profile for retrieval and reranking.

Input:
- BU Name (Chinese allowed): {bu_name}
- Focus (Chinese): {focus}
- Challenges (Chinese): {challenges}

Output requirements (English only):
1) challenge_sumary: A single concise, technical English paragraph (around 80-150 words) that abstracts the core challenges at a first-principles level; avoid marketing terms.
2) key_points_en: 3-7 bullet sentences, each a crisp technical point, not exceeding 25 words.
3) keywords_en: 10-20 short English keywords (single words or short phrases), domain-appropriate, comma-separated in JSON array form.
4) candidate_queries_en: 3-5 short English sub-queries (<= 15 words each) covering different views:
   - method-oriented (e.g., "efficient multi-agent coordination under partial observability")
   - problem-oriented (e.g., "robust long-context memory for tool-using agents")
   - application-oriented (e.g., "evaluation protocols for agentic RAG in production")

Strict JSON schema:
{{
  "bu_name": "<same as input BU name>",
  "challenge_sumary": "<single English paragraph>",
  "key_points_en": ["...", "..."],
  "keywords_en": ["...", "..."],
  "candidate_queries_en": ["...", "..."]
}}

Only output the JSON object. Do not include any other explanations."""

    try:
        completion = client.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise technical content structuring and retrieval expert.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format=BURichProfile,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"Error extracting rich profile for {bu_name}: {e}")
        # Return minimal structure on error
        return BURichProfile(
            bu_name=bu_name,
            challenge_sumary="",
            key_points_en=[],
            keywords_en=[],
            candidate_queries_en=[],
        )


def main():
    """Main execution function"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI(api_key=api_key)

    print("=" * 80)
    print(
        "Structured Challenge Summary + Rich Profile Extraction (Single summary per BU)"
    )
    print("=" * 80)

    # 1. Load research interests
    print(f"\n[Step 1] Loading research interests from {RESEARCH_INTEREST_FILE}...")
    teams = parse_research_interests(RESEARCH_INTEREST_FILE)
    print(f"Found {len(teams)} BUs")

    # 2. Extract rich profile for each BU
    print("\n[Step 2] Extracting challenge summaries and rich profiles...")
    structured_cache = {}

    for i, team in enumerate(teams, 1):
        bu_name = team["bu"]
        src_text = f"{team['focus']}\n{team['challenges']}"
        src_hash = hashlib.md5(src_text.encode("utf-8")).hexdigest()
        print(f"\n  [{i}/{len(teams)}] Processing {bu_name}...")

        # Extract rich profile using LLM
        bu_profile = extract_rich_profile(
            bu_name=bu_name,
            focus=team["focus"],
            challenges=team["challenges"],
            client=client,
        )

        # Convert to dict format for JSON with meta info
        structured_cache[bu_name] = {
            "challenge_sumary": bu_profile.challenge_sumary,
            "key_points_en": bu_profile.key_points_en,
            "keywords_en": bu_profile.keywords_en,
            "candidate_queries_en": bu_profile.candidate_queries_en,
            "meta": {
                "source_md_hash": src_hash,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        }

        preview = (bu_profile.challenge_sumary or "")[:120].replace("\n", " ")
        print(
            f"    Extracted summary (preview): {preview}{'...' if len(bu_profile.challenge_sumary) > 120 else ''}"
        )
        print(
            f"    key_points_en: {len(bu_profile.key_points_en)} | keywords_en: {len(bu_profile.keywords_en)} | candidate_queries_en: {len(bu_profile.candidate_queries_en)}"
        )

        time.sleep(0.2)  # Light rate limiting

    # 3. Save to JSON file
    print(f"\n[Step 3] Saving structured cache to {OUTPUT_CACHE_FILE}...")
    with open(OUTPUT_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(structured_cache, f, ensure_ascii=False, indent=2)

    # Print statistics
    non_empty = sum(1 for bu in structured_cache.values() if bu.get("challenge_sumary"))
    print("\n" + "=" * 80)
    print("Extraction Complete!")
    print(f"Total BUs: {len(structured_cache)}")
    print(f"BUs with non-empty summaries: {non_empty}")
    print(f"Output saved to: {OUTPUT_CACHE_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
