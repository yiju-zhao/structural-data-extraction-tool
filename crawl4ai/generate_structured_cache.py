#!/usr/bin/env python3
"""
Generate structured team_translations_cache.json with a single English challenge summary per BU
"""

import os
import json
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import time

load_dotenv()

# --- Configuration ---
RESEARCH_INTEREST_FILE = "research_interest_v2.md"
OUTPUT_CACHE_FILE = "team_translations_cache.json"

# --- Pydantic Models ---
class BUChallengeSummary(BaseModel):
    """Single English challenge summary for a BU"""
    bu_name: str
    # Per user requirement, keep the field name as 'challenge_sumary'
    # (intentional spelling kept to avoid downstream mismatch)
    challenge_sumary: str

def parse_research_interests(file_path: str) -> List[dict]:
    """Parse research_interest.md and extract BU information"""
    teams_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data_started = False
    for line in lines:
        line = line.strip()
        if '|:----' in line or '|----' in line:
            data_started = True
            continue
        if not data_started or not line.startswith('|'):
            continue

        parts = [p.strip() for p in line.split('|')[1:-1]]
        if len(parts) >= 3:
            bu, focus, challenges = parts[0], parts[1], parts[2]
            if not bu or not focus:
                continue

            if bu in teams_map:
                teams_map[bu]['focus'] += f"; {focus}"
                teams_map[bu]['challenges'] += f"\n{challenges}"
            else:
                teams_map[bu] = {'bu': bu, 'focus': focus, 'challenges': challenges}

    return list(teams_map.values())

def extract_challenge_summary(bu_name: str, focus: str, challenges: str, client: OpenAI) -> BUChallengeSummary:
    """
    Extract a single consolidated English challenge summary for a BU using LLM.

    Args:
        bu_name: Name of the BU
        focus: Focus area description (Chinese)
        challenges: Challenges description (Chinese)
        client: OpenAI client

    Returns:
        BUChallengeSummary object containing a single English summary
    """
    prompt = f"""你是一位顶级的AI科学家和领域专家。请深入分析以下BU的研究兴趣，并将所有难题整合为一段精准的英文技术性总结（单段文本）。

BU名称: {bu_name}
关注方向: {focus}
难题描述: {challenges}

任务与要求：
1) 将“关注方向”和“难题描述”中的关键技术挑战进行抽象与融合，用英文写出一段总结（单段），覆盖核心难题与第一性原理层面的技术问题。
2) 风格：客观、技术性强，避免营销语；便于用于技术匹配和检索。
3) 长度建议：大约 80–150 词（如文本内容较多，可适当超过，但保持紧凑）。
4) 输出为JSON，字段必须严格为：
   - "bu_name": BU名称（原样）
   - "challenge_sumary": 单段英文总结文本

仅输出JSON，不要包含其它解释。"""

    try:
        completion = client.chat.completions.parse(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "你是一个专业的技术文档分析专家以及AI领域专家。"},
                {"role": "user", "content": prompt}
            ],
            response_format=BUChallengeSummary,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"Error extracting challenge summary for {bu_name}: {e}")
        # Return empty structure on error
        return BUChallengeSummary(bu_name=bu_name, challenge_sumary="")

def main():
    """Main execution function"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI(api_key=api_key)

    print("=" * 80)
    print("Structured Challenge Summary Extraction (Single summary per BU)")
    print("=" * 80)

    # 1. Load research interests
    print(f"\n[Step 1] Loading research interests from {RESEARCH_INTEREST_FILE}...")
    teams = parse_research_interests(RESEARCH_INTEREST_FILE)
    print(f"Found {len(teams)} BUs")

    # 2. Extract single challenge summary for each BU
    print("\n[Step 2] Extracting and translating challenge summaries...")
    structured_cache = {}

    for i, team in enumerate(teams, 1):
        bu_name = team['bu']
        print(f"\n  [{i}/{len(teams)}] Processing {bu_name}...")

        # Extract single challenge summary using LLM
        bu_summary = extract_challenge_summary(
            bu_name=bu_name,
            focus=team['focus'],
            challenges=team['challenges'],
            client=client
        )

        # Convert to dict format for JSON
        structured_cache[bu_name] = {
            "challenge_sumary": bu_summary.challenge_sumary
        }

        preview = (bu_summary.challenge_sumary or "")[:120].replace("\n", " ")
        print(f"    Extracted summary (preview): {preview}{'...' if len(bu_summary.challenge_sumary) > 120 else ''}")

        time.sleep(0.3)  # Rate limiting

    # 3. Save to JSON file
    print(f"\n[Step 3] Saving structured cache to {OUTPUT_CACHE_FILE}...")
    with open(OUTPUT_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(structured_cache, f, ensure_ascii=False, indent=2)

    # Print statistics
    non_empty = sum(1 for bu in structured_cache.values() if bu.get('challenge_sumary'))
    print("\n" + "=" * 80)
    print("Extraction Complete!")
    print(f"Total BUs: {len(structured_cache)}")
    print(f"BUs with non-empty summaries: {non_empty}")
    print(f"Output saved to: {OUTPUT_CACHE_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()
