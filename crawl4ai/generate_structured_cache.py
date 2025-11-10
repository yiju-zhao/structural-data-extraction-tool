#!/usr/bin/env python3
"""
Generate structured team_translations_cache.json with individual challenge points
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
RESEARCH_INTEREST_FILE = "research_interest.md"
OUTPUT_CACHE_FILE = "team_translations_cache.json"

# --- Pydantic Models ---
class ChallengePoint(BaseModel):
    """Individual challenge point extracted from BU's challenges"""
    id: str
    challenge_zh: str
    challenge_en: str

class BUChallenges(BaseModel):
    """Structured challenges for a BU"""
    bu_name: str
    challenge_points: List[ChallengePoint]

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

def extract_challenge_points(bu_name: str, focus: str, challenges: str, client: OpenAI) -> BUChallenges:
    """
    Extract individual challenge points from BU's challenges text using LLM

    Args:
        bu_name: Name of the BU
        focus: Focus area description (Chinese)
        challenges: Challenges description (Chinese)
        client: OpenAI client

    Returns:
        BUChallenges object with structured challenge points
    """
    prompt = f"""你是一个技术文档分析专家。请从以下BU的研究兴趣中提取出独立的挑战点。

BU名称: {bu_name}
关注方向: {focus}
难题描述: {challenges}

任务：
1. 将"难题描述"中的每个独立问题/挑战拆分成单独的条目
2. 如果是编号列表（如1. 2. 3.），每个编号都是一个独立挑战点
3. 为每个挑战点生成：
   - 唯一ID（格式：bu_name_challenge_序号，例如：存储_challenge_1）
   - 中文原文（challenge_zh）- 保持原文完整
   - 英文翻译（challenge_en）- 保持技术术语的准确性
4. 确保每个挑战点都是完整的、独立的问题描述
5. 保留关注方向的上下文，使每个挑战点在其领域内有意义

请返回结构化的JSON格式。"""

    try:
        completion = client.chat.completions.parse(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "你是一个专业的技术文档分析专家。"},
                {"role": "user", "content": prompt}
            ],
            response_format=BUChallenges,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"Error extracting challenge points for {bu_name}: {e}")
        # Return empty structure on error
        return BUChallenges(bu_name=bu_name, challenge_points=[])

def main():
    """Main execution function"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI(api_key=api_key)

    print("=" * 80)
    print("Structured Challenge Points Extraction")
    print("=" * 80)

    # 1. Load research interests
    print(f"\n[Step 1] Loading research interests from {RESEARCH_INTEREST_FILE}...")
    teams = parse_research_interests(RESEARCH_INTEREST_FILE)
    print(f"Found {len(teams)} BUs")

    # 2. Extract challenge points for each BU
    print("\n[Step 2] Extracting and translating challenge points...")
    structured_cache = {}

    for i, team in enumerate(teams, 1):
        bu_name = team['bu']
        print(f"\n  [{i}/{len(teams)}] Processing {bu_name}...")

        # Extract challenge points using LLM
        bu_challenges = extract_challenge_points(
            bu_name=bu_name,
            focus=team['focus'],
            challenges=team['challenges'],
            client=client
        )

        # Convert to dict format for JSON
        structured_cache[bu_name] = {
            "challenge_points": [
                {
                    "id": point.id,
                    "challenge_zh": point.challenge_zh,
                    "challenge_en": point.challenge_en
                }
                for point in bu_challenges.challenge_points
            ]
        }

        print(f"    Extracted {len(bu_challenges.challenge_points)} challenge points:")
        for point in bu_challenges.challenge_points:
            print(f"      - {point.id}: {point.challenge_zh[:50]}...")

        time.sleep(0.3)  # Rate limiting

    # 3. Save to JSON file
    print(f"\n[Step 3] Saving structured cache to {OUTPUT_CACHE_FILE}...")
    with open(OUTPUT_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(structured_cache, f, ensure_ascii=False, indent=2)

    # Print statistics
    total_points = sum(len(bu['challenge_points']) for bu in structured_cache.values())
    print("\n" + "=" * 80)
    print("Extraction Complete!")
    print(f"Total BUs: {len(structured_cache)}")
    print(f"Total challenge points: {total_points}")
    print(f"Output saved to: {OUTPUT_CACHE_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()
