#!/usr/bin/env python3
"""
Session-Team Matching Script using OpenAI GPT-4
Matches NeurIPS 2025 sessions to research teams based on their interests
"""

import os
import json
import pandas as pd
from openai import OpenAI
from typing import List, Dict, Tuple
import time
from dotenv import load_dotenv
load_dotenv()

# Configuration
RESEARCH_INTEREST_FILE = "research_interest.md"
SESSIONS_CSV_FILE = "neurips_2025_sessions_SanDiego_detail.csv"
OUTPUT_CSV_FILE = "neurips_2025_sessions_SanDiego_matched.csv"

def parse_research_interests(file_path: str) -> List[Dict[str, str]]:
    """
    Parse the research interest markdown file and extract team profiles

    Returns:
        List of dicts with keys: 'bu', 'focus', 'challenges'
    """
    teams = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Skip header lines and parse the table
    data_started = False
    for line in lines:
        line = line.strip()

        # Skip until we find the table separator
        if '|:----' in line or '|----' in line:
            data_started = True
            continue

        if not data_started or not line.startswith('|'):
            continue

        # Parse table row
        parts = [p.strip() for p in line.split('|')[1:-1]]  # Remove empty first and last

        if len(parts) >= 3:
            bu = parts[0]
            focus = parts[1]
            challenges = parts[2]

            if bu and focus:  # Make sure it's not empty
                teams.append({
                    'bu': bu,
                    'focus': focus,
                    'challenges': challenges
                })

    return teams


def create_matching_prompt(session: Dict, teams: List[Dict[str, str]]) -> str:
    """
    Create a prompt for GPT-4 to match teams to a session

    Args:
        session: Dict containing session information
        teams: List of team profiles

    Returns:
        Formatted prompt string
    """
    # Build team profiles section
    team_profiles = []
    for i, team in enumerate(teams, 1):
        profile = f"{i}. BU: {team['bu']}\n"
        profile += f"   关注方向: {team['focus']}\n"
        profile += f"   难题: {team['challenges']}"
        team_profiles.append(profile)

    teams_text = "\n\n".join(team_profiles)

    # Build session information
    session_info = f"""
Session Information:
- Title: {session.get('title', 'N/A')}
- Type: {session.get('type', 'N/A')}
- Date & Time: {session.get('date', 'N/A')} {session.get('time', 'N/A')}
- Abstract: {session.get('abstract', 'N/A')}
- Overview: {session.get('overview', 'N/A')}
"""

    prompt = f"""你是一个严格的研究兴趣匹配专家。请分析以下NeurIPS 2025会议session，判断哪些研究团队应该参加这个session。

{session_info}

研究团队信息：
{teams_text}

请仔细分析该session的title、abstract和overview，与各团队的"关注方向"和"难题"进行匹配。

**判断标准（必须同时满足）**：
1. Session内容必须直接涉及团队的关注方向或具体难题
2. Session能为团队的难题提供解决思路、方法或技术
3. 不是泛泛的相关，而是有明确的技术关联点

**重要**：
- 如果session只是AI大领域相关，但与团队具体方向无关 → 不匹配
- 如果session可能"间接有用"但不直接相关 → 不匹配
- 宁可一个都不匹配，也不要勉强匹配
- 大多数session可能都不会有匹配的团队，这是正常的

请以JSON格式返回结果，格式如下：
{{
    "matched_teams": [
        {{
            "bu": "团队BU名称",
            "focus": "该团队的关注方向",
            "reason": "推荐理由（必须具体说明session哪个内容可以解决团队的哪个难题）"
        }}
    ]
}}

如果没有高度相关的团队，返回空数组 {{"matched_teams": []}}。
确保返回的是有效的JSON格式。
"""

    return prompt


def match_session_to_teams(session: Dict, teams: List[Dict[str, str]], client: OpenAI) -> Dict:
    """
    Use OpenAI API to match a session to relevant teams

    Args:
        session: Session information dict
        teams: List of team profiles
        client: OpenAI client instance

    Returns:
        Dict with matched teams information
    """
    prompt = create_matching_prompt(session, teams)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini", 
            messages=[
                {"role": "system", "content": "You are a research interest matching expert. Return only valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent results
            max_tokens=1000
        )

        result_text = response.choices[0].message.content.strip()

        # Try to extract JSON if wrapped in code blocks
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        result = json.loads(result_text)
        return result

    except Exception as e:
        print(f"Error matching session '{session.get('title', 'N/A')}': {str(e)}")
        return {"matched_teams": []}


def format_matched_results(matched_teams: List[Dict]) -> Tuple[str, str, str]:
    """
    Format matched teams into three strings for CSV columns

    Args:
        matched_teams: List of matched team dicts

    Returns:
        Tuple of (team_names, focuses, reasons) formatted with semicolon separators
    """
    if not matched_teams:
        return "", "", ""

    team_names = "; ".join([team['bu'] for team in matched_teams])
    focuses = "; ".join([f"{team['bu']}: {team['focus']}" for team in matched_teams])
    reasons = "; ".join([f"{team['bu']}: {team['reason']}" for team in matched_teams])

    return team_names, focuses, reasons


def main():
    """Main execution function"""
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it using: export OPENAI_API_KEY='your-api-key'")
        return

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    print("=" * 80)
    print("Session-Team Matching System (Incremental Write Mode)")
    print("=" * 80)

    # Step 1: Parse research interests
    print("\n[Step 1] Parsing research interests...")
    teams = parse_research_interests(RESEARCH_INTEREST_FILE)
    print(f"Found {len(teams)} teams:")
    for team in teams:
        print(f"  - {team['bu']}: {team['focus']}")

    # Step 2: Load sessions CSV
    print(f"\n[Step 2] Loading sessions from {SESSIONS_CSV_FILE}...")
    sessions_df = pd.read_csv(SESSIONS_CSV_FILE)
    print(f"Found {len(sessions_df)} sessions")

    # Step 3: Check if output file exists (for resume capability)
    start_idx = 0

    if os.path.exists(OUTPUT_CSV_FILE):
        print(f"\n[Notice] Output file {OUTPUT_CSV_FILE} already exists.")
        existing_df = pd.read_csv(OUTPUT_CSV_FILE)
        start_idx = len(existing_df)
        if start_idx >= len(sessions_df):
            print(f"[Notice] All sessions already processed. Delete {OUTPUT_CSV_FILE} to restart.")
            return
        print(f"[Notice] Resuming from session {start_idx + 1}/{len(sessions_df)}")
    else:
        # Initialize output CSV with header
        output_columns = list(sessions_df.columns) + ['匹配团队', '关注方向', '推荐理由']
        pd.DataFrame(columns=output_columns).to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"[Step 3] Created output file: {OUTPUT_CSV_FILE}")

    # Step 4: Match each session to teams and write incrementally
    print(f"\n[Step 4] Matching sessions to teams (writing incrementally)...")
    total_matches = 0

    for idx in range(start_idx, len(sessions_df)):
        row = sessions_df.iloc[idx]
        session = row.to_dict()
        session_title = session.get('title', 'N/A')[:60]

        print(f"  [{idx+1}/{len(sessions_df)}] Processing: {session_title}...")

        # Call OpenAI API
        match_result = match_session_to_teams(session, teams, client)
        matched_teams = match_result.get('matched_teams', [])

        # Format results
        team_names, focuses, reasons = format_matched_results(matched_teams)

        # Create output row
        output_row = row.to_dict()
        output_row['匹配团队'] = team_names
        output_row['关注方向'] = focuses
        output_row['推荐理由'] = reasons

        # Append to CSV immediately
        pd.DataFrame([output_row]).to_csv(
            OUTPUT_CSV_FILE,
            mode='a',
            header=False,
            index=False,
            encoding='utf-8-sig'
        )

        if matched_teams:
            print(f"      ✓ Matched {len(matched_teams)} team(s): {team_names}")
            print(f"      💾 Saved to CSV")
            total_matches += 1
        else:
            print(f"      - No matches")
            print(f"      💾 Saved to CSV")

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    # Summary statistics
    print("\n" + "=" * 80)
    print("Matching Complete!")
    print("=" * 80)
    print(f"Total sessions processed: {len(sessions_df)}")
    print(f"Sessions with matches: {total_matches}")
    print(f"Sessions without matches: {len(sessions_df) - total_matches}")
    print(f"\nOutput saved to: {OUTPUT_CSV_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
