#!/usr/bin/env python3
"""
Session-Team Matching Script using Sentence Embeddings and LLM Analysis
"""

import os
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict
import time
import argparse
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# --- Configuration ---
RESEARCH_INTEREST_FILE = "research_interest.md"
SESSIONS_CSV_FILE = "neurips_2025_sessions_SanDiego_detail.csv"
OUTPUT_CSV_FILE = "neurips_2025_sessions_SanDiego_matched_embedding_v1.csv"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
SIMILARITY_THRESHOLD = 0.4  # Cosine similarity threshold for a match

# --- Pydantic Models ---
class Recommendation(BaseModel):
    """Final recommendation containing reason and focus areas"""
    reason: str
    focus_areas: List[str]

# --- Huawei BU background information (from web research) ---
BU_CONTEXT_INFO = {
    "存储": "华为云三大核心业务之一（通算、智算、存储），负责数据中心存储系统、云存储解决方案的研发和优化，包括AI训练推理中的数据访问、存储架构创新等。",
    "CBG": "Consumer Business Group（消费者业务部门），负责华为智能手机、平板电脑、可穿戴设备、智慧屏等终端产品的研发、生产和销售，致力于全场景智慧生活体验。",
    "DCN": "Data Communication Network（数据通信网络部门），负责数据中心网络架构设计与优化，包括Spine-Leaf架构、VXLAN、SDN、数据中心互联、网络安全管控等技术的研发和部署。",
    "海思": "华为集成电路设计公司，中国最大的无晶圆厂半导体设计公司，主要产品包括麒麟系列移动处理器、AI芯片等，覆盖无线通信、智能视觉、智能媒体等领域的芯片设计。",
    "计算": "负责华为昇腾（Ascend）AI芯片和Atlas AI计算解决方案的研发，专注AI计算基础设施、高性能计算架构、AI训练推理加速等核心技术创新。",
    "温哥华云": "Huawei Cloud Vancouver研究团队，专注大语言模型（LLMs）的成本优化、微调推理技术、负责任AI（数据/模型水印、联邦学习）以及LLMs在运筹学、分析数据库等领域的实际应用。",
    "多伦多云": "Huawei Cloud分布式调度和数据引擎实验室，专注AI Agent技术研究，包括多智能体系统（Multi-Agent）、Agent编排（Agentic Orchestration）、Agent安全性以及GenAI云服务技术创新。",
    "诺亚": "华为诺亚方舟实验室，从事人工智能基础研究，主要方向包括大模型自演进、强化学习（RLHF）、LLM-based agent、深度强化学习、多智能体系统以及决策推理等前沿AI技术研究。",
}

def get_bu_context(bu_name: str) -> str:
    return BU_CONTEXT_INFO.get(bu_name, "该BU暂无背景信息")

def parse_research_interests(file_path: str) -> List[Dict[str, str]]:
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

def translate_to_english(text: str, client: OpenAI, cache: dict = None) -> str:
    """
    Translate Chinese text to English using LLM with caching

    Args:
        text: Chinese text to translate
        client: OpenAI client
        cache: Optional cache dict to avoid repeated translations

    Returns:
        English translation
    """
    # Check cache first
    if cache is not None and text in cache:
        return cache[text]

    prompt = f"""Translate the following Chinese technical description to English.
Preserve all technical terms, acronyms, and proper nouns.
Keep the translation concise and technical.

Chinese text:
{text}

English translation:"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1",  # Use cheapest model for translation
            messages=[
                {"role": "system", "content": "You are a technical translator specializing in AI and computer science."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # Low temperature for consistent translations
        )
        translation = completion.choices[0].message.content.strip()

        # Cache the result
        if cache is not None:
            cache[text] = translation

        return translation
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Fallback to original text

def generate_recommendation(session: Dict, team: Dict[str, str], similarity_score: float, client: OpenAI) -> Dict:
    """
    Generates a recommendation reason and focus areas for a session-team match using an LLM.
    """
    session_info = f"""
Session Information:
- Title: {session.get('title', 'N/A')}
- Type: {session.get('type', 'N/A')}
- Abstract: {session.get('abstract', 'N/A')}
- Overview: {session.get('overview', 'N/A')}
"""
    team_info = f"""
Team Information:
- BU: {team['bu']}
- Focus: {team['focus']}
- Challenges: {team['challenges']}
"""
    prompt = f"""你是一个研究兴趣匹配分析专家。一个Session和一个研究团队已经通过向量相似度（Cosine Similarity: {similarity_score:.2f}）被初步匹配。
你的任务是分析这个匹配，并提供推荐理由和重点关注方向。

{session_info}

{team_info}

---
**分析任务**:

1.  **撰写推荐理由 (reason)**:
    - 基于Session内容和团队的挑战，用50-80字解释为什么这个团队应该参加这个Session。
    - 理由需要精炼、自然，并明确指出技术连接点。

2.  **提取重点关注方向 (focus_areas)**:
    - 从Session内容中，提取出3-5个该团队应重点关注的具体技术点、算法或讨论议题。
    - 这应该是具体的、可操作的关注点列表。

请以JSON格式返回你的分析结果。
"""
    try:
        completion = client.chat.completions.parse(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "你是一个专业的技术匹配分析专家。"},
                {"role": "user", "content": prompt}
            ],
            response_format=Recommendation,
        )
        return completion.choices[0].message.parsed.model_dump()
    except Exception as e:
        print(f"Error generating recommendation for session '{session.get('title', 'N/A')}': {str(e)}")
        return {"reason": "Error", "focus_areas": []}

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Session-Team Matching System using Embeddings')
    parser.add_argument('--threshold', type=float, default=SIMILARITY_THRESHOLD,
                        help=f'Cosine similarity threshold for a match (default: {SIMILARITY_THRESHOLD})')
    args = parser.parse_args()

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI(api_key=api_key)

    print("=" * 80)
    print("Session-Team Matching System (Embeddings + LLM)")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1] Loading data and embedding model...")
    teams = parse_research_interests(RESEARCH_INTEREST_FILE)
    for team in teams:
        team['context'] = get_bu_context(team['bu'])
    sessions_df = pd.read_csv(SESSIONS_CSV_FILE)
    
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        print("Please make sure 'sentence-transformers' is installed (`pip install sentence-transformers`)")
        return
        
    print(f"Found {len(teams)} teams and {len(sessions_df)} sessions.")
    print(f"Using embedding model: {EMBEDDING_MODEL_NAME}")

    # 2. Translate team descriptions to English
    import json
    TRANSLATION_CACHE_FILE = "team_translations_cache.json"

    # Try to load existing translations from cache
    if os.path.exists(TRANSLATION_CACHE_FILE):
        print("\n[Step 2] Loading existing translations from cache...")
        with open(TRANSLATION_CACHE_FILE, 'r', encoding='utf-8') as f:
            cached_translations = json.load(f)

        # Apply cached translations to teams
        for team in teams:
            if team['bu'] in cached_translations:
                team['focus_en'] = cached_translations[team['bu']]['focus_en']
                team['challenges_en'] = cached_translations[team['bu']]['challenges_en']
                print(f"  ✓ Loaded translation for {team['bu']}")
        print("Loaded cached translations.")
    else:
        print("\n[Step 2] Translating team descriptions to English...")
        translation_cache = {}

        for i, team in enumerate(teams, 1):
            print(f"  [{i}/{len(teams)}] Translating {team['bu']}...")

            # Translate focus
            team['focus_en'] = translate_to_english(team['focus'], client, translation_cache)

            # Translate challenges
            team['challenges_en'] = translate_to_english(team['challenges'], client, translation_cache)

            print(f"    Focus: {team['focus_en'][:60]}...")
            print(f"    Challenges: {team['challenges_en'][:60]}...")

            time.sleep(0.3)  # Rate limiting

        print(f"Translated {len(teams)} teams to English.")

        # Save translations to cache file
        with open(TRANSLATION_CACHE_FILE, 'w', encoding='utf-8') as f:
            translations = {
                team['bu']: {
                    'focus_en': team['focus_en'],
                    'challenges_en': team['challenges_en']
                }
                for team in teams
            }
            json.dump(translations, f, ensure_ascii=False, indent=2)
        print(f"Translations saved to {TRANSLATION_CACHE_FILE}")

    # 3. Generate Embeddings (now both in English)
    print("\n[Step 3] Generating embeddings...")
    # Create combined text for teams and sessions - use English versions
    team_texts = [
        f"BU Context: {get_bu_context(t['bu'])}. Focus: {t['focus_en']}. Challenges: {t['challenges_en']}"
        for t in teams
    ]
    session_texts = [
        f"{row.get('title', '')}. {row.get('abstract', '')}. {row.get('overview', '')}"
        for _, row in sessions_df.iterrows()
    ]

    team_embeddings = model.encode(team_texts, show_progress_bar=True)
    session_embeddings = model.encode(session_texts, show_progress_bar=True)
    print("Embeddings generated successfully.")

    # 4. Calculate Cosine Similarity
    print("\n[Step 4] Calculating cosine similarity...")
    similarity_matrix = cosine_similarity(team_embeddings, session_embeddings)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

    # 5. Find matches above threshold
    print(f"\n[Step 5] Finding matches with similarity > {args.threshold}...")
    potential_matches_indices = np.where(similarity_matrix >= args.threshold)

    match_candidates = []
    for team_idx, session_idx in zip(*potential_matches_indices):
        match_candidates.append({
            "team_idx": team_idx,
            "session_idx": session_idx,
            "similarity": similarity_matrix[team_idx, session_idx]
        })

    print(f"Found {len(match_candidates)} potential matches.")

    # 6. Use LLM to generate final recommendations
    print("\n[Step 6] Generating final recommendations with LLM...")
    final_results = []
    for i, candidate in enumerate(match_candidates):
        team_idx = candidate['team_idx']
        session_idx = candidate['session_idx']
        
        team = teams[team_idx]
        session = sessions_df.iloc[session_idx].to_dict()
        similarity = candidate['similarity']

        print(f"  [{i+1}/{len(match_candidates)}] Analyzing match: {team['bu']} <-> {session['title'][:40]}... (Sim: {similarity:.2f})")
        
        recommendation = generate_recommendation(session, team, similarity, client)
        
        final_results.append({
            'bu_name': team['bu'],
            'session_title': session['title'],
            'session_type': session.get('type'),
            'session_date': session.get('date'),
            'session_time': session.get('time'),
            'cosine_similarity': similarity,
            'recommendation_reason': recommendation['reason'],
            'focus_areas': "; ".join(recommendation['focus_areas'])
        })
        time.sleep(0.3)

    # 7. Save final results
    if not final_results:
        print("\nNo final recommendations to save.")
        return
        
    output_df = pd.DataFrame(final_results)
    output_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 80)
    print("Matching Complete!")
    print(f"Total final recommendations: {len(final_results)}")
    print(f"Output saved to: {OUTPUT_CSV_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()