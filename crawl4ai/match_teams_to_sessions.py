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
RESEARCH_INTEREST_FILE = "research_interest_v2.md"
# SESSIONS_CSV_FILE = "neurips_2025_sessions_Sandiego_detail.csv"
# OUTPUT_CSV_FILE = "neurips_2025_sessions_Sandiego_match_research_interest_v2.csv"
SESSIONS_CSV_FILE = "neurips_2025_sessions_MexicoCity_detail.csv"
OUTPUT_CSV_FILE = "neurips_2025_sessions_MexicoCity_match_research_interest_v2.csv"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
SIMILARITY_THRESHOLD = 0.5  # Cosine similarity threshold for a match

# --- Pydantic Models ---
class Recommendation(BaseModel):
    """Final recommendation containing reason and focus areas"""
    reason: str
    focus_areas: List[str]

# --- Huawei BU background information (from web research) ---
# BU_CONTEXT_INFO = {
#     "存储": "华为云三大核心业务之一（通算、智算、存储），负责数据中心存储系统、云存储解决方案的研发和优化，包括AI训练推理中的数据访问、存储架构创新等。",
#     "CBG": "Consumer Business Group（消费者业务部门），负责华为智能手机、平板电脑、可穿戴设备、智慧屏等终端产品的研发、生产和销售，致力于全场景智慧生活体验。",
#     "DCN": "Data Communication Network（数据通信网络部门），负责数据中心网络架构设计与优化，包括Spine-Leaf架构、VXLAN、SDN、数据中心互联、网络安全管控等技术的研发和部署。",
#     "海思": "华为集成电路设计公司，中国最大的无晶圆厂半导体设计公司，主要产品包括麒麟系列移动处理器、AI芯片等，覆盖无线通信、智能视觉、智能媒体等领域的芯片设计。",
#     "计算": "负责华为昇腾（Ascend）AI芯片和Atlas AI计算解决方案的研发，专注AI计算基础设施、高性能计算架构、AI训练推理加速等核心技术创新。",
#     "温哥华云": "Huawei Cloud Vancouver研究团队，专注大语言模型（LLMs）的成本优化、微调推理技术、负责任AI（数据/模型水印、联邦学习）以及LLMs在运筹学、分析数据库等领域的实际应用。",
#     "多伦多云": "Huawei Cloud分布式调度和数据引擎实验室，专注AI Agent技术研究，包括多智能体系统（Multi-Agent）、Agent编排（Agentic Orchestration）、Agent安全性以及GenAI云服务技术创新。",
#     "诺亚": "华为诺亚方舟实验室，从事人工智能基础研究，主要方向包括大模型自演进、强化学习（RLHF）、LLM-based agent、深度强化学习、多智能体系统以及决策推理等前沿AI技术研究。",
# }

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
    - 从Session内容中，提取出1-2个该团队应重点关注的具体技术点、算法或讨论议题。
    - 这应该是具体的、可操作的关注点列表。

请以JSON格式返回你的分析结果。
"""
    try:
        completion = client.chat.completions.parse(
            model="gpt-4.1-mini",
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

def load_structured_cache(cache_file: str) -> Dict:
    """Load the structured challenge points cache"""
    import json
    with open(cache_file, 'r', encoding='utf-8') as f:
        return json.load(f)

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
    print("Session-Team Matching System (BU Summary Based)")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1] Loading data and embedding model...")
    TRANSLATION_CACHE_FILE = "team_translations_cache.json"

    # Load structured cache with challenge points
    if not os.path.exists(TRANSLATION_CACHE_FILE):
        print(f"Error: {TRANSLATION_CACHE_FILE} not found. Please run generate_structured_cache.py first.")
        return

    structured_cache = load_structured_cache(TRANSLATION_CACHE_FILE)
    sessions_df = pd.read_csv(SESSIONS_CSV_FILE)

    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        print("Please make sure 'sentence-transformers' is installed (`pip install sentence-transformers`)")
        return

    # 2. Prepare BU challenge summaries for embedding
    print("\n[Step 2] Preparing BU challenge summaries...")
    bu_summaries_list = []  # List of (bu_name, challenge_sumary)

    for bu_name, bu_data in structured_cache.items():
        summary = bu_data.get('challenge_sumary')
        if not summary:
            # Backward compatibility: derive summary from existing challenge_points if present
            points = bu_data.get('challenge_points', [])
            if isinstance(points, list) and points:
                summary = "; ".join([p.get('principle_en', '') for p in points if p.get('principle_en')])
            else:
                summary = ""
        bu_summaries_list.append({
            'bu_name': bu_name,
            'challenge_sumary': summary
        })

    print(f"Found {len(structured_cache)} BUs with summaries")
    print(f"Found {len(sessions_df)} sessions")
    print(f"Using embedding model: {EMBEDDING_MODEL_NAME}")

    # 3. Generate Embeddings for BU summaries
    print("\n[Step 3] Generating embeddings for BU summaries...")
    # Create text for each BU summary
    summary_texts = [entry['challenge_sumary'] for entry in bu_summaries_list]

    # Session texts
    session_texts = [
        f"{row.get('title', '')}. {row.get('abstract', '')}. {row.get('overview', '')}"
        for _, row in sessions_df.iterrows()
    ]

    print("  Encoding BU summaries...")
    bu_embeddings = model.encode(summary_texts, show_progress_bar=True)
    print("  Encoding sessions...")
    session_embeddings = model.encode(session_texts, show_progress_bar=True)
    print("Embeddings generated successfully.")

    # 4. Calculate Cosine Similarity
    print("\n[Step 4] Calculating cosine similarity...")
    similarity_matrix = cosine_similarity(bu_embeddings, session_embeddings)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"  (BUs x sessions) = ({len(bu_summaries_list)} x {len(sessions_df)})")

    # 5. Find matches above threshold
    print(f"\n[Step 5] Finding matches with similarity > {args.threshold}...")
    potential_matches_indices = np.where(similarity_matrix >= args.threshold)

    match_candidates = []
    for bu_idx, session_idx in zip(*potential_matches_indices):
        entry = bu_summaries_list[bu_idx]
        match_candidates.append({
            "bu_name": entry['bu_name'],
            "challenge_sumary": entry['challenge_sumary'],
            "session_idx": session_idx,
            "similarity": similarity_matrix[bu_idx, session_idx]
        })

    print(f"Found {len(match_candidates)} potential matches (BU summary <-> session).")

    # 6. Generate recommendations for matches (one similarity per BU-session)
    print("\n[Step 6] Generating recommendations for matches...")
    final_results = []

    for i, candidate in enumerate(match_candidates, 1):
        bu_name = candidate['bu_name']
        session_idx = candidate['session_idx']
        similarity = candidate['similarity']
        entry_summary = candidate.get('challenge_sumary', '')
        session = sessions_df.iloc[session_idx].to_dict()

        print(f"  [{i}/{len(match_candidates)}] {bu_name} <-> {session['title'][:40]}... (sim: {similarity:.2f})")

        recommendation_prompt = f"""你是一个研究兴趣匹配分析专家。一个Session和一个研究团队已经通过向量相似度（Cosine Similarity: {similarity:.2f}）被初步匹配。
你的任务是基于该团队的英文挑战总结进行分析，并提供推荐理由和重点关注方向。

Session Information:
- Title: {session.get('title', 'N/A')}
- Type: {session.get('type', 'N/A')}
- Abstract: {session.get('abstract', 'N/A')}
- Overview: {session.get('overview', 'N/A')}

BU: {bu_name}
BU Challenge Summary (English):
{entry_summary}

---
**分析任务**:

1.  **撰写推荐理由 (reason)**:
    - 基于Session内容与该BU挑战总结，用50-80字解释为什么这个团队应该参加这个Session。
    - 理由需要精炼、自然，并明确指出技术连接点。

2.  **提取重点关注方向 (focus_areas)**:
    - 从Session内容中，提取出3-5个该团队应重点关注的具体技术点、算法或讨论议题。
    - 这应该是具体的、可操作的关注点列表。

请以JSON格式返回你的分析结果。
"""

        try:
            completion = client.chat.completions.parse(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "你是一个专业的技术匹配分析专家。"},
                    {"role": "user", "content": recommendation_prompt}
                ],
                response_format=Recommendation,
            )
            recommendation = completion.choices[0].message.parsed.model_dump()
        except Exception as e:
            print(f"    Error generating recommendation: {e}")
            recommendation = {"reason": "Error", "focus_areas": []}

        final_results.append({
            'bu_name': bu_name,
            'session_title': session['title'],
            'session_type': session.get('type'),
            'session_date': session.get('date'),
            'session_time': session.get('time'),
            'cosine_similarity': similarity,
            'recommendation_reason': recommendation['reason'],
            'focus_areas': "; ".join(recommendation['focus_areas'])
        })

        time.sleep(0.3)  # Rate limiting

    # 8. Save final results
    if not final_results:
        print("\nNo final recommendations to save.")
        return

    output_df = pd.DataFrame(final_results)
    output_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 80)
    print("Matching Complete!")
    print(f"Total BU-Session pairs: {len(final_results)}")
    print(f"Total BU-summary matches: {len(match_candidates)}")
    print(f"Output saved to: {OUTPUT_CSV_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()
