#!/usr/bin/env python3
"""
Session-Team Matching Script (Agentic-only)
- Agentic RAG 1+3: multi-query recall + LLM multi-criteria reranking
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
import time
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# --- Configuration ---
TRANSLATION_CACHE_FILE = "team_translations_cache_v3.json"
# RESEARCH_INTEREST_FILE = "research_interest.md"
SESSIONS_CSV_FILE = "neurips_2025_sessions_MexicoCity_detail.csv"
OUTPUT_CSV_FILE = "neurips_2025_sessions_MexicoCity_match_research_interest_v3.csv"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
AGENTIC_SCORING_MODEL = os.getenv("OPENAI_AGENTIC_SCORING_MODEL", "gpt-4.1-mini")
# Xinference embedding endpoint (hardcoded as requested)
XINFERENCE_BASE_URL = "http://localhost:9997/v1"
# Batch size for Xinference embedding requests
EMBED_BATCH_SIZE = 64


# --- Pydantic Models ---
class Recommendation(BaseModel):
    """Final recommendation containing reason and focus areas"""

    reason: str
    focus_areas: List[str]


class CriteriaScores(BaseModel):
    """LLM multi-criteria scoring for a BU-session candidate"""

    relevance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Topical relevance between BU summary and session",
    )
    technical_overlap: float = Field(
        ..., ge=0.0, le=1.0, description="Technical method/stack overlap"
    )
    novelty_gain: float = Field(
        ..., ge=0.0, le=1.0, description="Potential novelty or learning gain for BU"
    )
    actionability: float = Field(
        ..., ge=0.0, le=1.0, description="Actionable value to BU (e.g., takeaways)"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Rater confidence")
    reason: str = Field(..., description="Short rationale (<= 50 words)")


# Embedding backend abstraction (SentenceTransformers or Xinference via OpenAI-compatible API)
class EmbeddingBackend:
    def __init__(self, backend: str, model_name: str):
        self.backend = backend
        self.model_name = model_name
        if backend == "st":
            # Lazy import so that Xinference users don't need sentence-transformers installed
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        elif backend == "xinference":
            # Use OpenAI-compatible client pointed to Xinference; api_key cannot be empty string
            import openai
            self.client = openai.Client(api_key="EMPTY", base_url=XINFERENCE_BASE_URL)
        else:
            raise ValueError(f"Unknown embed backend: {backend}")

    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        if self.backend == "st":
            return self.model.encode(texts, show_progress_bar=show_progress_bar)

        # Xinference: call /v1/embeddings with batching
        vectors = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            resp = self.client.embeddings.create(model=self.model_name, input=batch)
            vectors.extend([d.embedding for d in resp.data])
            time.sleep(0.01)  # light rate limit
        return np.asarray(vectors, dtype=np.float32)


def load_structured_cache(cache_file: str) -> Dict:
    with open(cache_file, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_weights(weights_str: str) -> Tuple[float, float, float, float]:
    try:
        parts = [float(x.strip()) for x in weights_str.split(",")]
        if len(parts) != 4:
            raise ValueError("weights must have 4 comma-separated numbers")
        s = sum(parts)
        return tuple([x / s if s > 0 else 0.0 for x in parts])  # normalize
    except Exception:
        # default: 0.45, 0.35, 0.10, 0.10
        return (0.45, 0.35, 0.10, 0.10)


def build_scoring_prompt(
    bu_name: str, challenge_sumary: str, key_points_en: List[str], session: Dict
) -> str:
    key_points_str = "\n".join([f"- {kp}" for kp in key_points_en[:7]])
    session_info = f"""
Session Information:
- Title: {session.get("title", "N/A")}
- Type: {session.get("type", "N/A")}
- Abstract: {session.get("abstract", "N/A")}
- Overview: {session.get("overview", "N/A")}
"""
    prompt = f"""You are a conference track matching judge. Score the match between a BU's challenge profile and a session.

BU: {bu_name}

BU Challenge Summary (English, concise):
{challenge_sumary}

Key Points (English, concise):
{key_points_str}

{session_info}

Scoring criteria (0.0 to 1.0, higher is better):
- relevance: topical/semantic relevance between the BU and the session
- technical_overlap: overlap in methods/architectures/techniques
- novelty_gain: likely novelty/learning benefit for the BU
- actionability: concrete takeaways the BU could apply soon
- confidence: how confident you are in the above (0-1)

Return strict JSON:
{{
  "relevance": 0-1 number,
  "technical_overlap": 0-1 number,
  "novelty_gain": 0-1 number,
  "actionability": 0-1 number,
  "confidence": 0-1 number,
  "reason": "<<= 50 words English rationale>"
}}
Only output the JSON. Do not include any other text."""
    return prompt


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Session-Team Matching System (Agentic RAG 1+3 only)"
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=4,
        help="Number of candidate queries to use per BU (agentic mode)",
    )
    parser.add_argument(
        "--topm",
        type=int,
        default=20,
        help="Top-M sessions per query to recall (agentic mode)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="0.45,0.35,0.10,0.10",
        help="Weights for [relevance, technical_overlap, novelty_gain, actionability]; comma-separated, will be normalized",
    )
    parser.add_argument(
        "--confidence_boost",
        type=float,
        default=0.5,
        help="Confidence boost factor (0~1): final *= (1 - cb) + cb * confidence",
    )
    parser.add_argument(
        "--rec_threshold",
        type=float,
        default=0.55,
        help="Minimum final_score (0-1) to emit recommendations in Agentic mode",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental mode: load existing output CSV and skip already matched (bu_name, session_title) pairs; only generate and append new matches.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=OUTPUT_CSV_FILE,
        help=f"Path to output CSV (default: {OUTPUT_CSV_FILE})",
    )
    parser.add_argument(
        "--sessions_csv",
        type=str,
        default=SESSIONS_CSV_FILE,
        help=f"Path to sessions CSV (default: {SESSIONS_CSV_FILE})",
    )
    parser.add_argument(
        "--score_topk_per_bu",
        type=int,
        default=30,
        help="Agentic only: prefilter candidates by base cosine and keep top-K per BU before LLM scoring. Larger K = higher cost & recall.",
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default=None,
        help=f"SentenceTransformer/Xinference embedding model name or UID (default: {EMBEDDING_MODEL_NAME})",
    )
    parser.add_argument(
        "--embed_backend",
        type=str,
        default="st",
        choices=["st", "xinference"],
        help="Embedding backend: 'st' (sentence-transformers) or 'xinference' (OpenAI-compatible via Xinference)",
    )
    args = parser.parse_args()

    # Determine recommendation threshold:
    # - Agentic: default 0.0 (recommend for all unless specified)
    # - Non-agentic: default equals --threshold
    rec_threshold = args.rec_threshold

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI(api_key=api_key)

    print("=" * 80)
    print("Session-Team Matching System (Agentic RAG 1+3)")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1] Loading data and embedding model...")

    if not os.path.exists(TRANSLATION_CACHE_FILE):
        print(
            f"Error: {TRANSLATION_CACHE_FILE} not found. Please run generate_structured_cache.py first."
        )
        return

    structured_cache = load_structured_cache(TRANSLATION_CACHE_FILE)
    sessions_df = pd.read_csv(args.sessions_csv)

    # Select embedding model (prefix modifiers removed)
    embed_model_name = args.embed_model or EMBEDDING_MODEL_NAME

    try:
        embedder = EmbeddingBackend(
            backend=getattr(args, "embed_backend", "st"),
            model_name=embed_model_name,
        )
    except Exception as e:
        print(f"Error initializing embedding backend: {e}")
        return

    # Prepare session texts and embeddings
    session_texts = []
    for _, row in sessions_df.iterrows():
        raw = f"{row.get('title', '')}. {row.get('abstract', '')}. {row.get('overview', '')}"
        session_texts.append(raw)
    print("  Encoding sessions...")
    session_embeddings = embedder.encode(session_texts, show_progress_bar=True)
    print("Embeddings generated successfully.")
    print(f"Found {len(structured_cache)} BUs")
    print(f"Found {len(sessions_df)} sessions")
    print(f"Using embedding backend: {args.embed_backend}, model: {embed_model_name}")

    # Incremental mode: load existing output pairs to skip
    existing_pairs = set()
    existing_df = None
    output_path = args.output_csv
    if args.incremental and os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            if {"bu_name", "session_title"}.issubset(set(existing_df.columns)):
                existing_pairs = set(
                    zip(
                        existing_df["bu_name"].astype(str).tolist(),
                        existing_df["session_title"].astype(str).tolist(),
                    )
                )
                print(f"Incremental mode: loaded {len(existing_pairs)} existing pairs from {output_path}")
            else:
                print(
                    f"Incremental mode: output CSV lacks required columns 'bu_name' and 'session_title'; proceeding without skipping."
                )
        except Exception as e:
            print(f"Incremental mode: failed to read existing output CSV: {e}. Proceeding without skipping.")

    final_results = []

    # Agentic RAG 1+3: always enabled
    if True:
        # Agentic RAG 1+3: multi-query recall + LLM multi-criteria rerank
        print("\n[Step 2] Agentic multi-query recall...")
        wR, wTO, wN, wA = parse_weights(args.weights)
        cb = max(0.0, min(1.0, args.confidence_boost))

        for i, (bu_name, bu_data) in enumerate(structured_cache.items(), 1):
            summary = bu_data.get("challenge_sumary", "")
            key_points = bu_data.get("key_points_en", [])
            queries = bu_data.get("candidate_queries_en", [])[
                : max(1, args.num_queries)
            ]

            if not queries:
                print(
                    f"  [{i}/{len(structured_cache)}] {bu_name} - ERROR: candidate_queries_en is empty. Re-generate cache with rich_profile."
                )
                continue

            print(f"\n  [{i}/{len(structured_cache)}] BU: {bu_name}")
            print(f"    Using {len(queries)} candidate queries:")
            for q in queries:
                print(f"      - {q}")

            # Encode queries and recall Top-M per query
            candidate_idx_set = set()
            for q in queries:
                q_emb = embedder.encode([q])[0]
                sims = cosine_similarity(
                    q_emb.reshape(1, -1), session_embeddings
                ).ravel()
                top_idx = np.argpartition(sims, -args.topm)[-args.topm :]
                candidate_idx_set.update(top_idx.tolist())

            print(f"    Candidate set size after union: {len(candidate_idx_set)}")

            # Compute base_cosine using BU summary
            if not summary:
                print(
                    "    WARNING: challenge_sumary is empty; base_cosine will be set to 0."
                )
                base_cosine_all = np.zeros(len(sessions_df), dtype=float)
            else:
                summary_emb = embedder.encode([summary])[0]
                base_cosine_all = cosine_similarity(
                    summary_emb.reshape(1, -1), session_embeddings
                ).ravel()

            # Multi-criteria scoring via LLM
            print("\n[Step 3] LLM multi-criteria scoring and rerank...")
            # Optional prefilter by base cosine to save LLM cost
            if args.score_topk_per_bu and summary:
                K = max(1, int(args.score_topk_per_bu))
                cand_with_scores = [(idx, float(base_cosine_all[idx])) for idx in candidate_idx_set]
                cand_with_scores.sort(key=lambda t: t[1], reverse=True)
                selected_idx_list = [idx for idx, _ in cand_with_scores[:K]]
                print(f"    Candidates after prefilter by base cosine: {len(selected_idx_list)} / {len(candidate_idx_set)} (K={K})")
            else:
                selected_idx_list = sorted(candidate_idx_set)
                print(f"    Candidates after prefilter by base cosine: {len(selected_idx_list)} / {len(candidate_idx_set)} (no prefilter)")

            per_bu_candidates = []
            for j, session_idx in enumerate(selected_idx_list, 1):
                session = sessions_df.iloc[session_idx].to_dict()
                base_cos = float(base_cosine_all[session_idx])

                # Skip already existing pairs in incremental mode (avoid LLM scoring)
                if args.incremental:
                    key = (str(bu_name), str(session.get("title")))
                    if key in existing_pairs:
                        continue

                scoring_prompt = build_scoring_prompt(
                    bu_name=bu_name,
                    challenge_sumary=summary,
                    key_points_en=key_points,
                    session=session,
                )

                try:
                    completion = client.chat.completions.parse(
                        model=AGENTIC_SCORING_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": "You provide precise numeric judgments with short rationales.",
                            },
                            {"role": "user", "content": scoring_prompt},
                        ],
                        response_format=CriteriaScores,
                    )
                    scores: CriteriaScores = completion.choices[0].message.parsed
                    linear = (
                        wR * scores.relevance
                        + wTO * scores.technical_overlap
                        + wN * scores.novelty_gain
                        + wA * scores.actionability
                    )
                    final_score = linear * ((1.0 - cb) + cb * scores.confidence)
                except Exception as e:
                    print(
                        f"    Error scoring BU '{bu_name}' with session '{session.get('title', 'N/A')[:40]}...': {e}"
                    )
                    # Hard fallback: base_cosine only
                    scores = CriteriaScores(
                        relevance=0.0,
                        technical_overlap=0.0,
                        novelty_gain=0.0,
                        actionability=0.0,
                        confidence=0.0,
                        reason="fallback: scoring error",
                    )
                    final_score = base_cos

                per_bu_candidates.append(
                    {
                        "bu_name": bu_name,
                        "session_idx": session_idx,
                        "session_title": session.get("title"),
                        "session_type": session.get("type"),
                        "session_date": session.get("date"),
                        "session_time": session.get("time"),
                        "base_cosine": base_cos,
                        "rel": scores.relevance,
                        "tech_overlap": scores.technical_overlap,
                        "novelty": scores.novelty_gain,
                        "actionability": scores.actionability,
                        "confidence": scores.confidence,
                        "final_score": final_score,
                        "scoring_reason": scores.reason,
                    }
                )

                time.sleep(0.15)  # mild rate limiting

            # Sort by final_score
            per_bu_candidates.sort(key=lambda x: x["final_score"], reverse=True)
            print(f"    Top-5 after rerank (final_score | base_cosine):")
            for k, c in enumerate(per_bu_candidates[:5], 1):
                print(
                    f"      {k}. {c['final_score']:.3f} | {c['base_cosine']:.3f} -> {c['session_title'][:60]}"
                )

            # Generate recommendations only for candidates above threshold
            print("\n[Step 4] Generating recommendations...")
            selected_cands = [c for c in per_bu_candidates if c["final_score"] >= rec_threshold]
            print(
                f"    Candidates above threshold ({rec_threshold:.2f}): {len(selected_cands)} / {len(per_bu_candidates)}"
            )
            for cand in selected_cands:
                session = sessions_df.iloc[cand["session_idx"]].to_dict()
                # Double-check skip before generating recommendation
                if args.incremental:
                    key = (str(cand["bu_name"]), str(session.get("title")))
                    if key in existing_pairs:
                        continue
                recommendation_prompt = f"""你是技术会议参谋。请以自然、简洁的人类口吻给出推荐：

Session Information:
- Title: {session.get("title", "N/A")}
- Type: {session.get("type", "N/A")}
- Abstract: {session.get("abstract", "N/A")}
- Overview: {session.get("overview", "N/A")}

BU: {bu_name}
BU Challenge Summary (English):
{summary}

输出要求：
1) reason：
   - 中文，口语化、精准，不要“AI口吻/套话/总结性开头”。
   - 40-60字，点出该BU与该Session最关键的1-2个技术连接点与收获。
   - 避免复述题目，避免空泛表述。
2) focus_areas：
   - 3-5条，每条≤12字，中文名词短语，直指要点；不加句号与编号。
   - 示例："对比学习损失"、"低秩适配"、"在线评测指标"。

仅输出严格JSON：{{"reason": "...", "focus_areas": ["...", "..."]}}
"""
                try:
                    completion = client.chat.completions.parse(
                        model="gpt-4.1-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "你是一个专业的技术匹配分析专家。",
                            },
                            {"role": "user", "content": recommendation_prompt},
                        ],
                        response_format=Recommendation,
                    )
                    recommendation = completion.choices[0].message.parsed.model_dump()
                except Exception as e:
                    print(f"    Error generating recommendation: {e}")
                    recommendation = {"reason": "Error", "focus_areas": []}

                final_results.append(
                    {
                        "bu_name": bu_name,
                        "session_title": session.get("title"),
                        "session_type": session.get("type"),
                        "session_date": session.get("date"),
                        "session_time": session.get("time"),
                        "base_cosine": cand["base_cosine"],
                        "rel": cand["rel"],
                        "tech_overlap": cand["tech_overlap"],
                        "novelty": cand["novelty"],
                        "actionability": cand["actionability"],
                        "confidence": cand["confidence"],
                        "final_score": cand["final_score"],
                        "scoring_reason": cand["scoring_reason"],
                        "recommendation_reason": recommendation["reason"],
                        "focus_areas": "; ".join(recommendation["focus_areas"]),
                    }
                )

            time.sleep(0.2)  # Rate limiting per BU

    # Save final results
    if not final_results:
        print("\nNo final recommendations to save.")
        return

    output_df = pd.DataFrame(final_results)
    # Save with incremental support
    if args.incremental and os.path.exists(output_path):
        try:
            # If columns match exactly, we can append without headers
            existing_df = pd.read_csv(output_path)
            if list(existing_df.columns) == list(output_df.columns):
                # Drop duplicates from new batch just in case
                output_df = output_df.drop_duplicates(subset=["bu_name", "session_title"])
                output_df.to_csv(output_path, mode="a", header=False, index=False, encoding="utf-8-sig")
                print(f"\nAppended {len(output_df)} new rows to: {output_path}")
            else:
                merged = pd.concat([existing_df, output_df], ignore_index=True)
                merged = merged.drop_duplicates(subset=["bu_name", "session_title"])  # safety
                merged.to_csv(output_path, index=False, encoding="utf-8-sig")
                print(
                    f"\nColumns mismatch between existing and new outputs. Rewrote merged file with {len(merged)} rows: {output_path}"
                )
        except Exception as e:
            print(f"\nIncremental save failed ({e}); falling back to overwrite.")
            output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Matching Complete!")
    print(f"Total BU-Session pairs (agentic): {len(final_results)}")
    print(f"Output saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
