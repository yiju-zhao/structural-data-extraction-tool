#!/bin/bash

# Match Teams to Sessions Script
# This script runs the agentic RAG matching system to match BUs with conference sessions

python match_teams_to_sessions.py \
    --sessions_csv neurips_2025_sessions_MexicoCity_detail.csv \
    --output_csv neurips_2025_sessions_MexicoCity_match_research_interest_v3.csv \
    --num_queries 4 \
    --topm 20 \
    --rec_threshold 0.55 \
    --score_topk_per_bu 30 \

# Parameter explanations:
# --embed_backend xinference \
# --embed_model bge-m3 \
# --num_queries: Number of candidate queries to use per BU (default: 4)
# --topm: Top-M sessions per query to recall (default: 20)
# --rec_threshold: Minimum final_score (0-1) to emit recommendations (default: 0.0)
# --weights: Weights for [relevance, technical_overlap, novelty_gain, actionability] (default: "0.45,0.35,0.10,0.10")
# --confidence_boost: Confidence boost factor (0~1) (default: 0.5)
# --score_topk_per_bu: Prefilter candidates by base cosine and keep top-K per BU before LLM scoring
# --incremental: Skip already matched pairs and append new matches
