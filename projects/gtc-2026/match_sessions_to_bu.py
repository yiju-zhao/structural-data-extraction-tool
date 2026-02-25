"""
Match GTC 2026 sessions to Business Unit challenges using LOTUS semantic operators.

For each of the 8 BUs, finds the top 50 most relevant sessions out of 904 total.
Pipeline: sem_search (904→150) → sem_topk (150→50) → sem_map (add reasons) → CSV export.
"""

import json
import os
import re

import pandas as pd
from dotenv import load_dotenv

import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store import FaissVS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_CSV = os.path.join(PROJECT_DIR, "gtc-2026-sessions-detailed.csv")
CHALLENGES_JSON = os.path.join(PROJECT_DIR, "challenge_collection.json")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "bu_recommendations")
INDEX_DIR = os.path.join(PROJECT_DIR, ".session_index")

SEARCH_K = 150  # embedding pre-filter size
TOPK = 50       # final top-K per BU

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# LOTUS setup
# ---------------------------------------------------------------------------
lm = LM(model="gpt-5-mini", max_batch_size=5, max_tokens=4096, rate_limit=30)
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
vs = FaissVS()
lotus.settings.configure(lm=lm, rm=rm, vs=vs)

# ---------------------------------------------------------------------------
# Load and prepare session data
# ---------------------------------------------------------------------------
print("Loading sessions...")
df = pd.read_csv(SESSIONS_CSV)
print(f"  Loaded {len(df)} sessions")


def parse_speakers(speakers_str):
    """Extract speaker names and companies from JSON string."""
    try:
        speakers = json.loads(speakers_str)
        return "; ".join(
            f"{s.get('name', '')} ({s.get('company', '')})" for s in speakers
        )
    except (json.JSONDecodeError, TypeError):
        return ""


def parse_key_takeaways(kt_str):
    """Join key takeaways list into a single string."""
    try:
        takeaways = json.loads(kt_str)
        if isinstance(takeaways, list):
            return " | ".join(takeaways)
    except (json.JSONDecodeError, TypeError):
        pass
    return str(kt_str) if pd.notna(kt_str) else ""


def safe_str(val):
    return str(val) if pd.notna(val) else ""


# Build combined session_text column
df["speakers_text"] = df["speakers"].apply(parse_speakers)
df["takeaways_text"] = df["key_takeaways"].apply(parse_key_takeaways)

def truncate(text, max_chars=500):
    """Truncate text to max_chars to keep LLM prompts manageable."""
    text = safe_str(text)
    return text[:max_chars] + "..." if len(text) > max_chars else text


df["session_text"] = df.apply(
    lambda r: (
        f"Title: {safe_str(r.get('title'))} | "
        f"Topic: {safe_str(r.get('topic'))} | "
        f"Industry: {safe_str(r.get('industry'))} | "
        f"Abstract: {truncate(r.get('abstract'), 500)} | "
        f"Key Takeaways: {truncate(r.get('takeaways_text'), 300)} | "
        f"NVIDIA Technology: {safe_str(r.get('nvidia_technology'))} | "
        f"Speakers: {safe_str(r.get('speakers_text'))} | "
        f"Technical Level: {safe_str(r.get('technical_level'))} | "
        f"Audience: {safe_str(r.get('intended_audience'))}"
    ),
    axis=1,
)

# Build semantic index
print("Building semantic index on session_text...")
df = df.sem_index("session_text", INDEX_DIR)
print("  Index built.")

# ---------------------------------------------------------------------------
# Load BU challenges
# ---------------------------------------------------------------------------
print("Loading BU challenges...")
with open(CHALLENGES_JSON, "r", encoding="utf-8") as f:
    challenges_data = json.load(f)

bu_queries = {}
for bu_name, categories in challenges_data.items():
    all_challenges = []
    for cat_name, challenge_list in categories.items():
        cat_label = cat_name.strip()
        for c in challenge_list:
            all_challenges.append(f"[{cat_label}] {c}")
    bu_queries[bu_name] = "\n".join(all_challenges)
    print(f"  {bu_name}: {len(all_challenges)} challenges")

# ---------------------------------------------------------------------------
# Per-BU matching pipeline
# ---------------------------------------------------------------------------
for bu_name, query_text in bu_queries.items():
    print(f"\n{'='*60}")
    print(f"Processing BU: {bu_name}")
    print(f"{'='*60}")

    # Step 1: sem_search — embedding pre-filter (904 → 150)
    print(f"  [1/4] sem_search: finding top {SEARCH_K} candidates...")
    candidates = df.sem_search("session_text", query_text, K=SEARCH_K)
    print(f"    → {len(candidates)} candidates")

    # Step 2: sem_topk — LLM-based ranking (150 → 50)
    print(f"  [2/4] sem_topk: ranking to top {TOPK}...")
    topk_instruction = (
        f"Given the following Business Unit technology challenges:\n"
        f"{query_text}\n\n"
        f"Rank the sessions by relevance to these challenges. "
        f"A session is more relevant if its {{session_text}} directly addresses, "
        f"provides insights into, or offers solutions for the BU's challenges."
    )
    top_sessions = candidates.sem_topk(topk_instruction, K=TOPK)
    print(f"    → {len(top_sessions)} sessions selected")

    # Step 3: sem_map — generate recommendation reasons
    print(f"  [3/4] sem_map: generating recommendation reasons...")
    reason_instruction = (
        f"Given the following Business Unit ({bu_name}) technology challenges:\n"
        f"{query_text}\n\n"
        f"For the session described by: {{session_text}}\n\n"
        f"Write a concise recommendation reason (2-3 sentences) explaining "
        f"why this session is relevant to the BU's challenges. "
        f"Be specific about which challenges it addresses."
    )
    top_sessions = top_sessions.sem_map(reason_instruction, suffix="recommendation_reason")
    print(f"    → Reasons generated")

    # Step 4: Build output CSV
    print(f"  [4/4] Exporting CSV...")
    top_sessions = top_sessions.reset_index(drop=True)
    top_sessions["rank"] = range(1, len(top_sessions) + 1)

    # Clean the recommendation_reason column name (LOTUS adds suffix)
    reason_col = [c for c in top_sessions.columns if "recommendation_reason" in c][0]

    output_df = top_sessions[["session_id", "rank", reason_col]].copy()
    output_df.columns = ["session_id", "rank", "recommendation_reason"]

    # Sanitize filename for filesystem
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', bu_name)
    output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_top50.csv")
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"    → Saved to {output_path}")
    print(f"    → {len(output_df)} rows, columns: {list(output_df.columns)}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("DONE — All BU recommendations generated!")
print(f"{'='*60}")
print(f"Output directory: {OUTPUT_DIR}")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    if fname.endswith(".csv"):
        fpath = os.path.join(OUTPUT_DIR, fname)
        row_count = len(pd.read_csv(fpath))
        print(f"  {fname}: {row_count} rows")

lm.print_total_usage()
