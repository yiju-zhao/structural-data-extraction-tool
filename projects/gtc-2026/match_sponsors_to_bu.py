"""
Match GTC 2026 sponsors to Business Unit challenges using LOTUS semantic operators.

For each of the 8 BUs, finds the top 20 most relevant sponsors out of ~340 total.
Pipeline: sem_search (340→80) → sem_topk (80→20) → sem_map (add reasons) → CSV export.
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
SPONSORS_CSV = os.path.join(PROJECT_DIR, "sponsors.csv")
CHALLENGES_JSON = os.path.join(PROJECT_DIR, "challenge_collection.json")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "bu_sponsor_recommendations")
INDEX_DIR = os.path.join(PROJECT_DIR, ".sponsor_index")

SEARCH_K = 80   # embedding pre-filter size
TOPK = 20       # final top-K per BU

# Xinference server configuration
XINFERENCE_BASE_URL = os.getenv("XINFERENCE_BASE_URL", "http://localhost:9997/v1")
XINFERENCE_MODEL = os.getenv("XINFERENCE_MODEL", "qwen3")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# LOTUS setup — using local Qwen3 via Xinference (OpenAI-compatible API)
# ---------------------------------------------------------------------------
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "not-needed")

lm = LM(
    model=f"openai/{XINFERENCE_MODEL}",
    max_batch_size=5,
    max_tokens=4096,
    api_base=XINFERENCE_BASE_URL,
)
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
vs = FaissVS()
lotus.settings.configure(lm=lm, rm=rm, vs=vs)

# ---------------------------------------------------------------------------
# Load and prepare sponsor data
# ---------------------------------------------------------------------------
print("Loading sponsors...")
df = pd.read_csv(SPONSORS_CSV)
print(f"  Loaded {len(df)} sponsors")


def safe_str(val):
    return str(val) if pd.notna(val) else ""


# Build combined sponsor_text column for semantic matching
df["sponsor_text"] = df.apply(
    lambda r: (
        f"Company: {safe_str(r.get('name'))} | "
        f"Tier: {safe_str(r.get('tier'))} | "
        f"Description: {safe_str(r.get('description'))}"
    ),
    axis=1,
)

# Build semantic index
print("Building semantic index on sponsor_text...")
df = df.sem_index("sponsor_text", INDEX_DIR)
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

    # Step 1: sem_search — embedding pre-filter (340 → 80)
    print(f"  [1/4] sem_search: finding top {SEARCH_K} candidates...")
    candidates = df.sem_search("sponsor_text", query_text, K=SEARCH_K)
    print(f"    → {len(candidates)} candidates")

    # Step 2: sem_topk — LLM-based ranking (80 → 20)
    print(f"  [2/4] sem_topk: ranking to top {TOPK}...")
    topk_instruction = (
        f"Given the following Business Unit technology challenges:\n"
        f"{query_text}\n\n"
        f"Rank the sponsors by relevance to these challenges. "
        f"A sponsor is more relevant if its {{sponsor_text}} directly addresses, "
        f"provides products/services for, or is a strong strategic fit with the BU's challenges."
    )
    top_sponsors = candidates.sem_topk(topk_instruction, K=TOPK)
    print(f"    → {len(top_sponsors)} sponsors selected")

    # Step 3: sem_map — generate recommendation reasons
    print(f"  [3/4] sem_map: generating recommendation reasons...")
    reason_instruction = (
        f"Given the following Business Unit ({bu_name}) technology challenges:\n"
        f"{query_text}\n\n"
        f"For the sponsor described by: {{sponsor_text}}\n\n"
        f"Write a concise recommendation reason (2-3 sentences) explaining "
        f"why this sponsor is relevant to the BU's challenges. "
        f"Be specific about which products or capabilities align with which challenges."
    )
    top_sponsors = top_sponsors.sem_map(reason_instruction, suffix="recommendation_reason")
    print(f"    → Reasons generated")

    # Step 4: Build output CSV
    print(f"  [4/4] Exporting CSV...")
    top_sponsors = top_sponsors.reset_index(drop=True)
    top_sponsors["rank"] = range(1, len(top_sponsors) + 1)

    reason_col = [c for c in top_sponsors.columns if "recommendation_reason" in c][0]

    output_df = top_sponsors[["name", "tier", "company_url", "rank", reason_col]].copy()
    output_df.columns = ["sponsor_name", "tier", "company_url", "rank", "recommendation_reason"]

    safe_name = re.sub(r'[<>:"/\\|?*]', '_', bu_name)
    output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_top20.csv")
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"    → Saved to {output_path}")
    print(f"    → {len(output_df)} rows, columns: {list(output_df.columns)}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("DONE — All BU sponsor recommendations generated!")
print(f"{'='*60}")
print(f"Output directory: {OUTPUT_DIR}")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    if fname.endswith(".csv"):
        fpath = os.path.join(OUTPUT_DIR, fname)
        row_count = len(pd.read_csv(fpath))
        print(f"  {fname}: {row_count} rows")

lm.print_total_usage()
