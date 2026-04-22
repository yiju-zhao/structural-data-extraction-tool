---
name: lotus-semantic-operators
description: Build LLM-powered data processing pipelines using LOTUS semantic operators over Pandas DataFrames. Use when the user wants to (1) filter, map, join, aggregate, rank, search, extract, deduplicate, or cluster data using natural language and LLMs, (2) build AI query pipelines over structured or unstructured data with a Pandas-like API, (3) use any LOTUS/lotus-ai semantic operator (sem_map, sem_filter, sem_join, sem_agg, sem_topk, sem_search, sem_extract, sem_sim_join, sem_dedup, sem_cluster_by, sem_partition_by). Triggers on phrases like "use LOTUS", "semantic filter", "semantic join", "sem_map", "LLM-powered data processing", or "natural language query over dataframe".
---

# LOTUS Semantic Operators

LOTUS provides declarative semantic operators that extend Pandas DataFrames with LLM-powered transformations parameterized by natural language expressions.

## Setup

```python
import pandas as pd
import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store import FaissVS

# Language model (required for most operators)
lm = LM(model="gpt-5.2")

# Retrieval model + vector store (required for search, sim_join, dedup, cluster)
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
vs = FaissVS()

lotus.settings.configure(lm=lm, rm=rm, vs=vs)
```

Alternative RM: `LiteLLMRM(model="text-embedding-3-small")` from `lotus.models`.

Optional reranker: `CrossEncoderReranker(model="mixedbread-ai/mxbai-rerank-large-v1")` from `lotus.models`, pass as `reranker=reranker` to `configure()`.

## Operators Quick Reference

| Operator | Requires | Purpose |
|---|---|---|
| `sem_map` | LM | Map each row via natural language projection |
| `sem_extract` | LM | Extract structured attributes from each row |
| `sem_filter` | LM | Keep rows matching a natural language predicate |
| `sem_agg` | LM | Aggregate/summarize across all rows |
| `sem_topk` | LM | Rank rows by natural language criteria, return top K |
| `sem_join` | LM (+RM,VS for cascade) | Join two DataFrames on a natural language predicate |
| `sem_sim_join` | RM, VS | Join two DataFrames by embedding similarity |
| `sem_search` | RM, VS | Semantic search over an indexed text column |
| `sem_dedup` | RM, VS | Deduplicate rows by semantic similarity threshold |
| `sem_cluster_by` | RM | Cluster rows by semantic similarity |
| `sem_partition_by` | RM | Partition rows (used with `sem_cluster_by` + `sem_agg`) |
| `sem_index` | RM, VS | Build a vector index on a column (prerequisite for search/sim_join/dedup) |

## Operator Details

See [references/api_reference.md](references/api_reference.md) for full parameter signatures and detailed examples of every operator.

### Language Expression Syntax

Use `{ColumnName}` to reference DataFrame columns in natural language instructions:

```python
df.sem_filter("The {abstract} discusses machine learning")
df.sem_map("Summarize {content} in one sentence")
df1.sem_join(df2, "{Course Name:left} teaches {Skill:right}")
```

For joins, use `:left` and `:right` suffixes to disambiguate columns from different DataFrames.

### Indexing (prerequisite for vector-based operators)

Before using `sem_search`, `sem_sim_join`, `sem_dedup`, or `sem_cluster_by`, index the target column:

```python
df = df.sem_index("column_name", "index_directory_name")
```

### Chaining Operators

Operators chain naturally since each returns a DataFrame:

```python
result = (
    df.sem_index("Text", "idx")
      .sem_search("Text", "machine learning", K=20)
      .sem_filter("The {Text} is about deep learning")
      .sem_topk("Which {Text} is most relevant to NLP?", K=5)
)
```

### Cascade Optimization (sem_filter, sem_join)

Use `CascadeArgs` to reduce LLM calls by pre-filtering with a cheaper model:

```python
from lotus.types import CascadeArgs
cascade_args = CascadeArgs(recall_target=0.9, precision_target=0.9)
df.sem_filter("...", cascade_args=cascade_args)
```

### Few-Shot Examples

`sem_map`, `sem_filter`, and `sem_extract` accept an `examples` DataFrame for few-shot prompting:

```python
examples_df = pd.DataFrame(
    [("Machine Learning 101", "Intro to AI")],
    columns=["Course Name", "Answer"]
)
df.sem_map("What is a similar course to {Course Name}?", examples=examples_df)
```
