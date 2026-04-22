# LOTUS Semantic Operators API Reference

## Table of Contents

1. [sem_map](#sem_map)
2. [sem_extract](#sem_extract)
3. [sem_filter](#sem_filter)
4. [sem_agg](#sem_agg)
5. [sem_topk](#sem_topk)
6. [sem_join](#sem_join)
7. [sem_sim_join](#sem_sim_join)
8. [sem_search](#sem_search)
9. [sem_dedup](#sem_dedup)
10. [sem_cluster_by / sem_partition_by](#sem_cluster_by--sem_partition_by)
11. [sem_index](#sem_index)
12. [Configuration](#configuration)

---

## sem_map

Map each row using a natural language projection. Creates a new column with the LLM's response.

```
df.sem_map(user_instruction, examples=None, suffix="_map", return_raw_outputs=False)
```

**Parameters:**
- `user_instruction` (str): Natural language instruction. Use `{ColumnName}` to reference columns.
- `examples` (DataFrame, optional): Few-shot examples with input columns + "Answer" column.
- `suffix` (str): Suffix for the new output column. Default: `"_map"`.
- `return_raw_outputs` (bool): Return raw LLM outputs. Default: `False`.

**Example:**
```python
import pandas as pd
import lotus
from lotus.models import LM

lm = LM(model="gpt-5.2")
lotus.settings.configure(lm=lm)

data = {"Course Name": [
    "Probability and Random Processes",
    "Optimization Methods in Engineering",
    "Digital Design and Integrated Circuits",
    "Computer Security",
]}
df = pd.DataFrame(data)

# Basic usage
df = df.sem_map("What is a similar course to {Course Name}? Be concise.")
print(df)

# With few-shot examples
examples_df = pd.DataFrame(
    [("Computer Graphics", "Computer Vision"), ("Real Analysis", "Complex Analysis")],
    columns=["Course Name", "Answer"]
)
df = df.sem_map(
    "Given {Course Name}, list a topic to explore next. Respond with just the topic name.",
    examples=examples_df,
    suffix="Next Topics"
)
```

---

## sem_extract

Extract one or more structured attributes from each row.

```
df.sem_extract(input_cols, output_cols, extract_quotes=None, postprocessor=None, suffix="_extract")
```

**Parameters:**
- `input_cols` (list[str]): Columns to extract from.
- `output_cols` (dict): Mapping of output column names to extraction instructions.
- `extract_quotes` (bool, optional): Whether to extract supporting quotes.
- `postprocessor` (callable, optional): Post-processing function for extracted values.
- `suffix` (str): Suffix for new columns. Default: `"_extract"`.

**Example:**
```python
df = pd.DataFrame({"review": [
    "The food was amazing but service was slow. Visited on March 5th.",
    "Great atmosphere, decent prices. Our waiter John was excellent.",
]})

df = df.sem_extract(
    input_cols=["review"],
    output_cols={
        "sentiment": "The overall sentiment (positive, negative, mixed)",
        "date_mentioned": "Any date mentioned in the review",
    }
)
```

---

## sem_filter

Keep rows that match a natural language predicate.

```
df.sem_filter(user_instruction, return_raw_outputs=False, default=True, suffix="_filter",
              examples=None, helper_examples=None, strategy=None, cascade_args=None, return_stats=False)
```

**Parameters:**
- `user_instruction` (str): Natural language predicate. Use `{ColumnName}` to reference columns.
- `return_raw_outputs` (bool): Return raw LLM outputs. Default: `False`.
- `default` (bool): Default value on parsing errors. Default: `True`.
- `suffix` (str): Suffix for new columns. Default: `"_filter"`.
- `examples` (DataFrame, optional): Few-shot examples.
- `helper_examples` (DataFrame, optional): Helper examples for cascade.
- `strategy` (str, optional): Reasoning strategy.
- `cascade_args` (CascadeArgs, optional): Arguments for cascade optimization.
  - `recall_target` (float): Target recall. Default: `None`.
  - `precision_target` (float): Target precision. Default: `None`.
  - `sampling_percentage` (float): Sampling percentage. Default: `0.1`.
  - `failure_probability` (float): Failure probability. Default: `0.2`.
- `return_stats` (bool): Return statistics. Default: `False`.

**Example:**
```python
df = pd.DataFrame({"abstract": [
    "This paper proposes a new method for training large language models...",
    "We study the migration patterns of Arctic birds...",
    "Our approach uses transformer architectures for code generation...",
]})

# Basic filter
filtered = df.sem_filter("The {abstract} is about natural language processing or LLMs")

# With cascade optimization
from lotus.types import CascadeArgs
cascade_args = CascadeArgs(recall_target=0.9, precision_target=0.9)
filtered, stats = df.sem_filter(
    "The {abstract} discusses deep learning",
    cascade_args=cascade_args,
    return_stats=True
)
```

---

## sem_agg

Aggregate/summarize across all rows in the DataFrame.

```
df.sem_agg(user_instruction, suffix="_agg")
```

**Parameters:**
- `user_instruction` (str): Aggregation instruction. Use `{ColumnName}` to reference columns.
- `suffix` (str): Suffix for output. Default: `"_agg"`.

**Returns:** DataFrame with a single row containing the aggregation result. Access via `._output[0]`.

**Example:**
```python
df = pd.DataFrame({"headline": [
    "Tech stocks surge amid AI optimism",
    "Federal Reserve holds rates steady",
    "New climate policy announced by EU",
    "Breakthrough in quantum computing research",
]})

result = df.sem_agg("Summarize the key themes across all {headline}")
print(result._output[0])
```

---

## sem_topk

Rank rows by natural language criteria and return the top K.

```
df.sem_topk(user_instruction, K, method="quick", return_stats=False)
```

**Parameters:**
- `user_instruction` (str): Ranking criteria. Use `{ColumnName}` to reference columns.
- `K` (int): Number of top rows to return.
- `method` (str): Ranking algorithm. Options: `"quick"`, `"heap"`, `"naive"`. Default: `"quick"`.
- `return_stats` (bool): Return statistics. Default: `False`.

**Methods:**
- `"quick"`: Quickselect-based, efficient for large datasets.
- `"heap"`: Heap-based sorting.
- `"naive"`: Pairwise comparison (most LLM calls, most accurate).

**Example:**
```python
df = pd.DataFrame({"Course Name": [
    "Probability and Random Processes",
    "Optimization Methods in Engineering",
    "Digital Design and Integrated Circuits",
    "Computer Security",
]})

# Get top 2 courses requiring least math
sorted_df, stats = df.sem_topk(
    "Which {Course Name} requires the least math?",
    K=2,
    method="quick",
    return_stats=True,
)
print(sorted_df)
print(stats)
```

---

## sem_join

Join two DataFrames based on a natural language predicate.

```
df1.sem_join(df2, join_instruction, cascade_args=None, return_stats=False)
```

**Parameters:**
- `df2` (DataFrame): Right DataFrame to join with.
- `join_instruction` (str): Natural language join condition. Use `{Col:left}` and `{Col:right}` syntax.
- `cascade_args` (CascadeArgs, optional): Cascade optimization arguments.
- `return_stats` (bool): Return statistics. Default: `False`.

**Note:** For cascade optimization, configure RM and VS in settings.

**Example:**
```python
import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.types import CascadeArgs
from lotus.vector_store import FaissVS

lm = LM(model="gpt-5.2")
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
vs = FaissVS()
lotus.settings.configure(lm=lm, rm=rm, vs=vs)

courses = pd.DataFrame({"Course Name": [
    "Digital Design and Integrated Circuits",
    "Data Structures and Algorithms",
    "Natural Language Processing",
]})
skills = pd.DataFrame({"Skill": ["Math", "Computer Science", "Machine Learning", "Writing"]})

# Exact join (all pairs evaluated by LLM)
result = courses.sem_join(skills, "By taking {Course Name:left} I will learn {Skill:right}")

# Cascade join (fewer LLM calls)
cascade_args = CascadeArgs(recall_target=0.7, precision_target=0.7)
result, stats = courses.sem_join(
    skills,
    "By taking {Course Name:left} I will learn {Skill:right}",
    cascade_args=cascade_args,
    return_stats=True,
)
print(f"LM calls: {stats['join_resolved_by_large_model']}")
```

---

## sem_sim_join

Join two DataFrames based on embedding similarity (no LLM calls).

```
df1.sem_sim_join(df2, left_on, right_on, K=1)
```

**Parameters:**
- `df2` (DataFrame): Right DataFrame (must be indexed on `right_on` column).
- `left_on` (str): Column name from left DataFrame.
- `right_on` (str): Column name from right DataFrame.
- `K` (int): Number of nearest neighbors per left row. Default: `1`.

**Prerequisite:** Index the right DataFrame column with `sem_index`.

**Example:**
```python
from lotus.models import LiteLLMRM
from lotus.vector_store import FaissVS

rm = LiteLLMRM(model="text-embedding-3-small")
vs = FaissVS()
lotus.settings.configure(rm=rm, vs=vs)

courses = pd.DataFrame({"Course Name": [
    "History of the Atlantic World",
    "Riemannian Geometry",
    "Operating Systems",
    "Compilers",
]})
departments = pd.DataFrame({"Skill": ["Math", "Computer Science"]})
departments = departments.sem_index("Skill", "skill_index")

result = courses.sem_sim_join(departments, left_on="Course Name", right_on="Skill", K=1)
print(result)
```

---

## sem_search

Semantic search over an indexed text column.

```
df.sem_search(col_name, query, K=None, n_rerank=None, return_scores=None, suffix=None)
```

**Parameters:**
- `col_name` (str): Indexed column to search over.
- `query` (str): Search query string.
- `K` (int, optional): Number of results to retrieve.
- `n_rerank` (int, optional): Number of candidates to rerank (requires reranker in settings).
- `return_scores` (bool, optional): Include similarity scores in output.
- `suffix` (str, optional): Suffix for the scores column.

**Prerequisite:** Index the column with `sem_index`.

**Example:**
```python
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
vs = FaissVS()
lotus.settings.configure(rm=rm, vs=vs)

df = pd.DataFrame({"Text": [
    "Introduction to machine learning algorithms",
    "History of ancient Rome",
    "Deep learning for natural language processing",
    "Cooking Italian cuisine",
]})
df = df.sem_index("Text", "text_index")

results = df.sem_search("Text", "AI and neural networks", K=2, return_scores=True)
print(results)
```

**With reranker:**
```python
from lotus.models import CrossEncoderReranker
reranker = CrossEncoderReranker(model="mixedbread-ai/mxbai-rerank-large-v1")
lotus.settings.configure(rm=rm, vs=vs, reranker=reranker)

results = df.sem_search("Text", "AI and neural networks", K=2, n_rerank=4)
```

---

## sem_dedup

Deduplicate rows by semantic similarity threshold.

```
df.sem_dedup(col_name, threshold=0.815)
```

**Parameters:**
- `col_name` (str): Indexed column to deduplicate on.
- `threshold` (float): Similarity threshold above which rows are considered duplicates. Default: `0.815`.

**Prerequisite:** Index the column with `sem_index`.

**Example:**
```python
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
vs = FaissVS()
lotus.settings.configure(rm=rm, vs=vs)

df = pd.DataFrame({"Text": [
    "Probability and Random Processes",
    "Optimization Methods in Engineering",
    "I don't know what day it is",
    "I don't know what time it is",  # similar to above → deduped
    "Harry Potter and the Sorcerer's Stone",
]})
df = df.sem_index("Text", "index_dir").sem_dedup("Text", threshold=0.815)
print(df)
```

---

## sem_cluster_by / sem_partition_by

Cluster and partition rows by semantic similarity, then aggregate per partition.

```
# Cluster utility function
lotus.utils.cluster(col_name, num_clusters)

# Partition by a clustering function
df.sem_partition_by(partition_fn)
```

**Parameters:**
- `col_name` (str): Column to cluster on.
- `num_clusters` (int): Number of clusters.
- `partition_fn` (callable): Function that assigns partition labels (e.g., `lotus.utils.cluster(...)`).

**Prerequisite:** Index the column with `sem_index`. Configure RM in settings.

**Example:**
```python
lm = LM(max_tokens=2048)
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
lotus.settings.configure(lm=lm, rm=rm)

df = pd.DataFrame({"Course Name": [
    "Probability and Random Processes",
    "Optimization Methods in Engineering",
    "Digital Design and Integrated Circuits",
    "Computer Security",
    "Cooking",
    "Food Sciences",
]})

df = (
    df.sem_index("Course Name", "course_name_index")
      .sem_partition_by(lotus.utils.cluster("Course Name", 2))
)
# Aggregate per partition
out = df.sem_agg("Summarize all {Course Name}")._output[0]
print(out)
```

---

## sem_index

Build a vector index on a column. Required before `sem_search`, `sem_sim_join`, `sem_dedup`, and `sem_cluster_by`.

```
df.sem_index(col_name, index_dir)
```

**Parameters:**
- `col_name` (str): Column to index.
- `index_dir` (str): Directory name to store the index.

**Returns:** The DataFrame (for chaining).

---

## Configuration

### Models

| Class | Import | Purpose |
|---|---|---|
| `LM` | `lotus.models` | Language model (wraps any LiteLLM-compatible model) |
| `SentenceTransformersRM` | `lotus.models` | Local embedding model |
| `LiteLLMRM` | `lotus.models` | API-based embedding model (e.g., OpenAI) |
| `CrossEncoderReranker` | `lotus.models` | Reranker for search refinement |

### Vector Stores

| Class | Import | Purpose |
|---|---|---|
| `FaissVS` | `lotus.vector_store` | FAISS-based vector store |

### Settings

```python
lotus.settings.configure(
    lm=lm,           # Language model (required for most operators)
    rm=rm,            # Retrieval/embedding model (required for vector ops)
    vs=vs,            # Vector store (required for vector ops)
    reranker=reranker # Optional reranker for sem_search
)
```

### CascadeArgs

```python
from lotus.types import CascadeArgs

cascade_args = CascadeArgs(
    recall_target=0.9,        # Target recall (0-1)
    precision_target=0.9,     # Target precision (0-1)
    sampling_percentage=0.1,  # Fraction of data to sample
    failure_probability=0.2,  # Acceptable failure probability
)
```

### Installation

```bash
pip install lotus-ai
```

Set API keys as environment variables for the models you use:
```bash
export OPENAI_API_KEY="..."        # For GPT models
export ANTHROPIC_API_KEY="..."     # For Claude models
```
