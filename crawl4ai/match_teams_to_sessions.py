#!/usr/bin/env python3
"""
Session-Team Matching Script using OpenAI GPT-5
Matches NeurIPS 2025 sessions to research teams based on their interests

Features:
- Uses GPT-5-mini with Pydantic structured output
- Type-safe parsing with Pydantic models
- Deduplicates matched teams automatically
- Limits to top 3 most relevant teams per session, sorted by relevance
"""

import os
import json
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict, Tuple
import time
import argparse
from dotenv import load_dotenv
load_dotenv()

# Configuration
RESEARCH_INTEREST_FILE = "research_interest.md"
SESSIONS_CSV_FILE = "neurips_2025_sessions_SanDiego_detail.csv"
OUTPUT_CSV_FILE = "neurips_2025_sessions_SanDiego_matched_v5.csv"
OUTPUT_REVIEW_FILE = "neurips_2025_sessions_SanDiego_matched_v3_review.csv"


# Pydantic models for structured output
class MatchedTeam(BaseModel):
    """Single matched team with focus and reason"""
    bu: str          # Team BU name
    focus: str       # Team's focus area
    reason: str      # Recommendation reason


class MatchResult(BaseModel):
    """Result containing list of matched teams"""
    matched_teams: List[MatchedTeam]


class ReviewDecision(BaseModel):
    """Review decision on whether rematch is needed"""
    needs_rematch: bool       # Whether rematch is needed
    review_notes: str         # Reason for the decision


class MatchScore(BaseModel):
    """Three-dimensional matching score for a session-team pair"""
    keyword_score: float      # Keyword matching score (0-10)
    directness_score: float   # Problem-solving directness score (0-10)
    relevance_score: float    # Technical relevance strength score (0-10)
    total_score: float        # Weighted average total score (0-10)
    score_reasoning: str      # Detailed reasoning for the scores


# Huawei BU background information (from web research)
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
    """
    Get BU background information from predefined dictionary

    Args:
        bu_name: Name of the BU

    Returns:
        Background information about the BU
    """
    return BU_CONTEXT_INFO.get(bu_name, "该BU暂无背景信息")


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


def create_prompt(session: Dict, teams: List[Dict[str, str]], review_mode: bool = False, old_matches: str = "") -> str:
    """
    Create a unified prompt for matching or reviewing matches

    Args:
        session: Dict containing session information
        teams: List of team profiles
        review_mode: If True, create review prompt; otherwise matching prompt
        old_matches: Original matched teams (used in review mode)

    Returns:
        Formatted prompt string
    """
    # Build team profiles section
    team_profiles = []
    for i, team in enumerate(teams, 1):
        profile = f"{i}. BU: {team['bu']}\n"
        # Add BU background context if available
        if 'context' in team and team['context']:
            profile += f"   BU背景: {team['context']}\n"
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

    # Detect information completeness
    abstract = str(session.get('abstract', 'N/A'))
    overview = str(session.get('overview', 'N/A'))
    has_abstract = abstract and abstract.strip() and abstract not in ('N/A', 'nan', 'None', '')
    has_overview = overview and overview.strip() and overview not in ('N/A', 'nan', 'None', '')
    only_title = not has_abstract and not has_overview

    # Create prompt based on mode
    if review_mode:
        # Review mode: stricter evaluation of existing matches
        prompt = f"""你是一个严格的研究兴趣匹配审核专家。你正在进行**REVIEW审核模式**，需要重新评估之前的匹配结果。

{session_info}

研究团队信息：
{teams_text}

**原始匹配结果**：{old_matches if old_matches else '无匹配'}

---

**REVIEW审核任务**：

你的任务是以**更高的标准**重新审核这个Session，判断是否应该匹配团队。

**审核原则（极度严格）**：

1. **质疑优先**：假设原始匹配可能存在错误，重新从零开始评估
2. **证据导向**：只有Session中有**明确的技术证据**才能匹配
3. **宁缺毋滥**：不确定时选择不匹配，过度匹配比遗漏更糟糕
4. **深度验证**：不要被表面的关键词相似性误导

---

**审核步骤**：

**第一步：Session技术内核提取**
- Session的核心技术点是什么？（提取3-5个关键技术术语）
- Session解决的具体技术问题是什么？
- Session提出的技术方法/算法是什么？

**第二步：团队需求严格匹配**
对于每个候选团队：
1. 团队"难题"中的具体技术术语是什么？
2. Session的技术点是否**直接命中**团队的技术术语？
3. 如果是宏观需求（如"理解趋势"），Session是否提供了**战略级洞察**？

**第三步：严格筛选**
- ✅ 匹配条件（必须满足）：
  * 具体技术术语完全对应 OR 宏观战略洞察明确
  * Session能直接解决团队的核心技术问题
  * 技术关联度高且明确

- ❌ 不匹配条件（任一即排除）：
  * 只是大领域相关，但技术细节不符
  * Session提到的技术与团队难题是"平行技术"（同领域但不同问题）
  * 关联度模糊或需要"脑补"才能建立联系
  * 只有间接或潜在的帮助

**第四步：最终决策**
- 最多匹配3个最相关的团队
- 如果没有高度相关的团队，返回空数组 `[]`
- 宁可漏掉一个正确匹配，也不要加入一个错误匹配

---

{'**【极度保守匹配】当前session信息不足（只有标题）**：' if only_title else ''}
{'''- 仅当标题中明确包含团队关注的具体技术术语时才匹配
- 不要根据标题推测可能的内容或做任何联想
- 标题笼统或宽泛 → 直接不匹配
- 只做最小的、合理的猜测，禁止过度联想''' if only_title else ''}

---

**推荐理由撰写要求**：
- 格式：Session讨论的[Session核心技术点]，可重点关注其在[团队技术难题]中的[具体应用方向或算法]
- 要求：50-80字，精炼自然，明确指出技术连接点

请返回JSON格式：
{{
    "matched_teams": [
        {{
            "bu": "团队BU名称",
            "focus": "该团队的关注方向",
            "reason": "推荐理由"
        }}
    ]
}}
"""
    else:
        # Matching mode: standard three-step analysis
        prompt = f"""你是一个严格的研究兴趣匹配专家。请分析以下NeurIPS 2025会议session，判断哪些研究团队应该参加这个session。

{session_info}

研究团队信息：
{teams_text}

请严格遵循以下三步分析法，为给定的Session匹配最多3个最相关的团队。核心原则是"宁缺毋滥"，只选择高度相关的匹配。

---

**第一步：解构Session - 深入技术内核**

1.  **核心议题 (What)：** 精炼总结Session的核心技术主题。它到底在讲什么？
2.  **目标问题 (Why)：** Session试图解决或优化的具体技术挑战是什么？
3.  **技术方案 (How)：** Session提出了什么具体的方法、模型、算法或系统设计？请列出关键技术术语。
4.  **底层原理 (First Principle)：** 这些技术方案的本质是什么？它是在哪个基础层面（如计算、存储、通信、算法）上进行了创新？

---

**第二步：拆解团队需求 - 聚焦核心瓶颈**

针对候选的每个团队，进行如下分析：

1.  **业务定位 (Business Context)：** 快速定位该团队的业务领域和核心职责。
2.  **核心难题 (Core Problem)：** 从"难题"描述中，提炼出1-2个最核心的技术挑战。
3.  **技术本质 (Technical Essence)：** 这个难题的底层技术逻辑是什么？（计算效率、内存瓶颈、算法精度、系统调度等）
4.  **关键术语 (Keywords)：** “难题”中出现了哪些必须关注的具体技术术语？
5.  **难题分类 (Problem Type)：** 根据难题性质，将其明确归类为：
    *   **A类：具体技术实现**
    *   **B类：宏观战略认知**

---

**第三步：执行匹配 - 依据底层逻辑**

1.  **选择匹配标准：**
    *   如果团队难题被分类为 **A类**：Session是否**直接**解决了团队的**关键术语**所指向的问题，并能提供**具体、可操作**的思路？
    *   如果团队难题被分类为 **B类**：Session所揭示的趋势，是否能帮助团队**预判**技术演进，并提供高阶的**战略性洞察**？

2.  **生成匹配结果：**
    *   按相关度从高到低排序，最多输出3个匹配团队。
    *   若无高度相关的团队，则返回空数组 `[]`。
    *   为每个匹配的团队，提供`bu`, `focus`, 和`reason`。

3.  **撰写推荐理由 (Reason)：**
    *   **格式：** Session讨论的[Session核心技术点]，可重点关注其在[团队技术难题]中的[具体应用方向或算法]。
    *   **要求：** 50-80字，语言精炼自然，明确指出技术连接点，给予清晰的关注建议。

---
{'**【极度保守匹配】当前session信息不足（只有标题）**：' if only_title else ''}
{'''- 仅当标题中明确包含团队关注的具体技术术语时才匹配
- 不要根据标题推测可能的内容或做任何联想
- 标题笼统或宽泛（如"AI Advances"、"Future of ML"、"Recent Progress"等）→ 直接不匹配
- 标题只提到大领域（如"Computer Vision"、"NLP"、"Robotics"）没有具体技术 → 不匹配
- 信心不足时 → 不匹配
- 只做最小的、合理的猜测，禁止过度联想''' if only_title else ''}
"""

    return prompt


def score_match(session: Dict, team: Dict[str, str], client: OpenAI) -> Dict:
    """
    Score a session-team match using three dimensions (0-10 scale each)

    Args:
        session: Dict containing session information
        team: Single team profile dict
        client: OpenAI client instance

    Returns:
        Dict with keyword_score, directness_score, relevance_score, total_score, score_reasoning
    """
    # Build session information
    session_info = f"""
Session Information:
- Title: {session.get('title', 'N/A')}
- Type: {session.get('type', 'N/A')}
- Abstract: {session.get('abstract', 'N/A')}
- Overview: {session.get('overview', 'N/A')}
"""

    # Build team information
    team_info = f"""
Team Information:
- BU: {team['bu']}
- BU Background: {team.get('context', 'N/A')}
- Focus: {team['focus']}
- Challenge/Problem: {team['challenges']}
"""

    prompt = f"""你是一个专业的技术匹配评分专家。请对以下Session和团队的匹配度进行三维度打分（0-10分制）。

{session_info}

{team_info}

---

**评分任务**：

请从以下三个维度对匹配度进行打分（每个维度0-10分）：

**维度1：关键术语匹配度（0-10分）**

评分标准：
1. 提取团队"难题"中的关键技术术语
2. 提取Session中的技术术语（从title/abstract/overview）
3. 计分方式：
   - 核心术语完全匹配：+3分/个
   - 相关术语匹配：+2分/个
   - 泛化术语匹配：+1分/个
   - **封顶10分**

示例：
- 团队难题："Ascend AI芯片的CANN算子优化"
- Session："Efficient Operator Fusion for AI Accelerators"
- 评分：AI accelerator(+2), Operator(+3) = 5分

**维度2：问题解决直接性（0-10分）**

评分标准：
- 9-10分：Session的主要内容**精确命中**团队难题的核心技术点
- 7-8分：Session回答了难题的**主要方面**，有明确可操作的思路
- 5-6分：Session提供了**部分相关**的解决思路或技术参考
- 3-4分：Session的思路可以**间接应用**到团队问题
- 1-2分：Session只在**概念层面**与问题相关
- 0分：Session与团队问题**完全不相关**

**维度3：技术相关性强度（0-10分）**

评分标准：
- 9-10分：Session和团队在**同一技术栈**工作（如都是Ascend芯片优化）
- 7-8分：Session和团队在**同一技术领域**（如都是AI推理加速）
- 5-6分：技术方向相关但**子领域不同**（如训练 vs 推理）
- 3-4分：**同大领域但技术路径不同**（如GPU优化 vs ASIC优化）
- 1-2分：只在**AI大领域**相关，技术细节完全不同
- 0分：完全不同的技术领域

---

**总分计算**：
总分 = (维度1 × 0.3) + (维度2 × 0.4) + (维度3 × 0.3)
（保留1位小数）

**评分理由**：
用一句话（80-150字）解释打分依据，格式为：
"关键术语：[匹配到的术语及分数]；直接性：[是否直接解决问题及分数]；相关性：[技术栈/领域关联度及分数]"

请返回JSON格式：
{{
    "keyword_score": X.X,
    "directness_score": X.X,
    "relevance_score": X.X,
    "total_score": X.X,
    "score_reasoning": "评分理由"
}}
"""

    try:
        # Use chat.completions.parse with Pydantic model for structured output
        completion = client.chat.completions.parse(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "你是一个专业的技术匹配评分专家，擅长量化评估session与团队需求的匹配度。"},
                {"role": "user", "content": prompt}
            ],
            response_format=MatchScore,
        )

        # Get the parsed result directly from Pydantic model
        score_result = completion.choices[0].message.parsed

        return {
            "keyword_score": score_result.keyword_score,
            "directness_score": score_result.directness_score,
            "relevance_score": score_result.relevance_score,
            "total_score": score_result.total_score,
            "score_reasoning": score_result.score_reasoning
        }

    except Exception as e:
        print(f"Error scoring session '{session.get('title', 'N/A')}' for team '{team['bu']}': {str(e)}")
        # Return zero scores on error
        return {
            "keyword_score": 0.0,
            "directness_score": 0.0,
            "relevance_score": 0.0,
            "total_score": 0.0,
            "score_reasoning": f"评分出错: {str(e)}"
        }


def review_match_decision(session: Dict, teams: List[Dict[str, str]], old_matches: str, client: OpenAI) -> Dict:
    """
    Review existing matches and decide if rematch is needed

    Args:
        session: Dict containing session information
        teams: List of team profiles
        old_matches: Original matched teams string (e.g., "BU1; BU2")
        client: OpenAI client instance

    Returns:
        Dict with keys: needs_rematch (bool), review_notes (str)
    """
    # Build team profiles section
    team_profiles = []
    for i, team in enumerate(teams, 1):
        profile = f"{i}. BU: {team['bu']}\n"
        if 'context' in team and team['context']:
            profile += f"   BU背景: {team['context']}\n"
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

    # Detect information completeness
    abstract = str(session.get('abstract', 'N/A'))
    overview = str(session.get('overview', 'N/A'))
    has_abstract = abstract and abstract.strip() and abstract not in ('N/A', 'nan', 'None', '')
    has_overview = overview and overview.strip() and overview not in ('N/A', 'nan', 'None', '')
    only_title = not has_abstract and not has_overview

    prompt = f"""你是一个严格的研究兴趣匹配审核专家。你的任务是判断原始匹配是否符合标准。

{session_info}

研究团队信息：
{teams_text}

**原始匹配结果**：{old_matches if old_matches else '无匹配'}

---

**审核任务**：

判断原始匹配是否符合以下匹配标准。如果不符合，则需要重新匹配。

**匹配标准检查清单**（提炼自matching标准）：

1. **A类问题（具体技术实现）检查**：
   - Session是否**直接**解决了团队的**关键技术术语**所指向的问题？
   - Session是否提供了**具体、可操作**的技术思路？

2. **B类问题（宏观战略认知）检查**：
   - Session是否能帮助团队**预判**技术演进？
   - Session是否提供了**战略性洞察**？

3. **保守原则检查**：
   - 如果只有标题信息，标题是否明确包含团队的具体技术术语？
   - 是否避免了大领域泛匹配（如"AI"、"ML"等宽泛概念）？

4. **相关性检查**：
   - Session的技术点与团队难题是否是"平行技术"（同领域但不同问题）？
   - 关联度是否明确，还是需要"脑补"才能建立联系？

---

{'**【极度保守匹配】当前session信息不足（只有标题）**：' if only_title else ''}
{'''- 仅当标题中明确包含团队关注的具体技术术语时才能匹配
- 标题笼统或宽泛 → 不应匹配
- 只做最小的、合理的推断''' if only_title else ''}

---

**判断逻辑**：
- 如果原始匹配**完全符合**以上标准 → needs_rematch = false
- 如果原始匹配**不符合**或**可能过度匹配** → needs_rematch = true，并在review_notes中说明原因

**review_notes要求**：
- 如果needs_rematch = true：用80-150字说明为什么原匹配不符合标准（例如："原匹配过于宽泛，Session讨论X技术，但团队需求Y技术，属于平行技术不直接相关"）
- 如果needs_rematch = false：简要说明"匹配符合标准"

请返回JSON格式：
{{
    "needs_rematch": true/false,
    "review_notes": "判断理由"
}}
"""

    try:
        # Use chat.completions.parse with Pydantic model for structured output
        completion = client.chat.completions.parse(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "你是一个严格的匹配审核专家。根据匹配标准判断原始匹配是否合格，需要时建议重新匹配。"},
                {"role": "user", "content": prompt}
            ],
            response_format=ReviewDecision,
        )

        # Get the parsed result directly from Pydantic model
        review_result = completion.choices[0].message.parsed

        return {
            "needs_rematch": review_result.needs_rematch,
            "review_notes": review_result.review_notes
        }

    except Exception as e:
        print(f"Error reviewing session '{session.get('title', 'N/A')}': {str(e)}")
        # Default to not rematch on error
        return {
            "needs_rematch": False,
            "review_notes": f"审核出错，保留原匹配: {str(e)}"
        }


def match_session_to_teams(session: Dict, teams: List[Dict[str, str]], client: OpenAI, review_mode: bool = False, old_matches: str = "", review_feedback: str = "") -> Dict:
    """
    Use OpenAI API to match a session to relevant teams with Pydantic structured output

    Args:
        session: Session information dict
        teams: List of team profiles
        client: OpenAI client instance
        review_mode: If True, use stricter review prompt
        old_matches: Original matched teams (for review mode)
        review_feedback: Feedback from review process (if rematching based on review)

    Returns:
        Dict with matched teams information
    """
    # Choose prompt based on mode
    if review_mode:
        prompt = create_prompt(session, teams, review_mode=True, old_matches=old_matches)
        system_msg = "你是一个极度严格的研究兴趣匹配审核专家。在REVIEW模式下，你必须以更高标准重新评估，质疑原始匹配，只保留有明确技术证据的匹配。宁可漏掉也不要过度匹配。"
    else:
        prompt = create_prompt(session, teams, review_mode=False)
        system_msg = "你是一个严格的研究兴趣匹配专家。Return data in the exact structured format requested. 推荐理由必须精简自然（50-80字），使用流畅的中文表达，格式为'Session讨论[具体技术点]，可重点关注[具体方向/技术/算法]'，给出明确的技术关注建议。重要：先判断团队难题的性质——如果是具体技术术语，则严格匹配；如果是宏观战略需求（如理解趋势、预判演进），则从宏观层面匹配。"

    # Add review feedback if provided
    if review_feedback:
        feedback_section = f"""
【评审建议 - 重新匹配说明】
原匹配经过评审发现需要调整：
{review_feedback}

请基于以上评审建议和下面的匹配标准，重新进行更精确的匹配。

---

"""
        prompt = feedback_section + prompt

    try:
        # Use chat.completions.parse with Pydantic model for structured output
        completion = client.chat.completions.parse(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            response_format=MatchResult,
        )

        # Get the parsed result directly from Pydantic model
        match_result = completion.choices[0].message.parsed

        # Convert Pydantic model to dict
        return {
            "matched_teams": [
                {
                    "bu": team.bu,
                    "focus": team.focus,
                    "reason": team.reason
                }
                for team in match_result.matched_teams
            ]
        }

    except Exception as e:
        print(f"Error matching session '{session.get('title', 'N/A')}': {str(e)}")
        return {"matched_teams": []}


def format_matched_results(matched_teams: List[Dict]) -> Tuple[str, str, str]:
    """
    Format matched teams into three strings for CSV columns

    Args:
        matched_teams: List of matched team dicts (already sorted by relevance)

    Returns:
        Tuple of (team_names, focuses, reasons) formatted with semicolon separators
    """
    if not matched_teams:
        return "", "", ""

    # Deduplicate teams while preserving order (first occurrence wins)
    seen_bus = set()
    unique_teams = []
    for team in matched_teams:
        bu = team['bu']
        if bu not in seen_bus:
            seen_bus.add(bu)
            unique_teams.append(team)

    # Limit to top 3 most relevant teams (already sorted by GPT-5)
    unique_teams = unique_teams[:3]

    team_names = "; ".join([team['bu'] for team in unique_teams])
    focuses = "; ".join([f"{team['bu']}: {team['focus']}" for team in unique_teams])
    reasons = "; ".join([f"{team['bu']}: {team['reason']}" for team in unique_teams])

    return team_names, focuses, reasons


def parse_old_matches(team_names_str: str, focuses_str: str, reasons_str: str) -> List[Dict]:
    """
    Parse old matches from CSV columns back to matched_teams format

    Args:
        team_names_str: String like "BU1; BU2; BU3"
        focuses_str: String like "BU1: focus1; BU2: focus2; BU3: focus3"
        reasons_str: String like "BU1: reason1; BU2: reason2; BU3: reason3"

    Returns:
        List of matched team dicts with keys: bu, focus, reason
    """
    if not team_names_str or pd.isna(team_names_str) or team_names_str.strip() == "":
        return []

    # Parse team names
    team_names = [name.strip() for name in str(team_names_str).split(';') if name.strip()]

    # Parse focuses into dict
    focuses_dict = {}
    if focuses_str and not pd.isna(focuses_str):
        for item in str(focuses_str).split(';'):
            if ':' in item:
                bu, focus = item.split(':', 1)
                focuses_dict[bu.strip()] = focus.strip()

    # Parse reasons into dict
    reasons_dict = {}
    if reasons_str and not pd.isna(reasons_str):
        for item in str(reasons_str).split(';'):
            if ':' in item:
                bu, reason = item.split(':', 1)
                reasons_dict[bu.strip()] = reason.strip()

    # Reconstruct matched_teams list
    matched_teams = []
    for bu in team_names:
        matched_teams.append({
            'bu': bu,
            'focus': focuses_dict.get(bu, ''),
            'reason': reasons_dict.get(bu, '')
        })

    return matched_teams


def scoring_pass(client: OpenAI, teams: List[Dict[str, str]], sessions_df: pd.DataFrame) -> None:
    """
    First pass: Score all session-team combinations and save to scored CSV

    Args:
        client: OpenAI client instance
        teams: List of team profiles with context
        sessions_df: DataFrame containing session information
    """
    # Define output file for scored results
    SCORED_OUTPUT = OUTPUT_CSV_FILE.replace('.csv', '_scored.csv')

    print("\n" + "=" * 80)
    print("SCORING PASS: Evaluating all session-team combinations")
    print("=" * 80)

    # Check if scored file exists (for resume capability)
    start_idx = 0
    if os.path.exists(SCORED_OUTPUT):
        print(f"\n[Notice] Scored file {SCORED_OUTPUT} already exists.")
        scored_df = pd.read_csv(SCORED_OUTPUT)
        start_idx = len(scored_df)
        if start_idx >= len(sessions_df) * len(teams):
            print(f"[Notice] All combinations already scored. Delete {SCORED_OUTPUT} to restart.")
            return
        print(f"[Notice] Resuming from combination {start_idx + 1}/{len(sessions_df) * len(teams)}")
    else:
        # Initialize scored CSV with header
        output_columns = list(sessions_df.columns) + [
            '团队BU', '关键词得分', '直接性得分', '相关性得分', '总分', '评分理由'
        ]
        pd.DataFrame(columns=output_columns).to_csv(SCORED_OUTPUT, index=False, encoding='utf-8-sig')
        print(f"[Step 1] Created scored file: {SCORED_OUTPUT}")

    # Calculate all combinations
    total_combinations = len(sessions_df) * len(teams)
    current_combination = 0

    print(f"\n[Step 2] Scoring {len(sessions_df)} sessions × {len(teams)} teams = {total_combinations} combinations...")
    print(f"Output will be saved to: {SCORED_OUTPUT}")

    # Iterate through all session-team combinations
    for session_idx, row in sessions_df.iterrows():
        session = row.to_dict()
        session_title = session.get('title', 'N/A')[:60]

        for team in teams:
            current_combination += 1

            # Skip if already processed (resume logic)
            if current_combination <= start_idx:
                continue

            team_bu = team['bu']
            print(f"\n[{current_combination}/{total_combinations}] Scoring: {session_title} × {team_bu}")

            # Call score_match to get scores
            score_result = score_match(session, team, client)

            # Create output row
            output_row = row.to_dict()
            output_row['团队BU'] = team_bu
            output_row['关键词得分'] = score_result['keyword_score']
            output_row['直接性得分'] = score_result['directness_score']
            output_row['相关性得分'] = score_result['relevance_score']
            output_row['总分'] = score_result['total_score']
            output_row['评分理由'] = score_result['score_reasoning']

            # Append to CSV immediately
            pd.DataFrame([output_row]).to_csv(
                SCORED_OUTPUT,
                mode='a',
                header=False,
                index=False,
                encoding='utf-8-sig'
            )

            print(f"  ✓ 关键词:{score_result['keyword_score']} 直接性:{score_result['directness_score']} 相关性:{score_result['relevance_score']} → 总分:{score_result['total_score']}")
            print(f"  💾 Saved to {SCORED_OUTPUT}")

            # Rate limiting
            time.sleep(0.3)

    # Summary
    print("\n" + "=" * 80)
    print("Scoring Complete!")
    print("=" * 80)
    print(f"Total combinations scored: {total_combinations}")
    print(f"\nScored results saved to: {SCORED_OUTPUT}")
    print("=" * 80)
    print("\nNext step: Run filtering pass with --filter flag")
    print(f"Example: python {__file__} --filter --min-score 6.0 --max-ratio 0.33")


def apply_allocation_constraints(scored_df: pd.DataFrame, max_ratio: float = 0.33) -> pd.DataFrame:
    """
    Apply 33% allocation constraint: each team can have at most max_ratio of total sessions

    Args:
        scored_df: DataFrame with scored matches (already filtered by min_score)
        max_ratio: Maximum allocation ratio per team (default 0.33 = 33%)

    Returns:
        Filtered DataFrame with allocation constraints applied
    """
    print("\n" + "=" * 80)
    print(f"Applying Allocation Constraints (max {max_ratio*100:.0f}% per team)")
    print("=" * 80)

    # Calculate max sessions per team
    total_sessions = scored_df['title'].nunique()
    max_per_team = int(total_sessions * max_ratio)

    print(f"\nTotal unique sessions: {total_sessions}")
    print(f"Max sessions per team: {max_per_team} ({max_ratio*100:.0f}%)")

    # Group by team and apply constraints
    constrained_rows = []
    team_stats = []

    for team_bu in scored_df['团队BU'].unique():
        team_df = scored_df[scored_df['团队BU'] == team_bu].copy()
        original_count = len(team_df)

        # Sort by total score descending
        team_df = team_df.sort_values('总分', ascending=False)

        # Apply constraint
        if original_count > max_per_team:
            team_df = team_df.head(max_per_team)
            status = f"截断 ({original_count} → {max_per_team})"
        else:
            status = f"未超限 ({original_count})"

        constrained_rows.append(team_df)
        team_stats.append({
            'team': team_bu,
            'original': original_count,
            'final': len(team_df),
            'status': status
        })

        print(f"[{team_bu}] {status}")

    # Combine all constrained teams
    result_df = pd.concat(constrained_rows, ignore_index=True)

    # Summary
    print("\n" + "=" * 80)
    print("Allocation Constraints Applied")
    print("=" * 80)
    print(f"Total matches before: {len(scored_df)}")
    print(f"Total matches after: {len(result_df)}")
    print(f"Filtered out: {len(scored_df) - len(result_df)}")

    return result_df


def filtering_pass(client: OpenAI, teams: List[Dict[str, str]], min_score: float = 6.0, max_ratio: float = 0.33) -> None:
    """
    Second pass: Filter by score, apply allocation constraints, and generate recommendations

    Args:
        client: OpenAI client instance
        teams: List of team profiles with context
        min_score: Minimum score threshold (default 6.0)
        max_ratio: Maximum allocation ratio per team (default 0.33)
    """
    # Define file paths
    SCORED_INPUT = OUTPUT_CSV_FILE.replace('.csv', '_scored.csv')
    FILTERED_OUTPUT = OUTPUT_CSV_FILE.replace('.csv', '_filtered.csv')

    print("\n" + "=" * 80)
    print(f"FILTERING PASS: Score threshold={min_score}, Max ratio={max_ratio*100:.0f}%")
    print("=" * 80)

    # Check if scored file exists
    if not os.path.exists(SCORED_INPUT):
        print(f"\nError: Scored file {SCORED_INPUT} not found.")
        print("Please run scoring pass first: python match_teams_to_sessions.py --score-only")
        return

    # Load scored results
    print(f"\n[Step 1] Loading scored results from {SCORED_INPUT}...")
    scored_df = pd.read_csv(SCORED_INPUT)
    print(f"  Loaded {len(scored_df)} scored combinations")

    # Step 2: Filter by minimum score
    print(f"\n[Step 2] Filtering by minimum score >= {min_score}...")
    high_score_df = scored_df[scored_df['总分'] >= min_score].copy()
    print(f"  High-score matches: {len(high_score_df)} (filtered out {len(scored_df) - len(high_score_df)})")

    if len(high_score_df) == 0:
        print(f"\nNo matches found with score >= {min_score}. Adjust --min-score threshold.")
        return

    # Step 3: Apply allocation constraints
    print(f"\n[Step 3] Applying allocation constraints...")
    constrained_df = apply_allocation_constraints(high_score_df, max_ratio)

    # Step 4: Group by session and generate final matched teams format
    print(f"\n[Step 4] Generating recommendations for {len(constrained_df)} final matches...")

    # Initialize output CSV
    filtered_columns = ['title', 'type', 'date', 'time', 'location', 'abstract', 'overview',
                       '匹配团队', '关注方向', '推荐理由', '总分']
    pd.DataFrame(columns=filtered_columns).to_csv(FILTERED_OUTPUT, index=False, encoding='utf-8-sig')

    # Group by session
    sessions_with_teams = constrained_df.groupby('title')

    total_sessions = len(sessions_with_teams)
    processed = 0

    for session_title, group_df in sessions_with_teams:
        processed += 1
        print(f"\n[{processed}/{total_sessions}] Processing: {session_title[:60]}...")

        # Get session info from first row
        session_row = group_df.iloc[0]
        session = session_row.to_dict()

        # Get teams for this session (sorted by score descending)
        matched_teams_info = []
        for _, row in group_df.sort_values('总分', ascending=False).iterrows():
            team_bu = row['团队BU']
            # Find team details
            team_details = next((t for t in teams if t['bu'] == team_bu), None)
            if team_details:
                matched_teams_info.append({
                    'bu': team_bu,
                    'focus': team_details['focus'],
                    'score': row['总分']
                })

        # Limit to top 3 teams per session
        matched_teams_info = matched_teams_info[:3]

        # Generate recommendation reasons using matching prompt
        # Create a list of matched teams to pass to match_session_to_teams
        teams_for_this_session = [next((t for t in teams if t['bu'] == tm['bu']), None)
                                  for tm in matched_teams_info]
        teams_for_this_session = [t for t in teams_for_this_session if t is not None]

        if teams_for_this_session:
            # Call match_session_to_teams to get detailed reasons
            match_result = match_session_to_teams(session, teams_for_this_session, client)
            matched_teams_with_reasons = match_result.get('matched_teams', [])

            # Format results
            team_names, focuses, reasons = format_matched_results(matched_teams_with_reasons)

            # Calculate average score for this session
            avg_score = group_df['总分'].mean()

            # Create output row
            output_row = {
                'title': session.get('title'),
                'type': session.get('type'),
                'date': session.get('date'),
                'time': session.get('time'),
                'location': session.get('location'),
                'abstract': session.get('abstract'),
                'overview': session.get('overview'),
                '匹配团队': team_names,
                '关注方向': focuses,
                '推荐理由': reasons,
                '总分': f"{avg_score:.1f}"
            }

            # Append to CSV
            pd.DataFrame([output_row]).to_csv(
                FILTERED_OUTPUT,
                mode='a',
                header=False,
                index=False,
                encoding='utf-8-sig'
            )

            print(f"  ✓ Matched {len(matched_teams_with_reasons)} team(s): {team_names}")
            print(f"  💾 Saved to {FILTERED_OUTPUT}")

        # Rate limiting
        time.sleep(0.5)

    # Summary
    print("\n" + "=" * 80)
    print("Filtering Complete!")
    print("=" * 80)
    print(f"Total sessions with matches: {total_sessions}")
    print(f"\nFiltered results saved to: {FILTERED_OUTPUT}")
    print("=" * 80)


def review_existing_matches(client: OpenAI, teams: List[Dict[str, str]], sessions_df: pd.DataFrame) -> None:
    """
    Review and update existing matches with incremental write to OUTPUT_REVIEW_FILE

    Args:
        client: OpenAI client instance
        teams: List of team profiles with context
        sessions_df: DataFrame containing session information
    """
    if not os.path.exists(OUTPUT_CSV_FILE):
        print(f"Error: Output file {OUTPUT_CSV_FILE} not found. Please run normal mode first.")
        return

    print("\n" + "=" * 80)
    print("REVIEW MODE: Re-evaluating existing matches (Incremental Write)")
    print("=" * 80)

    # Load existing results
    existing_df = pd.read_csv(OUTPUT_CSV_FILE)
    print(f"\nLoaded {len(existing_df)} existing matches from {OUTPUT_CSV_FILE}")

    # Check if review file exists (for resume capability)
    start_idx = 0
    if os.path.exists(OUTPUT_REVIEW_FILE):
        print(f"\n[Notice] Review file {OUTPUT_REVIEW_FILE} already exists.")
        reviewed_df = pd.read_csv(OUTPUT_REVIEW_FILE)
        start_idx = len(reviewed_df)
        if start_idx >= len(existing_df):
            print(f"[Notice] All sessions already reviewed. Delete {OUTPUT_REVIEW_FILE} to restart.")
            return
        print(f"[Notice] Resuming review from session {start_idx + 1}/{len(existing_df)}")
    else:
        # Initialize review CSV with header
        output_columns = list(existing_df.columns)
        pd.DataFrame(columns=output_columns).to_csv(OUTPUT_REVIEW_FILE, index=False, encoding='utf-8-sig')
        print(f"[Step 3] Created review file: {OUTPUT_REVIEW_FILE}")

    # Track statistics
    total_changed = 0
    total_rematched = 0
    total_confirmed = 0

    # Process each row from start_idx
    for idx in range(start_idx, len(existing_df)):
        row = existing_df.iloc[idx]
        session = row.to_dict()
        session_title = session.get('title', 'N/A')[:60]
        old_teams = str(row.get('匹配团队', ''))

        print(f"\n[{idx+1}/{len(existing_df)}] Reviewing: {session_title}...")
        print(f"  Original: {old_teams if old_teams else 'None'}")

        # Step 1: Review decision - check if rematch is needed
        review_decision = review_match_decision(session, teams, old_teams, client)
        needs_rematch = review_decision['needs_rematch']
        review_notes = review_decision['review_notes']

        print(f"  Review: {'需要重新匹配' if needs_rematch else '匹配符合标准'}")
        print(f"  理由: {review_notes[:80]}...")

        # Step 2: Conditional rematch
        if needs_rematch:
            # Rematch with review feedback
            match_result = match_session_to_teams(
                session, teams, client,
                review_feedback=review_notes
            )
            matched_teams = match_result.get('matched_teams', [])
            print(f"  ✏️  执行重新匹配...")
        else:
            # Keep original matches
            old_focuses = str(row.get('关注方向', ''))
            old_reasons = str(row.get('推荐理由', ''))
            matched_teams = parse_old_matches(old_teams, old_focuses, old_reasons)
            print(f"  ✓  保留原匹配")

        # Format new results
        new_team_names, new_focuses, new_reasons = format_matched_results(matched_teams)

        # Update row with new results
        output_row = row.to_dict()
        output_row['匹配团队'] = new_team_names
        output_row['关注方向'] = new_focuses
        output_row['推荐理由'] = new_reasons

        # Append to review CSV immediately
        pd.DataFrame([output_row]).to_csv(
            OUTPUT_REVIEW_FILE,
            mode='a',
            header=False,
            index=False,
            encoding='utf-8-sig'
        )

        # Update statistics
        if needs_rematch:
            total_rematched += 1
            if new_team_names != old_teams:
                total_changed += 1
                print(f"  结果: {new_team_names if new_team_names else 'None'}")
            else:
                print(f"  结果: 重新匹配后结果相同")
        else:
            total_confirmed += 1

        print(f"  💾 Saved to {OUTPUT_REVIEW_FILE}")

        # Rate limiting
        time.sleep(0.5)

    # Summary
    print("\n" + "=" * 80)
    print("Review Complete!")
    print("=" * 80)
    print(f"Total sessions reviewed: {len(existing_df)}")
    print(f"  - Rematched (需要重新匹配): {total_rematched}")
    print(f"  - Confirmed (保留原匹配): {total_confirmed}")
    print(f"  - Actually changed: {total_changed}")
    print(f"\nReviewed results saved to: {OUTPUT_REVIEW_FILE}")
    print("=" * 80)


def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Session-Team Matching System')
    parser.add_argument('--review', action='store_true',
                       help='Review and update existing matches')
    parser.add_argument('--score-only', action='store_true',
                       help='Run scoring pass only (evaluate all session-team combinations)')
    parser.add_argument('--filter', action='store_true',
                       help='Run filtering pass (filter by score and generate recommendations)')
    parser.add_argument('--min-score', type=float, default=6.0,
                       help='Minimum score threshold for filtering (default: 6.0)')
    parser.add_argument('--max-ratio', type=float, default=0.33,
                       help='Maximum allocation ratio per team (default: 0.33 = 33%%)')
    args = parser.parse_args()

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

    # Step 1.5: Add BU background context
    print("\n[Step 1.5] Adding BU background context...")
    for team in teams:
        team['context'] = get_bu_context(team['bu'])
        print(f"  - {team['bu']}: {team['context'][:50]}...")

    # Step 2: Load sessions CSV
    print(f"\n[Step 2] Loading sessions from {SESSIONS_CSV_FILE}...")
    sessions_df = pd.read_csv(SESSIONS_CSV_FILE)
    print(f"Found {len(sessions_df)} sessions")

    # Check mode and dispatch
    if args.score_only:
        # Run scoring pass only
        scoring_pass(client, teams, sessions_df)
        return
    elif args.filter:
        # Run filtering pass
        filtering_pass(client, teams, min_score=args.min_score, max_ratio=args.max_ratio)
        return
    elif args.review:
        # Run review mode
        review_existing_matches(client, teams, sessions_df)
        return

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
