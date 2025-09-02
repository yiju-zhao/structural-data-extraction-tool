# COLM 2025 Workshops – Data Extraction Plan

## Goals
- Extract structured information for every workshop listed at https://colmweb.org/workshops.html
- For each external workshop website, capture:
  - Title (as written on the website)
  - Abstract/description paragraph(s) from the home/landing page (verbatim)
  - Invited Speakers: names and organizations only (exclude program chairs, organizers, committee unless explicitly listed as invited speakers)
- If any data point is missing, write "N/A".
- Skip workshops whose external site is broken or lacks the desired information and note the reason.
- Respectful crawling: add ~3 seconds delay between visiting different workshop sites.

## Process
1) Enumerate all workshops from the COLM workshops page and collect their names and links. ✅
2) Create results.md and append one structured section per workshop. ✅ (initialized)
3) For each workshop:
   - Open its website (in a new tab)
   - Extract Title, Abstract (verbatim), Invited Speakers (names + org)
   - If invited speakers not present, set to "N/A"
   - Add ~3s wait between sites
   - Mark as done with brief notes (e.g., "no invited speakers listed")
4) Validate that all workshops were processed; list any skipped with reasons.
5) Deliver results to user.

## Checklist
- [x] Enumerate all workshop titles and links from the page
- [x] Initialize results.md file
- [x] Process: 1st Workshop on Multilingual Data Quality Signals (WMDQS) — extracted Title, Abstract, and Invited Speakers
- [x] Process: COLM 2025 Workshop on AI Agents: Capabilities and Safety (AIA)
- [x] Process: First Workshop on Bridging NLP and Public Opinion Research (NLPOR)
- [x] Process: First Workshop on Optimal Reliance and Accountability in Interactions with Generative Language Models (ORIGen)
- [x] Process: INTERPLAY25: First Workshop on the Interplay of Model Behavior and Model Internals
- [x] Process: LLM for Scientific Discovery: Reasoning, Assistance, and Collaboration (LM4SCI) — speakers not yet announced (set N/A)
- [x] Process: Multilingual and Equitable Language Technologies (MELT)
- [x] Process: NLP for Democracy (NLP4Democracy)
- [x] Process: Pragmatic Reasoning in Language Models: Language Models as Language Users (PragLM)
- [x] Process: RAM 2: Reasoning, Attention & Memory – 10 Years On
- [x] Process: SCALR: The 1st Workshop on Test-time Scaling and Reasoning Models
- [x] Process: Social Sim'25
- [x] Process: SoLaR: Socially Responsible Language Modelling Research
- [x] Process: The First Workshop on the Application of LLM Explainability to Reasoning and Planning
- [x] Process: Visions of Language Modeling
- [x] Process: XTempLLMs: The 1st Workshop on Large Language Models for Cross-Temporal Research
- [ ] Validate coverage and finalize

## Tracking
- Total workshops found: 16
- Completed: 16
- Skipped (with reasons): 0
- Notes: All workshops processed; next step is to validate coverage and finalize.