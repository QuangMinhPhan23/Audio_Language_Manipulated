---
type: paper
node_id: paper:lee2026_beyond_binary_instrument
title: "Beyond Binary Instrument QA: Probing Instrument Grounding in Music Audio-Language Models"
authors: ["Yujun Lee", "Joonhyeok Shin", "Hyoeun Kim", "Kyuhong Shim"]
year: 2026
venue: "arXiv 2026"
external_ids:
  arxiv: "2606.31338"
  doi: null
  s2: null
tags: ["benchmark-validity", "instrument-grounding", "position-bias"]
added: 2026-10-20T00:47:19Z
---

# Beyond Binary Instrument QA: Probing Instrument Grounding in Music Audio-Language Models

## One-line thesis
High binary instrument-QA accuracy hides option-position bias, confusable-instrument errors and temporal bias; argues for multi-axis diagnostics.

## Problem / Gap
_TODO._

## Method
_TODO._

## Key Results
_TODO._

## Assumptions
_TODO._

## Limitations / Failure Modes
_TODO._

## Reusable Ingredients
_TODO._

## Open Questions
_TODO._

## Claims
_TODO._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
_TODO._

## Abstract (original)

> Recent music audio-language models achieve high accuracy on instrument question-answering benchmarks, but it remains unclear whether this reflects robust audio grounding or benchmark-specific shortcuts. In this paper, we introduce an OpenMIC-derived diagnostic benchmark sequence for instrument grounding in music audio-language models, extending binary instrument-presence QA to genre-prior-reduced examples, confusable instrument discrimination, longer audio context, and temporal localization. Across these settings, high binary QA accuracy often fails to predict model behavior: models can exhibit option-position bias, confusable-instrument errors, and temporal response bias. These results suggest that instrument grounding should be evaluated with multi-axis diagnostic benchmarks rather than a single aggregate accuracy.

