---
type: paper
node_id: paper:zang2025_are_you_really
title: "Are you really listening? Boosting Perceptual Awareness in Music-QA Benchmarks"
authors: ["Yongyi Zang", "Sean O'Brien", "Taylor Berg-Kirkpatrick", "Julian McAuley", "Zachary Novack"]
year: 2025
venue: "ISMIR 2025"
external_ids:
  arxiv: "2504.00369"
  doi: null
  s2: null
tags: ["benchmark-validity", "text-shortcut", "blind-baseline", "distractors"]
added: 2026-10-20T00:47:14Z
---

# Are you really listening? Boosting Perceptual Awareness in Music-QA Benchmarks

## One-line thesis
Text-only LLMs reach 56.4% on MuChoMusic; Perceptual Index + synthetic distractors make questions require listening.

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

> Large Audio Language Models (LALMs), where pretrained text LLMs are finetuned with audio input, have made remarkable progress in music understanding. However, current evaluation methodologies exhibit critical limitations: on the leading Music Question Answering benchmark, MuchoMusic, text-only LLMs without audio perception capabilities achieve surprisingly high accuracy of up to 56.4%, on par or above most LALMs. Furthermore, when presented with random Gaussian noise instead of actual audio, LALMs still perform significantly above chance. These findings suggest existing benchmarks predominantly assess reasoning abilities rather than audio perception. To overcome this challenge, we present RUListening: Robust Understanding through Listening, a framework that enhances perceptual evaluation in Music-QA benchmarks. We introduce the Perceptual Index (PI), a quantitative metric that measures a question's reliance on audio perception by analyzing log probability distributions from text-only language models. Using this metric, we generate synthetic, challenging distractors to create QA pairs that necessitate genuine audio perception. When applied to MuchoMusic, our filtered dataset successfully forces models to rely on perceptual information-text-only LLMs perform at chance levels, while LALMs similarly deteriorate when audio inputs are replaced with noise. These results validate our framework's effectiveness in creating benchmarks that more accurately evaluate audio perception capabilities.

