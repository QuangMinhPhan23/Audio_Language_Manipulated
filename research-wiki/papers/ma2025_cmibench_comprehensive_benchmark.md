---
type: paper
node_id: paper:ma2025_cmibench_comprehensive_benchmark
title: "CMI-Bench: A Comprehensive Benchmark for Evaluating Music Instruction Following"
authors: ["Yinghao Ma", "Siyou Li", "Juntao Yu", "Emmanouil Benetos", "Akira Maezawa"]
year: 2025
venue: "ISMIR 2025"
external_ids:
  arxiv: "2506.12285"
  doi: null
  s2: null
tags: ["benchmark", "mir", "emotion", "bias"]
added: 2026-10-20T00:47:16Z
---

# CMI-Bench: A Comprehensive Benchmark for Evaluating Music Instruction Following

## One-line thesis
Recasts MIR annotations (incl. emotion regression/tagging) as instruction-following tasks; large LLM-vs-supervised gaps and cultural/chronological bias.

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

> Recent advances in audio-text large language models (LLMs) have opened new possibilities for music understanding and generation. However, existing benchmarks are limited in scope, often relying on simplified tasks or multi-choice evaluations that fail to reflect the complexity of real-world music analysis. We reinterpret a broad range of traditional MIR annotations as instruction-following formats and introduce CMI-Bench, a comprehensive music instruction following benchmark designed to evaluate audio-text LLMs on a diverse set of music information retrieval (MIR) tasks. These include genre classification, emotion regression, emotion tagging, instrument classification, pitch estimation, key detection, lyrics transcription, melody extraction, vocal technique recognition, instrument performance technique detection, music tagging, music captioning, and (down)beat tracking: reflecting core challenges in MIR research. Unlike previous benchmarks, CMI-Bench adopts standardized evaluation metrics consistent with previous state-of-the-art MIR models, ensuring direct comparability with supervised approaches. We provide an evaluation toolkit supporting all open-source audio-textual LLMs, including LTU, Qwen-audio, SALMONN, MusiLingo, etc. Experiment results reveal significant performance gaps between LLMs and supervised models, along with their culture, chronological and gender bias, highlighting the potential and limitations of current models in addressing MIR tasks. CMI-Bench establishes a unified foundation for evaluating music instruction following, driving progress in music-aware LLMs.

