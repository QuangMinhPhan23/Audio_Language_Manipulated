---
type: paper
node_id: paper:ma2025_mmar_challenging_benchmark
title: "MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix"
authors: ["Ziyang Ma", "Yinghao Ma", "Yanqiao Zhu", "Chen Yang", "Yi-Wen Chao", "Ruiyang Xu", "Wenxi Chen", "Yuanzhe Chen", "Zhuo Chen", "Jian Cong", "Kai Li", "Keliang Li", "Siyou Li", "Xinfeng Li", "Xiquan Li", "Zheng Lian", "Yuzhe Liang", "Minghao Liu", "Zhikang Niu", "Tianrui Wang", "Yuping Wang", "Yuxuan Wang", "Yihao Wu", "Guanrou Yang", "Jianwei Yu", "Ruibin Yuan", "Zhisheng Zheng", "Ziya Zhou", "Haina Zhu", "Wei Xue", "Emmanouil Benetos", "Kai Yu", "Eng-Siong Chng", "Xie Chen"]
year: 2025
venue: "NeurIPS 2025 D&B"
external_ids:
  arxiv: "2505.13032"
  doi: null
  s2: null
tags: ["benchmark", "audio-reasoning", "cultural-layer", "neurips-db"]
added: 2026-10-20T00:47:16Z
---

# MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix

## One-line thesis
1,000 real-world audio QA items layered Signal/Perception/Semantic/Cultural with CoT rationales.

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

> We introduce MMAR, a new benchmark designed to evaluate the deep reasoning capabilities of Audio-Language Models (ALMs) across massive multi-disciplinary tasks. MMAR comprises 1,000 meticulously curated audio-question-answer triplets, collected from real-world internet videos and refined through iterative error corrections and quality checks to ensure high quality. Unlike existing benchmarks that are limited to specific domains of sound, music, or speech, MMAR extends them to a broad spectrum of real-world audio scenarios, including mixed-modality combinations of sound, music, and speech. Each question in MMAR is hierarchically categorized across four reasoning layers: Signal, Perception, Semantic, and Cultural, with additional sub-categories within each layer to reflect task diversity and complexity. To further foster research in this area, we annotate every question with a Chain-of-Thought (CoT) rationale to promote future advancements in audio reasoning. Each item in the benchmark demands multi-step deep reasoning beyond surface-level understanding. Moreover, a part of the questions requires graduate-level perceptual and domain-specific knowledge, elevating the benchmark's difficulty and depth. We evaluate MMAR using a broad set of models, including Large Audio-Language Models (LALMs), Large Audio Reasoning Models (LARMs), Omni Language Models (OLMs), Large Language Models (LLMs), and Large Reasoning Models (LRMs), with audio caption inputs. The performance of these models on MMAR highlights the benchmark's challenging nature, and our analysis further reveals critical limitations of understanding and reasoning capabilities among current models. We hope MMAR will serve as a catalyst for future advances in this important but little-explored area.

