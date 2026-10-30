---
type: paper
node_id: paper:ghosh2025_music_flamingo_scaling
title: "Music Flamingo: Scaling Music Understanding in Audio Language Models"
authors: ["Sreyan Ghosh", "Arushi Goel", "Lasha Koroshinadze", "Sang-gil Lee", "Zhifeng Kong", "Joao Felipe Santos", "Ramani Duraiswami", "Dinesh Manocha", "Wei Ping", "Mohammad Shoeybi", "Bryan Catanzaro"]
year: 2025
venue: "arXiv 2025"
external_ids:
  arxiv: "2511.10289"
  doi: null
  s2: null
tags: ["model", "music-lalm", "baseline-candidate", "cultural-context"]
added: 2026-10-20T00:47:17Z
---

# Music Flamingo: Scaling Music Understanding in Audio Language Models

## One-line thesis
Audio Flamingo 3-based music LALM trained on MF-Skills (harmony, structure, timbre, lyrics, culture) + MF-Think CoT + GRPO; SOTA on 10+ music benchmarks.

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

> We introduce Music Flamingo, a novel large audio-language model designed to advance music (including song) understanding in foundational audio models. While audio-language research has progressed rapidly, music remains challenging due to its dynamic, layered, and information-dense nature. Progress has been further limited by the difficulty of scaling open audio understanding models, primarily because of the scarcity of high-quality music data and annotations. As a result, prior models are restricted to producing short, high-level captions, answering only surface-level questions, and showing limited generalization across diverse musical cultures. To address these challenges, we curate MF-Skills, a large-scale dataset labeled through a multi-stage pipeline that yields rich captions and question-answer pairs covering harmony, structure, timbre, lyrics, and cultural context. We fine-tune an enhanced Audio Flamingo 3 backbone on MF-Skills and further strengthen multiple skills relevant to music understanding. To improve the model's reasoning abilities, we introduce a post-training recipe: we first cold-start with MF-Think, a novel chain-of-thought dataset grounded in music theory, followed by GRPO-based reinforcement learning with custom rewards. Music Flamingo achieves state-of-the-art results across 10+ benchmarks for music understanding and reasoning, establishing itself as a generalist and musically intelligent audio-language model. Beyond strong empirical results, Music Flamingo sets a new standard for advanced music understanding by demonstrating how models can move from surface-level recognition toward layered, human-like perception of songs. We believe this work provides both a benchmark and a foundation for the community to build the next generation of models that engage with music as meaningfully as humans do.

