---
type: paper
node_id: paper:weck2024_muchomusic_evaluating_music
title: "MuChoMusic: Evaluating Music Understanding in Multimodal Audio-Language Models"
authors: ["Benno Weck", "Ilaria Manco", "Emmanouil Benetos", "Elio Quinton", "George Fazekas", "Dmitry Bogdanov"]
year: 2024
venue: "ISMIR 2024"
external_ids:
  arxiv: "2408.01337"
  doi: null
  s2: null
tags: ["benchmark", "music-qa", "mcq", "closest-competitor"]
added: 2026-10-20T00:47:14Z
---

# MuChoMusic: Evaluating Music Understanding in Multimodal Audio-Language Models

## One-line thesis
Human-validated 1,187-question music MCQ benchmark on 644 tracks; finds audio LLMs over-rely on the language modality.

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

> Multimodal models that jointly process audio and language hold great promise in audio understanding and are increasingly being adopted in the music domain. By allowing users to query via text and obtain information about a given audio input, these models have the potential to enable a variety of music understanding tasks via language-based interfaces. However, their evaluation poses considerable challenges, and it remains unclear how to effectively assess their ability to correctly interpret music-related inputs with current methods. Motivated by this, we introduce MuChoMusic, a benchmark for evaluating music understanding in multimodal language models focused on audio. MuChoMusic comprises 1,187 multiple-choice questions, all validated by human annotators, on 644 music tracks sourced from two publicly available music datasets, and covering a wide variety of genres. Questions in the benchmark are crafted to assess knowledge and reasoning abilities across several dimensions that cover fundamental musical concepts and their relation to cultural and functional contexts. Through the holistic analysis afforded by the benchmark, we evaluate five open-source models and identify several pitfalls, including an over-reliance on the language modality, pointing to a need for better multimodal integration. Data and code are open-sourced.

