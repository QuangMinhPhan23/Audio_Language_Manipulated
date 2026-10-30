# Vision, text and image-generation analogues of "a famous entity's identity overrides the evidence" (checked 2026-10-22)

Scope: prior work that reviewers could cite to call two music ideas incremental. Idea A gives audio-LLMs edited versions of famous songs, with and without the title. Idea B prompts music generators with famous song titles. For each paper, four features are scored: (i) counterfactual stimuli built from famous entities, (ii) a correct-name vs wrong-name vs no-name manipulation, (iii) a dose-response on entity popularity or recognition, (iv) mechanistic localisation or mitigation.

Method note: every number below comes from the arXiv abstract or HTML page fetched in this session, unless it is marked otherwise. Where a page did not confirm a venue, the venue is listed as unconfirmed. About 37 search and fetch calls were made, so some appendix figures were read only in part. Those cases are flagged.

## Q1. Vision-language models: what the closest designs do, and whether they have features (i)-(iv)

### Takeaway
"Vision Language Models are Biased" (VLMs-are-Biased) is the closest vision precedent. It edits images of famous subjects (feature i) and shows that adding the correct subject name, either in the image or in the prompt, lowers accuracy by about 4.5-4.75 points (part of feature ii). It has no wrong-name arm and no per-item popularity dose-response. Its mitigations are prompting and background removal, not a mechanism. Visual CounterFact / Pixels-vs-Priors (PvP) and the 2026 "Vision-Default, Prior-Override" paper own the mechanistic and mitigation angle (feature iv), but they work on everyday object categories, not named famous entities, and they report no analysis of how strong the prior is. HallusionBench, NaturalBench, BlindTest, CounterCurate and POPEv2 are general counterfactual or blind-answer benchmarks without name or popularity manipulations. I found no vision paper that combines famous-entity counterfactuals, a wrong-name arm and a popularity dose-response.

### Cited Findings

**VLMs are Biased (Vo et al., arXiv 2505.23941)**
- Stimuli: 1,392 counterfactual images across 7 domains: animals 273 (91 animals with an extra leg added), logos 207, flags 120, chess pieces 144, game boards 84, optical illusions 396, and patterned grids 168 (made from scratch) (Sec. 3, Table 1). — [arXiv HTML 2505.23941](https://arxiv.org/html/2505.23941)
- Headline: mean counting accuracy on the counterfactual images is 17.05% across five VLMs (Table 2, Sec. 4.1). Per domain: animals 2.12%, logos 6.13%, flags 9.25%, chess 26.25%, game boards 2.26%, illusions 50.87%, grids 22.44%. 75.70% of answers match the biased (expected) answer (Fig. 4, Sec. 5). On the yes/no question Q3, accuracy is 25.11% on counterfactual images vs 100% on the originals (Table 3). — [arXiv HTML 2505.23941](https://arxiv.org/html/2505.23941)
- Name written in the image (App. A.9, Table 9b), baseline → with in-image subject name: Gemini-2.5 Pro 16.02→12.04 (−3.98); Sonnet-3.7 16.59→13.78 (−2.81); GPT-4.1 13.88→11.21 (−2.67); o3 18.50→12.09 (−6.41); o4-mini 20.25→13.69 (−6.56); **mean 17.05→12.56 (−4.49)**. Thinking models drop more. — [arXiv HTML 2505.23941](https://arxiv.org/html/2505.23941)
- Name in the text prompt (App. A.21, Tables 24-25, "non-neutral prompts with object names"; only 2 models). Accuracy: Sonnet-3.7 16.59→14.18 (−2.41); o4-mini 20.25→13.16 (−7.09); **mean 18.42→13.67 (−4.75)**. Bias rate (share of expected answers): Sonnet-3.7 76.63→80.03 (+3.40); o4-mini 73.66→80.90 (+7.24); **mean 75.14→80.46 (+5.32)**. — [arXiv HTML 2505.23941](https://arxiv.org/html/2505.23941)
- No wrong-name or misleading-name condition appeared in the parts read (characters 0-250k of a 267k-character page). — [arXiv HTML 2505.23941](https://arxiv.org/html/2505.23941)
- Popularity: the domains are presented in order of decreasing popularity (Sec. 1), but there is no per-item or per-domain dose-response analysis, and accuracy does not fall in step with that order (illusions score highest). — [arXiv HTML 2505.23941](https://arxiv.org/html/2505.23941)
- Mitigation (App. A.10, Table 9c): a "debiased" instruction gives a mean of +1.87 points (17.05→18.92), and a "double-check" prompt gives +2.70 (→19.75). Removing the background raises accuracy by +21.09 points (17.05→38.14%) and lowers the bias rate by −40.58 points (75.70→35.12%) (Sec. 4.4, Table 4). — [arXiv HTML 2505.23941](https://arxiv.org/html/2505.23941)
- A linear probe on the vision encoders reaches 95.26% on leg counting, so the visual information is present but gets overridden (App. A.8). This is a probe, not a localisation of where the override happens. o4-mini with tools rises only from 23.18% to 25.08% (Sec. 5). — [arXiv HTML 2505.23941](https://arxiv.org/html/2505.23941)
- Features: (i) YES; (ii) correct-name vs no-name only, with no wrong-name arm; (iii) NO; (iv) mitigation by prompting and background removal plus a probe, with no localisation inside the model.

**Visual CounterFact / "Pixels Versus Priors" (Golovanevsky, Rudman, Lepori, Bar, Singh, Eickhoff; arXiv 2505.17127)**
- 2,904 images: 575 colour originals and 575 colour counterfactuals, 877 size originals and 877 size counterfactuals. The objects come from McRae norms (190 objects where at least 30% of people listed a colour) plus ImageNet and CIFAR-100 classes coloured by GPT-4o, giving 622 objects and 575 after filtering. Size pairs differ by at least 10x (App. A). — [arXiv HTML 2505.17127](https://arxiv.org/html/2505.17127)
- Baseline (Table 1): when asked what colour/size an object "most" commonly is, while shown the counterfactual image, accuracy collapses. Examples: LLaVA-Next colour 47.26% (vs 92.09% on the original image), Qwen2-VL size 28.34% (vs 96.29%), Janus-Pro size 18.02% (vs 96.01%). Here the image overrides the prior, which is the opposite direction from the music concern. — [arXiv HTML 2505.17127](https://arxiv.org/html/2505.17127)
- Mechanism: in layer-by-layer decoding, the world-knowledge answer rises in mid-to-late layers and then flips to the counterfactual answer in the final layers (Sec. 5.2). The PvP steering vectors flip world-knowledge answers to counterfactual ones in 99.3% (colour) and 80.8% (size) of cases on average (Table 3; for example Qwen2-VL colour 99.7% at layers 17-19). — [arXiv abs 2505.17127](https://arxiv.org/abs/2505.17127); [HTML](https://arxiv.org/html/2505.17127)
- No named famous entities, no name manipulation, and no analysis of prior strength or frequency. The fetched pages did not confirm the venue (v2 dated 29 Sep 2025). "EMNLP 2025" is UNVERIFIED here. — [arXiv abs 2505.17127](https://arxiv.org/abs/2505.17127)
- Features: (i) NO (object categories); (ii) NO; (iii) NO; (iv) YES.

**"Vision-Default, Prior-Override" (Lietzow, Bitterman, Eickhoff, Rudman, Golovanevsky; arXiv 2606.28273, 26 Jun 2026)**
- A follow-up on Visual CounterFact recoloured objects (a search summary says 469 counterfactual colour images). Heads that are causally necessary make up 2.5-4.8% of all heads across three VLM families. Ablating them flips knowledge-based answers to vision-based ones in 68-96% of cases, but changes only 0.8-7.5% of vision-based predictions. These are late-layer "routing" and "writing" heads. — [arXiv abs 2606.28273](https://arxiv.org/abs/2606.28273)
- Features: (i) NO; (ii) NO; (iii) not mentioned in the abstract; (iv) YES (localisation by activation patching).

**HallusionBench (Guan et al., arXiv 2310.14566; the page says "Accepted to CVPR 2024")**
- 346 images (165 original, 181 human-edited) and 1,129 questions: 178 with no image, 447 on original images, 504 on edited images. 591 questions are visual-dependent and 538 visual-supplement. Edits include flipping, reversing order, masking, editing text in the image, editing objects and editing colour. The examples include edited maps, charts and famous illusions (Sec. 3.3, Table 1). — [arXiv HTML 2310.14566](https://arxiv.org/html/2310.14566)
- GPT-4V: 31.42% question-pair accuracy, and every other model is below 16% (abstract). On easy vs hard questions ("Hard questions refer to the edited images"), GPT-4V scores 79.56% vs 38.37% (Table 2). Failure types for GPT-4V: language hallucination 21.86%, visual illusion 46.17% (Table 3). — [arXiv abs](https://arxiv.org/abs/2310.14566); [HTML](https://arxiv.org/html/2310.14566)
- Features: (i) PARTIAL (well-known charts, maps and illusions, edited); (ii) NO; (iii) NO; (iv) NO.

**NaturalBench (Li, Lin, ..., Neubig, Ramanan; arXiv 2410.14669; NeurIPS 2024)**
- 10,000 human-verified VQA samples. Each question is paired with two images that have different answers, so it cannot be answered without looking. 53 VLMs were tested, and models such as GPT-4o, Qwen2-VL and Molmo trail humans (over 90%) by 50-70%. Models often give the same answer whatever the image. — [arXiv abs 2410.14669](https://arxiv.org/abs/2410.14669)
- Features: (i) NO; (ii) NO; (iii) NO; (iv) NO. This is relevant only as a design for defeating blind answering. Note: arXiv 2410.04884 is a different paper.

**"Vision language models are blind" / BlindTest (Rahmanzadehgervi, Bolton, Taesiri, Nguyen; arXiv 2407.06581)**
- 7 simple geometric tasks, such as whether two circles overlap, counting line crossings, identifying a circled letter, and counting circles in an Olympic-like logo. Four leading VLMs average 58.07%, the best (Claude 3.5 Sonnet) reaches 77.84%, and humans are expected to reach 100%. — [arXiv abs 2407.06581](https://arxiv.org/abs/2407.06581)
- Features: (i) NO beyond the Olympic-like logo; (ii) NO; (iii) NO; (iv) NO. The abstract page lists no venue.

**CounterCurate (Zhang, Cai, Xie, Lee; Findings of ACL 2024; arXiv 2402.13254)**
- Physical counterfactuals (positions via keyword swap plus image flip; counting via GLIGEN inpainting) and semantic counterfactuals (GPT-4V/DALLE-3) used for *training* augmentation. CLIP and LLaVA start near chance; the method adds +33% (CLIP) and +37% (LLaVA) on Flickr30k-Positions. — [arXiv 2402.13254](https://arxiv.org/html/2402.13254v4); [ACL Anthology](https://preview.aclanthology.org/setup/2024.findings-acl.915)
- Features: (i) NO; (ii) NO; (iii) NO; (iv) mitigation by training only.

**POPEv2 / "Obliviate" (Li, Zhou, Zhao, Fang, Wen; arXiv 2508.04567)**
- Counterfactual images made from LVLM *training* images with some objects masked. Models hallucinate more on images they saw in training and often answer "yes" for the masked objects. Probing places the bias in the language-model head, and the fix (Obliviate) fine-tunes about 2% of the parameters (2B-72B models). — [arXiv abs 2508.04567](https://arxiv.org/abs/2508.04567)
- Features: (i) NO (but "seen-in-training" is a kind of familiarity contrast); (ii) NO; (iii) NO; (iv) YES. No venue on the page.

**MMKC-Bench (Jia et al., arXiv 2505.19509)**
- 1,573 knowledge instances and 3,381 images across 23 types, covering multimodal context-memory and inter-context conflicts built by counterfactual editing. A search summary says this includes changing the entity name. The finding: LMMs can recognise conflicts but "tend to favor internal parametric knowledge over external evidence." No popularity analysis appears in the abstract. — [arXiv abs 2505.19509](https://arxiv.org/abs/2505.19509); [search snippet of HTML](https://arxiv.org/html/2505.19509v1)
- Features: (i) PARTIAL; (ii) PARTIAL (name or knowledge swapped, but not a crossed correct/wrong/no-name design, as far as the abstract shows); (iii) NO; (iv) unknown.

**Popularity in vision (recognition, not override)**
- PopVQA ("Performance Gap in Entity Knowledge Extraction Across Modalities in VLMs", arXiv 2412.14133, ACL 2025): 15,395 entity-image pairs covering celebrities, landmarks, paintings and brand logos. Factual QA accuracy drops by up to 18% when the entity is shown in an image instead of named in text. Image information flows in only at deeper layers. — [arXiv 2412.14133](https://arxiv.org/pdf/2412.14133); [ACL listing](https://papers.cool/venue/2025.acl-long.1411@ACL)
- InfoSeek (arXiv 2302.11713) splits entities by Wikipedia monthly page views (following Mallen et al.). End-to-end models do better than pipelines on rarely viewed (tail) entities, but pipelines win on popular ones. — [arXiv 2302.11713](https://arxiv.org/pdf/2302.11713)
- SnapNTell (Findings of EMNLP 2024): 7,568 entities in 22 categories. VLMs struggle with long-tail entities. — [arXiv 2403.04735](https://arxiv.org/pdf/2403.04735)

**ROHE (NAACL 2025)**: UNVERIFIABLE. Searches found no paper by this name. Give the full title or authors to resolve it.

**Other 2025-2026 famous-entity edits**: one targeted search for edited famous paintings or landmarks and one for celebrity-face or name conflicts found no paper that edits famous artworks, landmarks or celebrities *and* varies the name (beyond VLMs-are-Biased and MMKC-Bench). — [search results incl. 2606.28273, 2505.23941](https://arxiv.org/html/2606.28273)

### Inferences
- A reviewer will cite VLMs-are-Biased as the direct precedent for Idea A ("edited famous stimulus + name cue lowers accuracy"). Idea A's room for novelty is (a) a wrong-name arm (an edited song paired with a different famous title) that separates "any famous name" from "the matching identity", and (b) a per-item popularity dose-response. Neither appeared in VLMs-are-Biased in the parts read.
- The vision mechanistic line (PvP, Vision-Default/Prior-Override) uses generic object priors (a strawberry is red), not identity priors. A music paper that only adds a popularity curve, without mechanism or mitigation, would sit behind that line on feature (iv).
- The direction of the effect matters. In Visual CounterFact, the *image* overrides the prior when the question asks about the prior. In VLMs-are-Biased, the *prior* overrides the image. Audio models could go either way, so both question framings ("what do you hear" vs "what is the song usually like") are worth including.

### Gaps
- The VLMs-are-Biased venue (ICLR 2026) was not confirmed on the fetched page. The last ~17k characters of the appendix were not read, so a wrong-name arm in that tail cannot be ruled out with certainty.
- The Visual CounterFact venue (EMNLP 2025) was not confirmed.
- ROHE could not be identified.
- MMKC-Bench: it is unclear whether it crosses name with image the way Idea A would (full text not read).
- "VLMs see what they expect" and "seeing is not believing" were not found as specific papers within the search budget.

## Q2. Text LLMs: knowledge conflicts and popularity or memorisation dose-response

### Takeaway
In text, it is already established that popularity modulates how often the model's memory overrides the context. Xie et al. ("Adaptive Chameleon or Stubborn Sloth") show the memorisation ratio rising with PopQA popularity (GPT-4 about 80% for the most popular questions). Mallen et al. (PopQA) show accuracy tracking log popularity and scaling helping mainly popular entities. Chang et al. ("Speak, Memory") show name-cloze memorisation tracking web frequency (Spearman ρ up to 0.84) and memorised books inflating downstream scores. Longpre et al. 2021 started the entity-substitution knowledge-conflict paradigm and also bucket substitute popularity. A popularity dose-response therefore cannot be claimed as conceptually new. Only its transfer to audio and its pairing with counterfactual stimuli can be.

### Cited Findings

**Longpre et al. 2021, "Entity-Based Knowledge Conflicts for QA" (EMNLP 2021, arXiv 2109.05052)**
- The answer entity in the context passage is replaced in four ways: corpus substitution, type swap, *popularity substitution* (a same-type Wikidata entity within set Wikipedia monthly page-view bounds) and alias substitution (§2.2). — [arXiv HTML 2109.05052v2](https://arxiv.org/html/2109.05052v2)
- Memorisation ratio = p_o / (p_o + p_s), the rate of giving the original answer over the original plus substitute. T5 reader before mitigation: NQ train 29.5, NQ dev with answer overlap 27.1, dev without overlap 1.5, NewsQA 9.3 (Table 4). Across T5 sizes from 60M to 11B, the ratio rises from below 15% to at least 50% (Fig. 5). — [arXiv HTML 2109.05052v2](https://arxiv.org/html/2109.05052v2)
- Popularity (§4.2, Fig. 8, PER answers only, 5 buckets): substitutes with higher popularity are predicted more often on NQ train and dev with overlap; popularity predicts less on dev without overlap. The text gives no percentages. Note: this is the popularity of the *substitute*, not of the original entity. — [arXiv HTML 2109.05052v2](https://arxiv.org/html/2109.05052v2)
- Mitigation: training on substituted data (about 25% extra) cuts the ratio from 29.5→2.6 (train), 27.1→1.9, 1.5→0.0 and 9.3→0.6, while exact match on out-of-domain data rises by +7.1 (NQ no overlap) and +4.4 (NewsQA) (Table 4). — [arXiv HTML 2109.05052v2](https://arxiv.org/html/2109.05052v2)
- Features: (i) YES (text counterfactual); (ii) not a name arm (the answer is swapped); (iii) PARTIAL (substitute popularity); (iv) mitigation YES.

**Xie et al., "Adaptive Chameleon or Stubborn Sloth" (arXiv 2305.13300; the page did not show a venue, but it is commonly cited as ICLR 2024, UNVERIFIED here)**
- Counter-memory is built by eliciting the model's closed-book answer, then substituting the entity (PopQA) or flipping the answer (StrategyQA), with a ChatGPT-written supporting passage checked by an NLI model. — [arXiv HTML 2305.13300](https://arxiv.org/html/2305.13300)
- **Popularity dose-response (§4.2, Fig. 3):** the memorisation ratio rises with question popularity (popularity = monthly Wikipedia page views). GPT-4 reaches about 80% for the most popular questions, read as stronger confirmation bias for facts seen more often in pretraining (ChatGPT, GPT-4, PaLM2, Llama2-7B). — [arXiv HTML 2305.13300](https://arxiv.org/html/2305.13300)
- Order effects: apart from GPT-4, models change by more than 5% with evidence order, and PaLM2 and Llama2-7B by more than 30% (Table 5). The memorisation ratio also rises with the amount of evidence agreeing with memory (Table 6). — [arXiv HTML 2305.13300](https://arxiv.org/html/2305.13300)
- Features: (i) YES (text); (ii) NO; (iii) **YES**; (iv) NO.

**Mallen et al. 2023, PopQA, "When Not to Trust Language Models" (ACL 2023, arXiv 2212.10511)**
- About 14k questions (14,282 run) over 16 relations. Popularity = monthly Wikipedia page views of the subject. — [arXiv HTML 2212.10511](https://arxiv.org/html/2212.10511)
- The correlation between log popularity and correctness is about 0.4 for GPT-3 davinci-003 and about 0.1 for GPT-Neo 1.3B. Scaling helps mainly when log10(popularity) is above 4. On the 4,000 least popular questions, GPT-3 scores 19% (35% overall). — [arXiv HTML 2212.10511](https://arxiv.org/html/2212.10511)
- Retrieval hurts in 10% of questions (GPT-3 right without retrieval, wrong with it) and fixes 17%. Adaptive retrieval (retrieving only below a popularity threshold) reaches 46.5%, +5.3 over the best non-adaptive method. — [arXiv HTML 2212.10511](https://arxiv.org/html/2212.10511)
- Features: (i) NO; (ii) NO; (iii) **YES** (the standard popularity dose-response design); (iv) mitigation by adaptive retrieval.

**Chang et al. 2023, "Speak, Memory" (EMNLP 2023, arXiv 2305.00118)**
- 571 novels (1749-2020). Name-cloze (predict a masked character name from a passage): Alice in Wonderland GPT-4 0.98 and ChatGPT 0.82; Harry Potter 0.76 and 0.43; the most-frequent-name baseline is 0.6% and humans score 0% (Table 1, §3.2). — [arXiv HTML 2305.00118](https://arxiv.org/html/2305.00118)
- **Web-frequency dose-response (Table 3, §6.2)**: Spearman ρ between GPT-4 accuracy and hit counts. Pre-1928 books: Google 0.74, Bing 0.70, C4 0.71, Pile 0.84. Post-1928: 0.37, 0.41, 0.36, 0.21. — [arXiv HTML 2305.00118](https://arxiv.org/html/2305.00118)
- Downstream contamination (Table 5, §7.1), predicting year of publication: GPT-4 mean absolute error is 0.6 years for the top 10% most memorised books vs 14.5 years for the bottom 10%; ChatGPT 3.3 vs 29.8 years. Narrative-time prediction shows no significant gap (§7.2). — [arXiv HTML 2305.00118](https://arxiv.org/html/2305.00118)
- Features: (i) NO; (ii) the name *is* the probe target (name-cloze); (iii) **YES**; (iv) NO.

**ConflictBank (Su et al., arXiv 2408.12076; the page shows no venue, "NeurIPS 2024 D&B" UNVERIFIED here)**
- 2,863,205 claims from Wikidata, 7,453,853 claim-evidence pairs and 553,117 QA pairs. Conflict types: misinformation, temporal and semantic (§2). — [arXiv HTML 2408.12076](https://arxiv.org/html/2408.12076)
- The memorisation ratio is below 50% with only conflicting evidence and above 50% when default evidence is also present (§3.2). Larger LLaMA2 models have lower ratios (−5.12 from 7B to 13B, −6.37 from 13B to 70B). Adding descriptions to questions cuts LLaMA3-70B's ratio from 67.52% to 8.31% (§3.5). **No entity-popularity analysis.** — [arXiv HTML 2408.12076](https://arxiv.org/html/2408.12076)
- Features: (i) YES (text); (ii) NO; (iii) NO; (iv) NO.

**Knowledge-conflict survey (Xu et al., "Knowledge Conflicts for LLMs: A Survey", EMNLP 2024 main, 2024.emnlp-main.486, pp. 8541-8565)**
- Sorts conflicts into context-memory, inter-context and intra-memory, and reviews their causes, model behaviour and remedies. — [ACL Anthology](https://preview.aclanthology.org/setup/2024.emnlp-main.486); [arXiv 2403.08319](https://arxiv.org/html/2403.08319)

### Inferences
- "Prior strength rises with popularity, so the override grows" is a known text result (Xie et al.; Mallen et al.; Chang et al.). A reviewer will expect the music papers to cite these and frame popularity as a test of whether that known law carries over to audio. They should not present it as a discovery.
- What the text literature does *not* do is change the perceptual signal of the entity itself while holding or swapping its name. That crossed design (edited stimulus × correct/wrong/no title × popularity) is the only combination not found in text or vision.
- Speak-Memory's name-cloze is a ready recognition measure. A "title-from-lyrics/description" probe for songs would be its direct analogue and would give the popularity axis independently of chart data.

### Gaps
- Exact per-bucket numbers for Longpre's Fig. 8 and Xie's Fig. 3 exist only in the plots, apart from Xie's "about 80%".
- Venues for Xie et al. (ICLR 2024) and ConflictBank (NeurIPS 2024 D&B) were not confirmed on the fetched pages.

## Q3. Image and video generation analogues for Idea B (prompting generators with famous titles or names)

### Takeaway
Memorisation in text-to-image models is tied to how often an item is repeated in the training data, which is a frequency dose-response. Carlini et al. find extraction mostly for images duplicated at least 100 times. Somepalli et al. find replicated images had about 34 duplicates vs about 12 for random ones. "How Many Van Goghs" estimates an imitation threshold of about 200-700 training images per concept (faces, art styles). "Memory Triggers" shows title-like phrases ("starry night") pulling in a memorised artwork. "Art-Free" shows that title knowledge comes from art in the training data, not from the text encoder alone. So "generators reproduce famous items from their names, and more so for more frequent items" is established for images. Idea B is new mainly in the music medium. I found no video-generation analogue within the search budget.

### Cited Findings
- **Carlini et al. 2023, "Extracting Training Data from Diffusion Models" (arXiv 2301.13188; the page shows no venue).** Stable Diffusion: the 350,000 most-duplicated training captions were used as prompts, with 500 generations each (about 175M images). 94 images met the (ℓ2, 0.15) memorisation definition and manual review added 13 more, giving 109. About 58% are photos of a recognisable person, 17% products, and 14% logos or posters (Sec. 4.2.2). There was little memorisation for duplication counts below 100, and most extraction came from images duplicated at least 100 times (Fig. 5). Imagen: 23 of the top 1,000 duplicated prompts were memorised (Sec. 4.3). Example: the prompt "Ann Graham Lotz" (a person's name) reproduces her training photo (ℓ2 0.031, Fig. 1). Prompts were chosen by *duplication*, not by name. — [arXiv HTML 2301.13188v1](https://arxiv.org/html/2301.13188v1)
- **Somepalli et al., "Diffusion Art or Digital Forgery?" (arXiv 2212.03860; the page shows no venue, CVPR 2023 UNVERIFIED here).** 9,000 LAION-Aesthetics captions with one generation each. About 1.88% of generations have SSCD similarity above 0.5 to a training image in the 12M-image subset searched, which the authors call an underestimate (Sec. 7). Famous artworks: prompts based on The Great Wave reproduce its wave structure, and Starry Night and The Scream also appear (Fig. 10, Fig. 12, Tables 3-4). Duplication: close matches average about 34.1 duplicates vs about 11.6 for random images (Fig. 11). — [arXiv HTML 2212.03860v3](https://arxiv.org/html/2212.03860v3)
- **"How Many Van Goghs Does It Take to Van Gogh? Finding the Imitation Threshold" (Verma et al., arXiv 2410.15002; NeurIPS 2024 workshop listing).** For human faces and art styles, an imitation score is computed per concept and plotted against the estimated count of that concept in the training data (for example about 213K for Van Gogh). A change-point method then finds the threshold: about 200-600 images (v1) or 200-700 (later version), depending on domain and model. — [arXiv 2410.15002](https://arxiv.org/abs/2410.15002v1); [NeurIPS listing](https://neurips.cc/virtual/2024/106951); [AI2 blog](https://allenai.org/blog/imitation-threshold)
- **"Memory Triggers" (Naseh, Roh, Houmansadr; arXiv 2312.03692).** Duplication of single words or phrases in captions triggers memorisation. A search summary (not checked against the paper) says the prompt "Van Gogh starry night" gave 64.6% of images with similarity above 0.83 to the original. — [arXiv 2312.03692](https://arxiv.org/pdf/2312.03692); [search summary source](https://lacuna.tiptreesystems.com/work/memory-triggers-unveiling-memorization-in-text-to-image-generative-models/wrk_c8e9e91a2134765a40c6036ed655144d)
- **"Art-Free Generative Models" / "Opt-In Art" (Ren, Materzynska, Gandikota, Bau, Torralba; arXiv 2412.00176).** A text-to-image model trained without art content. A search summary (not checked against the paper) says it fails to produce the Mona Lisa or Starry Night from their titles, while SD1.4 reproduces them. That would mean title-to-artwork recall comes from art in the image training data, not from the text encoder. — [arXiv 2412.00176](https://arxiv.org/abs/2412.00176v1); [HF page](https://huggingface.co/papers/2412.00176)
- Related mechanism and mitigation: cross-attention concentrates on "trigger" token embeddings during memorised generations, which is used for detection and mitigation (arXiv 2403.11052). "Finding Dori" (arXiv 2507.16880) argues memorisation is not localised and can be re-triggered by small changes to the text embedding. — [arXiv 2403.11052](https://arxiv.org/pdf/2403.11052); [arXiv 2507.16880](https://arxiv.org/abs/2507.16880v2)

### Inferences
- For Idea B, a reviewer can cite Carlini, Somepalli and the imitation threshold as "generators regurgitate frequent training items, with a frequency threshold". A music version needs a reason why music differs (melody/lyrics identity vs timbre/style) or a new measurement, such as which musical dimensions are reproduced (melody, harmony, timbre, lyrics) as a function of chart popularity. A plain title-prompt replication study would likely be judged incremental.
- The Art-Free result gives a ready control for Idea B: compare title prompts with matched descriptive prompts (no title) to separate title-triggered memory from following the description.

### Gaps
- No study prompting video generators with famous titles was found within the budget.
- Memory Triggers' 64.6% and Art-Free's Mona Lisa result come from search summaries and were not checked against the papers.
- A paper titled "knowing without seeing", and concept-erasure work on text-encoder knowledge (e.g., Gandikota et al. ESD), were not checked.

## Q4. Does any prior work already establish that popularity or recognition modulates the override?

### Takeaway
In text and in image generation, yes. In vision understanding (the closest analogue to Idea A), not with counterfactual stimuli: popularity has been studied only for recognition and knowledge accuracy (PopVQA, InfoSeek, SnapNTell), not for how strongly a prior overrides an edited image. VLMs-are-Biased orders its domains by popularity but does not test a dose-response.

### Cited Findings
- Text, override grows with popularity: Xie et al. §4.2/Fig. 3 (memorisation ratio rises with popularity, GPT-4 about 80% at the top) — [arXiv HTML 2305.13300](https://arxiv.org/html/2305.13300). Longpre et al. Fig. 8 (substitute popularity buckets) — [arXiv HTML 2109.05052v2](https://arxiv.org/html/2109.05052v2).
- Text, memorisation or accuracy grows with popularity: PopQA (correlation about 0.4, scaling helps when log10 popularity is above 4) — [arXiv HTML 2212.10511](https://arxiv.org/html/2212.10511). Speak-Memory (ρ 0.74-0.84 pre-1928) — [arXiv HTML 2305.00118](https://arxiv.org/html/2305.00118).
- Image generation, replication grows with frequency: Carlini (at least 100 duplicates) — [arXiv HTML 2301.13188v1](https://arxiv.org/html/2301.13188v1). Somepalli (34.1 vs 11.6 duplicates) — [arXiv HTML 2212.03860v3](https://arxiv.org/html/2212.03860v3). Imitation threshold of 200-700 images — [arXiv 2410.15002](https://arxiv.org/abs/2410.15002v1).
- Vision understanding, popularity affects recognition but not override: InfoSeek (page-view split) — [arXiv 2302.11713](https://arxiv.org/pdf/2302.11713). PopVQA (up to 18% image-vs-text gap) — [arXiv 2412.14133](https://arxiv.org/pdf/2412.14133). VLMs-are-Biased, domains ordered by popularity with no dose-response test — [arXiv HTML 2505.23941](https://arxiv.org/html/2505.23941).

### Summary matrix (features i-iv)

| Work | (i) famous-entity counterfactual | (ii) correct / wrong / no name | (iii) popularity dose-response | (iv) mechanism / mitigation |
|---|---|---|---|---|
| VLMs are Biased (2505.23941) | Yes | Correct vs none only (−4.49 in-image; −4.75 in-prompt) | No (domains only) | Prompting and background removal (+21.09); probe |
| Visual CounterFact / PvP (2505.17127) | No (object categories) | No | No | Yes (steering 99.3% / 80.8%) |
| Vision-Default/Prior-Override (2606.28273) | No | No | No (abstract) | Yes (2.5-4.8% of heads; 68-96% flips) |
| HallusionBench (2310.14566) | Partial (edited charts, maps, illusions) | No | No | No |
| NaturalBench (2410.14669) | No | No | No | No |
| BlindTest (2407.06581) | No | No | No | No |
| CounterCurate (2402.13254) | No | No | No | Training augmentation |
| POPEv2 (2508.04567) | No (masked training images) | No | No | Yes (LM head, about 2% of parameters) |
| MMKC-Bench (2505.19509) | Partial | Partial (name edits) | No | Unknown |
| PopVQA (2412.14133) | No | No (image vs text name) | Recognition only | Information-flow analysis |
| Longpre 2021 (2109.05052) | Text | No | Partial (substitute) | Yes (ratio 29.5→2.6) |
| Xie et al. (2305.13300) | Text | No | **Yes** | No |
| PopQA (2212.10511) | No | No | **Yes** | Adaptive retrieval |
| Speak, Memory (2305.00118) | No | Name-cloze probe | **Yes** | No |
| ConflictBank (2408.12076) | Text | No | No | No |
| Carlini 2023 (2301.13188) | n/a (generation) | Name prompts incidental | **Yes** (duplication) | Deduplication |
| Somepalli (2212.03860) | n/a | Title-like prompts | **Yes** (duplication) | No |
| Imitation threshold (2410.15002) | n/a | Concept names | **Yes** (threshold) | No |

### Inferences
- For Idea A, novelty can be claimed only for the *combination*: edited famous stimulus × correct/wrong/no title × per-item popularity (chart or streaming data) × audio. Each part alone has a precedent: (i)+(ii-partial) in VLMs-are-Biased, (iii) in Xie/PopQA, (iv) in PvP. A wrong-name arm is the cleanest unclaimed piece in the vision literature checked here.
- For Idea B, the frequency-to-memorisation dose-response is established in images. A music paper should present popularity as a replication of that law in a new medium, or move the claim to what is reproduced (melody vs timbre vs lyrics) and to controls such as title vs description.

### Gaps
- No exhaustive 2025-2026 sweep was possible within the budget. A vision paper with a wrong-name arm or a popularity curve on counterfactual famous images may exist and was not found. A targeted search of OpenReview for ICLR 2026 / CVPR 2026 using "famous", "counterfactual" and "popularity" is recommended before submission.
