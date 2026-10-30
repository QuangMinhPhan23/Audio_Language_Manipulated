# Legal record on AI music generators reproducing named songs, and legal risk for an Australian student researcher running Ideas A and B

Status date: 2026-10-22. These notes are research notes, not legal advice. Each item is marked either **[Fact]** (backed by the cited source) or **[Opinion/Inference]** (my reasoning). Wherever the answer depends on how UNSW reads the law, the researcher must ask UNSW: copyright@unsw.edu.au for copyright, humanethics@unsw.edu.au for ethics, and the UNSW Legal Office for specific legal advice.

## Q1. GEMA v. Suno, UMG v. Suno, and the wider legal record (as of 2026-10-22)

### Takeaway
The novelty doc's GEMA v. Suno claims check out: 31 July 2026, Case 42 O 763/25, prompts of lyrics, style and title, 4 to 176 repetitions, the court did the comparison itself, and fines of up to €250,000 per breach. The prompts **included the original lyrics**, though, so GEMA is not a "title-only" result. In UMG v. Suno, the "Johnny B. Goode" (29 outputs) and "Sway" examples come from the **original June 2024 complaint**. That complaint said outright that it was *not* claiming the outputs themselves infringe; the outputs were offered as evidence that the songs were in the training data. As of late September 2026 there was no US ruling on the merits of AI music outputs.

### Cited Findings

**GEMA v. Suno (Landgericht München I, the Munich I Regional Court)**
- [Fact] Judgment of 31 July 2026, Case No. 42 O 763/25, Munich I Regional Court — [KPW Law](https://kpw.law/en/gema-also-wins-against-suno/); same date and case number in [GEMA press release](https://www.gema.de/en/w/suno-entscheidung-2026) / [MIZ](https://miz.org/en/news/gema-sieht-durch-urteil-gegen-suno-massstabe-fur-internationales-urheberrecht-gesetzt).
- [Fact] Timeline: claim filed 21 January 2025; oral hearing 9 March 2026; judgment 31 July 2026 — [CMS case tracker](https://cms.law/en/int/publication/artificial-intelligence-and-copyright-case-tracker/germany-gema-v.-suno-31-july-2026).
- [Fact] Six works: "Atemlos durch die Nacht", "Rasputin", "Big in Japan", "Forever Young", the chorus of "Mambo No. 5", and "Daddy Cool". KPW says infringements caused by the lyrics were "expressly excluded" from the claim — [KPW Law](https://kpw.law/en/gema-also-wins-against-suno/).
- [Fact] Prompts: GEMA staff entered "the respective original text, the desired music style, and the title of the work" — [KPW Law](https://kpw.law/en/gema-also-wins-against-suno/). Other sources agree: "the original song lyrics, the title of the work and a desired musical style … No specifications were given regarding melody, harmony, rhythm or arrangement" — [CMS](https://cms.law/en/int/publication/artificial-intelligence-and-copyright-case-tracker/germany-gema-v.-suno-31-july-2026); "simple prompts" naming title, lyrics and style — [Simmons & Simmons](https://www.simmons-simmons.com/en/publications/cmszvaaz6003oug849s2sqawl/gema-v-suno-ai-key-eu-decision-on-ai-training-outputs-and-copyright); [Bristows](https://inquisitiveminds.bristows.com/post/102ngsl/gema-v-suno-munich-court-finds-ai-music-training-and-outputs-infringe-copyright).
- [Fact] Repetitions: identical prompts were repeated 4 to 176 times. "Atemlos" took 176, "Big in Japan" 124, and the other works 14, 12, 8 and 4 — [PPC Land](https://ppc.land/suno-loses-gema-case-as-munich-court-sets-250-000-euro-penalty-per-breach/); "between 4 and 176 times" — [Simmons & Simmons](https://www.simmons-simmons.com/en/publications/cmszvaaz6003oug849s2sqawl/gema-v-suno-ai-key-eu-decision-on-ai-training-outputs-and-copyright).
- [Fact] The court made the musical comparison itself and refused the defendant's requests for expert evidence — [PPC Land](https://ppc.land/suno-loses-gema-case-as-munich-court-sets-250-000-euro-penalty-per-breach/). KPW, Simmons and CMS do not say who did the comparison, so this detail rests on one secondary source.
- [Fact] Findings:
  - "The musical works at issue are reproduced in the models." The court ruled out chance given how complex the outputs were, so it attributed them to memorisation.
  - "The original elements of the musical pieces are recognizable in the output." Suno, not the user, is responsible for the outputs. — [KPW](https://kpw.law/en/gema-also-wins-against-suno/)
  - Infringing outputs are "attributable to systemic causes" (the training data and the model architecture). Suno's argument that only the prompting user is responsible was rejected. — [Simmons & Simmons](https://www.simmons-simmons.com/en/publications/cmszvaaz6003oug849s2sqawl/gema-v-suno-ai-key-eu-decision-on-ai-training-outputs-and-copyright); [Bristows](https://inquisitiveminds.bristows.com/post/102ngsl/gema-v-suno-munich-court-finds-ai-music-training-and-outputs-infringe-copyright)
- [Fact] Text-and-data-mining exception rejected. "Anyone who gains access by circumventing an effective technical protection measure cannot invoke this exception": Suno allegedly got past YouTube's "Rolling Cipher" protection by stream ripping — [KPW](https://kpw.law/en/gema-also-wins-against-suno/). Simmons adds that reproductions inside the model "do not serve the purpose of further data analysis" — [Simmons & Simmons](https://www.simmons-simmons.com/en/publications/cmszvaaz6003oug849s2sqawl/gema-v-suno-ai-key-eu-decision-on-ai-training-outputs-and-copyright).
- [Fact] For training done in the US, the court applied US law and rejected fair use: the use was not transformative, was commercial, and substitutes for the originals in the market. The injunction also covers copying "within the territory of the United States of America" — [KPW](https://kpw.law/en/gema-also-wins-against-suno/); [Bristows](https://inquisitiveminds.bristows.com/post/102ngsl/gema-v-suno-munich-court-finds-ai-music-training-and-outputs-infringe-copyright).
- [Fact] Remedies:
  - an injunction covering reproduction, training use, and public performance through the generator;
  - disclosure of the extent of infringement and related revenue;
  - liability for damages, amount not yet set — [KPW](https://kpw.law/en/gema-also-wins-against-suno/); [Simmons](https://www.simmons-simmons.com/en/publications/cmszvaaz6003oug849s2sqawl/gema-v-suno-ai-key-eu-decision-on-ai-training-outputs-and-copyright).
- [Fact] Penalty: "a court-determined fine of up to 250,000 euros, or alternatively custodial detention of up to six months" for each future breach — [PPC Land](https://ppc.land/suno-loses-gema-case-as-munich-court-sets-250-000-euro-penalty-per-breach/). This is the standard German order-penalty threat attached to an injunction, not a fine already imposed. KPW, Simmons and Bristows do not mention it.
- [Fact, conflicting] Making-available claim. KPW says the on-demand "making available" claim **failed** because up to 176 tries were needed, so users could not get the work "at any time of their choosing"; the public-performance claim, pleaded in the alternative, succeeded — [KPW](https://kpw.law/en/gema-also-wins-against-suno/). CMS instead says the court found that outputs infringe the making-available right — [CMS](https://cms.law/en/int/publication/artificial-intelligence-and-copyright-case-tracker/germany-gema-v.-suno-31-july-2026). KPW's account is more detailed; the full judgment text would settle it.
- [Fact] Publicity of details: Variety reported that the ruling's specifics were not made public. Law-firm summaries nonetheless describe its reasoning in detail — [Variety](https://variety.com/2026/digital/news/suno-loses-ai-lawsuit-gema-1236825010/) (as summarised in search results).
- [Fact] Appeal status: the judgment is "not final and may be appealed" — [Bristows](https://inquisitiveminds.bristows.com/post/102ngsl/gema-v-suno-munich-court-finds-ai-music-training-and-outputs-infringe-copyright); [PPC Land](https://ppc.land/suno-loses-gema-case-as-munich-court-sets-250-000-euro-penalty-per-breach/). Digital Music News reported that Suno is "set to appeal" (from search summary). I found no confirmation that an appeal was filed or a hearing set at the Munich Higher Regional Court.

**GEMA v. OpenAI (lyrics)**
- [Fact] Munich I Regional Court, Case 42 O 14139/24, 11 November 2025. Memorising song lyrics in a model, and outputting them in response to prompts, counts as reproduction. The text-and-data-mining exception does not cover this. The personality-rights claim was dismissed — [Bird & Bird](https://cm.twobirds.com/en/insights/2025/landmark-ruling-of-the-munich-regional-court-(gema-v-openai)-on-copyright-and-ai-training); [CMS](https://cms.law/en/deu/legal-updates/gema-vs.-openai-munich-regional-court-i-issues-landmark-copyright-decision).
- [Fact] OpenAI appealed, and the case is pending before the Munich Higher Regional Court — [KPW](https://kpw.law/en/gema-also-wins-against-suno/); [Simmons & Simmons](https://www.simmons-simmons.com/en/publications/cmszvaaz6003oug849s2sqawl/gema-v-suno-ai-key-eu-decision-on-ai-training-outputs-and-copyright). I found no appeal decision.
- [Fact] A related EU case, *Like Company v Google* (C-250/25), may clarify the text-and-data-mining exception; the Advocate General's opinion was expected 3 September 2026 — [Simmons & Simmons](https://www.simmons-simmons.com/en/publications/cmszvaaz6003oug849s2sqawl/gema-v-suno-ai-key-eu-decision-on-ai-training-outputs-and-copyright). Not checked whether it has issued.

**UMG Recordings et al. v. Suno (D. Mass., 1:24-cv-11611-FDS)**
- [Fact] The complaint was filed 24 June 2024. Plaintiffs sought statutory damages "of up to $150,000 per infringing work", and the theory was **training** infringement — [Music Business Worldwide, 25 Jun 2024](https://www.musicbusinessworldwide.com/major-labels-plan-to-cost-suno-and-udio-billions-and-why-it-all-rests-on-michael-buble/).
- [Fact] "Sway": the complaint compares the Suno output "When Marimba Rhythms Start To Play" with Michael Bublé's "Sway". The prompt was "canadian smooth male singer 2004 jazz pop buble sway latin mambo minor key" **plus lyrics from the original**, so it combined artist name, song title, style and lyrics — [MBW](https://www.musicbusinessworldwide.com/major-labels-plan-to-cost-suno-and-udio-billions-and-why-it-all-rests-on-michael-buble/).
- [Fact] "Johnny B. Goode": the labels generated "29 different outputs that contain the style of Chuck Berry's 'Johnny B. Goode'" — [404 Media](https://404media.co/listen-to-the-ai-generated-ripoff-songs-that-got-udio-and-suno-sued). For testing, "Plaintiffs specified the lyrics for the output"; the style prompt was "1950s rock and roll, rhythm & blues, 12 bar blues, rockabilly, energetic male vocalist, singer guitarist", which has no title or artist name — [404 Media](https://404media.co/listen-to-the-ai-generated-ripoff-songs-that-got-udio-and-suno-sued).
  - Correction to the novelty doc: these examples are in the original 2024 complaint. The doc ties the 29 outputs to the September 2025 *proposed amended* complaint; I did not verify whether that filing repeats them.
- [Fact] Suno's Answer (Doc. 28, 1 August 2024):
  - It quotes the complaint's own disclaimer that plaintiffs are not "alleging that these outputs themselves infringe the Copyrighted Recordings".
  - It says outputs "are, as a rule, non-infringing".
  - It says training data "includes essentially all music files of reasonable quality that are accessible on the open Internet", and argues fair use for "a back-end technological process".
  - On "Johnny B. Goode", it says the features at issue belong to the composition, which the plaintiffs don't own. — [Docket Alarm, Doc. 28](https://www.docketalarm.com/cases/Massachusetts_District_Court/1--24-cv-11611/UMG_Recordings_Inc._et_al_v._Suno_Inc._et_al/28/)
  - So Doc. 28 is Suno's *Answer*, not a "motion".
- [Fact] Later procedure:
  - 21 May 2026: UMG and Sony moved for leave to file a Second Amended Complaint. It added 61,026 works found in discovery and a claim under DMCA §1201(a), the US ban on circumventing copy protection, over "stream ripping" — [MBW (court doc)](https://www.musicbusinessworldwide.com/files/2026/05/UMG_Sony_Suno.pdf); [proposed SAC](https://chatgptiseatingtheworld.com/wp-content/uploads/2026/05/Proposed-Second-Amended-Complaint-in-UMG-v.-Suno-May-21-2026-1.pdf).
  - Per a tracker blog, the court allowed the §1201(a) claim but refused to add the 61,026 recordings. UMG then filed a **second lawsuit** on 18 September 2026 covering about 60,202 works — [ChatGPT Is Eating the World](https://chatgptiseatingtheworld.com/2026/09/18/umg-recordings-files-2d-lawsuit-v-suno-as-expected-adds-60202-works-to-copyright-claim/). This is a secondary blog; the docket should be checked.
  - Schedule: summary judgment on fair use was reported pushed to 8 January 2027 — [ChatGPT Is Eating the World](https://chatgptiseatingtheworld.com/2026/03/10/summary-judgment-in-umg-recordings-v-suno-pushed-back-to-jan-8-2027/). Another search summary says dispositive motions are now due 9 April 2027. Sources conflict; a fair-use ruling before 2027 is unlikely either way.

**Settlements and deals**
- [Fact] Warner Music settled with Suno on 25 November 2025 and partnered with it. Suno is to launch licensed models in 2026 that replace its current ones, with limits on downloads — [Bloomberg Law](https://news.bloomberglaw.com/ip-law/warner-music-settles-lawsuit-with-ai-music-startup-suno); [MarketScreener](https://uk.marketscreener.com/news/warner-music-group-settles-copyright-case-with-suno-for-licensed-ai-music-ce7d5ed2db8bff22).
- [Fact] Universal settled with Udio in October 2025 to build a licensed platform — [Yahoo/Reuters](https://www.yahoo.com/news/articles/universal-music-settles-copyright-dispute-004804979.html). Warner also settled with Udio — [search summary, Rareform](https://www.rareformaudio.com/blog/ai-music-lawsuits-suno-udio-licensing).
- [Fact] Suno is fighting to keep the Warner settlement terms away from UMG and Sony — [MBW](https://www.musicbusinessworldwide.com/suno-fights-to-keep-warner-music-settlement-terms-away-from-umg-and-sony/).
- [Fact] As of the latest sources (late September 2026), UMG and Sony have not settled with Suno, and Sony has not settled with Udio — [AI Lawsuit Tracker](https://ailawsuittracker.com/cases/umg-v-suno/); [CourtDocket](https://courtdocket.org/suno-lawsuit-explained-warner-settlement-umg-and-sony-next/).
- [Fact] A US musicians' union has sued UMG and WMG over their AI settlements — [Billboard](https://www.billboard.com/pro/musicians-union-lawsuit-umg-wmg-ai-settlements/).
- [Fact] Isbell, Lowery and others filed a proposed class action against Suno on 31 August 2026 in D. Mass. It raises publicity and biometric-privacy claims, not copyright — [MBW](https://musicbusinessworldwide.com/jason-isbell-and-david-lowery-are-suing-suno-in-a-class-action-suit-importantly-theyre-hitting-mikey-shulmans-company-with-identity-claims-not-copyright); [Boston Globe](https://www.bostonglobe.com/2026/09/02/business/musicians-sue-ai-music-company-suno-imitating-voices/).

**Jamendo v. Nvidia and v. Suno**
- [Fact] Jamendo sued NVIDIA on 22 June 2026 (N.D. Cal.). It alleged that Fugatto and Audio Flamingo were trained on the MTG-Jamendo research dataset (CC BY-NC) without authorisation, with claims for copyright infringement, breach of contract/licence, unjust enrichment and unfair competition — [Business Wire / Winamp](https://www.businesswire.com/news/home/20260901637423/en/Winamp-Group-Provides-an-Update-on-the-U.S.-Proceedings-Involving-NVIDIA-and-Suno); [KR Law](https://kr.law/news/article-detail/jamendos-copyright-suit-against-nvidia-and-suno).
- [Fact] Jamendo sued Suno on 29 June 2026 (D. Mass.) — [Business Wire](https://www.businesswire.com/news/home/20260629046020/en/Winamp-Group-Announces-the-Filing-of-a-U.S.-Lawsuit-Against-Suno); [Billboard](https://www.billboard.com/pro/suno-ai-training-copyright-lawsuit-production-music-firm/).
  - It voluntarily dismissed the Suno suit in August 2026 — [Music Times](https://www.musictimes.com/articles/112601/20260818/jamendo-drops-copyright-lawsuit-against-suno-six-weeks-after-filing-it.htm).
  - Winamp's 1 September update said an updated complaint would be refiled — [Business Wire](https://www.businesswire.com/news/home/20260901637423/en/Winamp-Group-Provides-an-Update-on-the-U.S.-Proceedings-Involving-NVIDIA-and-Suno).
  - A 9 October 2026 blog headline says Jamendo has also dropped the NVIDIA suit — [ChatGPT Is Eating the World](https://chatgptiseatingtheworld.com/2026/10/09/sa-jamendo-has-dropped-its-suit-v-nvidia-2d-suit-jamendo-has-dropped/). Not confirmed from the docket.

**2026 US rulings on AI music outputs**
- [Fact] I found no 2026 US ruling deciding whether AI music outputs infringe or are fair use. A judge denied Udio's bid to dismiss a DMCA claim (Law360, April 2026); that is procedural — [Law360](https://www.law360.co.uk/articles/2466018) (headline only).
- [Fact] In Sony v. Uncharted Labs (Udio, S.D.N.Y. No. 24-cv-04777), leave to amend was denied on 29 June 2026 (Dkt. 183) — cited in Suno's July 2026 filing, [MBW court doc](https://www.musicbusinessworldwide.com/files/2026/07/UMG-Recordings-Inc.-et-al.-v.-Suno-Inc.-et-al.-No.-1-24-cv-11611-FDS-D.-Mass.pdf).

### Inferences
- [Opinion] For Idea B, the novelty doc's framing of GEMA as "lyrics, style and title" is accurate. GEMA's evidence was a lyrics-plus-title-plus-style probe on a closed model, repeated up to 176 times, with the comparison done by eye and ear. A **title-only, quantitative, open-model** study is therefore still distinct from it. The UMG "Sway" prompt also mixed artist, title, style and lyrics.
- [Opinion] Both cases show that "named prompt leads to recognisable output" is already used as *evidence of memorisation*. That supports the scientific relevance of Idea B. It also means the researcher's outputs could, in principle, become the same kind of evidence about the open models used.

### Gaps
- The full GEMA v. Suno judgment text was not read. Two points remain open: whether making-available succeeded (KPW and CMS conflict), and whether lyrics were excluded from the claim (KPW says yes; PPC Land is silent).
- Whether Suno actually filed an appeal (Berufung) has not been confirmed.
- I did not verify whether the "29 outputs" figure also appears in the September 2025 proposed amended complaint.
- I could not fetch the Bird & Bird GEMA v. Suno page (HTTP 402) or the Akin Gump tracker (HTTP 403).
- A Sony–Suno or Sony–Udio settlement after May to September 2026 cannot be ruled out.

## Q2. Australian law: text-and-data-mining, fair dealing, YouTube terms, copy protection, UNSW guidance

### Takeaway
Australia has **no text-and-data-mining exception**; the government ruled one out on 26–27 October 2025. The only plausible defence for copying commercial recordings is **fair dealing for research or study**. For sound recordings that is s103C (s40 covers the underlying musical work), and it turns on five fairness factors with no safe-harbour amount for audio. Downloading from YouTube also breaches YouTube's Terms of Service. It may also amount to circumventing an access-control technical protection measure under s116AN, which fair dealing does not excuse. UNSW guidance says fair dealing covers assessments but **not published material**.

### Cited Findings
- [Fact] Attorney-General Michelle Rowland ruled out a text-and-data-mining exception for AI training (media release 26 October 2025; ABC AM comments 27 October 2025). Consultation continues through the Copyright and AI Reference Group on a paid licensing model — [ACS Information Age](https://ia.acs.org.au/article/2025/govt-rules-out-ai-copyright-exemption-for-tech-giants.html); [Billboard](https://www.billboard.com/pro/australia-government-shuts-down-copyright-exemptions-ai/); [Bird & Bird](https://cm.twobirds.com/en/insights/2025/australia/3-key-takeaways-from-australia's-latest-ai-copyright-law-reform-announcement); [King & Wood Mallesons](https://pulse.mallesons.com/ip-whiteboard/creators-first-no-text-and-data-mining-for-australia/).
- [Fact] The idea had come from the Productivity Commission's interim report, which recommended a fair dealing exception for text-and-data-mining. ARIA, PPCA and APRA AMCOS welcomed the decision to rule it out — [Limelight](https://limelight-arts.com.au/news/australian-government-rules-out-text-and-data-mining-tdm-exception-for-ai-training/).
- [Fact] s103C(1): "A fair dealing with an audio-visual item does not constitute an infringement of the copyright in the item or in any work or other audio-visual item included in the item if it is for the purpose of research or study." Audio-visual items include sound recordings and films. s40 covers literary, dramatic, musical and artistic works — [AustLII s103C](https://www6.austlii.edu.au/au/legis/cth/consol_act/ca1968133/s103c.html); [ALRC DP79 "Current law"](https://www.alrc.gov.au/publication/copyright-and-the-digital-economy-dp-79/7-fair-dealing/current-law/).
- [Fact] Fairness factors (ss40(2), 103C(2)), a list that is not closed:
  - purpose and character of the dealing;
  - nature of the item;
  - whether it "could be obtained within a reasonable time at an ordinary commercial price";
  - effect on the potential market for, or value of, the item;
  - amount and substantiality of the part taken.
  — [ALRC](https://www.alrc.gov.au/publication/copyright-and-the-digital-economy-dp-79/7-fair-dealing/current-law/)
- [Fact] The "reasonable portion" safe harbour (10% / one chapter) is defined for written works, not for audio. Sound-recording use is judged case by case — [ALRC](https://www.alrc.gov.au/publication/copyright-and-the-digital-economy-dp-79/7-fair-dealing/current-law/); [Curtin / UTS library guides via search](https://www.lib.uts.edu.au/sites/default/files/2020-11/Factsheet%20-%20What%20can%20you%20copy.pdf).
- [Fact] Courts have read "research" and "study" narrowly (*De Garis v Neville Jeffress Pidler* (1990) 37 FCR 99). Current law looks at the purpose of the person making the copy — [ALRC](https://www.alrc.gov.au/publication/copyright-and-the-digital-economy-dp-79/7-fair-dealing/current-law/).
- [Fact] I found no Australian court decision applying fair dealing to AI training or AI evaluation — [Spruson & Ferguson](https://www.spruson.com/australia-on-ai-copyright-text-and-data-mining-exemption/) (as summarised).
- [Fact] UNSW Library guidance:
  - Fair dealing for research or study covers students and musical/audio-visual material, but with strict limits.
  - It lets students include copyright material in assessments, but does **not** cover material that will be published, including theses on UNSWorks.
  - Questions go to copyright@unsw.edu.au; specific advice comes from the UNSW Legal Office.
  — [UNSW copyright: study and research](https://subjectguides.library.unsw.edu.au/copyright/study_and_research); [UNSW fair dealing](https://subjectguides.library.unsw.edu.au/copyright/fair_dealing). Both pages were summarised via search; direct fetch returned 404.
- [Fact] YouTube Terms of Service (effective 6 November 2023). Users may not "access, reproduce, download … or otherwise use any part of the Service or any Content except: (a) as expressly authorized by the Service; or (b) with prior written permission from YouTube and, if applicable, the respective rights holders". Users also may not "access the Service using any automated means (such as robots, botnets or scrapers)" without permission — [YouTube Terms](https://www.youtube.com/static?template=terms).
- [Fact] s116AN gives the copyright owner a civil action against anyone who circumvents an access-control technical protection measure (copy protection that controls access) and knew or ought to have known it. The listed exceptions are narrow: owner permission, software interoperability, encryption research and similar. I found **no general research exception** — [AustLII s116AN (search summary)](https://20.austlii.edu.au/au/legis/cth/consol_act/ca1968133/s116an.html).
- [Fact] In GEMA v. Suno, the Munich court treated YouTube's "Rolling Cipher" as an effective technical protection measure that Suno circumvented by stream ripping — [KPW](https://kpw.law/en/gema-also-wins-against-suno/). UMG's proposed amendment likewise pleads stream ripping as a DMCA §1201 circumvention claim — [MBW court doc](https://www.musicbusinessworldwide.com/files/2026/05/UMG_Sony_Suno.pdf).

### Inferences
- [Opinion] The MGPHot authors' legal basis was EU DSM Directive Art. 3, the research-organisation text-and-data-mining exception. It does not carry over to an Australian researcher. Australia has no equivalent, and the government ruled one out in October 2025.
- [Opinion] **Does fair dealing cover the 1,712 YouTube clips?**
  - In favour: the purpose is non-commercial research; each clip is short (30 s); the clips are used for measurement, not listening for pleasure; and nothing is redistributed.
  - Against: the songs are available at an ordinary commercial price (iTunes and similar); there are 1,712 whole-song excerpts; and they were obtained in breach of YouTube's Terms of Service, possibly by circumventing copy protection.
  - Overall: an arguable but **untested** defence for *private storage and analysis*. It is weak for redistributing the clips or publishing them as a dataset (UNSW says fair dealing does not cover publication).
- [Opinion] The YouTube download step is the weakest point, and fair dealing cannot cure it:
  - (a) Breaching the Terms of Service is a contract matter between the user and YouTube, separate from copyright.
  - (b) If yt-dlp-style tools defeat YouTube's signature cipher, that may count as circumventing an access-control technical protection measure under s116AN, and research fair dealing is not a defence to that.
  - Practical risk of enforcement against a student doing private analysis is low. The *reputational* and *institutional* risk is real (paper reviewers, the UNSW integrity office), because the leading 2026 court decision named stream ripping specifically.
- [Opinion] Safer handling for the clips:
  - keep them on Katana, access-restricted;
  - never redistribute audio;
  - publish only derived numbers or embeddings and the MGPHot IDs;
  - document the legal basis;
  - get written guidance from copyright@unsw.edu.au before the NeurIPS submission.
  - Reproducing the 30-s clips from a lawfully purchased copy would remove both the Terms-of-Service and copy-protection issues.

### Gaps
- I could not fetch the current s40, s103C or s116AN text from legislation.gov.au or AustLII (HTTP 403). The quotes come from search snippets and the ALRC paper.
- I did not check Copyright Regulations 2017 for any *prescribed* circumvention exception covering research.
- I found no Australian case or regulator statement on whether YouTube's cipher is an access-control technical protection measure.
- I found no UNSW page that specifically addresses YouTube downloading, AI evaluation, or text-and-data-mining.

## Q3. Idea B legal profile (prompt open generators with song titles; analyse outputs)

### Takeaway
Idea B needs **no copyrighted input audio** and runs on open, permissively licensed or non-commercial-licensed models. Its main legal exposure is producing outputs that may reproduce protected songs, plus the question of what is released afterwards. Under GEMA's reasoning, liability for outputs fell on the **model operator** because of "systemic causes", not on the prompting user. The risk is low if the outputs stay private and only features and statistics are published.

### Cited Findings
- [Fact] The GEMA court rejected Suno's argument that responsibility "lies solely with the claimant as the user". It attributed infringing outputs to "systemic causes" (training data, architecture) and treated Suno as the infringer — [Simmons & Simmons](https://www.simmons-simmons.com/en/publications/cmszvaaz6003oug849s2sqawl/gema-v-suno-ai-key-eu-decision-on-ai-training-outputs-and-copyright); [Bristows](https://inquisitiveminds.bristows.com/post/102ngsl/gema-v-suno-munich-court-finds-ai-music-training-and-outputs-infringe-copyright).
- [Fact] GEMA staff themselves generated the infringing outputs (up to 176 times) as evidence. That was treated as proof-gathering; the court did not treat it as the plaintiffs' own infringement. The making-available claim failed partly *because* so many tries were needed — [KPW](https://kpw.law/en/gema-also-wins-against-suno/); [PPC Land](https://ppc.land/suno-loses-gema-case-as-munich-court-sets-250-000-euro-penalty-per-breach/).
- [Fact] UMG's lawyers likewise generated look-alike outputs with "targeted prompts" to show what was in the training data — [404 Media](https://404media.co/listen-to-the-ai-generated-ripoff-songs-that-got-udio-and-suno-sued); [MBW](https://www.musicbusinessworldwide.com/major-labels-plan-to-cost-suno-and-udio-billions-and-why-it-all-rests-on-michael-buble/).
- [Fact] Published memorisation audits handle this unevenly. *Bob's Confetti* (arXiv 2507.17937v3, Suno/YuE lyric-substitution attacks) has **no** ethics, broader-impact or legal section in the HTML version, yet links "Demo examples … at our project page" — [arXiv 2507.17937v3](https://arxiv.org/html/2507.17937v3).
- [Fact] Other audit lines of work exist: MusicGen exact or near-exact fragment reproduction ([arXiv 2412.08549](https://www.arxiv.org/pdf/2412.08549)) and black-box membership inference for music models ([arXiv 2605.29202](https://arxiv.org/pdf/2605.29202)). I did not confirm how either handles releasing outputs.
- [Fact] Jamendo's suit against NVIDIA alleged that a CC BY-NC research dataset was used to build commercial models, and included a licence-breach claim — [KR Law](https://kr.law/news/article-detail/jamendos-copyright-suit-against-nvidia-and-suno). This is relevant to keeping MusicGen (CC-BY-NC weights) and the MUSDB/MoisesDB research licences strictly non-commercial.

### Inferences
- [Opinion] Generating outputs privately on Katana for measurement is low risk for the researcher.
  - Australia: if an output reproduces a substantial part of a musical work, making it is technically a reproduction. Private, non-commercial research analysis is a strong fair dealing case under s40/s103C: the purpose is research, no market substitution, and nothing is distributed.
  - In practice, the GEMA reasoning points enforcement at whoever *trained and serves* the model.
- [Opinion] **What to publish:**
  - Low risk: aggregate statistics, similarity scores, melody or chroma features, embeddings, and short descriptive excerpts (e.g. "matches the chorus melody of X").
  - Higher risk: posting the audio outputs that most resemble famous songs (a demo page, a dataset). That is distribution or communication to the public of possibly infringing material, and fair dealing for research does not cover publication (UNSW guidance).
  - If audio examples are wanted for reviewers, keep them few, short, access-restricted (e.g. anonymous review only), and clear them with UNSW first.
- [Opinion] Report findings about open models neutrally ("the model produces X% melodic match to the named song"). Do not assert that a named developer "infringed": that is a legal conclusion and could raise defamation-type concerns. Consider notifying model maintainers before publication, as responsible disclosure.
- [Opinion] Prompting with titles and artist names only, with no lyrics, is legally cleaner than GEMA's and UMG's prompts. It also avoids reproducing lyrics in prompts, which was the issue in GEMA v. OpenAI.

### Gaps
- I found no published music-memorisation paper with an explicit legal or ethics protocol for withholding outputs. A targeted read of ethics statements (e.g. arXiv 2412.08549, 2605.29202, Carlini-style audits) would help.
- I found no Australian authority on user liability for AI outputs.

## Q4. Idea A legal profile (stem-swapped versions of famous songs)

### Takeaway
Idea A needs the **original commercial audio**, source separation, and the creation of new versions. That stacks several acts: copying the recording, extracting parts, and combining them into a new recording, which may be an adaptation. All of them rest only on fair dealing in Australia. Buying the tracks avoids the YouTube terms and copy-protection problems but does not authorise making altered versions. The legally cleanest option is licensed or open multitrack collections (MUSDB18-HQ, MoisesDB, MedleyDB, Cambridge-MT), which the project already partly has. Most of their songs are not famous.

### Cited Findings
- [Fact] The Open Multitrack Testbed paper notes that stems from the Rock Band video game, used in earlier experiments, were "readily extracted from the game but not shareable as audio only, since its use is restricted by copyright" — [Open Multitrack Testbed, ISMIR/academia.edu](https://www.academia.edu/32237897/The_Open_Multitrack_Testbed).
- [Fact] Widely circulated "multitrack masters" of famous songs are described as **leaked** material — [Premier Guitar](https://www.premierguitar.com/articles/22144-recording-guitarist-outlaw-recording-lessons).
- [Fact] MedleyDB was created because royalty-free multitracks shareable for research are scarce; it has over 250 multitracks — [MedleyDB paper](https://bpb-us-e1.wpmucdn.com/wp.nyu.edu/dist/2/2294/files/2016/08/bittner-medleydb.pdf).
- [Fact] A 2023 ISMIR paper notes that studios may oppose publishing stems, and that available stems may come with licence restrictions — [ISMIR 2023](https://archives.ismir.net/ismir2023/paper/000073.pdf).
- [Fact] One fairness factor under s103C/s40 is whether the item "could be obtained within a reasonable time at an ordinary commercial price" — [ALRC](https://www.alrc.gov.au/publication/copyright-and-the-digital-economy-dp-79/7-fair-dealing/current-law/).

### Inferences
- [Opinion] Purchased tracks versus YouTube:
  - Buying (e.g. DRM-free files from iTunes or Bandcamp) removes the YouTube Terms-of-Service breach and the copy-protection issue. It also shows good faith and pays the rights holder, which helps the "market effect" factor.
  - The purchase licence is for personal use. Separating, altering and recombining the audio is still a reproduction or adaptation that needs fair dealing.
  - So purchasing is **safer than YouTube, but not a licence**.
  - The "ordinary commercial price" factor cuts both ways: it shows the work was available to buy, which counts against copying freely, but buying it is exactly what the researcher did.
- [Opinion] A stem-swapped version reproduces substantial parts of the sound recording and the musical work. Whether Australian law calls it an "adaptation" (for music, defined around arrangement or transcription) or simply a reproduction of a substantial part, it needs a defence either way. Fair dealing for research is plausible for **private** generation and model testing. It is weak for **publishing** the altered audio or using it in a public listening study, especially with recruited outside listeners. Ask copyright@unsw.edu.au.
- [Opinion] Rock Band or leaked multitracks: avoid them. They are copyright-restricted and often unlawfully sourced, and using them in a 2027 paper would draw reviewer objections.
- [Opinion] Lower-risk sources:
  - MUSDB18-HQ, MoisesDB and MedleyDB (already research-licensed or CC);
  - Cambridge-MT multitracks (educational terms; check them, the page returned 403);
  - official stem or remix-contest releases issued by labels or artists, whose contest terms usually allow non-commercial remixing. Check each one.
- [Opinion] If Idea A must use famous songs: the GEMA-style "recognisability" tests and listening studies are where audio leaves the researcher's control, so keep those to the licensed material.

### Gaps
- I did not verify the text of the Cambridge-MT licence (HTTP 403), the MedleyDB licence text (believed to be CC BY-NC-SA, unverified), or any specific official stem or remix-contest licence.
- I did not verify the iTunes/Apple Media Services or Bandcamp terms on personal use.
- I found no Australian case on whether separating stems or making remixes for research is fair dealing.

## Q5. Human-subjects: does a listener "naturalness" study need UNSW ethics approval; can a student be Chief Investigator?

### Takeaway
Yes, approval is needed. UNSW's Human Research Ethics Procedure (v2.3, effective 27 March 2025) requires approval **before** any human research at minimal risk or above. A low-risk listening study goes to the school or faculty panel (Human Research Ethics Advisory Panel, HREAP) through the iRECS portal. Research students working with human participants must be supervised. I found no UNSW source saying an unsupervised undergraduate can be Chief Investigator, and the researcher should assume they cannot without a staff sponsor.

### Cited Findings
- [Fact] "Human ethics approval must be obtained before commencing any human research at minimal or greater risk and involving human participants, their data, and biospecimens." The procedure "applies to all staff and research students at UNSW" — [UNSW Human Research Ethics Procedure v2.3, effective 27 Mar 2025](https://unsw.edu.au/content/dam/pdfs/governance/policy/2022-01-policies/humanresearchethicsprocedure.pdf).
- [Fact] Minimal-risk and low-risk proposals by staff and students go to the Human Research Ethics Advisory Panels (HREAPs); more-than-low-risk proposals go to the Human Research Ethics Committees (HRECs) — [same procedure](https://unsw.edu.au/content/dam/pdfs/governance/policy/2022-01-policies/humanresearchethicsprocedure.pdf).
- [Fact] "Research students must be supervised when working with human participants, conducting fieldwork, or collecting their data or tissue overseas" — [same procedure](https://unsw.edu.au/content/dam/pdfs/governance/policy/2022-01-policies/humanresearchethicsprocedure.pdf). The sentence sits in the overseas-research section, so how far it reaches is ambiguous.
- [Fact] "Low risk" means research where the only foreseeable risk is discomfort. Applications go through iRECS to the panel responsible for the school — [UNSW Research (search summary)](https://research.unsw.edu.au/higher-risk-research); [UNSW Psychology ethics resources](https://unsw.edu.au/science/our-schools/psychology/our-research/research-resources/ethics-and-sona-resources).
- [Fact] Contact: humanethics@unsw.edu.au — [same procedure](https://unsw.edu.au/content/dam/pdfs/governance/policy/2022-01-policies/humanresearchethicsprocedure.pdf).

### Inferences
- [Opinion] A small online naturalness-rating study counts as human research (collecting participants' responses) and is almost certainly "low risk". It needs HREAP approval before any data collection, including informal pilot ratings by friends if the data are used in the paper.
- [Opinion] As a solo undergraduate, the researcher will very likely need a UNSW staff member as Chief Investigator (supervisor or sponsor) with themselves as student investigator. The iRECS system normally ties applications to staff Chief Investigators.
  - This is consistent with the 2026-10-22 recall-design review in project memory, which said a text-only identity paper should be chosen unless a staff ethics Chief Investigator is found.
  - Confirm with humanethics@unsw.edu.au.
- [Opinion] Playing copyrighted stem-swapped audio to outside participants also falls outside private research use. This links Q4 and Q5: listening studies are both an ethics and a copyright step-up.

### Gaps
- I could not access UNSW's current SharePoint ethics pages (they redirect to an authenticated SharePoint site). The specific rule on who may be Chief Investigator, and on undergraduate (non-research-student) applicants, is unverified.
- I did not check whether a crowdsourced rating study (e.g. Prolific) has additional UNSW requirements.
- I did not check whether "negligible risk" exemptions under the National Statement (the national ethics guidelines) are available at UNSW for anonymous online ratings. The procedure text suggests minimal risk and above must be reviewed.
