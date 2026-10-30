"""Question generation for the MusicWhy benchmark across five evaluation tasks.

Each task targets a distinct aspect of musical reasoning:
  Task 1 -- Reasoning QA (multiple-choice + open-ended)
  Task 2 -- Source-Aware Emotion Attribution
  Task 3 -- Song Relationship Explanation
  Task 4 -- Symbolic Sound Identification
  Task 5 -- Sound-World Connection
"""

from __future__ import annotations

import hashlib
import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Question = Dict[str, Any]
"""Canonical question dict with keys:
    task_id, question_text, audio_reference, options (for MC),
    correct_answer, reference_explanation, metadata
"""

FeatureDict = Dict[str, Any]
StemFeatures = Dict[str, FeatureDict]  # stem_name -> features
SongRecord = Dict[str, Any]           # full song metadata + features


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_EMOTION_MAP: Dict[str, Dict[str, str]] = {
    "energy": {
        "description": "high energy / excitement",
        "low_desc": "calm / subdued",
        "feature_keys": ["tempo", "spectral_centroid", "dynamic_range"],
    },
    "melancholy": {
        "description": "sadness / melancholy",
        "low_desc": "happiness / uplift",
        "feature_keys": ["estimated_key", "harmonic_percussive_ratio", "spectral_centroid"],
    },
    "tension": {
        "description": "tension / unease",
        "low_desc": "resolution / relaxation",
        "feature_keys": ["spectral_bandwidth", "dynamic_range", "harmonic_percussive_ratio"],
    },
    "nostalgia": {
        "description": "nostalgia / warmth",
        "low_desc": "coldness / detachment",
        "feature_keys": ["spectral_centroid", "dynamic_range", "tempo"],
    },
    "aggression": {
        "description": "aggression / intensity",
        "low_desc": "gentleness / softness",
        "feature_keys": ["spectral_centroid", "tempo", "dynamic_range"],
    },
}

_CULTURAL_SOUND_PROFILES: List[Dict[str, Any]] = [
    {
        "id": "atlanta_hiphop",
        "sound_description": (
            "Deep 808 sub-bass with sustained decay, rapid trap hi-hats with "
            "triplet patterns, sparse melodic loops, heavy use of reverb on vocals"
        ),
        "correct_association": "Atlanta hip-hop / trap music",
        "distractors": [
            "West Coast hip-hop / G-funk",
            "London club music / UK garage",
            "Detroit electronic / techno",
        ],
        "explanation": (
            "The 808-centric bass, triplet hi-hat patterns, and sparse melodic "
            "approach are hallmarks of Atlanta trap production, pioneered by "
            "producers like Lex Luger and Metro Boomin."
        ),
    },
    {
        "id": "lofi_nostalgia",
        "sound_description": (
            "Reverbed piano chords with slight detuning, vinyl crackle and hiss "
            "layered throughout, muffled lo-pass filtered drums, slow tempo "
            "around 70-85 BPM"
        ),
        "correct_association": "Lo-fi hip-hop / study beats nostalgia",
        "distractors": [
            "Jazz fusion / virtuosic improvisation",
            "Vaporwave / critique of consumerism",
            "Impressionism / classical tone colour",
        ],
        "explanation": (
            "The combination of detuned piano, vinyl artifacts, and lo-pass "
            "filtering deliberately evokes analog warmth and imperfection, "
            "creating an idealized auditory memory. The slow tempo matches "
            "resting heart rate, promoting relaxation."
        ),
    },
    {
        "id": "metal_aggression",
        "sound_description": (
            "Heavily distorted down-tuned guitar riffs, blast-beat drumming at "
            "200+ BPM, growled or screamed vocals, very high spectral energy "
            "across the frequency range"
        ),
        "correct_association": "Extreme metal / death metal aggression and catharsis",
        "distractors": [
            "Punk rock / anti-establishment rebellion and urgency",
            "Industrial music / mechanised anger and dystopian noise",
            "Grunge / Gen-X apathy and disillusioned introspection",
        ],
        "explanation": (
            "Down-tuned distorted guitars create dense harmonic overtones perceived "
            "as aggressive. Blast beats exceed normal physiological rhythms, creating "
            "intensity. The combination channels cathartic release through sonic extremity."
        ),
    },
    {
        "id": "ambient_space",
        "sound_description": (
            "Long sustained synthesizer pads with slow filter sweeps, heavy reverb "
            "creating a sense of vast space, no discernible beat or pulse, "
            "frequencies concentrated in the mid-low range"
        ),
        "correct_association": "Ambient music / meditative soundscapes",
        "distractors": [
            "New age music / holistic healing rituals",
            "Drone metal / sustained heaviness and doom",
            "Minimalism / repetitive classical process",
        ],
        "explanation": (
            "The absence of rhythmic pulse removes temporal urgency. Long reverb tails "
            "simulate large physical spaces, triggering spatial awareness. Slow timbral "
            "evolution engages sustained attention without demanding active processing."
        ),
    },
    {
        "id": "bossa_nova",
        "sound_description": (
            "Nylon-string guitar with syncopated fingerpicking patterns, soft brushed "
            "drums with a distinctive 'long-short-short' rhythm, warm breathy vocals, "
            "moderate tempo around 120 BPM, jazz-influenced chord extensions"
        ),
        "correct_association": "Brazilian bossa nova / sophisticated urban romance",
        "distractors": [
            "Spanish flamenco / passionate Andalusian lament",
            "Caribbean calypso / carnival festivity and satire",
            "American cool jazz / restrained West Coast detachment",
        ],
        "explanation": (
            "The syncopated nylon guitar pattern is the signature bossa nova rhythm. "
            "Jazz harmony extensions (7ths, 9ths) add sophistication. The intimate "
            "vocal delivery emerged as a reaction against loud samba, reflecting "
            "1950s-60s Brazilian urban middle-class aesthetics."
        ),
    },
    {
        "id": "edm_euphoria",
        "sound_description": (
            "Saw-wave supersaw synthesizer stacks, side-chain compressed kick creating "
            "a pumping effect, tempo at 128 BPM, long build-ups with rising white noise "
            "sweeps followed by a bass drop"
        ),
        "correct_association": "EDM / festival euphoria and collective experience",
        "distractors": [
            "Trance / hypnotic meditation and transcendence",
            "House music / underground club community and soul",
            "Synthwave / retro-futurist 1980s film nostalgia",
        ],
        "explanation": (
            "The build-and-drop structure exploits anticipation and release cycles. "
            "128 BPM matches brisk walking pace and entrains group movement. "
            "Side-chain pumping creates a breathing effect that synchronizes "
            "a crowd's physical response."
        ),
    },
    {
        "id": "reggae_resistance",
        "sound_description": (
            "Off-beat rhythmic guitar/keyboard 'skank', deep bass lines with melodic "
            "movement, one-drop drum pattern emphasizing beat 3, slow-moderate tempo "
            "around 70-90 BPM, reverb-heavy mix"
        ),
        "correct_association": "Jamaican reggae / resistance and spiritual consciousness",
        "distractors": [
            "Jamaican ska / upbeat dancehall celebration and optimism",
            "Dub / studio-as-instrument echo and bass experimentation",
            "Nigerian Afrobeat / political protest and big-band groove",
        ],
        "explanation": (
            "The one-drop pattern creates a deliberate, grounded feel that resists "
            "hurried Western rhythmic norms. Off-beat emphasis inverts conventional "
            "strong-beat structure, a sonic metaphor for counter-cultural positioning. "
            "Heavy reverb evokes spiritual spaciousness rooted in Rastafari tradition."
        ),
    },
    {
        "id": "kpop_production",
        "sound_description": (
            "Dense multi-layered production with frequent genre shifts within a single "
            "track, precise vocal harmonies with pitch correction, EDM-influenced drops "
            "alternating with R&B verses, high spectral energy, 100-130 BPM"
        ),
        "correct_association": "K-pop / maximalist production and global appeal engineering",
        "distractors": [
            "J-pop / idol cuteness culture and domestic fan devotion",
            "Western Top-40 pop / radio-formatted verse-chorus formula",
            "Bollywood / cinematic song-and-dance drama and spectacle",
        ],
        "explanation": (
            "Genre-switching within one song reflects K-pop's strategy of offering "
            "multiple hooks to capture diverse listener preferences. Dense layering and "
            "precision engineering reflect the high-investment idol production system. "
            "The blend of Western electronic elements with precise group vocals is "
            "distinctive to the Korean pop industry's global export model."
        ),
    },
    {
        "id": "afrobeat_groove",
        "sound_description": (
            "Interlocking polyrhythmic guitar and percussion patterns, call-and-response "
            "vocal sections, prominent horn arrangements, extended song structures "
            "often exceeding 10 minutes, 100-130 BPM with heavy rhythmic emphasis"
        ),
        "correct_association": "West African Afrobeat / communal resistance music",
        "distractors": [
            "Caribbean soca / carnival party music and road march",
            "South African jazz / township swing and exile longing",
            "Colombian cumbia / coastal folk dance and courtship",
        ],
        "explanation": (
            "Polyrhythmic interlocking creates a groove-based temporal experience "
            "distinct from Western verse-chorus structure. Extended durations reflect "
            "communal gathering rather than radio formatting. Horn arrangements add "
            "a big-band collective energy rooted in the Fela Kuti tradition."
        ),
    },
    {
        "id": "city_pop",
        "sound_description": (
            "Lush synthesizer pads with chorus effects, slap bass with tight funk "
            "grooves, airy female vocals with reverb, jazz-influenced chord progressions "
            "with 7th and 9th extensions, bright mix around 100-120 BPM"
        ),
        "correct_association": "Japanese city pop / 1980s urban sophistication",
        "distractors": [
            "American yacht rock / smooth West Coast leisure",
            "French house / filtered disco loops and club chic",
            "Korean R&B / heartfelt ballads and vocal runs",
        ],
        "explanation": (
            "The fusion of Western funk-jazz harmony with Japanese pop vocal aesthetics "
            "defined city pop as the soundtrack of Japan's economic bubble. Lush "
            "synthesizers and chorus effects signal studio sophistication, while the "
            "bright mix reflected optimistic urban consumer culture of 1980s Tokyo."
        ),
    },
    {
        "id": "uk_drill",
        "sound_description": (
            "Sliding 808 bass patterns, sparse hi-hats with rhythmic triplet variations, "
            "dark minor-key melodies, UK-accented vocal flow with rapid syllabic delivery, "
            "tempo around 140-143 BPM, reverb-heavy atmospheric pads"
        ),
        "correct_association": "UK drill / London street narrative",
        "distractors": [
            "Chicago drill / South Side street reality",
            "Grime / East London MC battle culture",
            "Dancehall / Jamaican sound clash culture",
        ],
        "explanation": (
            "UK drill adapted Chicago drill's 808 patterns to a slightly faster tempo "
            "with distinctive sliding bass. The dark atmospheric pads create a cinematic "
            "tension absent in the original Chicago style. The vocal delivery fuses "
            "grime's rapid-fire flow with drill's rhythmic template."
        ),
    },
    {
        "id": "nordic_ambient",
        "sound_description": (
            "Glacial synthesizer textures with slow granular evolution, field recordings "
            "of wind and water, very sparse arrangement with long silences, extremely "
            "low tempo or no perceptible beat, frequencies concentrated below 500 Hz"
        ),
        "correct_association": "Nordic ambient / landscape-as-music tradition",
        "distractors": [
            "New age / meditation and wellness soundtracks",
            "Berlin school / sequenced cosmic synthesizer music",
            "Japanese onkyo / near-silent sound minimalism",
        ],
        "explanation": (
            "Field recordings anchor the music to specific geography. Glacial pace "
            "and low-frequency concentration mirror the Nordic landscape's vastness "
            "and stillness. The tradition treats environmental sound as compositional "
            "material rather than background."
        ),
    },
    {
        "id": "flamenco",
        "sound_description": (
            "Nylon-string guitar with rapid rasgueado strumming and picado runs, "
            "hand-clapping (palmas) in complex rhythmic patterns, passionate vocal "
            "melisma with raw timbre, Phrygian mode harmony, dramatic tempo changes "
            "between sections"
        ),
        "correct_association": "Spanish flamenco / Andalusian emotional intensity",
        "distractors": [
            "Portuguese fado / saudade and maritime melancholy",
            "Latin American nueva canción / protest folk song",
            "Romani brass band / Balkan wedding celebration",
        ],
        "explanation": (
            "The Phrygian mode creates a distinctive harmonic tension unique to "
            "Andalusian tradition. Palmas provide rhythmic counterpoint that demands "
            "active physical engagement. Dramatic tempo shifts mirror the emotional "
            "arc of duende — the ineffable quality of raw emotional authenticity."
        ),
    },
    {
        "id": "vaporwave",
        "sound_description": (
            "Slowed-down 1980s pop and smooth jazz samples pitched down 20-40%, "
            "heavy reverb creating a dreamlike underwater quality, chopped and screwed "
            "vocal fragments, lo-fi digital artifacts and compression, 60-80 BPM"
        ),
        "correct_association": "Vaporwave / critique of consumer capitalism and digital nostalgia",
        "distractors": [
            "Lo-fi hip-hop / relaxation soundtrack for studying and focus",
            "Synthwave / retro-futurist homage to 1980s action film scores",
            "Chillwave / sun-faded indie escapism and summer memory haze",
        ],
        "explanation": (
            "Pitch-shifted corporate muzak becomes uncanny, transforming elevator music "
            "into critique. The underwater reverb creates distance from the original "
            "commercial context. Digital artifacts emphasize the mediated nature of "
            "the experience — nostalgia for a past filtered through technology."
        ),
    },
    {
        "id": "gospel_choir",
        "sound_description": (
            "Large vocal ensemble with layered harmonies, call-and-response between "
            "lead singer and choir, organ and piano accompaniment with rich harmonic "
            "movement, dynamic builds from whisper to full-throated crescendo, "
            "heavy use of melisma and vocal ornamentation"
        ),
        "correct_association": "African-American gospel / spiritual transcendence through collective voice",
        "distractors": [
            "European choral tradition / liturgical polyphony and sacred harmony balance",
            "Motown soul / polished vocal-group pop crafted for crossover radio success",
            "Contemporary worship / arena-scale praise songs for megachurch congregations",
        ],
        "explanation": (
            "Call-and-response structure mirrors the participatory worship tradition. "
            "Dynamic builds from quiet to explosive volume create a physical experience "
            "of spiritual ascent. Melismatic ornamentation derives from West African "
            "vocal traditions and expresses emotion beyond what words carry."
        ),
    },
    {
        "id": "gamelan",
        "sound_description": (
            "Interlocking metallic percussion patterns with shimmering sustain, "
            "cyclical rhythmic structures that repeat with gradual variation, "
            "non-Western tuning with audible beating between partials, "
            "layered temporal cycles from fast surface to slow deep pulse"
        ),
        "correct_association": "Indonesian gamelan / cyclical time and communal music-making",
        "distractors": [
            "Indian classical raga / improvised melodic journey over a drone",
            "Chinese silk-and-bamboo / refined ensemble music of the literati",
            "West African drumming / polyrhythmic dialogue and community dance",
        ],
        "explanation": (
            "Interlocking patterns require collective coordination without a conductor. "
            "Non-Western tuning creates the characteristic shimmering from beating "
            "between closely-spaced partials. Cyclical structures embody Javanese "
            "and Balinese concepts of time as circular rather than linear."
        ),
    },
    {
        "id": "phonk",
        "sound_description": (
            "Chopped Memphis rap vocal samples with heavy reverb, distorted cowbell "
            "patterns, dark detuned synthesizer bass, lo-fi tape saturation throughout, "
            "aggressive side-chain pumping, 130-160 BPM"
        ),
        "correct_association": "Phonk / internet-era Memphis rap revival and car culture",
        "distractors": [
            "Southern trap / Atlanta 808 street music and club dominance",
            "Industrial techno / warehouse rave hardness and machine pulse",
            "Cloud rap / hazy internet-era melancholy and ethereal beats",
        ],
        "explanation": (
            "Chopped Memphis samples reference 1990s underground hip-hop while the "
            "aggressive processing transforms them for modern contexts. The cowbell "
            "patterns and tape saturation are deliberate stylistic markers. The pumping "
            "compression is designed for car speaker systems and drift culture videos."
        ),
    },
    {
        "id": "shoegaze",
        "sound_description": (
            "Heavily distorted guitars with extreme reverb and chorus creating a "
            "wall of sound, buried vocals mixed low beneath the guitar wash, "
            "droning sustain with minimal rhythmic variation, mid-tempo around "
            "90-110 BPM, dense mid-frequency saturation"
        ),
        "correct_association": "Shoegaze / introverted immersion and textural beauty",
        "distractors": [
            "Dream pop / ethereal escapism and soft-focus romance",
            "Post-punk / angular tension and urban alienation",
            "Noise rock / confrontational volume and abrasion",
        ],
        "explanation": (
            "The wall of guitar texture submerges melody beneath timbre, prioritizing "
            "immersion over communication. Buried vocals signal introversion — the "
            "voice as another texture rather than a narrative vehicle. The dense "
            "mid-frequency wash creates a cocoon-like listening experience."
        ),
    },
    {
        "id": "cumbia_digital",
        "sound_description": (
            "Traditional cumbia rhythm (steady kick-snare pattern with guira shuffle), "
            "synthesizer replacing accordion melody, digital bass drops and effects, "
            "call-and-response vocal hooks, moderate tempo around 90-100 BPM, "
            "blend of acoustic and electronic textures"
        ),
        "correct_association": "Digital cumbia / Latin American dance tradition meeting electronic production",
        "distractors": [
            "Reggaeton / Puerto Rican urban pop built on the dembow rhythm for global clubs",
            "Baile funk / Rio favela party music with booming bass and call-and-response",
            "Bachata / Dominican romantic guitar dance music about heartbreak and longing",
        ],
        "explanation": (
            "The steady cumbia rhythm is preserved as cultural anchor while electronic "
            "production updates the sound for club contexts. The guira shuffle pattern "
            "is the signature element distinguishing cumbia from other Latin dance forms. "
            "Digital bass drops bridge traditional and contemporary dance cultures."
        ),
    },
    {
        "id": "delta_blues",
        "sound_description": (
            "Solo acoustic guitar with bottleneck slide technique, raw unprocessed "
            "vocal delivery with rough timbre, simple I-IV-V chord progressions, "
            "rhythmic flexibility with push-pull against the beat, low spectral "
            "energy concentrated in the guitar's fundamental range"
        ),
        "correct_association": "Delta blues / American South individual hardship and resilience",
        "distractors": [
            "Folk protest / 1960s acoustic songs for civil rights and peace",
            "Country / rural storytelling about family, faith and heartbreak",
            "Jazz / urban improvisation and ensemble virtuosity in the speakeasy",
        ],
        "explanation": (
            "The solo performance format reflects individual expression against "
            "adversity. Bottleneck slide technique creates vocal-like pitch bends "
            "that blur the line between singing and playing. Raw timbre signals "
            "authenticity — the opposite of polished studio production."
        ),
    },
    {
        "id": "hyperpop",
        "sound_description": (
            "Pitch-shifted vocals to extreme highs, abrupt genre transitions within "
            "seconds, distorted maximalist bass drops, auto-tuned vocal manipulation "
            "as aesthetic choice, very high spectral energy across all frequencies, "
            "unpredictable song structure, 140-170 BPM"
        ),
        "correct_association": "Hyperpop / post-ironic digital maximalism and genre deconstruction",
        "distractors": [
            "EDM / festival euphoria and stadium-scale drops for collective release",
            "Nightcore / fan-made sped-up remixes of anime and trance favourites",
            "Bubblegum pop / sugary teen-market hooks and carefree innocence",
        ],
        "explanation": (
            "Extreme pitch-shifting detaches the voice from human identity. Abrupt "
            "genre shifts reject coherent genre allegiance. Maximalist distortion "
            "pushes pop conventions past their breaking point — simultaneously "
            "embracing and satirizing excess. The aesthetic emerged from internet-native "
            "music communities unbound by genre gatekeeping."
        ),
    },
    {
        "id": "indian_raga",
        "sound_description": (
            "Sitar or sarangi with ornamental microtonal bends (gamak), sustained "
            "drone from tanpura providing a constant tonal reference, tabla providing "
            "cyclical rhythmic patterns (tala), gradual acceleration from slow alap "
            "to fast gat section, non-tempered tuning"
        ),
        "correct_association": "North Indian classical (Hindustani) / spiritual-temporal journey through raga",
        "distractors": [
            "Middle Eastern maqam / modal improvisation and ecstatic tarab in Arab art music",
            "Indonesian gamelan / cyclical interlocking time and communal court ritual",
            "West African griot / hereditary oral history and genealogy sung with the kora",
        ],
        "explanation": (
            "The tanpura drone establishes a gravitational tonal center against which "
            "the raga unfolds. The alap-to-gat acceleration structure mirrors a "
            "journey from contemplation to ecstasy. Microtonal ornaments (gamak) carry "
            "emotional meaning that tempered tuning cannot express, connecting specific "
            "ragas to times of day and emotional states."
        ),
    },
    {
        "id": "synthwave",
        "sound_description": (
            "Analog-style synthesizer arpeggios with warm detuned oscillators, "
            "gated reverb drums reminiscent of 1980s production, driving bass "
            "sequences, cinematic pads, bright high-frequency content, 80-118 BPM"
        ),
        "correct_association": "Synthwave / retro-futurist nostalgia for 1980s media aesthetics",
        "distractors": [
            "Vaporwave / slowed-down critique of consumer culture and muzak",
            "Electropop / bright synth-driven dance pop for radio and clubs",
            "Film score / orchestral drama and leitmotif storytelling on screen",
        ],
        "explanation": (
            "The analog synthesis aesthetic recreates a remembered rather than actual "
            "1980s. Gated reverb drums are a sonic timestamp for that decade. The "
            "cinematic quality references action and sci-fi film soundtracks, "
            "constructing nostalgia for media experiences rather than lived ones."
        ),
    },
    {
        "id": "fado",
        "sound_description": (
            "Portuguese guitar (guitarra portuguesa) with tremolo picking and "
            "ornamented melodic lines, classical guitar accompaniment, solo vocal "
            "with dark timbre and controlled vibrato, minor-key melodies with "
            "chromatic descending lines, slow tempo around 60-80 BPM"
        ),
        "correct_association": "Portuguese fado / saudade and melancholic longing",
        "distractors": [
            "Spanish flamenco / duende and passionate intensity",
            "Italian opera / bel canto drama and tragic grandeur",
            "Greek rebetiko / urban underclass blues and exile",
        ],
        "explanation": (
            "The guitarra portuguesa's tremolo creates a shimmering, tear-like "
            "texture unique to the genre. Chromatic descending lines embody the "
            "concept of saudade — longing for something absent. The dark vocal "
            "timbre and slow tempo create an intimate confessional atmosphere "
            "rooted in Lisbon's maritime farewell culture."
        ),
    },
]

_ERA_SOUND_WORLD_TEMPLATES: List[Dict[str, Any]] = [
    {
        "era": "1950s",
        "region": "United States",
        "audio_characteristics": {
            "tempo_range": (100, 160),
            "spectral_centroid_range": (1000, 2500),
            "description": (
                "Simple arrangements with prominent vocals, limited frequency range "
                "due to recording technology, moderate to fast tempos, major-key "
                "tonality, clear beat emphasis"
            ),
        },
        "world_context": (
            "Post-WWII economic boom, suburbanization, rise of teenager as consumer "
            "demographic, early television, Cold War anxieties, racial segregation "
            "being challenged"
        ),
        "connection_explanation": (
            "Rock and roll's energetic tempo and simple structures reflected youthful "
            "optimism and rebellion in a prosperous but socially repressive era. "
            "The merging of Black and white musical traditions mirrored emerging "
            "civil rights consciousness."
        ),
    },
    {
        "era": "1970s",
        "region": "United States / United Kingdom",
        "audio_characteristics": {
            "tempo_range": (110, 130),
            "spectral_centroid_range": (2000, 4000),
            "description": (
                "Lush orchestral disco arrangements, four-on-the-floor kick patterns, "
                "prominent bass lines, bright mix with high spectral energy, "
                "synthesizer experimentation"
            ),
        },
        "world_context": (
            "Oil crisis and stagflation, Watergate disillusionment, Vietnam aftermath, "
            "disco as escapism, rise of LGBTQ+ visibility, punk as counter-reaction, "
            "early personal computing"
        ),
        "connection_explanation": (
            "Disco's escapist glamour provided relief from economic malaise and "
            "political cynicism. Its inclusive dance floors reflected liberation "
            "movements. Meanwhile, punk's stripped-down aggression was a direct "
            "sonic reaction to both economic hardship and disco's perceived excess."
        ),
    },
    {
        "era": "1990s",
        "region": "United States",
        "audio_characteristics": {
            "tempo_range": (80, 140),
            "spectral_centroid_range": (1500, 3500),
            "description": (
                "Grunge: distorted guitars, dynamic quiet-loud shifts, raw production. "
                "Hip-hop: sampled breakbeats, boom-bap drums, jazz samples. "
                "Wide dynamic range in alternative, compressed in pop"
            ),
        },
        "world_context": (
            "End of Cold War optimism, dot-com boom beginnings, Gulf War, "
            "LA riots, rise of the internet, Gen-X disaffection, "
            "multiculturalism discourse"
        ),
        "connection_explanation": (
            "Grunge's raw, unpolished sound rejected the artifice of 1980s excess, "
            "reflecting Gen-X disillusionment despite economic prosperity. "
            "Hip-hop's sampling culture repurposed existing music as commentary, "
            "and its regional diversification mirrored decentralizing cultural production."
        ),
    },
    {
        "era": "2008-2012",
        "region": "Global",
        "audio_characteristics": {
            "tempo_range": (126, 140),
            "spectral_centroid_range": (3000, 5000),
            "description": (
                "EDM crossover with pop, heavy side-chain compression, bright super-saw "
                "synths, build-and-drop structures, loudness war peaks, auto-tuned vocals"
            ),
        },
        "world_context": (
            "Global financial crisis 2008, Occupy movements, social media explosion, "
            "smartphone adoption, streaming beginning to replace physical media, "
            "Arab Spring"
        ),
        "connection_explanation": (
            "EDM's communal festival culture offered physical togetherness as social "
            "media created digital isolation. Its maximalist production reflected both "
            "the loudness war's technological arms race and a desire for overwhelming "
            "sensation during economic uncertainty. The drop structure mirrors "
            "tension-release cycles amplified by crisis-era anxiety."
        ),
    },
    {
        "era": "2020-2022",
        "region": "Global",
        "audio_characteristics": {
            "tempo_range": (60, 90),
            "spectral_centroid_range": (800, 2000),
            "description": (
                "Lo-fi aesthetics, bedroom production, muted/warm timbres, slow tempos, "
                "vinyl simulation artifacts, reduced dynamic range through gentle "
                "compression, intimate vocal recording"
            ),
        },
        "world_context": (
            "COVID-19 pandemic and lockdowns, social isolation, remote work shift, "
            "mental health crisis, BLM protests, political polarization, "
            "economic stimulus and uncertainty"
        ),
        "connection_explanation": (
            "Lo-fi's slow tempo and warm timbres matched the enforced stillness of "
            "lockdowns. Bedroom production aesthetics reflected literal home confinement. "
            "The vinyl crackle and analog simulation evoked a nostalgic longing for "
            "pre-pandemic normalcy. Reduced dynamic range created a comforting sonic "
            "cocoon during a period of heightened anxiety."
        ),
    },
    {
        "era": "1960s",
        "region": "United Kingdom / United States",
        "audio_characteristics": {
            "tempo_range": (100, 150),
            "spectral_centroid_range": (1500, 3500),
            "description": (
                "British Invasion guitar-driven pop evolving into psychedelic "
                "experimentation, studio as instrument (tape loops, reversed audio, "
                "panning effects), modal harmony, sitars and unconventional instruments"
            ),
        },
        "world_context": (
            "Vietnam War escalation, counterculture and hippie movement, Civil Rights Act, "
            "moon landing, sexual revolution, LSD and consciousness exploration, "
            "student protests worldwide"
        ),
        "connection_explanation": (
            "Psychedelic music's expanded sonic palette directly reflected consciousness "
            "expansion through hallucinogens and Eastern philosophy. Studio experimentation "
            "mirrored a generation questioning established structures. The movement from "
            "simple pop to complex forms tracked the decade's political radicalization."
        ),
    },
    {
        "era": "1980s",
        "region": "United States / United Kingdom / Japan",
        "audio_characteristics": {
            "tempo_range": (100, 130),
            "spectral_centroid_range": (2500, 5000),
            "description": (
                "Digital synthesizers (DX7, Fairlight), gated reverb drums, "
                "bright polished production, MIDI sequencing precision, heavy chorus "
                "and delay effects, wide stereo imaging, compressed vocals"
            ),
        },
        "world_context": (
            "Reagan/Thatcher economic liberalism, Cold War resurgence, MTV launch, "
            "consumer electronics boom, yuppie culture, AIDS crisis, "
            "Japanese economic dominance, Live Aid"
        ),
        "connection_explanation": (
            "Digital synthesizers and sequenced precision reflected faith in technology "
            "and consumer futurism. The polished, expensive production aesthetic matched "
            "Reagan-era conspicuous consumption. MTV made music visual, pushing sonic "
            "choices toward dramatic, attention-grabbing textures."
        ),
    },
    {
        "era": "1930s-1940s",
        "region": "United States",
        "audio_characteristics": {
            "tempo_range": (120, 200),
            "spectral_centroid_range": (800, 2000),
            "description": (
                "Big band swing with large horn sections, dance-oriented rhythms, "
                "crooner vocals with early microphone technique, limited frequency "
                "response from 78 RPM recording, call-and-response between sections"
            ),
        },
        "world_context": (
            "Great Depression and New Deal, rise of fascism in Europe, World War II, "
            "radio as dominant mass medium, Harlem Renaissance aftermath, "
            "wartime rationing and factory labor"
        ),
        "connection_explanation": (
            "Swing's exuberant energy served as Depression-era escapism through "
            "communal dance halls. Big band's military-like precision and large-group "
            "coordination mirrored wartime collective effort. The democratic "
            "interplay of soloists within the ensemble reflected New Deal ideals."
        ),
    },
    {
        "era": "1975-1982",
        "region": "United Kingdom / United States",
        "audio_characteristics": {
            "tempo_range": (140, 200),
            "spectral_centroid_range": (2000, 4500),
            "description": (
                "Short songs under three minutes, stripped-down three-chord structures, "
                "aggressive distorted guitars, shouted vocals with minimal technique, "
                "raw lo-fi recording quality, fast tempos"
            ),
        },
        "world_context": (
            "UK economic decline and unemployment, class stratification, "
            "disillusionment with hippie idealism, Thatcher election, "
            "No Future ethos, DIY zine culture, squatter communities"
        ),
        "connection_explanation": (
            "Punk's deliberate primitivism rejected both progressive rock's virtuosity "
            "and disco's glamour as inaccessible to working-class youth. Raw recording "
            "quality was both economic necessity and ideological statement. Short, fast "
            "songs reflected urgency and the refusal to elaborate or justify."
        ),
    },
    {
        "era": "1985-1995",
        "region": "Jamaica / United Kingdom",
        "audio_characteristics": {
            "tempo_range": (60, 80),
            "spectral_centroid_range": (500, 1500),
            "description": (
                "Heavy sub-bass below 80 Hz, sparse arrangements with wide spaces, "
                "dub echo and reverb effects, digital dancehall riddims, "
                "toasting vocal style, sound system engineering for physical bass impact"
            ),
        },
        "world_context": (
            "Jamaican urban poverty and political violence, Windrush generation in UK, "
            "sound system culture as community institution, Thatcher-era inner city "
            "decline, early rave scene cross-pollination"
        ),
        "connection_explanation": (
            "Sub-bass as physical experience originated in sound system competitions "
            "where bass power determined reputation. Dub's echo and space reflected "
            "displacement and diaspora. The sparse arrangement aesthetic turned "
            "economic limitation into a production philosophy that later seeded "
            "UK garage, dubstep, and grime."
        ),
    },
    {
        "era": "1988-1992",
        "region": "United Kingdom / Germany",
        "audio_characteristics": {
            "tempo_range": (120, 150),
            "spectral_centroid_range": (2500, 5000),
            "description": (
                "Four-on-the-floor kick, Roland TR-808/909 drum machines, "
                "acid squelch from TB-303, repetitive hypnotic structures, "
                "ecstatic builds, minimal vocals, 303 filter sweeps"
            ),
        },
        "world_context": (
            "Fall of the Berlin Wall, acid house moral panic in UK, "
            "Thatcher's Criminal Justice Bill, warehouse raves, "
            "ecstasy culture, second Summer of Love, early internet"
        ),
        "connection_explanation": (
            "Repetitive hypnotic structures and ecstasy created transcendent communal "
            "experiences in abandoned industrial spaces — reclaiming post-industrial "
            "decay as liberation. The anonymous, DJ-centered format rejected rock's "
            "individualism. Detroit techno's origins in Black futurism and Berlin's "
            "post-reunification techno scene both used electronic music to imagine "
            "futures beyond existing social structures."
        ),
    },
    {
        "era": "1997-2003",
        "region": "United States / Global",
        "audio_characteristics": {
            "tempo_range": (85, 105),
            "spectral_centroid_range": (2000, 4000),
            "description": (
                "TRL-era pop: maximalist vocal production, teen pop harmonies, "
                "boy-band choreography-synced rhythms, Neptunes/Timbaland syncopated "
                "beats, early Auto-Tune as effect, CD-quality brightness"
            ),
        },
        "world_context": (
            "Dot-com bubble peak and crash, Clinton-era prosperity, pre-9/11 optimism "
            "then post-9/11 anxiety, Napster and file-sharing revolution, "
            "reality TV emergence, globalization debates"
        ),
        "connection_explanation": (
            "Turn-of-millennium pop's glossy production reflected peak consumer optimism "
            "before the twin shocks of the dot-com crash and 9/11. Napster's disruption "
            "mirrored the internet's challenge to gatekeepers. The shift from teen pop's "
            "innocence to post-9/11 introspection in R&B tracks the era's emotional arc."
        ),
    },
    {
        "era": "2013-2018",
        "region": "United States / Global",
        "audio_characteristics": {
            "tempo_range": (60, 85),
            "spectral_centroid_range": (1000, 3000),
            "description": (
                "Trap hi-hat rolls with 808 sub-bass, mumble rap vocal style, "
                "heavy Auto-Tune as expressive tool, streaming-optimized short intros, "
                "playlist-friendly mood consistency, Latin trap fusion"
            ),
        },
        "world_context": (
            "Streaming dominance replacing album sales, social media as career launcher, "
            "opioid crisis, political polarization, smartphone ubiquity, "
            "rise of global south pop (reggaeton, K-pop crossover)"
        ),
        "connection_explanation": (
            "Trap's dominance paralleled streaming's reward of mood-consistent playlists "
            "over album narratives. Auto-Tuned mumble vocals prioritized vibe over lyrical "
            "clarity, matching scroll-culture attention spans. The opioid crisis found "
            "expression in lean-culture references and sedated vocal delivery."
        ),
    },
    {
        "era": "1950s-1960s",
        "region": "Brazil",
        "audio_characteristics": {
            "tempo_range": (80, 130),
            "spectral_centroid_range": (1000, 2500),
            "description": (
                "Nylon-string guitar with syncopated samba-derived rhythms, "
                "whispered intimate vocal delivery, cool jazz harmonies with "
                "extended chords, minimal percussion, gentle dynamic range"
            ),
        },
        "world_context": (
            "Brazilian modernization under Kubitschek, construction of Brasilia, "
            "Rio de Janeiro beach culture, emerging middle class, "
            "cultural exchange with US jazz scene, pre-military-dictatorship optimism"
        ),
        "connection_explanation": (
            "Bossa nova's quiet intimacy rejected both American jazz's volume and "
            "traditional samba's carnival energy, creating a sophisticated urbane sound "
            "for Brazil's aspiring cosmopolitan class. The whispered vocals and cool "
            "harmonies reflected beachside leisure culture and mid-century optimism "
            "about Brazilian modernity."
        ),
    },
    {
        "era": "2022-2025",
        "region": "Global",
        "audio_characteristics": {
            "tempo_range": (90, 140),
            "spectral_centroid_range": (1500, 4000),
            "description": (
                "AI-assisted production tools, genre-fluid compositions blending "
                "regional styles, hyperpop influence on mainstream, vinyl-digital "
                "hybrid aesthetics, spatial audio mixing for headphones, "
                "micro-genre proliferation"
            ),
        },
        "world_context": (
            "Post-pandemic reopening, generative AI disruption, TikTok as primary "
            "discovery platform, 15-second hook optimization, vinyl revival alongside "
            "streaming dominance, creator economy, climate anxiety"
        ),
        "connection_explanation": (
            "AI tools democratized production while raising authenticity questions. "
            "TikTok's short-form discovery pushed songs toward immediate hooks, "
            "compressing the time to emotional payoff. Genre fluidity reflected "
            "algorithm-driven listening that dissolved traditional genre communities "
            "into individualized taste profiles."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _stable_id(task_id: int, *parts: str) -> str:
    """Generate a deterministic question ID from task number and content parts."""
    raw = f"task{task_id}:" + "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _as_float(value: Any) -> Optional[float]:
    """Scalar from a feature stored as a number, a list, or a numpy array (as loaded from h5)."""
    if value is None:
        return None
    if hasattr(value, "mean") and hasattr(value, "size"):
        return float(value.mean()) if value.size else None
    if isinstance(value, (list, tuple)):
        return sum(value) / len(value) if value else None
    return float(value)


def _centroid(features: FeatureDict) -> Optional[float]:
    sc = _as_float(features.get("spectral_centroid_mean"))
    return sc if sc is not None else _as_float(features.get("spectral_centroid"))


def _describe_features(features: FeatureDict) -> str:
    """Produce a concise natural-language summary of audio features."""
    parts: List[str] = []

    tempo = _as_float(features.get("tempo")) or 0.0
    if tempo < 70:
        parts.append(f"very slow tempo ({tempo:.0f} BPM)")
    elif tempo < 100:
        parts.append(f"slow-moderate tempo ({tempo:.0f} BPM)")
    elif tempo < 130:
        parts.append(f"moderate tempo ({tempo:.0f} BPM)")
    elif tempo < 150:
        parts.append(f"fast tempo ({tempo:.0f} BPM)")
    else:
        parts.append(f"very fast tempo ({tempo:.0f} BPM)")

    sc = _centroid(features)
    if sc is not None:
        if sc < 1500:
            parts.append(f"dark/warm timbre (spectral centroid {sc:.0f} Hz)")
        elif sc < 3000:
            parts.append(f"balanced brightness (spectral centroid {sc:.0f} Hz)")
        else:
            parts.append(f"bright/sharp timbre (spectral centroid {sc:.0f} Hz)")

    hp = _as_float(features.get("harmonic_percussive_ratio"))
    if hp is not None:
        if hp > 5:
            parts.append("heavily tonal/harmonic content")
        elif hp > 1:
            parts.append("slightly more harmonic than percussive")
        else:
            parts.append("dominated by percussive elements")

    dr = _as_float(features.get("dynamic_range_db"))
    if dr is not None:
        if dr < 10.0:
            parts.append("compressed dynamics")
        elif dr < 14.0:
            parts.append("moderate dynamic range")
        else:
            parts.append("wide dynamic range")

    key_idx = _as_float(features.get("estimated_key"))
    if key_idx is not None and 0 <= key_idx < 12:
        parts.append(f"key center of {_KEY_NAMES[int(key_idx)]}")

    return ", ".join(parts) if parts else "no feature data available"


def _pick_distractors(
    correct: str,
    pool: List[str],
    n: int = 3,
    rng: random.Random | None = None,
) -> List[str]:
    """Select *n* distractor strings from *pool*, excluding *correct*."""
    rng = rng or random.Random()
    available = [d for d in pool if d != correct]
    if len(available) < n:
        generic = [
            "The interaction of these features produces cognitive dissonance "
            "that listeners resolve as calm detachment rather than engagement.",
            "The spectral and rhythmic characteristics cancel each other out, "
            "resulting in an emotionally neutral listening experience.",
            "The production techniques used create a sense of artificiality "
            "that distances the listener from any genuine emotional response.",
            "The timbral warmth overrides the rhythmic energy, producing "
            "a meditative rather than arousing effect.",
        ]
        available.extend(g for g in generic if g not in available)
    return rng.sample(available, min(n, len(available)))


# ---------------------------------------------------------------------------
# Task 1: feature-grounded reasoning options
# ---------------------------------------------------------------------------
# All four options share one template and name no emotion, so neither wording nor writing
# style identifies the answer. The
# correct option describes every feature accurately; each distractor flips one or two features
# to their opposites along with the opposite's perceptual mechanism, so it reads as coherent
# but contradicts the measurements in the question.
# Only features well clear of their split are flipped, so no distractor is borderline.

# Medians over the 389 songs, so each level is right about half the time. With hp split at 1.0, 311 of
# 319 correct answers said "sustained, pitched", and a model guessing without audio scored 34%.
_TASK1_SPLITS: Dict[str, float] = {"tempo": 112.0, "sc": 2110.0, "hp": 3.0, "dr": 12.0}
_TASK1_MARGIN = 0.15

# High and low (phrase + mechanism) have identical total length per feature, so every option
# has the same length and "pick the longest" carries no signal (enforced in tests).
_TASK1_PHRASES: Dict[str, Dict[str, Tuple[str, str]]] = {
    "tempo": {
        "high": ("brisk tempo", "drives physiological arousal through motor entrainment"),
        "low": ("unhurried tempo", "leaves space for reflective, low-arousal listening"),
    },
    "sc": {
        "high": ("bright timbre", "adds a perceptual sharpness that heightens alertness"),
        "low": ("dark, warm timbre", "softens every attack and draws attention inwards"),
    },
    "hp": {
        "high": ("more sustained, pitched texture", "holds tonal content that invites emotional dwelling"),
        "low": ("more percussive texture", "puts sharp transients up front so the beat keeps pushing on"),
    },
    "dr": {
        "high": ("wide dynamic range", "creates swells and contrasts that build and release"),
        "low": ("compressed dynamics", "keeps the energy levels dense and even all the way"),
    },
}

# No emotion is named: it was derived from the same feature levels, so naming it let a model pick
# the option consistent with it without listening (blind accuracy on the audio version was 92%).
_TASK1_TEMPLATES = [
    "The {tempo} {tempo_m}, and the {sc} {sc_m}. The {hp} {hp_m}, while the {dr} "
    "{dr_m}. Together these shape how the track feels.",
    "The {sc} {sc_m}, and the {hp} {hp_m}. Alongside them, the {tempo} {tempo_m} and "
    "the {dr} {dr_m}, and that combination sets the track's feel.",
    "Start with the {dr}: it {dr_m}. The {tempo} {tempo_m}, the {sc} {sc_m}, and the "
    "{hp} {hp_m}. Together they set the track's character.",
]


def _task1_measurements(features: FeatureDict) -> Optional[Dict[str, float]]:
    values = {
        "tempo": _as_float(features.get("tempo")),
        "sc": _centroid(features),
        "hp": _as_float(features.get("harmonic_percussive_ratio")),
        "dr": _as_float(features.get("dynamic_range_db")),
    }
    return None if any(v is None for v in values.values()) else values  # type: ignore[return-value]


def _task1_emotion(levels: Dict[str, str]) -> str:
    """Transparent arousal/brightness rules, so the target emotion follows from the features."""
    if levels["tempo"] == "high":
        if levels["hp"] == "low" and levels["dr"] == "low":
            return "aggression"
        return "energy"
    if levels["sc"] == "low":
        return "melancholy"
    if levels["hp"] == "high":
        return "nostalgia"
    return "tension"


def _task1_option(template: str, levels: Dict[str, str], emotion_desc: str) -> str:
    fills: Dict[str, str] = {}
    for feat, level in levels.items():
        phrase, mechanism = _TASK1_PHRASES[feat][level]
        fills[feat] = phrase
        fills[f"{feat}_m"] = mechanism
    text = template.format(emotion=emotion_desc, **fills)
    return text[0].upper() + text[1:]


def _build_task1_item(
    features: FeatureDict, rng: random.Random,
) -> Optional[Tuple[str, Dict[str, float], List[str], int, List[List[str]]]]:
    """Return (emotion_key, measurements, options, correct_idx, flips per distractor) or None."""
    values = _task1_measurements(features)
    if values is None:
        return None
    levels = {k: "high" if v >= _TASK1_SPLITS[k] else "low" for k, v in values.items()}
    clear = [k for k, v in values.items() if abs(v - _TASK1_SPLITS[k]) / _TASK1_SPLITS[k] >= _TASK1_MARGIN]
    if len(clear) < 2:
        return None
    # Flip sets are chosen so each feature is right in exactly 2 of 4 options and all options are
    # equally far apart; with single flips the answer would be the per-feature majority option.
    if len(clear) >= 3:
        a, b, c = rng.sample(clear, 3)
        flips = [[a, b], [b, c], [a, c]]
    else:
        flips = [[clear[0]], [clear[1]], list(clear)]

    emotion_key = _task1_emotion(levels)
    emotion_desc = _EMOTION_MAP[emotion_key]["description"]
    template = rng.choice(_TASK1_TEMPLATES)
    correct = _task1_option(template, levels, emotion_desc)
    distractors = []
    for flip in flips:
        alt = dict(levels)
        for k in flip:
            alt[k] = "low" if levels[k] == "high" else "high"
        distractors.append(_task1_option(template, alt, emotion_desc))

    order = list(range(4))
    rng.shuffle(order)
    texts = [correct] + distractors
    options = [texts[i] for i in order]
    flips_by_option = [([] if i == 0 else flips[i - 1]) for i in order]
    return emotion_key, values, options, order.index(0), flips_by_option


# ---------------------------------------------------------------------------
# Task 2: source attribution from stem-ablation measurements
# ---------------------------------------------------------------------------
# Ground truth comes from jobs/phase2/extract_ablation.pbs: the full mix and each "minus one stem"
# mix are measured, and the stem whose removal changes an attribute most is its driver. An earlier
# version picked the answer with a hand-written heuristic that rewarded beat-tracker errors, and
# text models scored below chance on it.
# direction "drop": removing the driver lowers the measure; "rise": removing it raises the measure
_TASK2_ATTRS: Dict[str, Dict[str, str]] = {
    "loudness": {"desc": "overall loudness and energy", "key": "rms_mean", "direction": "drop",
                 "measure": "mix RMS loudness"},
    "rhythm": {"desc": "rhythmic drive (the push of the beat)", "key": "onset_mean", "direction": "drop",
               "measure": "spectral flux (energy of new sound events)"},
    "brightness": {"desc": "brightness (bright, treble-heavy sparkle)", "key": "centroid_weighted",
                   "direction": "drop", "measure": "energy-weighted spectral centroid"},
    "warmth": {"desc": "warmth and low-end weight", "key": "centroid_weighted", "direction": "rise",
               "measure": "energy-weighted spectral centroid (it rises when a warm stem is removed)"},
    "tonality": {"desc": "tonal, pitched character (as opposed to noise-like texture)", "key": "flatness_weighted",
                 "direction": "rise", "measure": "spectral flatness (it rises when a tonal stem is removed)"},
}
_TASK2_MIN_EFFECT = 0.05   # driver's removal must change the measure by at least 5%
_TASK2_MIN_RATIO = 1.5     # ... and by 1.5x the runner-up, so the label isn't a coin flip


def _task2_effects(ablation: Dict[str, Any], attr: str) -> Dict[str, float]:
    spec = _TASK2_ATTRS[attr]
    full = float(ablation["full"][spec["key"]])
    if abs(full) < 1e-12:
        return {}
    sign = -1.0 if spec["direction"] == "drop" else 1.0
    return {k[len("minus_"):]: sign * (float(v[spec["key"]]) - full) / abs(full)
            for k, v in ablation.items() if k.startswith("minus_")}


def _task2_clear_driver(effects: Dict[str, float]) -> Optional[Tuple[str, str, float, float]]:
    """(driver, runner_up, driver_effect, runner_up_effect) if the driver is unambiguous."""
    if len(effects) < 2:
        return None
    ranked = sorted(effects.items(), key=lambda kv: kv[1], reverse=True)
    (best, e1), (second, e2) = ranked[0], ranked[1]
    if e1 < _TASK2_MIN_EFFECT or e1 < _TASK2_MIN_RATIO * max(e2, 0.0) or e1 - e2 < 0.03:
        return None
    return best, second, e1, e2


def _describe_stem_for_task2(sf_: FeatureDict, mix_rms: float) -> str:
    parts = []
    rms = _as_float(sf_.get("rms_mean"))
    if rms is not None and mix_rms > 0:
        parts.append(f"{rms / mix_rms * 100:.0f}% of the mix's RMS level")
    act = _as_float(sf_.get("activity_ratio"))
    if act is not None:
        parts.append(f"audible in {act * 100:.0f}% of the song")
    sc = _centroid(sf_)
    if sc is not None:
        parts.append(f"energy-weighted spectral centroid {sc:.0f} Hz")
    onset = _as_float(sf_.get("onset_strength_mean"))
    if onset is not None:
        parts.append(f"mean onset strength {onset:.2f}")
    hp = _as_float(sf_.get("harmonic_percussive_ratio"))
    if hp is not None:
        parts.append(f"harmonic/percussive ratio {hp:.2f}")
    return ", ".join(parts) or "no measurements"


def _task3_reference(song_a: Dict[str, Any], song_b: Dict[str, Any], pair: Dict[str, Any]) -> str:
    """Measured similarities and differences between two songs, as a grading reference."""
    fa, fb = song_a.get("features", {}), song_b.get("features", {})
    same, diff = [], []

    def compare(label: str, a: Optional[float], b: Optional[float], rel_tol: float, fmt: str) -> None:
        if a is None or b is None:
            return
        target = same if abs(a - b) <= rel_tol * max(abs(a), abs(b), 1e-9) else diff
        target.append(f"{label} ({fmt.format(a)} vs {fmt.format(b)})")

    compare("tempo", _as_float(fa.get("tempo")), _as_float(fb.get("tempo")), 0.08, "{:.0f} BPM")
    compare("brightness / spectral centroid", _centroid(fa), _centroid(fb), 0.15, "{:.0f} Hz")
    compare("harmonic-to-percussive balance", _as_float(fa.get("harmonic_percussive_ratio")),
            _as_float(fb.get("harmonic_percussive_ratio")), 0.35, "{:.2f}")
    compare("dynamic range", _as_float(fa.get("dynamic_range_db")), _as_float(fb.get("dynamic_range_db")), 0.2,
            "{:.1f} dB")
    compare("spectral bandwidth", _as_float(fa.get("spectral_bandwidth")), _as_float(fb.get("spectral_bandwidth")),
            0.15, "{:.0f} Hz")
    parts = []
    if pair.get("same_artist"):
        parts.append(f"Both songs are by {song_a.get('artist')}.")
    ga, gb = song_a.get("genre"), song_b.get("genre")
    if ga and gb and ga != "unknown" and gb != "unknown":
        parts.append(f"Genres: {ga} vs {gb}{' (same genre)' if ga == gb else ''}.")
    parts.append("Shared: " + ("; ".join(same) if same else "no single measured feature is close") + ".")
    parts.append("Different: " + ("; ".join(diff) if diff else "none of the measured features differ much") + ".")
    parts.append("The embedding similarity reflects overall timbre and texture, which these summary "
                 "features capture only partly.")
    return " ".join(parts)


def _task1_measurement_text(values: Dict[str, float]) -> str:
    return (
        f"tempo {values['tempo']:.0f} BPM (median track 112); spectral centroid {values['sc']:.0f} Hz "
        f"(median 2110); harmonic-to-percussive energy ratio {values['hp']:.2f} (above 1 means more "
        f"harmonic than percussive; median 3.1); dynamic range {values['dr']:.1f} dB (gap between the "
        f"loud and quiet passages; median 12)"
    )


# ---------------------------------------------------------------------------
# QuestionGenerator
# ---------------------------------------------------------------------------

class QuestionGenerator:
    """Generates benchmark questions for all five MusicWhy evaluation tasks.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self.rng = random.Random(seed)
        self._question_count = 0

    def _task_rng(self, task_id: int) -> random.Random:
        # One stream per task, so changing one task's generator doesn't reshuffle the others (a Task 1
        # fix once replaced 248 of 358 Task 2 questions through the shared stream)
        return random.Random(f"{self._seed}-task{task_id}")

    # ------------------------------------------------------------------
    # Task 1: Reasoning QA
    # ------------------------------------------------------------------

    def generate_task_1_questions(
        self,
        data: List[SongRecord],
        n_questions: int = 100,
    ) -> List[Question]:
        """Generate multiple-choice reasoning QA questions.

        For each song, present measured features and an emotion derived from them, and ask
        which explanation is consistent with the measurements. All options argue for that
        emotion; distractors misdescribe one or two clear-cut features (see _build_task1_item).
        Songs with fewer than two clear-cut features are skipped.

        Parameters
        ----------
        data : list of SongRecord
            Each record must contain ``features`` (FeatureDict) and optionally
            ``song_id``, ``emotion_label``, ``audio_path``.
        n_questions : int
            Number of questions to generate (capped by ``len(data)``).

        Returns
        -------
        list of Question
        """
        self.rng = self._task_rng(1)
        questions: List[Question] = []
        sampled = self.rng.sample(data, min(n_questions, len(data)))

        skipped = 0
        for record in sampled:
            features = record.get("features", {})
            song_id = record.get("song_id", str(uuid.uuid4()))
            audio_ref = record.get("audio_path", "")
            item = _build_task1_item(features, self.rng)
            if item is None:
                skipped += 1
                continue
            emotion_key, values, options, correct_idx, flips = item
            emotion_desc = _EMOTION_MAP[emotion_key]["description"]
            measurements = _task1_measurement_text(values)
            correct_reasoning = options[correct_idx]
            option_labels = ["A", "B", "C", "D"]

            question_text = (
                f"A track was measured with: {measurements}.\n\n"
                f"Which explanation of how this track's sound shapes its feel is consistent "
                f"with these measurements?\n\n"
                + "\n".join(f"{option_labels[i]}) {options[i]}" for i in range(len(options)))
            )

            q: Question = {
                "task_id": 1,
                "question_id": _stable_id(1, song_id, emotion_key),
                "modality": "text",
                "question_text": question_text,
                "audio_reference": "",
                "options": {option_labels[i]: options[i] for i in range(len(options))},
                "correct_answer": option_labels[correct_idx],
                "reference_explanation": correct_reasoning,
                "metadata": {
                    "song_id": song_id,
                    "emotion": emotion_key,
                    "features_summary": measurements,
                    "flipped_features": {option_labels[i]: flips[i] for i in range(len(options))},
                },
            }
            questions.append(q)
            # Same options without the measurements: the text version can be solved by checking numbers
            # against thresholds (text-only models scored 96%), so audio models also get one that
            # requires listening
            audio_q = dict(q, metadata=dict(q["metadata"]))
            audio_q.update({
                "question_id": _stable_id(1, song_id, emotion_key, "audio"),
                "modality": "audio",
                "audio_reference": audio_ref,
                "question_text": (
                    f"Listen to the track. Which explanation of how its sound shapes its feel "
                    f"matches what you hear?\n\n"
                    + "\n".join(f"{option_labels[i]}) {options[i]}" for i in range(len(options)))
                ),
            })
            questions.append(audio_q)

        logger.info("Task 1: generated %d reasoning QA questions (%d songs skipped: <2 clear-cut features)",
                    len(questions), skipped)
        return questions

    # ------------------------------------------------------------------
    # Task 2: Source-Aware Emotion Attribution
    # ------------------------------------------------------------------

    def generate_task_2_questions(
        self,
        data: List[SongRecord],
        n_questions: int = 100,
    ) -> List[Question]:
        """Generate source-aware emotion attribution questions.

        Each song record must include ``stem_features`` mapping stem names
        to their extracted features, plus an ``emotion_label``.  The question
        asks which stem is most responsible for the target emotion.

        Parameters
        ----------
        data : list of SongRecord
            Must contain ``stem_features``, ``features``, and optionally
            ``emotion_label``, ``correct_stem``, ``audio_path``, ``song_id``.
        n_questions : int
            Target question count.
        """
        self.rng = self._task_rng(2)
        questions: List[Question] = []
        eligible = [r for r in data if r.get("ablation") and len(r.get("stem_features") or {}) == 4]
        sampled = self.rng.sample(eligible, min(n_questions, len(eligible)))
        driver_counts: Dict[str, int] = {}
        no_clear = 0

        for record in sampled:
            ablation = record["ablation"]
            stem_features: StemFeatures = record["stem_features"]
            song_id = record.get("song_id", str(uuid.uuid4()))
            audio_ref = record.get("audio_path", "")

            # Attributes with an unambiguous driver; among them prefer the one whose driver stem is
            # rarest so far, so answers spread across stems instead of "drums" for every rhythm item
            clear = {}
            for attr in _TASK2_ATTRS:
                found = _task2_clear_driver(_task2_effects(ablation, attr))
                if found:
                    clear[attr] = found
            if not clear:
                no_clear += 1
                continue
            attr = min(clear, key=lambda a: (driver_counts.get(clear[a][0], 0), self.rng.random()))
            driver, runner_up, e1, e2 = clear[attr]
            driver_counts[driver] = driver_counts.get(driver, 0) + 1
            spec = _TASK2_ATTRS[attr]

            stem_names = sorted(stem_features)
            option_order = list(stem_names)
            self.rng.shuffle(option_order)
            option_labels = ["A", "B", "C", "D"]
            options = {option_labels[i]: option_order[i] for i in range(4)}
            correct_label = option_labels[option_order.index(driver)]
            options_text = "\n".join(f"{option_labels[i]}) {option_order[i]}" for i in range(4))
            ask = (f"Which stem is most responsible for the song's {spec['desc']}? In other words, "
                   f"removing which stem would reduce it the most?")

            mix_rms = float(ablation["full"]["rms_mean"])
            stem_lines = "\n".join(f"  - {sn}: {_describe_stem_for_task2(stem_features[sn], mix_rms)}"
                                   for sn in stem_names)
            explanation = (
                f"Measured by removing each stem and re-analysing the mix: without the {driver}, "
                f"the {spec['measure']} changes by {e1 * 100:.0f}% in the direction that weakens "
                f"{spec['desc']}, versus {e2 * 100:.0f}% for the next most influential stem ({runner_up})."
            )
            effects = {s: round(v, 4) for s, v in _task2_effects(ablation, attr).items()}
            base_meta = {"song_id": song_id, "attribute": attr, "stems": stem_names,
                         "correct_stem": driver, "removal_effects": effects}

            variants = {
                "text": (f"A song has been separated into four stems, measured as follows:\n{stem_lines}\n\n"
                         f"{ask}\n\n{options_text}"),
                "audio": (f"Listen to the song. It can be separated into four stems: bass, drums, vocals and "
                          f"other (everything else, e.g. guitars, keys).\n\n{ask}\n\n{options_text}"),
            }
            for modality, text in variants.items():
                questions.append({
                    "task_id": 2,
                    "question_id": _stable_id(2, song_id, attr, modality),
                    "modality": modality,
                    "question_text": text,
                    "audio_reference": audio_ref if modality == "audio" else "",
                    "options": options,
                    "correct_answer": correct_label,
                    "reference_explanation": explanation,
                    "metadata": dict(base_meta),
                })

        logger.info("Task 2: generated %d source-attribution questions from %d songs (%d songs had no "
                    "unambiguous driver); drivers %s", len(questions), len(questions) // 2, no_clear, driver_counts)
        return questions

    # ------------------------------------------------------------------
    # Task 3: Song Relationship Explanation
    # ------------------------------------------------------------------

    def generate_task_3_questions(
        self,
        data: List[Dict[str, Any]],
        n_questions: int = 100,
    ) -> List[Question]:
        """Generate song-relationship explanation questions.

        Parameters
        ----------
        data : list of dict
            Each dict describes a pair of similar songs:
            ``song_a`` and ``song_b`` (SongRecord), ``similarity_score`` (float),
            and optionally ``reference_explanation``.
        n_questions : int
            Target question count.
        """
        self.rng = self._task_rng(3)
        questions: List[Question] = []
        # Rank-1 neighbours only: the clearest relationships, and one MC item per anchor song
        usable = [p for p in data if p.get("similarity_rank", 1) == 1 and len(p.get("distractors") or []) >= 3]
        sampled = self.rng.sample(usable, min(n_questions, len(usable)))
        labels = ["A", "B", "C", "D"]

        for pair in sampled:
            song_a = pair["song_a"]
            song_b = pair["song_b"]
            sim_score = pair.get("similarity_score", 0.0)
            a_id = song_a.get("song_id", "unknown_a")
            b_id = song_b.get("song_id", "unknown_b")
            a_feats = song_a.get("features", {})
            b_feats = song_b.get("features", {})
            a_desc = _describe_features(a_feats)
            b_desc = _describe_features(b_feats)
            a_title = song_a.get("title", a_id)
            b_title = song_b.get("title", b_id)
            meta = {"song_a_id": a_id, "song_b_id": b_id, "similarity_score": sim_score,
                    "same_artist": bool(pair.get("same_artist")),
                    "same_genre": bool(song_a.get("genre") and song_a.get("genre") == song_b.get("genre")
                                       and song_a.get("genre") != "unknown")}
            reference = _task3_reference(song_a, song_b, pair)

            # Multiple choice: which candidate is the embedding nearest neighbour of song A?
            candidates = [song_b] + list(pair["distractors"][:3])
            order = list(range(4))
            self.rng.shuffle(order)
            opts = [candidates[i] for i in order]
            correct = labels[order.index(0)]
            options = {labels[i]: f"\"{c.get('title', '')}\" by {c.get('artist', '')}" for i, c in enumerate(opts)}
            option_desc = "\n".join(f"{labels[i]}) \"{c.get('title', '')}\": {_describe_features(c.get('features', {}))}"
                                    for i, c in enumerate(opts))
            mc_meta = dict(meta, candidate_ids=[c.get("song_id") for c in opts])
            questions.append({
                "task_id": 3,
                "question_id": _stable_id(3, a_id, b_id, "mc_text"),
                "modality": "text",
                "question_text": (f"Song A (\"{a_title}\"): {a_desc}\n\nWhich of these songs sounds most "
                                  f"similar to Song A overall (timbre, texture, production)?\n\n{option_desc}"),
                "audio_reference": "",
                "options": options,
                "correct_answer": correct,
                "reference_explanation": reference,
                "metadata": mc_meta,
            })
            questions.append({
                "task_id": 3,
                "question_id": _stable_id(3, a_id, b_id, "mc_audio"),
                "modality": "audio",
                "question_text": ("Listen to Song A and to the four candidate songs. Which candidate sounds most "
                                  "similar to Song A overall (timbre, texture, production)?\n\n"
                                  + "\n".join(f"{k}) {v}" for k, v in options.items())),
                "audio_reference": {"song_a": song_a.get("audio_path", ""),
                                    "candidates": {labels[i]: c.get("audio_path", "") for i, c in enumerate(opts)}},
                "options": options,
                "correct_answer": correct,
                "reference_explanation": reference,
                "metadata": mc_meta,
            })

            # Open-ended: explain the relationship; graded against the measured reference
            questions.append({
                "task_id": 3,
                "question_id": _stable_id(3, a_id, b_id),
                "modality": "text",
                "question_text": (
                    f"Two songs were found to be close neighbours in a music-audio embedding space.\n\n"
                    f"Song A (\"{a_title}\"): {a_desc}\n"
                    f"Song B (\"{b_title}\"): {b_desc}\n\n"
                    f"Explain what connects these two songs and how they differ. Which specific audio "
                    f"features, production choices, or musical characteristics are shared?"
                ),
                "audio_reference": {"song_a": song_a.get("audio_path", ""), "song_b": song_b.get("audio_path", "")},
                "options": None,
                "correct_answer": None,
                "reference_explanation": reference,
                "metadata": meta,
            })

        logger.info("Task 3: generated %d relationship questions from %d pairs", len(questions), len(sampled))
        return questions

    # ------------------------------------------------------------------
    # Task 4: Symbolic Sound Identification
    # ------------------------------------------------------------------

    def generate_task_4_questions(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        n_questions: int = 50,
    ) -> List[Question]:
        """Generate symbolic-sound identification questions.

        Uses built-in cultural sound profiles and any additional profiles
        supplied in *data* (each with ``sound_description``,
        ``correct_association``, ``distractors``, ``explanation``).

        Parameters
        ----------
        data : list of dict, optional
            Additional cultural sound profiles beyond the built-in set.
        n_questions : int
            Target question count.
        """
        self.rng = self._task_rng(4)
        profiles = list(_CULTURAL_SOUND_PROFILES)
        if data:
            profiles.extend(data)

        questions: List[Question] = []
        # One question per profile: repeats are the same item with reshuffled letters and the
        # same question_id, which inflates counts and leaks across train/test splits
        selected = self.rng.sample(range(len(profiles)), min(n_questions, len(profiles)))

        for idx in selected:
            profile = profiles[idx]
            pid = profile.get("id", str(idx))
            sound_desc = profile["sound_description"]
            correct = profile["correct_association"]
            distractors = list(profile.get("distractors", []))

            # Ensure exactly 3 distractors
            while len(distractors) < 3:
                distractors.append("No specific cultural association can be identified.")
            distractors = distractors[:3]

            options_list = [correct] + distractors
            self.rng.shuffle(options_list)
            correct_idx = options_list.index(correct)
            labels = ["A", "B", "C", "D"]

            question_text = (
                f"Consider a track with the following sonic characteristics:\n"
                f"\"{sound_desc}\"\n\n"
                f"What cultural or psychological association does this sound most "
                f"strongly evoke?\n\n"
                + "\n".join(f"{labels[i]}) {options_list[i]}" for i in range(len(options_list)))
            )

            q: Question = {
                "task_id": 4,
                "question_id": _stable_id(4, pid),
                "question_text": question_text,
                "audio_reference": profile.get("audio_path", ""),
                "options": {labels[i]: options_list[i] for i in range(len(options_list))},
                "correct_answer": labels[correct_idx],
                "reference_explanation": profile.get("explanation", ""),
                "metadata": {
                    "profile_id": pid,
                    "cultural_context": correct,
                },
            }
            questions.append(q)

        logger.info("Task 4: generated %d symbolic-sound questions", len(questions))
        return questions

    # ------------------------------------------------------------------
    # Task 5: Sound-World Connection
    # ------------------------------------------------------------------

    def generate_task_5_questions(
        self,
        data: Optional[List[Dict[str, Any]]] = None,
        n_questions: int = 50,
    ) -> List[Question]:
        """Generate sound-world connection questions.

        Pairs audio features from a specific era with world context and asks
        why this music emerged at that time.

        Parameters
        ----------
        data : list of dict, optional
            Additional era-sound-world templates beyond built-in ones.  Each
            must have ``era``, ``region``, ``audio_characteristics``,
            ``world_context``, and ``connection_explanation``.
        n_questions : int
            Target question count.
        """
        self.rng = self._task_rng(5)
        templates = list(_ERA_SOUND_WORLD_TEMPLATES)
        if data:
            templates.extend(data)

        questions: List[Question] = []
        # One question per template: repeats would be verbatim copies with the same question_id
        selected = self.rng.sample(range(len(templates)), min(n_questions, len(templates)))

        for idx in selected:
            tmpl = templates[idx]
            era = tmpl["era"]
            region = tmpl["region"]
            audio_chars = tmpl["audio_characteristics"]
            world_ctx = tmpl["world_context"]
            ref_explanation = tmpl["connection_explanation"]

            audio_desc = audio_chars.get("description", "")
            tempo_lo, tempo_hi = audio_chars.get("tempo_range", (100, 130))
            sc_lo, sc_hi = audio_chars.get("spectral_centroid_range", (1500, 3000))

            question_text = (
                f"During the {era} in {region}, popular music had these characteristics:\n"
                f"\"{audio_desc}\"\n"
                f"(Typical tempo: {tempo_lo}-{tempo_hi} BPM, spectral centroid: "
                f"{sc_lo}-{sc_hi} Hz)\n\n"
                f"World context at the time:\n\"{world_ctx}\"\n\n"
                f"Why did music with these sonic characteristics emerge during this "
                f"specific period? Explain the connection between the sound and the "
                f"cultural/economic/political context."
            )

            q: Question = {
                "task_id": 5,
                "question_id": _stable_id(5, era, region),
                "question_text": question_text,
                "audio_reference": tmpl.get("audio_path", ""),
                "options": None,  # open-ended
                "correct_answer": None,  # open-ended
                "reference_explanation": ref_explanation,
                "metadata": {
                    "era": era,
                    "region": region,
                    "tempo_range": (tempo_lo, tempo_hi),
                    "spectral_centroid_range": (sc_lo, sc_hi),
                },
            }
            questions.append(q)

        logger.info("Task 5: generated %d sound-world questions", len(questions))
        return questions

    # ------------------------------------------------------------------
    # Convenience: generate all tasks at once
    # ------------------------------------------------------------------

    def generate_all(
        self,
        task1_data: List[SongRecord],
        task2_data: List[SongRecord],
        task3_data: List[Dict[str, Any]],
        task4_data: Optional[List[Dict[str, Any]]] = None,
        task5_data: Optional[List[Dict[str, Any]]] = None,
        n_per_task: int = 100,
    ) -> List[Question]:
        """Generate questions for all five tasks and return a combined list."""
        all_questions: List[Question] = []
        all_questions.extend(self.generate_task_1_questions(task1_data, n_per_task))
        all_questions.extend(self.generate_task_2_questions(task2_data, n_per_task))
        all_questions.extend(self.generate_task_3_questions(task3_data, n_per_task))
        all_questions.extend(self.generate_task_4_questions(task4_data, n_per_task))
        all_questions.extend(self.generate_task_5_questions(task5_data, n_per_task))
        logger.info("Generated %d total questions across 5 tasks", len(all_questions))
        return all_questions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_emotion_stem(
        emotion_key: str,
        stem_features: StemFeatures,
        full_features: FeatureDict,
    ) -> str:
        """Heuristic: pick the stem whose features best explain *emotion_key*.

        Uses simple rules based on known psychoacoustic relationships.
        """
        scores: Dict[str, float] = {}
        emotion_feature_keys = _EMOTION_MAP.get(emotion_key, {}).get("feature_keys", [])

        for stem_name, sf in stem_features.items():
            score = 0.0

            # Higher tempo contribution -> energy/aggression
            if "tempo" in emotion_feature_keys:
                stem_tempo = sf.get("tempo", 0)
                full_tempo = full_features.get("tempo", 120)
                if full_tempo > 0:
                    score += abs(stem_tempo - full_tempo) / full_tempo

            # Higher spectral centroid -> brightness, energy
            if "spectral_centroid" in emotion_feature_keys:
                sc_raw = sf.get("spectral_centroid")
                stem_sc = (sum(sc_raw) / len(sc_raw)) if isinstance(sc_raw, list) and sc_raw else 0
                score += stem_sc / 5000.0

            # Harmonic content -> melancholy, nostalgia
            if "harmonic_percussive_ratio" in emotion_feature_keys:
                hp = sf.get("harmonic_percussive_ratio", 1.0)
                if emotion_key in ("melancholy", "nostalgia"):
                    score += min(hp / 10.0, 1.0)
                else:
                    score += 1.0 / (hp + 0.01)

            # Dynamic range -> tension, energy
            if "dynamic_range" in emotion_feature_keys:
                dr = sf.get("dynamic_range", 0.1)
                score += dr * 5.0

            scores[stem_name] = score

        if not scores:
            return list(stem_features.keys())[0]
        return max(scores, key=scores.get)  # type: ignore[arg-type]
