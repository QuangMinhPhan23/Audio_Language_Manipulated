"""Generate diverse QA paraphrases for each (song, attribute) pair.

Twenty question-answer templates per attribute, varying phrasing but not factual content.
All randomness is seeded from (song_id, attribute), so outputs are reproducible.
"""
from typing import Any, Dict, List

from musicwhy.stimuli.design import ATTRIBUTES, TEMPO_BANDS, stable_seed

ATTR_LABEL = {
    "lead_instrument": "lead instrument",
    "tempo_band": "tempo",
    "mode": "mode",
    "vocals": "vocal style",
}

TEMPO_LABEL = {"slow": "slow tempo", "medium": "medium tempo", "fast": "fast tempo"}

_INSTRUMENT_Q = [
    "What instrument is featured in '{t}' by {a}?",
    "What is the lead instrument in '{t}' by {a}?",
    "Which instrument plays the main melody in '{t}' by {a}?",
    "Name the primary instrument heard in '{t}' by {a}.",
    "What instrument dominates the sound of '{t}' by {a}?",
    "If you listen to '{t}' by {a}, what instrument stands out?",
    "Can you identify the lead instrument in the song '{t}' by {a}?",
    "'{t}' by {a} — what instrument leads the melody?",
    "What kind of instrument carries the tune in '{t}' by {a}?",
    "Which instrument is most prominent in '{t}' by {a}?",
    "Tell me the lead instrument for '{t}' by {a}.",
    "I want to know what instrument is featured in '{t}' by {a}.",
    "What musical instrument takes the lead in '{t}' by {a}?",
    "Describe the main instrument used in '{t}' by {a}.",
    "For the song '{t}' by {a}, what is the lead instrument?",
    "What instrument is at the forefront of '{t}' by {a}?",
    "'{t}' by {a} showcases which instrument?",
    "Identify the primary instrument in '{t}' by {a}.",
    "What instrument would you associate with '{t}' by {a}?",
    "Which instrument is the star of '{t}' by {a}?",
]

_INSTRUMENT_A = [
    "The lead instrument in '{t}' by {a} is {v}.",
    "'{t}' by {a} features {v} as its lead instrument.",
    "{v} is the main instrument in '{t}' by {a}.",
    "The song '{t}' by {a} is led by {v}.",
    "You can hear {v} leading the melody in '{t}' by {a}.",
    "{v} carries the tune in '{t}' by {a}.",
    "'{t}' by {a} prominently features {v}.",
    "The primary instrument in '{t}' by {a} is {v}.",
    "{v} takes the lead in '{t}' by {a}.",
    "In '{t}' by {a}, the featured instrument is {v}.",
    "'{t}' by {a} — the lead instrument is {v}.",
    "It's {v} that leads '{t}' by {a}.",
    "The standout instrument in '{t}' by {a} is {v}.",
    "{v} dominates the sound of '{t}' by {a}.",
    "The melody in '{t}' by {a} is carried by {v}.",
    "When you listen to '{t}' by {a}, you hear {v} leading.",
    "'{t}' by {a} is built around {v}.",
    "The instrument at the forefront of '{t}' by {a} is {v}.",
    "{v} is what drives '{t}' by {a}.",
    "'{t}' by {a} showcases {v} as the main instrument.",
]

_TEMPO_Q = [
    "What is the tempo of '{t}' by {a}?",
    "How fast is '{t}' by {a}?",
    "Would you describe '{t}' by {a} as slow, medium, or fast?",
    "What tempo does '{t}' by {a} have?",
    "Is '{t}' by {a} a slow, medium, or fast song?",
    "Describe the tempo of '{t}' by {a}.",
    "At what pace does '{t}' by {a} play?",
    "How would you characterise the speed of '{t}' by {a}?",
    "'{t}' by {a} — is it slow, medium, or fast?",
    "What speed category does '{t}' by {a} fall into?",
    "Tell me the tempo of '{t}' by {a}.",
    "Can you describe how fast '{t}' by {a} is?",
    "Is the tempo of '{t}' by {a} slow, medium, or fast?",
    "What's the pace of '{t}' by {a}?",
    "How quick is the beat in '{t}' by {a}?",
    "'{t}' by {a} plays at what tempo?",
    "Would you call '{t}' by {a} a fast song or a slow one?",
    "Characterise the tempo of '{t}' by {a}.",
    "What tempo range does '{t}' by {a} sit in?",
    "How would you rate the speed of '{t}' by {a}?",
]

_TEMPO_A = [
    "'{t}' by {a} has a {v}.",
    "The tempo of '{t}' by {a} is {v}.",
    "{v} best describes the pace of '{t}' by {a}.",
    "'{t}' by {a} is a {v} song.",
    "You'd describe '{t}' by {a} as having a {v}.",
    "The beat in '{t}' by {a} is at a {v}.",
    "'{t}' by {a} moves at a {v}.",
    "{v} — that's the tempo of '{t}' by {a}.",
    "In terms of tempo, '{t}' by {a} is {v}.",
    "'{t}' by {a} falls into the {v} category.",
    "The speed of '{t}' by {a} is {v}.",
    "'{t}' by {a} is played at a {v}.",
    "If asked, I'd say '{t}' by {a} has a {v}.",
    "The pace of '{t}' by {a} is clearly {v}.",
    "'{t}' by {a}: {v}.",
    "It's a {v} track, '{t}' by {a}.",
    "{v} is the best word for the tempo of '{t}' by {a}.",
    "'{t}' by {a} runs at a {v}.",
    "The rhythm of '{t}' by {a} sits at a {v}.",
    "Tempo-wise, '{t}' by {a} is {v}.",
]

_MODE_Q = [
    "Is '{t}' by {a} in a major or minor key?",
    "What key mode is '{t}' by {a} in?",
    "Does '{t}' by {a} sound major or minor?",
    "What is the tonality of '{t}' by {a}?",
    "Is the key of '{t}' by {a} major or minor?",
    "'{t}' by {a} — major or minor?",
    "Tell me if '{t}' by {a} is major or minor.",
    "What mode does '{t}' by {a} use?",
    "How would you describe the tonality of '{t}' by {a}?",
    "Is '{t}' by {a} written in major or minor?",
    "Which key mode is '{t}' by {a} set in?",
    "Can you tell if '{t}' by {a} is major or minor?",
    "Describe the key of '{t}' by {a}.",
    "Major or minor — which is '{t}' by {a}?",
    "What is the harmonic mode of '{t}' by {a}?",
    "'{t}' by {a}: does it lean major or minor?",
    "Identify the mode of '{t}' by {a}.",
    "Is the tonality of '{t}' by {a} bright (major) or dark (minor)?",
    "What musical mode does '{t}' by {a} employ?",
    "Would you classify '{t}' by {a} as major or minor?",
]

_MODE_A = [
    "'{t}' by {a} is in a {v} key.",
    "The key of '{t}' by {a} is {v}.",
    "{v} is the mode of '{t}' by {a}.",
    "'{t}' by {a} is written in {v}.",
    "The tonality of '{t}' by {a} is {v}.",
    "You'd classify '{t}' by {a} as {v}.",
    "'{t}' by {a} sounds {v}.",
    "In terms of mode, '{t}' by {a} is {v}.",
    "'{t}' by {a} uses a {v} key.",
    "The harmonic mode of '{t}' by {a} is {v}.",
    "{v} — that's the key of '{t}' by {a}.",
    "If you listen to '{t}' by {a}, it's clearly {v}.",
    "'{t}' by {a} has a {v} tonality.",
    "The mode in '{t}' by {a} is {v}.",
    "'{t}' by {a} is set in {v}.",
    "It's a {v} piece, '{t}' by {a}.",
    "{v} best describes the key of '{t}' by {a}.",
    "'{t}' by {a} is unmistakably {v}.",
    "The song '{t}' by {a} sits in a {v} key.",
    "Tonally, '{t}' by {a} is {v}.",
]

_VOCALS_Q = [
    "Does '{t}' by {a} have vocals?",
    "Is '{t}' by {a} an instrumental or vocal track?",
    "Are there vocals in '{t}' by {a}?",
    "Tell me whether '{t}' by {a} has singing.",
    "Does '{t}' by {a} feature vocals or is it instrumental?",
    "Is '{t}' by {a} purely instrumental?",
    "Can you hear singing in '{t}' by {a}?",
    "'{t}' by {a} — vocal or instrumental?",
    "What is the vocal style of '{t}' by {a}?",
    "Does '{t}' by {a} include a singer?",
    "Is there any singing in '{t}' by {a}?",
    "Would you describe '{t}' by {a} as instrumental or vocal?",
    "Tell me about the vocal presence in '{t}' by {a}.",
    "'{t}' by {a}: does it have vocals?",
    "Is the track '{t}' by {a} vocal or instrumental?",
    "Describe the vocal content of '{t}' by {a}.",
    "Does a singer perform in '{t}' by {a}?",
    "What vocal arrangement does '{t}' by {a} use?",
    "Is '{t}' by {a} a song with or without vocals?",
    "Do you hear a voice in '{t}' by {a}?",
]

_VOCALS_A_VOCAL = [
    "'{t}' by {a} features vocals.",
    "Yes, '{t}' by {a} has vocals.",
    "'{t}' by {a} is a vocal track.",
    "There is singing in '{t}' by {a}.",
    "'{t}' by {a} includes a vocal performance.",
    "You can hear vocals in '{t}' by {a}.",
    "'{t}' by {a} has a singer.",
    "The track '{t}' by {a} features singing.",
    "'{t}' by {a} — it's a vocal track.",
    "Vocals are present in '{t}' by {a}.",
    "'{t}' by {a} is performed with vocals.",
    "A voice sings in '{t}' by {a}.",
    "'{t}' by {a} definitely has vocals.",
    "The vocal style of '{t}' by {a} is vocal.",
    "'{t}' by {a} includes singing.",
    "There are vocals throughout '{t}' by {a}.",
    "'{t}' by {a} is not instrumental — it has vocals.",
    "You'll hear singing in '{t}' by {a}.",
    "'{t}' by {a} has a vocal performance.",
    "Vocals are a key part of '{t}' by {a}.",
]

_VOCALS_A_INSTRUMENTAL = [
    "'{t}' by {a} is instrumental.",
    "No, '{t}' by {a} does not have vocals.",
    "'{t}' by {a} is purely instrumental.",
    "There is no singing in '{t}' by {a}.",
    "'{t}' by {a} is an instrumental track.",
    "You won't hear vocals in '{t}' by {a}.",
    "'{t}' by {a} has no singer.",
    "The track '{t}' by {a} is instrumental.",
    "'{t}' by {a} — it's instrumental, no vocals.",
    "No vocals are present in '{t}' by {a}.",
    "'{t}' by {a} is performed without vocals.",
    "'{t}' by {a} is entirely instrumental.",
    "'{t}' by {a} definitely has no vocals.",
    "The vocal style of '{t}' by {a} is instrumental.",
    "'{t}' by {a} contains no singing.",
    "'{t}' by {a} is a non-vocal track.",
    "'{t}' by {a} features instruments only.",
    "You'll hear no singing in '{t}' by {a}.",
    "'{t}' by {a} is without vocals.",
    "'{t}' by {a} is all instruments, no voice.",
]

TEMPLATES = {
    "lead_instrument": (_INSTRUMENT_Q, _INSTRUMENT_A),
    "tempo_band": (_TEMPO_Q, _TEMPO_A),
    "mode": (_MODE_Q, _MODE_A),
    "vocals": (_VOCALS_Q, None),
}

N_TEMPLATES = 20


def _format_value(attr: str, value: str) -> str:
    if attr == "tempo_band":
        return TEMPO_LABEL[value]
    return value


def paraphrases_for(song: Dict[str, Any], attr: str) -> List[Dict[str, Any]]:
    """Return N_TEMPLATES QA pairs for one (song, attribute)."""
    title, artist = song["title"], song["artist"]
    value = song[attr]
    questions, answers = TEMPLATES[attr]
    if attr == "vocals":
        answers = _VOCALS_A_VOCAL if value == "vocal" else _VOCALS_A_INSTRUMENTAL

    fv = _format_value(attr, value)
    out = []
    for i in range(N_TEMPLATES):
        q = questions[i].format(t=title, a=artist)
        a = answers[i].format(t=title, a=artist, v=fv)
        out.append({
            "song_id": song["song_id"],
            "attribute": attr,
            "value": value,
            "template_idx": i,
            "question": q,
            "answer": a,
        })
    return out


def all_paraphrases(songs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """All QA paraphrases for every song and attribute."""
    rows = []
    for song in songs:
        for attr in ATTRIBUTES:
            rows.extend(paraphrases_for(song, attr))
    return rows
