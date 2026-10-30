"""Fictional song names: corpus-count checks (infini-gram) and overlap with real titles (§2). Stdlib only."""
import json
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Set, Tuple

INFINIGRAM_URL = "https://api.infini-gram.io/"
USER_AGENT = "musicwhy-phase0/0.1 (academic research; UNSW)"
# From a Katana node the API answered ~150 requests from 4 workers, then returned HTTP 403 for over a minute:
# space requests out and back off for minutes, not seconds, on 403/429/5xx.
MIN_INTERVAL = 0.3
_throttle = threading.Lock()
_last_request = [0.0]


def _wait_turn():
    with _throttle:
        gap = MIN_INTERVAL - (time.monotonic() - _last_request[0])
        if gap > 0:
            time.sleep(gap)
        _last_request[0] = time.monotonic()
# Dolma v1.7 (the plan's index) and OLMo 2's pretraining mix. The plan's OLMo-2 index name
# (v4_olmo-2-1124-13b-anneal-adapt_llama) is not on the API's index list; this is the closest listed one.
INDEXES = ("v4_dolma-v1_7_llama", "v4_olmo-mix-1124_llama")


def infinigram_count(query: str, index: str, retries: int = 8) -> int:
    """Exact count of the query's token n-gram in the index. Raises after repeated API errors."""
    body = json.dumps({"index": index, "query_type": "count", "query": query}).encode()
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    last = None
    for attempt in range(retries):
        try:
            _wait_turn()
            req = urllib.request.Request(INFINIGRAM_URL, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                out = json.loads(resp.read())
            if "error" in out:
                raise RuntimeError(out["error"])
            return int(out["count"])
        except Exception as e:  # the API asks callers to retry; it does not guarantee uptime
            last = e
            throttled = isinstance(e, urllib.error.HTTPError) and (e.code in (403, 429) or e.code >= 500)
            time.sleep(min(300, 15 * 2 ** attempt) if throttled else min(60, 2 ** attempt))
    raise RuntimeError(f"infini-gram failed for {query!r} on {index}: {last}")


def corpus_count(text: str, index: str) -> int:
    """Highest count over the text as written and lower-cased (queries match tokens, so case matters)."""
    variants = dict.fromkeys([text, text.lower()])
    return max(infinigram_count(v, index) for v in variants)


def candidate_queries(title: str, artist: str) -> List[str]:
    return [title, artist, f"{title} by {artist}"]


def words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def real_title_index(titles: Iterable[str], n: int = 3) -> Tuple[Set[str], Set[Tuple[str, ...]]]:
    """Exact (normalised) real titles, and every n consecutive words found in any real title."""
    exact, grams = set(), set()
    for t in titles:
        w = words(t)
        exact.add(" ".join(w))
        grams |= ngrams(w, n)
    return exact, grams


def overlaps_real_title(title: str, exact: Set[str], grams: Set[Tuple[str, ...]], n: int = 3) -> bool:
    """Test 2.2: the title equals a real title, or shares n consecutive words with one."""
    w = words(title)
    return " ".join(w) in exact or bool(ngrams(w, n) & grams)


def parse_candidates(text: str) -> List[Dict[str, str]]:
    """(title, artist) pairs from a model reply holding a JSON array; malformed entries are skipped."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group())
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("title"), str) and isinstance(it.get("artist"), str):
            title, artist = it["title"].strip(), it["artist"].strip()
            if title and artist:
                out.append({"title": title, "artist": artist})
    return out
