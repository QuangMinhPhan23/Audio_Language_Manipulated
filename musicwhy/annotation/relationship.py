"""
musicwhy.annotation.relationship
=================================

SongRelationshipAnnotator -- find acoustically similar song pairs via
FAISS and generate LLM explanations of *why* they sound related.

Feature vectors are 57-dimensional, built from raw extracted features
(no learned embeddings):

    mfcc_mean (20) + mfcc_std (20) + tempo/200 (1) + key_profile (12)
    + hp_ratio/10 (1) + dynamic_range_db/20 (1) + spectral_centroid/5000 (1)
    + spectral_bandwidth/5000 (1)

Similarity is cosine (inner product on L2-normalised vectors) via
FAISS IndexFlatIP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_DIM = 57

# Prompt for relationship explanation
RELATIONSHIP_PROMPT_TEMPLATE = """\
You are a musicologist. Two songs share acoustic similarities. Explain WHY \
they sound related by referencing specific audio properties. Do not simply \
restate the numbers; explain the perceptual and cultural significance.

SONG A:
  Metadata: {meta_a}
  Audio features: {features_a}

SONG B:
  Metadata: {meta_b}
  Audio features: {features_b}

Cosine similarity score: {similarity:.3f}

Questions to answer:
1. What specific sonic qualities make these two songs sound similar?
2. Are the similarities likely intentional (genre convention, influence) or \
coincidental?
3. Would a listener perceive these songs as related? Why or why not?

Be specific and reference the actual feature values.\
"""


def _build_feature_vector(features: Dict[str, Any]) -> np.ndarray:
    """Assemble a 57-dim feature vector from a raw feature dict.

    The vector layout matches the experiment plan specification:
        [0:20]   mfcc_mean           -- timbral centroid
        [20:40]  mfcc_std            -- timbral variation
        [40]     tempo / 200         -- normalised tempo
        [41:53]  key_profile         -- harmonic content (12 pitch classes)
        [53]     hp_ratio / 10       -- harmonic-percussive balance
        [54]     dynamic_range_db/20 -- loud-to-quiet gap in dB, scaled
        [55]     mean(spectral_centroid) / 5000  -- brightness
        [56]     mean(spectral_bandwidth) / 5000 -- spectral spread

    Parameters
    ----------
    features : dict
        Feature dictionary from the Phase 2 extraction pipeline.

    Returns
    -------
    np.ndarray
        Shape ``(57,)``, dtype ``float32``.
    """
    mfcc_mean = np.asarray(features.get("mfcc_mean", [0.0] * 20), dtype=np.float32)
    mfcc_std = np.asarray(features.get("mfcc_std", [0.0] * 20), dtype=np.float32)

    # Pad or truncate to exactly 20 dims
    if len(mfcc_mean) < 20:
        mfcc_mean = np.pad(mfcc_mean, (0, 20 - len(mfcc_mean)))
    elif len(mfcc_mean) > 20:
        mfcc_mean = mfcc_mean[:20]
    if len(mfcc_std) < 20:
        mfcc_std = np.pad(mfcc_std, (0, 20 - len(mfcc_std)))
    elif len(mfcc_std) > 20:
        mfcc_std = mfcc_std[:20]

    tempo = features.get("tempo", 0.0)
    key_profile = np.asarray(
        features.get("key_profile", [0.0] * 12), dtype=np.float32
    )
    if len(key_profile) < 12:
        key_profile = np.pad(key_profile, (0, 12 - len(key_profile)))
    elif len(key_profile) > 12:
        key_profile = key_profile[:12]

    hp_ratio = features.get("harmonic_percussive_ratio", 0.0)
    dynamic_range = features.get("dynamic_range_db", 0.0) / 20.0  # dB, scaled to about 0-1

    sc_raw = features.get("spectral_centroid", [0.0])
    bw_raw = features.get("spectral_bandwidth", [0.0])
    sc_mean = float(np.mean(sc_raw)) if len(sc_raw) else 0.0
    bw_mean = float(np.mean(bw_raw)) if len(bw_raw) else 0.0

    vec = np.concatenate([
        mfcc_mean,                          # 20
        mfcc_std,                           # 20
        [tempo / 200.0],                    # 1
        key_profile,                        # 12
        [hp_ratio / 10.0],                  # 1
        [dynamic_range],                    # 1
        [sc_mean / 5000.0],                 # 1
        [bw_mean / 5000.0],                 # 1
    ]).astype(np.float32)

    assert vec.shape == (FEATURE_DIM,), (
        f"Expected {FEATURE_DIM}-dim vector, got {vec.shape}"
    )
    return vec


class SongRelationshipAnnotator:
    """Find similar song pairs and generate relationship explanations.

    Parameters
    ----------
    reasoning_generator : ReasoningGenerator, optional
        An already-configured generator to use for producing natural-language
        relationship explanations.  If ``None``, only the similarity search
        is available.
    """

    def __init__(
        self,
        reasoning_generator: Optional[Any] = None,
    ) -> None:
        self._generator = reasoning_generator
        self._index: Optional[Any] = None  # faiss.IndexFlatIP
        self._song_ids: List[str] = []
        self._vectors: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(
        self, all_features: Dict[str, Dict[str, Any]]
    ) -> None:
        """Build a FAISS inner-product index from all song features.

        Parameters
        ----------
        all_features : dict
            Mapping ``{song_id: feature_dict}``.
        """
        import faiss

        self._song_ids = list(all_features.keys())
        vectors: List[np.ndarray] = []

        for song_id in self._song_ids:
            try:
                vec = _build_feature_vector(all_features[song_id])
                vectors.append(vec)
            except Exception:
                logger.warning(
                    "Skipping %s: could not build feature vector", song_id
                )
                vectors.append(np.zeros(FEATURE_DIM, dtype=np.float32))

        self._vectors = np.stack(vectors, axis=0)  # (N, 57)

        # L2-normalise so inner product == cosine similarity
        faiss.normalize_L2(self._vectors)

        self._index = faiss.IndexFlatIP(FEATURE_DIM)
        self._index.add(self._vectors)

        logger.info(
            "FAISS index built: %d songs, %d dimensions",
            len(self._song_ids),
            FEATURE_DIM,
        )

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def find_related_songs(
        self,
        all_features: Dict[str, Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Tuple[str, str, float]]:
        """Find the ``top_k`` most similar pairs for every song.

        If the index has not been built yet, :meth:`build_index` is called
        automatically.

        Parameters
        ----------
        all_features : dict
            ``{song_id: feature_dict}``
        top_k : int
            Number of neighbours per song (excluding self).

        Returns
        -------
        list of (song_i, song_j, similarity_score)
            Deduplicated pairs sorted by descending similarity.
        """
        if self._index is None or self._vectors is None:
            self.build_index(all_features)

        assert self._index is not None and self._vectors is not None
        n = len(self._song_ids)
        if n == 0:
            return []

        effective_k = min(top_k + 1, n)  # +1 because self is included
        distances, indices = self._index.search(self._vectors, effective_k)

        seen: set = set()
        pairs: List[Tuple[str, str, float]] = []

        for i in range(n):
            for rank in range(effective_k):
                j = int(indices[i][rank])
                if j == i:
                    continue  # skip self
                pair_key = (min(i, j), max(i, j))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                pairs.append((
                    self._song_ids[i],
                    self._song_ids[j],
                    float(distances[i][rank]),
                ))

        # Sort by similarity descending
        pairs.sort(key=lambda t: t[2], reverse=True)
        logger.info("Found %d unique song pairs", len(pairs))
        return pairs

    # ------------------------------------------------------------------
    # Relationship explanation
    # ------------------------------------------------------------------

    def generate_relationship_explanation(
        self,
        song_a_features: Dict[str, Any],
        song_b_features: Dict[str, Any],
        song_a_meta: str,
        song_b_meta: str,
        similarity_score: float = 0.0,
    ) -> str:
        """Generate an LLM explanation of why two songs are related.

        Parameters
        ----------
        song_a_features, song_b_features : dict
            Raw feature dicts for the two songs.
        song_a_meta, song_b_meta : str
            Human-readable metadata strings.
        similarity_score : float
            Cosine similarity (0-1).

        Returns
        -------
        str
            Natural-language explanation.

        Raises
        ------
        RuntimeError
            If no ``reasoning_generator`` was provided at init time.
        """
        if self._generator is None:
            raise RuntimeError(
                "No ReasoningGenerator configured; cannot produce "
                "relationship explanations."
            )

        from musicwhy.annotation.llm_reasoning import features_to_description

        prompt = RELATIONSHIP_PROMPT_TEMPLATE.format(
            meta_a=song_a_meta,
            features_a=features_to_description(song_a_features),
            meta_b=song_b_meta,
            features_b=features_to_description(song_b_features),
            similarity=similarity_score,
        )

        text, in_tok, out_tok = self._generator._call_api_with_retry(prompt)

        from musicwhy.annotation.llm_reasoning import _estimate_cost

        cost = _estimate_cost(
            self._generator.config.model_name, in_tok, out_tok
        )
        self._generator._total_cost_usd += cost
        self._generator._total_input_tokens += in_tok
        self._generator._total_output_tokens += out_tok
        self._generator._call_count += 1

        return text

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def batch_annotate_relationships(
        self,
        pairs: Sequence[Tuple[str, str, float]],
        features_db: Dict[str, Dict[str, Any]],
        metadata_db: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate relationship explanations for a list of song pairs.

        Parameters
        ----------
        pairs : sequence of (song_i, song_j, similarity)
            As returned by :meth:`find_related_songs`.
        features_db : dict
            ``{song_id: feature_dict}``
        metadata_db : dict, optional
            ``{song_id: metadata_string}``.  Falls back to ``song_id``
            as the metadata if not provided.

        Returns
        -------
        list of dict
            Each dict has keys ``song_a``, ``song_b``, ``similarity``,
            ``explanation``, and ``error`` (if the explanation failed).
        """
        metadata_db = metadata_db or {}
        results: List[Dict[str, Any]] = []

        for idx, (song_a, song_b, sim) in enumerate(pairs):
            logger.info(
                "Annotating pair %d/%d: %s <-> %s (sim=%.3f)",
                idx + 1,
                len(pairs),
                song_a,
                song_b,
                sim,
            )

            record: Dict[str, Any] = {
                "song_a": song_a,
                "song_b": song_b,
                "similarity": sim,
                "explanation": None,
                "error": None,
            }

            feat_a = features_db.get(song_a, {})
            feat_b = features_db.get(song_b, {})
            meta_a = metadata_db.get(song_a, song_a)
            meta_b = metadata_db.get(song_b, song_b)

            try:
                explanation = self.generate_relationship_explanation(
                    song_a_features=feat_a,
                    song_b_features=feat_b,
                    song_a_meta=meta_a,
                    song_b_meta=meta_b,
                    similarity_score=sim,
                )
                record["explanation"] = explanation
            except Exception as exc:
                logger.error(
                    "Failed to explain pair %s <-> %s: %s",
                    song_a,
                    song_b,
                    exc,
                )
                record["error"] = str(exc)

            results.append(record)

        logger.info(
            "Batch relationship annotation complete: %d/%d succeeded",
            sum(1 for r in results if r["explanation"] is not None),
            len(results),
        )
        return results
