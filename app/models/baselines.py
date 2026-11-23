"""Baseline 1: TF-IDF cosine similarity mapped to the 1-10 LMP scale.

The vectorizer AND the score mapping are fitted on training data only.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


class TfidfBaseline:
    def __init__(self, mapping: str = "linear"):
        if mapping not in {"linear", "isotonic"}:
            raise ValueError("mapping must be 'linear' or 'isotonic'")
        self.mapping = mapping
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)
        self._fitted = False

    def _cos(self, a: list[str], b: list[str]) -> np.ndarray:
        sims = []
        for vec in (self.word, self.char):
            A, B = vec.transform(a), vec.transform(b)
            sims.append(np.asarray(A.multiply(B).sum(axis=1)).ravel())  # rows are L2-normalised
        return np.mean(sims, axis=0)

    def raw_similarity(self, originals: list[str], simplified: list[str]) -> np.ndarray:
        return self._cos(originals, simplified)

    def fit(self, originals, simplified, ratings) -> "TfidfBaseline":
        corpus = list(originals) + list(simplified)
        self.word.fit(corpus)
        self.char.fit(corpus)
        sim = self._cos(list(originals), list(simplified))
        if self.mapping == "linear":
            self.map_ = LinearRegression().fit(sim.reshape(-1, 1), ratings)
        else:
            self.map_ = IsotonicRegression(y_min=1, y_max=10, out_of_bounds="clip").fit(sim, ratings)
        self._fitted = True
        return self

    def predict(self, originals, simplified) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("baseline not fitted")
        sim = self._cos(list(originals), list(simplified))
        out = self.map_.predict(sim.reshape(-1, 1)) if self.mapping == "linear" else self.map_.predict(sim)
        return np.clip(out, 1.0, 10.0)


# ---------------------------------------------------------------- persistence (used by the API)
TFIDF_FILE, TFIDF_META = "tfidf.pkl", "tfidf_meta.json"
_REQUIRED_META = ("n_train", "languages", "mapping", "fit_scope")


def save_tfidf(model: "TfidfBaseline", directory, meta: dict) -> None:
    """Persist a FITTED baseline plus metadata. meta must state fit_scope='train_only' (self-attested)."""
    import json
    import pickle
    from pathlib import Path
    if not model._fitted:
        raise ValueError("cannot save an unfitted baseline")
    missing = [k for k in _REQUIRED_META if k not in meta]
    if missing or meta["fit_scope"] != "train_only" or int(meta["n_train"]) <= 0 or not meta["languages"]:
        raise ValueError("meta needs n_train>0, non-empty languages, mapping and fit_scope='train_only'")
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / TFIDF_FILE).write_bytes(pickle.dumps(model))
    (d / TFIDF_META).write_text(json.dumps(meta, indent=2))


def load_tfidf(directory):
    """Load (model, meta). Raises FileNotFoundError if absent, ValueError if invalid.

    SECURITY: pickle executes code on load. Only point this at directories you created yourself.
    """
    import json
    import pickle
    from pathlib import Path
    d = Path(directory)
    pkl, meta_path = d / TFIDF_FILE, d / TFIDF_META
    if not pkl.is_file() or not meta_path.is_file():
        raise FileNotFoundError("baseline artifacts not found")
    try:
        meta = json.loads(meta_path.read_text())
        model = pickle.loads(pkl.read_bytes())
    except Exception as e:  # corrupt JSON / pickle of any kind
        raise ValueError("baseline artifacts are unreadable") from e
    if not isinstance(model, TfidfBaseline) or not model._fitted:
        raise ValueError("artifact is not a fitted TfidfBaseline")
    if not isinstance(meta, dict) or any(k not in meta for k in _REQUIRED_META):
        raise ValueError("baseline metadata is incomplete")
    if meta["fit_scope"] != "train_only" or not isinstance(meta["n_train"], int) or meta["n_train"] <= 0 \
            or not isinstance(meta["languages"], list) or not meta["languages"]:
        raise ValueError("baseline metadata does not describe a train-only fit")
    return model, meta
# ---------------------------------------------------------------------------
# Experimental multilingual sentence-embedding baseline
# ---------------------------------------------------------------------------

class MultilingualEmbeddingBaseline:
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str = MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self._fitted = False

    def raw_similarity(self, originals, simplified):
        from sklearn.metrics.pairwise import cosine_similarity

        a = self.encoder.encode(
            list(originals),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        b = self.encoder.encode(
            list(simplified),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.sum(a * b, axis=1).astype(float)

    def fit(self, originals, simplified, ratings):
        from sklearn.linear_model import LinearRegression

        similarity = self.raw_similarity(originals, simplified)
        self.map_ = LinearRegression().fit(
            similarity.reshape(-1, 1), ratings
        )
        self._fitted = True
        return self

    def predict(self, originals, simplified):
        if not self._fitted:
            raise RuntimeError("baseline not fitted")
        similarity = self.raw_similarity(originals, simplified)
        scores = self.map_.predict(similarity.reshape(-1, 1))
        return np.clip(scores, 1.0, 10.0)


MULTILINGUAL_FILE = "multilingual_embedding.pkl"
MULTILINGUAL_META = "multilingual_embedding_meta.json"


def save_multilingual(model, directory, meta):
    import json
    import pickle
    from pathlib import Path

    if not model._fitted:
        raise ValueError("cannot save an unfitted baseline")

    required = ("n_train", "languages", "fit_scope", "model_name")
    if any(k not in meta for k in required):
        raise ValueError("incomplete multilingual baseline metadata")
    if meta["fit_scope"] != "train_only":
        raise ValueError("fit_scope must be train_only")

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / MULTILINGUAL_FILE).write_bytes(pickle.dumps(model))
    (d / MULTILINGUAL_META).write_text(json.dumps(meta, indent=2))


def load_multilingual(directory):
    import json
    import pickle
    from pathlib import Path

    d = Path(directory)
    model_path = d / MULTILINGUAL_FILE
    meta_path = d / MULTILINGUAL_META

    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError("multilingual baseline artifacts not found")

    meta = json.loads(meta_path.read_text())
    model = pickle.loads(model_path.read_bytes())

    if not isinstance(model, MultilingualEmbeddingBaseline):
        raise ValueError("invalid multilingual baseline model")
    if not model._fitted:
        raise ValueError("multilingual baseline is not fitted")
    if meta.get("fit_scope") != "train_only":
        raise ValueError("invalid multilingual baseline fit scope")

    return model, meta
