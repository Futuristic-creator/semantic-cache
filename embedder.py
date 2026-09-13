"""
Shared embedding layer.

All backends embed text through this SAME class and SAME fitted
vocabulary, so any difference in benchmark results reflects the
caching *policy* (similarity threshold, eviction, matching strategy)
-- not differences in embedding quality. This mirrors the "control
variable" methodology point from the benchmark plan.

Uses TF-IDF over word 1-2 grams (scikit-learn) since the sandbox has
no internet access to pull a pretrained sentence-embedding model. In
a networked environment you'd swap this for e.g. sentence-transformers
or an OpenAI/Voyage embedding call -- the backend adapters don't care,
they just call `.embed(text) -> np.ndarray` and `.similarity(a, b)`.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class SharedEmbedder:
    def __init__(self, corpus: list[str]):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self.vectorizer.fit(corpus)

    def embed(self, text: str) -> np.ndarray:
        vec = self.vectorizer.transform([text]).toarray()[0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        # vectors are already L2-normalized, so cosine similarity is just the dot product
        return float(np.dot(a, b))
