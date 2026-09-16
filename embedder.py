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
        return float(np.dot(a, b))
