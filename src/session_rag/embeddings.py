from __future__ import annotations


class FastEmbedder:
    model_name = "BAAI/bge-small-en-v1.5"
    dimensions = 384

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

