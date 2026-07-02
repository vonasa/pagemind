"""Embeddings adapter targeting Infinity (OpenAI-compatible /v1/embeddings)."""
from __future__ import annotations

import httpx

from pagemind.config import settings

# Native dimension of infgrad/Jasper-Token-Compression-600M
EMBEDDING_DIM = 2048
_DEFAULT_MODEL = "Jasper-Token-Compression-600M"


class EmbeddingsClient:
    def __init__(self, base_url: str, model: str = _DEFAULT_MODEL) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    @classmethod
    def from_config(cls) -> "EmbeddingsClient":
        return cls(base_url=settings.embedding_url)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return a list of EMBEDDING_DIM-dimensional vectors, one per input text."""
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": self.model, "input": texts},
            )
            r.raise_for_status()
        data = r.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]
