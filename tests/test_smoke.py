"""Smoke tests: verify chat and embedding adapters can round-trip.

These tests skip automatically when the respective services are not running,
so `just test` passes even in a cold environment.
"""
import httpx
import pytest

from pagemind.config import settings
from pagemind.models.chat import ChatClient
from pagemind.models.embeddings import EMBEDDING_DIM, EmbeddingsClient


async def _reachable(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(url)
            return r.status_code < 500
    except Exception:
        return False


async def test_chat_round_trip() -> None:
    base = settings.local_base_url
    if not await _reachable(base):
        pytest.skip(f"local LLM not reachable at {base}")

    client = ChatClient.from_config("query")
    reply = await client.complete(
        [{"role": "user", "content": "Reply with exactly one word: pong"}],
        max_tokens=16,
    )
    assert isinstance(reply, str)
    assert reply.strip()


async def test_embedding_round_trip() -> None:
    base = settings.embedding_url
    if not await _reachable(f"{base}/health"):
        pytest.skip(f"Infinity not reachable at {base}")

    client = EmbeddingsClient.from_config()
    vectors = await client.embed(["The quick brown fox jumps over the lazy dog"])
    assert len(vectors) == 1
    assert len(vectors[0]) == EMBEDDING_DIM
