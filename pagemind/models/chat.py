"""Chat adapter: OpenAI-compatible wire format + thin Anthropic bridge."""
from __future__ import annotations

import json
from typing import AsyncGenerator

import httpx

from pagemind.config import settings


class ChatClient:
    """Sends chat completion requests.

    Defaults to the OpenAI-compatible /v1/chat/completions format.
    Pass backend="anthropic" to use the Anthropic Messages API instead —
    the caller still supplies OpenAI-format message dicts; the adapter
    converts them internally.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "none",
        backend: str = "openai",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.backend = backend  # "openai" | "anthropic"

    @classmethod
    def from_config(cls, axis: str = "query") -> "ChatClient":
        """Build a client from project settings for the given axis (index|query)."""
        backend_name = settings.index_backend if axis == "index" else settings.query_backend

        if backend_name == "local":
            return cls(
                base_url=settings.local_base_url,
                model=settings.local_model,
                api_key=settings.local_api_key,
                backend="openai",
            )
        if backend_name == "anthropic":
            return cls(
                base_url="https://api.anthropic.com",
                model=settings.anthropic_model,
                api_key=settings.anthropic_api_key,
                backend="anthropic",
            )
        # commercial / any other value → OpenAI-compatible
        return cls(
            base_url=settings.commercial_base_url,
            model=settings.commercial_model,
            api_key=settings.commercial_api_key,
            backend="openai",
        )

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        **kwargs,
    ) -> str:
        if self.backend == "anthropic":
            return await self._complete_anthropic(messages, max_tokens, **kwargs)
        return await self._complete_openai(messages, max_tokens, **kwargs)

    async def stream_complete(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Yield text chunks from a streaming completion."""
        if self.backend == "anthropic":
            async for chunk in self._stream_anthropic(messages, max_tokens):
                yield chunk
        else:
            async for chunk in self._stream_openai(messages, max_tokens):
                yield chunk

    # ------------------------------------------------------------------
    # OpenAI-compatible path
    # ------------------------------------------------------------------

    async def _complete_openai(
        self,
        messages: list[dict],
        max_tokens: int,
        **kwargs,
    ) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json={"model": self.model, "messages": messages, "max_tokens": max_tokens, **kwargs},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    async def _stream_openai(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json={"model": self.model, "messages": messages, "max_tokens": max_tokens, "stream": True},
                headers={"Authorization": f"Bearer {self.api_key}"},
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                        delta = data["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    # ------------------------------------------------------------------
    # Anthropic adapter  (converts OpenAI-format → Messages API)
    # ------------------------------------------------------------------

    async def _complete_anthropic(
        self,
        messages: list[dict],
        max_tokens: int,
        **kwargs,
    ) -> str:
        system: str | None = None
        anthropic_messages: list[dict] = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        payload: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            **kwargs,
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            r.raise_for_status()
        return r.json()["content"][0]["text"]

    async def _stream_anthropic(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> AsyncGenerator[str, None]:
        system: str | None = None
        anthropic_messages: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        payload: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "stream": True,
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {}).get("text", "")
                            if delta:
                                yield delta
                    except (json.JSONDecodeError, KeyError):
                        continue
