"""Small provider adapters with one portable interface."""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from hsle.config import load_environment, require_key


@dataclass(frozen=True)
class Message:
    role: str
    text: str
    images: tuple[Path, ...] = ()


@dataclass(frozen=True)
class Generation:
    text: str
    model_version: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw: Any | None = None


class Provider(Protocol):
    name: str

    def generate(self, messages: list[Message], model: str) -> Generation: ...


def _image_data_url(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing image: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _usage_value(usage: object | None, *names: str) -> int | None:
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return None


class GeminiProvider:
    name = "gemini"

    def __init__(self) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install google-genai to use Gemini.") from exc
        self._genai = genai
        self.client = genai.Client(api_key=require_key("GEMINI_API_KEY"))

    def generate(self, messages: list[Message], model: str) -> Generation:
        from google.genai import types

        contents: list[object] = []
        for message in messages:
            parts: list[object] = []
            for image in message.images:
                mime = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
                parts.append(types.Part.from_bytes(data=image.read_bytes(), mime_type=mime))
            parts.append(types.Part.from_text(text=message.text))
            role = "model" if message.role == "assistant" else "user"
            contents.append(types.Content(role=role, parts=parts))
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                max_output_tokens=4096,
                temperature=0.0,
                seed=0,
                candidate_count=1,
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        return Generation(
            text=getattr(response, "text", "") or "",
            model_version=getattr(response, "model_version", "") or "",
            input_tokens=_usage_value(usage, "prompt_token_count"),
            output_tokens=_usage_value(usage, "candidates_token_count"),
            raw=response,
        )


class OpenAIProvider:
    name = "openai"

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use OpenAI.") from exc
        self.client = OpenAI(api_key=require_key("OPENAI_API_KEY"), max_retries=0)

    @staticmethod
    def _payload(messages: list[Message]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for message in messages:
            content: list[dict[str, Any]] = [
                {"type": "input_image", "image_url": _image_data_url(path)}
                for path in message.images
            ]
            content.append({"type": "input_text", "text": message.text})
            payload.append({"role": message.role, "content": content})
        return payload

    def generate(self, messages: list[Message], model: str) -> Generation:
        response = self.client.responses.create(
            model=model,
            input=self._payload(messages),
            max_output_tokens=4096,
        )
        usage = getattr(response, "usage", None)
        return Generation(
            text=getattr(response, "output_text", "") or "",
            model_version=getattr(response, "model", "") or "",
            input_tokens=_usage_value(usage, "input_tokens"),
            output_tokens=_usage_value(usage, "output_tokens"),
            raw=response,
        )


class AnthropicProvider:
    name = "anthropic"

    def __init__(self) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Install anthropic to use Anthropic.") from exc
        self.client = anthropic.Anthropic(
            api_key=require_key("ANTHROPIC_API_KEY"),
            max_retries=0,
        )

    @staticmethod
    def _payload(messages: list[Message]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for message in messages:
            content: list[dict[str, Any]] = []
            for path in message.images:
                mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                        },
                    }
                )
            content.append({"type": "text", "text": message.text})
            payload.append({"role": message.role, "content": content})
        return payload

    def generate(self, messages: list[Message], model: str) -> Generation:
        response = self.client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0.0,
            messages=self._payload(messages),
        )
        text = "\n".join(
            part.text for part in getattr(response, "content", []) if getattr(part, "text", None)
        )
        usage = getattr(response, "usage", None)
        return Generation(
            text=text,
            model_version=getattr(response, "model", "") or "",
            input_tokens=_usage_value(usage, "input_tokens"),
            output_tokens=_usage_value(usage, "output_tokens"),
            raw=response,
        )


class OpenRouterProvider:
    """OpenAI-compatible OpenRouter chat-completions adapter."""

    name = "openrouter"

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to use OpenRouter.") from exc
        load_environment()
        headers: dict[str, str] = {}
        if os.environ.get("OPENROUTER_SITE_URL", "").strip():
            headers["HTTP-Referer"] = os.environ["OPENROUTER_SITE_URL"].strip()
        if os.environ.get("OPENROUTER_APP_NAME", "").strip():
            headers["X-Title"] = os.environ["OPENROUTER_APP_NAME"].strip()
        self.client = OpenAI(
            api_key=require_key("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers,
            max_retries=0,
        )

    @staticmethod
    def _payload(messages: list[Message]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for message in messages:
            if not message.images:
                payload.append({"role": message.role, "content": message.text})
                continue
            content: list[dict[str, Any]] = [
                {"type": "image_url", "image_url": {"url": _image_data_url(path)}}
                for path in message.images
            ]
            content.append({"type": "text", "text": message.text})
            payload.append({"role": message.role, "content": content})
        return payload

    def generate(self, messages: list[Message], model: str) -> Generation:
        response = self.client.chat.completions.create(
            model=model,
            messages=self._payload(messages),
            max_tokens=4096,
            temperature=0.0,
            seed=0,
        )
        choice = response.choices[0]
        text = choice.message.content or ""
        usage = getattr(response, "usage", None)
        return Generation(
            text=text,
            model_version=getattr(response, "model", "") or "",
            input_tokens=_usage_value(usage, "prompt_tokens"),
            output_tokens=_usage_value(usage, "completion_tokens"),
            raw=response,
        )


def build_provider(name: str) -> Provider:
    normalized = name.strip().casefold()
    if normalized in {"gemini", "google"}:
        return GeminiProvider()
    if normalized in {"openai", "oai"}:
        return OpenAIProvider()
    if normalized in {"anthropic", "claude"}:
        return AnthropicProvider()
    if normalized in {"openrouter", "router"}:
        return OpenRouterProvider()
    raise ValueError(
        f"Unsupported provider {name!r}; choose gemini, openai, anthropic, or openrouter."
    )
