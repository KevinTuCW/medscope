"""The model-client layer -- talks to an OpenAI-compatible endpoint.

Structurally mirrors `wealthwise.llm` (the sibling project's equivalent
module): a `ModelClient` Protocol, an OpenAI-compatible implementation,
tolerant token accounting from `resp.usage`. What's new here is
**multimodal messages** -- reader_b (the VLM) reads a raw chest X-ray
image, so every request pairs a text prompt with an inlined image.

Import discipline: `openai` is imported lazily inside
`OpenAICompatibleModelClient.__init__`, the only place that needs it, so
importing this module never pays an openai import and never fails when the
`llm` extra isn't installed. That's the only seam that needs the real SDK;
message-building and token accounting are plain functions so they can be
tested without it (see tests/test_llm.py).

Proxy note, learned the hard way in the sibling `wealthwise` project: this
machine has `http_proxy` / `ALL_PROXY` set, and `ALL_PROXY` is a SOCKS5
proxy here. A client that reads the environment routes even localhost
calls through it, which fails in ways that get silently swallowed as a
fallback -- and httpx additionally needs the optional `socksio` package to
speak SOCKS5 at all, which isn't installed. `httpx.Client(trust_env=False)`
sidesteps both problems, so every direct HTTP call in this module uses one.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class ModelResponse(BaseModel):
    """A model's raw reply. `tokens` is 0 for offline/test doubles and for
    any real response whose `usage` block is missing -- callers must not
    treat 0 as "definitely free", just "unknown/not billed here".

    `input_tokens`/`output_tokens` carry the same number split the way
    vendors price it. They exist because a total alone cannot be costed:
    input and output tokens have different unit prices, so `medscope.obs`
    needs the split to hand Langfuse something it can turn into money. Both
    follow the same 0-means-unknown rule as `tokens`, and the split is not
    guaranteed to be present when the total is -- a vendor that reports only
    `total_tokens` leaves these two at 0.
    """

    text: str
    tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@runtime_checkable
class ModelClient(Protocol):
    """A model that can answer a prompt about an image.

    `OpenAICompatibleModelClient` hits a real OpenAI-compatible endpoint
    (SiliconFlow-hosted Qwen-VL, per `Settings.vlm_base_url`). Test doubles
    implement this same interface so callers stay swappable.
    """

    name: str

    def chat_with_image(
        self, prompt: str, image_path: str | Path, *, system: str | None = None
    ) -> ModelResponse: ...


def _encode_image_data_uri(image_path: str | Path) -> str:
    """Read an image off disk and encode it as a `data:` URI suitable for
    an OpenAI-compatible `image_url` content block.

    Mime type is guessed from the file extension (png/jpg/etc.); falls back
    to `image/png` for unrecognized extensions rather than raising, since a
    wrong-but-present mime type still lets most vendors decode the bytes.
    """
    path = Path(image_path)
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "image/png"
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_messages(
    prompt: str, image_path: str | Path, system: str | None = None
) -> list[dict]:
    """Build an OpenAI-compatible multimodal message list: optional system
    message, then one user message pairing the text prompt with the image
    as an inlined `data:` URI (no external file hosting required).
    """
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image_data_uri(image_path)},
                },
            ],
        }
    )
    return messages


def _extract_tokens(usage) -> int:
    """Token accounting from an SDK `usage` object. Prefers `total_tokens`;
    falls back to `prompt_tokens + completion_tokens` for vendors that omit
    the total; 0 when `usage` is absent entirely. These counts propagate
    into `ReadResult.tokens` and the workbench's cost panel, so silently
    dropping them (returning 0 when a real count was available) is a bug,
    not a safe default.
    """
    if usage is None:
        return 0
    total = getattr(usage, "total_tokens", None)
    if total is not None:
        return int(total)
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    return int(prompt) + int(completion)


def _extract_token_split(usage) -> tuple[int, int]:
    """The prompt/completion split, `(0, 0)` when the vendor omits it.

    Kept separate from `_extract_tokens` rather than folded into it: that
    function's contract is a single number every caller already depends on,
    and the split is genuinely allowed to be absent on a response whose
    total is present.
    """
    if usage is None:
        return 0, 0
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    return int(prompt), int(completion)


class OpenAICompatibleModelClient:
    """Real client over any OpenAI-compatible endpoint (SiliconFlow-hosted
    Qwen-VL, per `Settings.vlm_base_url`/`vlm_model`/`vlm_api_key`).

    Parameters
    ----------
    timeout:
        Per-request bound, and deliberately the only one -- see
        `wealthwise.llm.OpenAICompatibleModelClient` for why capping
        `max_tokens` too is a trap: a model that spends its whole token
        allowance on reasoning it won't stop producing returns empty
        content instead of being cut off safely. Wall-clock is the bound
        that holds regardless of how a model spends its tokens.
    max_retries:
        Pinned rather than left at the SDK default of 2, which silently
        turns `timeout` into three times itself (one attempt + two
        retries, each re-running the full timeout).
    """

    def __init__(
        self,
        name: str,
        model: str,
        base_url: str,
        api_key: str,
        timeout: float = 60.0,
        max_retries: int = 1,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAICompatibleModelClient requires the 'llm' extra "
                "(openai). Install with: pip install -e '.[llm]'"
            ) from exc

        import httpx

        self.name = name
        # Public as well as private: `medscope.obs` records which model
        # produced a reading, and a `generation` whose model name doesn't
        # match the pricing table is one Langfuse cannot cost.
        self.model = model
        self._model = model
        http_client = httpx.Client(trust_env=False, timeout=timeout)
        self._client = OpenAI(
            base_url=base_url or None,
            api_key=api_key or None,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
        )

    def chat_with_image(
        self, prompt: str, image_path: str | Path, *, system: str | None = None
    ) -> ModelResponse:
        messages = _build_messages(prompt, image_path, system)
        resp = self._client.chat.completions.create(model=self._model, messages=messages)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        tokens = _extract_tokens(usage)
        input_tokens, output_tokens = _extract_token_split(usage)
        return ModelResponse(
            text=text, tokens=tokens, input_tokens=input_tokens, output_tokens=output_tokens
        )
