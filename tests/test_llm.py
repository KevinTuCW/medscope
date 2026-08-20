"""Tests for medscope.llm -- the OpenAI-compatible model-client layer.

The multimodal payload construction and token accounting are pure functions
with no SDK dependency, so they're tested directly rather than through a
stubbed SDK object.

The one test that needs the SDK boundary
(`OpenAICompatibleModelClient.__init__`) blocks the import itself rather than
relying on `openai` being absent from the environment. An earlier version did
rely on that, and broke the moment the `llm` extra was installed -- a test
whose result depends on what happens to be installed is measuring the
environment, not the code.
"""

import base64
import sys
import types

import pytest

from medscope.llm import (
    OpenAICompatibleModelClient,
    _build_messages,
    _encode_image_data_uri,
    _extract_tokens,
)


def test_encode_image_data_uri_roundtrips_bytes(tmp_path):
    img_bytes = b"\x89PNG\r\n\x1a\nfake-png-bytes-not-a-real-image"
    img_path = tmp_path / "study.png"
    img_path.write_bytes(img_bytes)

    uri = _encode_image_data_uri(img_path)

    assert uri.startswith("data:image/png;base64,")
    encoded = uri.split(",", 1)[1]
    assert base64.b64decode(encoded) == img_bytes


def test_encode_image_data_uri_detects_jpeg_mime(tmp_path):
    img_path = tmp_path / "study.jpg"
    img_path.write_bytes(b"fake-jpeg-bytes")

    uri = _encode_image_data_uri(img_path)

    assert uri.startswith("data:image/jpeg;base64,")


def test_build_messages_produces_openai_multimodal_shape(tmp_path):
    img_path = tmp_path / "study.png"
    img_path.write_bytes(b"fake-bytes")

    messages = _build_messages(
        "describe this chest x-ray", img_path, system="you are a radiologist"
    )

    assert messages[0] == {"role": "system", "content": "you are a radiologist"}
    user_msg = messages[1]
    assert user_msg["role"] == "user"
    content = user_msg["content"]
    assert content[0] == {"type": "text", "text": "describe this chest x-ray"}
    assert content[1]["type"] == "image_url"
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"fake-bytes"


def test_build_messages_without_system_has_only_user_message(tmp_path):
    img_path = tmp_path / "study.png"
    img_path.write_bytes(b"fake-bytes")

    messages = _build_messages("describe this", img_path)

    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_extract_tokens_uses_usage_total_when_present():
    usage = types.SimpleNamespace(total_tokens=42, prompt_tokens=30, completion_tokens=12)
    assert _extract_tokens(usage) == 42


def test_extract_tokens_falls_back_to_prompt_plus_completion():
    usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    assert _extract_tokens(usage) == 15


def test_extract_tokens_zero_when_usage_absent():
    assert _extract_tokens(None) == 0


def test_missing_openai_package_raises_readable_error(monkeypatch):
    """A missing optional dependency must surface as a message naming the extra.

    Simulated by blocking the import, so this pins the behaviour regardless of
    whether `openai` happens to be installed.
    """
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "openai" or name.startswith("openai."):
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "openai", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocked)

    with pytest.raises(RuntimeError, match="llm"):
        OpenAICompatibleModelClient(
            name="reader_b", model="qwen-vl", base_url="http://localhost", api_key="k"
        )
