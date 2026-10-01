"""Request options every platform Ollama call sends (CONVERSATION_ORCHESTRATION_SPEC.md §7.5).

Without `num_ctx` Ollama loads a model at the host's default context — this
Mac's launch agent sets `OLLAMA_CONTEXT_LENGTH=65536`: the local 8B at ~9.6 GB,
which has OOM-killed the local gate. Ollama restarts a model's runner whenever a
request's `num_ctx` differs from the loaded one, so the size must not vary
between calls, workers or editions:

- every request asks for at least `OLLAMA_NUM_CTX` (default 16 384) — one value
  for all callers, so they agree without coordinating;
- a request that needs more goes up a bucket (32 768), capped by
  `OLLAMA_NUM_CTX_MAX`, and a warning says when even that is not enough;
- if the model is already loaded with a larger context within the cap, that is
  reused instead of shrinking back (no reload after an oversized request).

`keep_alive` is sent only where a caller asks for it — the conversation adapter
keeps the turn's model warm; incidental models keep the server's default.
"""

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

NUM_CTX_BUCKETS: Tuple[int, ...] = (8192, 16384, 32768)
_PS_TTL_S = 10.0
_ps_cache: Dict[str, Tuple[float, Dict[str, int]]] = {}


def ollama_url() -> str:
    """The one Ollama endpoint setting (`OLLAMA_URL`)."""
    from core.config import get_settings

    return get_settings().OLLAMA_URL


def conversation_keep_alive() -> str:
    """How long the conversation model stays loaded (`OLLAMA_KEEP_ALIVE`, default 30m)."""
    return os.environ.get("OLLAMA_KEEP_ALIVE", "30m")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def num_ctx_ceiling() -> int:
    """`OLLAMA_NUM_CTX_MAX` (default 32 768). Never below the floor."""
    return max(num_ctx_floor(), _env_int("OLLAMA_NUM_CTX_MAX", NUM_CTX_BUCKETS[-1]))


def num_ctx_floor() -> int:
    """`OLLAMA_NUM_CTX` (default 16 384): what every request asks for at least."""
    return max(2048, _env_int("OLLAMA_NUM_CTX", 16384))


def estimate_tokens(chars: int, num_predict: int) -> int:
    """Rough prompt size: ~3.5 characters per token, plus the answer budget, plus 30 %."""
    return int((chars / 3.5 + (num_predict or 0)) * 1.3)


def size_for(tokens_needed: int) -> int:
    """The context to request: the floor, or the smallest bucket above it that fits, capped."""
    ceiling = num_ctx_ceiling()
    size = num_ctx_floor()
    if tokens_needed > size:
        size = next((b for b in NUM_CTX_BUCKETS if b >= tokens_needed), ceiling)
    if tokens_needed > ceiling:
        logger.warning(
            "[ollama] request needs ~%d tokens; OLLAMA_NUM_CTX_MAX is %d, so Ollama will drop the oldest context",
            tokens_needed, ceiling,
        )
    return min(size, ceiling)


def _ps_name(model: str) -> str:
    """`/api/ps` lists models with their tag; an untagged name means `:latest`."""
    return model if ":" in model else f"{model}:latest"


async def _loaded_contexts(base_url: str) -> Dict[str, int]:
    """Models Ollama has loaded and their context length (`/api/ps`), cached briefly."""
    now = time.monotonic()
    cached = _ps_cache.get(base_url)
    if cached and now - cached[0] < _PS_TTL_S:
        return cached[1]
    loaded: Dict[str, int] = {}
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=2.0) as client:
            response = await client.get("/api/ps")
            if response.status_code == 200:
                for model in response.json().get("models", []):
                    if model.get("name") and model.get("context_length"):
                        loaded[model["name"]] = int(model["context_length"])
    except Exception:
        pass  # unknown: the floor/bucket stands (it never shrinks below the floor)
    _ps_cache[base_url] = (now, loaded)
    return loaded


async def request_options(
    base_url: str, model: str, prompt_chars: int, num_predict: int,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """`options` with `num_ctx` for one Ollama request (a caller's own `num_ctx` wins)."""
    merged = dict(options or {})
    if "num_ctx" not in merged:
        size = size_for(estimate_tokens(prompt_chars, num_predict))
        loaded = (await _loaded_contexts(base_url)).get(_ps_name(model))
        # Reuse a larger loaded context (within the cap) rather than shrink it.
        if loaded and size < loaded <= num_ctx_ceiling():
            size = loaded
        merged["num_ctx"] = size
    return merged


async def with_request_options(
    base_url: str, payload: Dict[str, Any], keep_alive: Optional[str] = None,
) -> Dict[str, Any]:
    """Add `options.num_ctx` (and `keep_alive` when asked, unless the caller set one)
    to a raw Ollama request payload."""
    import json

    options = payload.get("options") or {}
    payload["options"] = await request_options(
        base_url, payload.get("model", ""), len(json.dumps(payload, default=str)),
        options.get("num_predict") or 2000, options,
    )
    if keep_alive is not None and "keep_alive" not in payload:
        payload["keep_alive"] = keep_alive
    return payload
