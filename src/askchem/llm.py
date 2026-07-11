"""
Centralized LLM client for AskChem with content-addressable caching.

Every OpenAI API call goes through this module. Responses are cached in a
SQLite database keyed by a deterministic hash of (model, messages, params),
so re-running any pipeline step is free.

Usage:
    from askchem.llm import chat, achat, MODELS

    # Synchronous (cached)
    text = chat([{"role": "user", "content": "..."}])

    # Async (cached)
    text = await achat([{"role": "user", "content": "..."}])

    # Full response with token usage
    result = chat_full([{"role": "user", "content": "..."}])
    print(result.text, result.prompt_tokens, result.completion_tokens)

Env vars:
    CHEMTREE_LLM_CACHE=0          Disable caching (always call API)
    CHEMTREE_LLM_CACHE_DB=<path>  Custom cache DB path
"""

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from openai import OpenAI, AsyncOpenAI

MODELS = {
    "fast": "gpt-5-mini",
    "strong": "gpt-5.4",
}

DEFAULT_MODEL = MODELS["fast"]

_CACHE_ENABLED = os.environ.get("CHEMTREE_LLM_CACHE", "1") != "0"
_CACHE_DB_PATH = os.environ.get(
    "CHEMTREE_LLM_CACHE_DB",
    str(Path(__file__).parent.parent.parent / "data" / "llm_cache.db"),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    request_json TEXT NOT NULL,
    response_text TEXT NOT NULL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_model ON llm_cache(model);
CREATE INDEX IF NOT EXISTS idx_cache_created ON llm_cache(created_at);
"""

_local = threading.local()


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached: bool


def _get_cache_conn() -> sqlite3.Connection:
    """Thread-local SQLite connection for the cache."""
    if not hasattr(_local, "cache_conn") or _local.cache_conn is None:
        os.makedirs(os.path.dirname(_CACHE_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(_CACHE_DB_PATH, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        _local.cache_conn = conn
    return _local.cache_conn


def _make_cache_key(model: str, messages: list[dict], response_format: dict | None,
                    max_completion_tokens: int) -> str:
    """Deterministic SHA-256 hash of the request parameters."""
    canonical = json.dumps({
        "model": model,
        "messages": messages,
        "response_format": response_format,
        "max_completion_tokens": max_completion_tokens,
    }, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _make_prompt_hash(messages: list[dict]) -> str:
    """Hash of just the messages (for dedup analysis across models)."""
    return hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()[:16]


def _cache_get(key: str) -> LLMResult | None:
    if not _CACHE_ENABLED:
        return None
    try:
        conn = _get_cache_conn()
        row = conn.execute(
            "SELECT response_text, prompt_tokens, completion_tokens, total_tokens "
            "FROM llm_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row:
            return LLMResult(
                text=row[0], prompt_tokens=row[1],
                completion_tokens=row[2], total_tokens=row[3], cached=True,
            )
    except Exception:
        pass
    return None


def _cache_put(key: str, model: str, messages: list[dict],
               response_format: dict | None, max_completion_tokens: int,
               result: LLMResult) -> None:
    if not _CACHE_ENABLED:
        return
    try:
        conn = _get_cache_conn()
        request_json = json.dumps({
            "model": model, "messages": messages,
            "response_format": response_format,
            "max_completion_tokens": max_completion_tokens,
        }, sort_keys=True, ensure_ascii=True)
        conn.execute(
            "INSERT OR IGNORE INTO llm_cache "
            "(cache_key, model, prompt_hash, request_json, response_text, "
            " prompt_tokens, completion_tokens, total_tokens, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, model, _make_prompt_hash(messages), request_json,
             result.text, result.prompt_tokens, result.completion_tokens,
             result.total_tokens, datetime.now().isoformat()),
        )
        conn.commit()
    except Exception:
        pass


def get_client() -> OpenAI:
    """Get a synchronous OpenAI client."""
    return OpenAI()


def get_async_client() -> AsyncOpenAI:
    """Get an async OpenAI client."""
    return AsyncOpenAI()


def chat_full(messages: list[dict], model: str = None,
              max_completion_tokens: int = 8192,
              json_mode: bool = False, **kwargs) -> LLMResult:
    """Synchronous chat completion with caching. Returns full result with token usage."""
    model = model or DEFAULT_MODEL
    response_format = {"type": "json_object"} if json_mode else None

    cache_key = _make_cache_key(model, messages, response_format, max_completion_tokens)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    create_kwargs = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        **kwargs,
    }
    if response_format:
        create_kwargs["response_format"] = response_format

    client = get_client()
    response = client.chat.completions.create(**create_kwargs)
    text = response.choices[0].message.content or ""
    usage = response.usage

    result = LLMResult(
        text=text,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
        cached=False,
    )
    _cache_put(cache_key, model, messages, response_format, max_completion_tokens, result)
    return result


def chat(messages: list[dict], model: str = None,
         max_completion_tokens: int = 8192,
         json_mode: bool = False, **kwargs) -> str:
    """Synchronous chat completion with caching. Returns response text."""
    return chat_full(messages, model, max_completion_tokens, json_mode, **kwargs).text


async def achat_full(messages: list[dict], model: str = None,
                     max_completion_tokens: int = 8192,
                     json_mode: bool = False, **kwargs) -> LLMResult:
    """Async chat completion with caching. Returns full result with token usage."""
    model = model or DEFAULT_MODEL
    response_format = {"type": "json_object"} if json_mode else None

    cache_key = _make_cache_key(model, messages, response_format, max_completion_tokens)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    create_kwargs = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        **kwargs,
    }
    if response_format:
        create_kwargs["response_format"] = response_format

    aclient = get_async_client()
    response = await aclient.chat.completions.create(**create_kwargs)
    text = response.choices[0].message.content or ""
    usage = response.usage

    result = LLMResult(
        text=text,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
        cached=False,
    )
    _cache_put(cache_key, model, messages, response_format, max_completion_tokens, result)
    return result


async def achat(messages: list[dict], model: str = None,
                max_completion_tokens: int = 8192,
                json_mode: bool = False, **kwargs) -> str:
    """Async chat completion with caching. Returns response text."""
    result = await achat_full(messages, model, max_completion_tokens, json_mode, **kwargs)
    return result.text


def cache_stats() -> dict:
    """Return cache statistics."""
    try:
        conn = _get_cache_conn()
        total = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
        by_model = dict(conn.execute(
            "SELECT model, COUNT(*) FROM llm_cache GROUP BY model"
        ).fetchall())
        tokens = conn.execute(
            "SELECT SUM(prompt_tokens), SUM(completion_tokens), SUM(total_tokens) "
            "FROM llm_cache"
        ).fetchone()
        return {
            "total_cached_calls": total,
            "by_model": by_model,
            "total_prompt_tokens": tokens[0] or 0,
            "total_completion_tokens": tokens[1] or 0,
            "total_tokens": tokens[2] or 0,
        }
    except Exception:
        return {"total_cached_calls": 0, "by_model": {}, "total_tokens": 0}
