from __future__ import annotations

import json
import logging
import time
from typing import Optional, Dict, Any, List

import requests

log = logging.getLogger(__name__)


def _with_v1(base: str) -> str:
    base = (base or "").rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


class OpenAICompatibleClient:
    """
    OpenAI-compatible client for llama-server.

    Key differences vs your previous version:
    - No requests.Session(): avoids sticky TCP connections behind a TCP load balancer.
    - Forces "Connection: close" to make each request use a fresh TCP connection.
    - Retries on 503 "Loading model" and on transient disconnects.
    - Keeps the same interface for the final extraction CLI
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        use_grammar: bool = False,
        use_session: bool = False,
    ):
        self.base_url = _with_v1(base_url)
        self.api_key = api_key
        self.model = model
        self.use_grammar = use_grammar
        self.use_session = use_session
        self._session: Optional[requests.Session] = requests.Session() if use_session else None

        # Static headers (no Session, but we still centralize headers)
        self.headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # With TCP load balancer:
        # - close: spread requests across backends (default)
        # - keep-alive + Session: stick to one backend (needed for prompt cache reuse)
        self.headers["Connection"] = "keep-alive" if use_session else "close"
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        response_format: Optional[Dict[str, Any]] = None,
        grammar: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: int = 600,
        max_retries: int = 12,
    ) -> str:
        url = f"{self.base_url}/chat/completions"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if self.use_grammar and grammar:
            payload["grammar"] = grammar
        if extra_body:
            payload.update(extra_body)

        # Exponential-ish backoff (fast start, then slower)
        def _sleep(attempt: int) -> None:
            # 0 -> 1.0s, 1 -> 1.5s, ... capped
            t = min(8.0, 1.0 + attempt * 0.5)
            time.sleep(t)

        last_err: Optional[BaseException] = None

        for attempt in range(max_retries):
            try:
                http = self._session if self._session is not None else requests
                resp = http.post(url, headers=self.headers, json=payload, timeout=timeout)

                # Common llama-server transient state
                if resp.status_code == 503:
                    txt = ""
                    try:
                        txt = resp.text or ""
                    except Exception:
                        pass
                    if "Loading model" in txt:
                        log.warning("LLM 503 Loading model (attempt %d/%d). Retrying...", attempt + 1, max_retries)
                        _sleep(attempt)
                        continue

                # If request body isn't accepted (some backends)
                if resp.status_code in (400, 404, 422):
                    stripped = dict(payload)
                    stripped.pop("response_format", None)
                    stripped.pop("grammar", None)
                    resp = http.post(url, headers=self.headers, json=stripped, timeout=timeout)

                if resp.status_code >= 400:
                    try:
                        log.error("LLM %s: %s", resp.status_code, (resp.text or "")[:2000])
                    except Exception:
                        pass

                resp.raise_for_status()

                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"No choices in response: {data}")
                msg = choices[0].get("message") or {}
                content = msg.get("content", "")
                if not content:
                    raise RuntimeError(f"No content in response: {data}")
                return content

            except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
                # Includes RemoteDisconnected / connection reset issues
                last_err = e
                log.warning("LLM connection error (attempt %d/%d): %s", attempt + 1, max_retries, e)
                _sleep(attempt)
                continue
            except requests.HTTPError as e:
                # Non-retriable unless it's explicitly a transient 5xx; we keep it simple
                last_err = e
                # Retry on 5xx (except if already handled above)
                status = None
                try:
                    status = e.response.status_code  # type: ignore[union-attr]
                except Exception:
                    pass
                if status is not None and 500 <= status < 600 and attempt + 1 < max_retries:
                    log.warning("LLM HTTP %s (attempt %d/%d). Retrying...", status, attempt + 1, max_retries)
                    _sleep(attempt)
                    continue
                raise
            except Exception as e:
                last_err = e
                # retry once or twice for generic transient issues
                if attempt + 1 < max_retries:
                    log.warning("LLM error (attempt %d/%d): %s", attempt + 1, max_retries, e)
                    _sleep(attempt)
                    continue
                raise

        raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")
