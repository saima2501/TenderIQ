"""
ollama_client.py - Local Ollama REST API Client
Part of Step 2G.3: Local AI Explanation Service

Provides communication with a local Ollama instance running Qwen (qwen2.5:1.5b).
Gracefully handles offline states, timeouts, and connection errors without raising
unhandled exceptions to callers.
"""

import os
import logging
from typing import Optional, Dict, Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_ENDPOINT = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
DEFAULT_CONNECT_TIMEOUT = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "2.0"))
DEFAULT_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "45.0"))


class OllamaClient:
    """Synchronous and safe client for Ollama local API."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_ENDPOINT,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self._timeout = httpx.Timeout(
            timeout=self.request_timeout,
            connect=self.connect_timeout,
        )

    def is_available(self) -> bool:
        """Checks if local Ollama daemon is reachable and responding."""
        try:
            with httpx.Client(timeout=httpx.Timeout(2.0, connect=1.0)) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 350,
    ) -> Optional[str]:
        """Sends chat generation request to Ollama.

        Returns string completion on success, or None on any connection error,
        timeout, or invalid status code.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                res = client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

                if res.status_code != 200:
                    logger.warning(
                        "Ollama API returned non-200 status code: %d - %s",
                        res.status_code,
                        res.text[:200],
                    )
                    return None

                data = res.json()
                content = data.get("message", {}).get("content", "").strip()
                return content if content else None

        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            logger.info("Ollama is not reachable at %s: %s", self.base_url, exc)
            return None
        except httpx.ReadTimeout as exc:
            logger.warning("Ollama request timed out after %.1fs: %s", self.request_timeout, exc)
            return None
        except Exception as exc:
            logger.warning("Unexpected error communicating with Ollama: %s", exc)
            return None
