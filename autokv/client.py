"""Small standard-library client for the vLLM HTTP server."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping


class VllmHttpError(RuntimeError):
    def __init__(self, status: int | None, body: str):
        self.status = status
        self.body = body
        super().__init__(f"vLLM HTTP request failed (status={status}): {body[:500]}")


class VllmClient:
    def __init__(
        self,
        base_url: str,
        model_id: str,
        *,
        timeout: float = 120.0,
        max_response_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http or https")
        if not model_id:
            raise ValueError("model_id cannot be empty")

    @staticmethod
    def completion_payload(
        model_id: str, prompt: str, max_tokens: int
    ) -> dict[str, Any]:
        return {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "stream": False,
        }

    def _read(self, response: Any) -> bytes:
        body = response.read(self.max_response_bytes + 1)
        if len(body) > self.max_response_bytes:
            raise VllmHttpError(response.status, "response exceeded size limit")
        return body

    def _request_bytes(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> bytes:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return self._read(response)
        except urllib.error.HTTPError as exc:
            body = exc.read(4096).decode("utf-8", errors="replace")
            raise VllmHttpError(exc.code, body) from exc
        except urllib.error.URLError as exc:
            raise VllmHttpError(None, str(exc.reason)) from exc

    def _request_json(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        body = self._request_bytes(method, path, payload)
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise VllmHttpError(200, "response was not valid JSON") from exc
        if not isinstance(parsed, Mapping):
            raise VllmHttpError(200, "response JSON was not an object")
        return parsed

    def health(self) -> bool:
        try:
            self._request_bytes("GET", "/health")
            return True
        except VllmHttpError:
            return False

    def tokenize(self, prompt: str) -> int:
        response = self._request_json(
            "POST",
            "/tokenize",
            {
                "model": self.model_id,
                "prompt": prompt,
                "add_special_tokens": True,
                "return_token_strs": False,
            },
        )
        count = response.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise VllmHttpError(200, "tokenize response has no valid count")
        return count

    def complete(self, prompt: str, max_tokens: int) -> Mapping[str, Any]:
        return self._request_json(
            "POST",
            "/v1/completions",
            self.completion_payload(self.model_id, prompt, max_tokens),
        )

    def chat_complete(self, user_prompt: str, max_tokens: int) -> Mapping[str, Any]:
        """使用服务端冻结的 chat template 生成，供 v2 已计数消息使用。"""
        return self._request_json(
            "POST",
            "/v1/chat/completions",
            {
                "model": self.model_id,
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "top_p": 1,
                "seed": 42,
                "stream": False,
            },
        )

    def echo_logprobs(self, prompt: str) -> Mapping[str, Any]:
        payload = self.completion_payload(self.model_id, prompt, 0)
        payload.update({"echo": True, "logprobs": 1})
        return self._request_json("POST", "/v1/completions", payload)


def wait_until_ready(
    client: VllmClient,
    *,
    timeout_seconds: float = 900.0,
    interval_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if client.health():
            return
        sleep(interval_seconds)
    raise TimeoutError(f"vLLM server was not healthy after {timeout_seconds:.0f} seconds")
