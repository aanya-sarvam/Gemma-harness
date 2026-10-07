"""Minimal client for an OpenAI-compatible chat endpoint (e.g. vLLM) serving the extraction model."""
import json
import time

import requests


class GemmaClient:
    def __init__(self, base_url, model, api_key=None, timeout=600, retries=4,
                 auth_header="api-subscription-key", stream=False):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            if auth_header.lower() == "authorization":
                self.headers["Authorization"] = f"Bearer {api_key}"
            else:
                self.headers[auth_header] = api_key
        self.timeout = timeout
        self.retries = retries
        self.stream = stream

    def _read_stream(self, r):
        text, finish, usage = [], None, {}
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            usage = d.get("usage") or usage
            for ch in d.get("choices", []):
                text.append((ch.get("delta") or {}).get("content") or "")
                finish = ch.get("finish_reason") or finish
        return "".join(text), finish, usage

    def chat(self, messages, temperature=0.0, max_tokens=4096, json_mode="none", json_schema=None):
        body = {"model": self.model, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens, "stream": self.stream}
        if json_mode == "object" or (json_mode == "schema" and not json_schema):
            body["response_format"] = {"type": "json_object"}
        elif json_mode == "schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "deed", "schema": json_schema}}

        t0 = time.time()
        last = None
        for a in range(self.retries):
            try:
                r = requests.post(self.url, json=body, headers=self.headers,
                                  timeout=self.timeout, stream=self.stream)
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {e}"
                time.sleep(2 ** (a + 1))
                continue
            if r.status_code == 200 and self.stream:
                text, finish, usage = self._read_stream(r)
                return {"text": text, "finish_reason": finish, "usage": usage,
                        "latency_s": round(time.time() - t0, 2), "error": None}
            if r.status_code == 200:
                d = r.json()
                ch = d["choices"][0]
                return {"text": ch["message"].get("content") or "",
                        "finish_reason": ch.get("finish_reason"),
                        "usage": d.get("usage", {}), "latency_s": round(time.time() - t0, 2),
                        "error": None}
            last = f"HTTP {r.status_code}: {r.text[:500]}"
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** (a + 1))
                continue
            break  # 400s (e.g. context overflow, too many images) — don't retry
        return {"text": "", "finish_reason": None, "usage": {},
                "latency_s": round(time.time() - t0, 2), "error": last}
