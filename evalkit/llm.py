"""Thin recording wrapper over the harness GemmaClient."""
import os

from common import extract_json, load_env
from gemma_client import GemmaClient

load_env()


class LLM:
    def __init__(self, timeout=300):
        self.client = GemmaClient(os.environ["GEMMA_BASE_URL"],
                                  os.environ.get("GEMMA_MODEL", "gemma4"), os.environ.get("GEMMA_API_KEY"), timeout=timeout)

    def call(self, messages, tag="", temperature=0.0, max_tokens=4096, mode="object", json_schema=None):
        """Returns (parsed_or_None, call_record). Blank output with response_format (a known API quirk) is retried once without it."""
        r = self.client.chat(messages, temperature, max_tokens, mode, json_schema)
        fallback = False
        if not r["text"] and not r["error"] and mode != "none":
            msgs = [dict(m) for m in messages]
            last = msgs[-1]
            tail = "\n\nReturn only one JSON object, no commentary."
            if isinstance(last["content"], str):
                last["content"] += tail
            else:
                last["content"] = list(last["content"]) + [{"type": "text", "text": tail}]
            r = self.client.chat(msgs, temperature, max_tokens, "none")
            fallback = True
        parsed = extract_json(r["text"])
        u = r.get("usage") or {}
        return parsed, dict(tag=tag, prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0),
                            latency_s=r["latency_s"], error=r["error"], finish_reason=r.get("finish_reason"),
                            valid_json=parsed is not None, blank_fallback=fallback, text=(r["text"] or "")[:8000])
