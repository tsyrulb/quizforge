import os, httpx, json
from typing import List, Dict, Optional, Union

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY  = os.getenv("LLM_API_KEY", "ollama")
MODEL_GENERAL = os.getenv("MODEL_GENERAL", "qwen2.5:7b-instruct")
MODEL_CODER   = os.getenv("MODEL_CODER", "qwen2.5-coder:7b")
MODEL_REASON  = os.getenv("MODEL_REASONER", "deepseek-r1:7b")
TIMEOUT_S = int(os.getenv("LLM_TIMEOUT", "600"))

HEADERS = {"Authorization": f"Bearer {API_KEY}"}

async def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.4,
    response_format_json: bool = False,
    max_tokens: Optional[int] = None,
    stop: Optional[list[str]] = None
) -> str:
    payload: Dict[str, Union[str, float, Dict, List]] = {
        "model": model or MODEL_GENERAL,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if stop: payload["stop"] = stop
    # Use explicit httpx.Timeout object
    timeout = httpx.Timeout(connect=10.0, read=TIMEOUT_S, write=60.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

def pick_model(kind: str) -> str:
    if kind == "coding": return MODEL_CODER
    if kind == "reason": return MODEL_REASON
    return MODEL_GENERAL