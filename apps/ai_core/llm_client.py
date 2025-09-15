import os, httpx, json
from typing import List, Dict, Optional

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY  = os.getenv("LLM_API_KEY", "ollama")
MODEL_GENERAL = os.getenv("MODEL_GENERAL", "qwen2.5:7b-instruct")
MODEL_CODER   = os.getenv("MODEL_CODER", "qwen2.5-coder:7b")
MODEL_REASON  = os.getenv("MODEL_REASONER", "deepseek-r1:7b")

HEADERS = {"Authorization": f"Bearer {API_KEY}"}

async def chat(
    messages: List[Dict[str,str]],
    model: Optional[str] = None,
    temperature: float = 0.4,
    response_format_json: bool = False,
) -> str:
    payload = {
        "model": model or MODEL_GENERAL,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format_json:
        # OpenAI-style schema hint
        payload["response_format"] = {"type": "json_object"}
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{BASE_URL}/chat/completions", headers=HEADERS, json=payload)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return content

def pick_model(kind: str) -> str:
    if kind == "coding": return MODEL_CODER
    if kind == "reason": return MODEL_REASON
    return MODEL_GENERAL
