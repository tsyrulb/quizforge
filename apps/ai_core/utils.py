import json, yaml
from typing import Any
import re

def normalize_short(d: dict) -> dict:
    if not isinstance(d, dict):
        return d
    out = dict(d)

    # Fix type
    t = str(out.get("type","")).lower()
    if t in ("short-answer", "short_answer", "free_text", "open", "open-ended"):
        out["type"] = "short"
    else:
        out["type"] = "short"

    # Coerce rubric_points -> list[str] (3–6)
    rp = out.get("rubric_points")
    points: list[str] = []
    if isinstance(rp, list):
        points = [str(x).strip() for x in rp if str(x).strip()]
    elif isinstance(rp, int):
        n = max(3, min(6, rp))
        points = [f"Must include key point {i+1}." for i in range(n)]
    elif isinstance(rp, str):
        parts = [p.strip() for p in re.split(r"[;\n•,-]", rp) if p.strip()]
        points = parts[:6]

    # Ensure at least 3 items
    while len(points) < 3:
        points.append(f"Must include key point {len(points)+1}.")

    out["rubric_points"] = points[:6]
    return out

def normalize_mcq(d: dict) -> dict:
    """Coerce common LLM deviations into our MCQ schema."""
    if not isinstance(d, dict):
        return d
    out = dict(d)

    # type → "mcq"
    t = str(out.get("type", "")).lower()
    if t in ("multiple_choice", "multiple-choice", "mcq"):
        out["type"] = "mcq"
    else:
        out["type"] = "mcq"

    # choices: strings like "A) Text" → objects
    choices = out.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], str):
        fixed = []
        for i, s in enumerate(choices[:4]):
            s = str(s)
            m = re.match(r'^\s*([ABCD])[\)\.\:]\s*(.+)$', s)
            cid = m.group(1) if m else "ABCD"[i]
            txt = m.group(2).strip() if m else s.strip()
            fixed.append({"id": cid, "text": txt, "correct": False})
        out["choices"] = fixed

    # ensure 4 choices, ids A..D, set correct flags
    if isinstance(out.get("choices"), list):
        out["choices"] = out["choices"][:4]
        ids = ["A", "B", "C", "D"]
        for i, ch in enumerate(out["choices"]):
            if not isinstance(ch, dict):
                out["choices"][i] = {"id": ids[i], "text": str(ch), "correct": False}
            else:
                ch["id"] = ids[i]  # normalize order
                ch.setdefault("text", "")
                ch.setdefault("correct", False)

    # correct_id normalization
    corr = out.get("correct_id") or out.get("answer") or out.get("correct")
    if isinstance(corr, str):
        corr = corr.strip().upper()[:1]
    if corr not in {"A", "B", "C", "D"}:
        corr = "A"
    out["correct_id"] = corr

    # flip correct flag on the right choice
    if isinstance(out.get("choices"), list):
        for ch in out["choices"]:
            if isinstance(ch, dict):
                ch["correct"] = (ch.get("id") == corr)

    return out

def ensure_json(s: str) -> dict:
    # Convert YAML → JSON if needed; or parse JSON directly
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        data = yaml.safe_load(s)
        return data

def truncate(text: str, n: int = 4000) -> str:
    return text if len(text) <= n else text[:n] + "…"
