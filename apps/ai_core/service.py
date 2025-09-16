import os
import json
from typing import Optional
from utils import normalize_mcq, normalize_short, strip_cot, first_sentence
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx 

from schemas import (
    GenerateRequest, MCQQuestion, CodingQuestion, SQLQuestion, ShortQuestion,
    GradeMCQRequest, GradeShortRequest, GradeResult
)
from llm_client import chat, pick_model
from prompts import (
    GENERIC_SYSTEM, MCQ_USER_TMPL, CODING_USER_TMPL, SQL_USER_TMPL,
    SHORT_USER_TMPL, GRADE_SHORT_SYSTEM, GRADE_SHORT_USER_TMPL
)
from rag.retriever import get_index, ingest_example_docs
from utils import ensure_json, truncate

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------
load_dotenv()
app = FastAPI(title="QuizForge AI Core", version="0.2.0")

# Fast startup: do NOT bootstrap RAG unless explicitly requested.
# If you want to seed demo chunks, set RAG_BOOTSTRAP=1 in .env.
if os.getenv("RAG_BOOTSTRAP", "0") == "1":
    persist = os.getenv("RAG_PERSIST", "./rag_store")
    if not os.path.exists(persist) or (os.path.isdir(persist) and not os.listdir(persist)):
        try:
            ingest_example_docs()
        except Exception:
            # Seeding is optional; ignore failures to keep startup resilient
            pass


# ------------------------------------------------------------------------------
# Models for small helper endpoints
# ------------------------------------------------------------------------------
class IngestDoc(BaseModel):
    title: str
    text: str
    source: Optional[str] = None


# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_base": os.getenv("LLM_BASE_URL"),
        "models": {
            "general": os.getenv("MODEL_GENERAL"),
            "coder": os.getenv("MODEL_CODER"),
            "reasoner": os.getenv("MODEL_REASONER"),
        },
        "rag": {
            "persist": os.getenv("RAG_PERSIST", "./rag_store"),
            "bootstrap_on_start": os.getenv("RAG_BOOTSTRAP", "0") == "1",
        },
    }


# ------------------------------------------------------------------------------
# RAG ingestion (manual; lazy by design)
# ------------------------------------------------------------------------------
@app.post("/rag/ingest")
def rag_ingest(doc: IngestDoc):
    try:
        get_index().add_documents([doc.dict()])
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, f"RAG ingest failed: {e}")


# ------------------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------------------
@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Generate a single item (mcq | short | coding | sql).
    If req.use_rag is True, we lazily initialize embeddings on first retrieval.
    """
    try:
        context = get_index().retrieve(req.topic, top_k=6) if req.use_rag else ""
    except Exception:
        # If embedding init fails (e.g., first-run downloads), fall back to no context
        context = ""

    if req.qtype == "mcq":
        user = MCQ_USER_TMPL.format(topic=req.topic, difficulty=req.difficulty, context=context)
        model = pick_model("general")
    elif req.qtype == "short":
        user = SHORT_USER_TMPL.format(topic=req.topic, difficulty=req.difficulty, context=context)
        model = pick_model("general")
    elif req.qtype == "coding":
        lang = req.language or "python"
        tags = req.tags or [req.topic]
        user = CODING_USER_TMPL.format(language=lang, tags=tags, difficulty=req.difficulty)
        model = pick_model("coding")
    elif req.qtype == "sql":
        user = SQL_USER_TMPL.format(dataset="sakila", difficulty=req.difficulty, context=context)
        model = pick_model("general")
    else:
        raise HTTPException(400, "Unsupported qtype")

    try:
        out = await chat(
            messages=[{"role": "system", "content": GENERIC_SYSTEM},
                      {"role": "user", "content": truncate(user)}],
            model=model,
            temperature=0.5,
            response_format_json=True,
        )
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {e}")

    try:
        data = ensure_json(out)
        
        if req.qtype == "mcq":    
            data = normalize_mcq(data)
            return MCQQuestion(**data)
        if req.qtype == "coding": return CodingQuestion(**data)
        if req.qtype == "sql":    return SQLQuestion(**data)
        if req.qtype == "short":  
            data = normalize_short(data)
            return ShortQuestion(**data)
        raise HTTPException(500, "Unexpected qtype after generation.")
    except Exception as ve:
        # Surface the model output snippet to help debug schema issues
        snippet = (out[:400] + "…") if isinstance(out, str) and len(out) > 400 else out
        raise HTTPException(422, f"Model output did not match schema: {ve}. Output snippet: {snippet}")


# ------------------------------------------------------------------------------
# Grading
# ------------------------------------------------------------------------------
@app.post("/grade/mcq", response_model=GradeResult)
def grade_mcq(req: GradeMCQRequest):
    ok = (req.answer_id == req.question.correct_id)
    return GradeResult(
        correct=ok,
        score=1.0 if ok else 0.0,
        feedback=("Correct." if ok else f"Incorrect. Correct answer is {req.question.correct_id}."),
    )


@app.post("/grade/short", response_model=GradeResult)
async def grade_short(req: GradeShortRequest):
    """
    Grade short-answer strictly by rubric using a reasoning model; fallback to keyword proportion.
    """
    try:
        user = GRADE_SHORT_USER_TMPL.format(
            question_json=json.dumps(req.question.dict(), ensure_ascii=False),
            answer=req.answer_text
        )
        out = await chat(
            messages=[{"role": "system", "content": GRADE_SHORT_SYSTEM},
                      {"role": "user", "content": truncate(user)}],
            model=pick_model("reason"),
            temperature=0.0,
            response_format_json=True,
        )
        data = ensure_json(out)
        # Validate expected keys exist
        _ = data["correct"]; _ = data["score"]; _ = data["feedback"]
        return GradeResult(**data)
    except Exception:
        # Deterministic fallback: simple rubric keyword coverage
        pts = req.question.rubric_points
        hits = sum(int(p.lower() in req.answer_text.lower()) for p in pts)
        score = hits / max(1, len(pts))
        missing = [p for p in pts if p.lower() not in req.answer_text.lower()]
        return GradeResult(
            correct=score >= 0.8,
            score=score,
            feedback=("Good coverage." if score >= 0.8 else f"Missing: {', '.join(missing[:4])}…"),
        )


# ------------------------------------------------------------------------------
# Hints (no separate prompt constants required)
# ------------------------------------------------------------------------------
@app.post("/hint")
async def hint(body: dict):
    q = body.get("question", {}) or {}
    failed = body.get("failed_tests", [])[:5]
    partial = (body.get("partial_answer", "") or "")[:1200]

    brief = {
        "type": q.get("type"), "title": q.get("title"), "language": q.get("language"),
        "difficulty": q.get("difficulty"), "signature": q.get("signature"),
        "prompt": q.get("prompt"), "constraints": (q.get("constraints") or [])[:4],
        "failed_tests": failed,
    }
    prompt = (
        "Provide ONE concise, actionable hint (no solutions, no reasoning dumps).\n"
        "Reply with the hint sentence only.\n\n"
        f"{json.dumps(brief, ensure_ascii=False)}\n\n"
        f"Partial answer: {partial}"
    )

    try:
        out = await chat(
            messages=[
                {"role":"system","content":"Reply with ONE actionable hint, ≤25 words. No preamble, no examples, no lists, no reasoning tags."},
                {"role":"user","content":truncate(prompt)}
            ],
            model=pick_model("reason"),
            temperature=0.1,
            max_tokens=80,
            stop=["</think>", "\n\n"]   
        )

        clean = first_sentence(strip_cot(out))
        if not clean:  # fallback to general model if empty
            out2 = await chat(
                messages=[{"role":"system","content":"ONE concise hint, ≤25 words."},
                          {"role":"user","content":truncate(prompt)}],
                model=pick_model("general"),
                temperature=0.1, max_tokens=80, stop=["\n\n"]
            )
            clean = first_sentence(strip_cot(out2)) or "Allocate count[max(nums)+1], increment for each value, return the count array."
        return {"hint": clean}

    except httpx.HTTPStatusError as e:
        detail = f"{e.response.status_code} {e.response.reason_phrase} - {e.response.text[:300]}"
        raise HTTPException(502, f"LLM hint failed: {detail}")
    except httpx.TimeoutException:
        raise HTTPException(502, "LLM hint failed: timeout (try smaller model or raise LLM_TIMEOUT).")
    except Exception as e:
        raise HTTPException(502, f"LLM hint failed: {repr(e)}")