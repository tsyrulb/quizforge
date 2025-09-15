from __future__ import annotations
from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, field_validator

Difficulty = Literal["easy", "medium", "hard"]
QType = Literal["mcq", "coding", "sql", "short"]
Lang = Literal["python", "javascript", "typescript", "csharp"]


class MCQChoice(BaseModel):
    id: Literal["A", "B", "C", "D"]
    text: str
    correct: Optional[bool] = False


class MCQQuestion(BaseModel):
    type: Literal["mcq"] = "mcq"
    topic: str
    difficulty: Difficulty
    question: str
    choices: List[MCQChoice]
    correct_id: Literal["A", "B", "C", "D"]
    explanation: str
    citations: Optional[List[str]] = None

    @field_validator("choices")
    @classmethod
    def exactly_four(cls, v: List[MCQChoice]) -> List[MCQChoice]:
        if len(v) != 4:
            raise ValueError("MCQ must have exactly 4 choices")
        return v


class CodingQuestion(BaseModel):
    type: Literal["coding"] = "coding"
    title: str
    language: Lang
    difficulty: Difficulty
    tags: List[str]
    prompt: str
    signature: str
    starter_code: str
    tests: List[Dict[str, str]]  # [{name, input, expected}]
    constraints: Optional[List[str]] = None
    explanation: Optional[str] = None


class SQLQuestion(BaseModel):
    type: Literal["sql"] = "sql"
    title: str
    dataset: str
    difficulty: Difficulty
    prompt: str
    canonical_query: str
    expected_result_hash: Optional[str] = None
    hints: Optional[List[str]] = None


class ShortQuestion(BaseModel):
    type: Literal["short"] = "short"
    topic: str
    difficulty: Difficulty
    prompt: str
    rubric_points: List[str]
    citations: Optional[List[str]] = None


class GenerateRequest(BaseModel):
    qtype: QType
    topic: str
    difficulty: Difficulty = "easy"
    language: Optional[Lang] = None  # <-- no introspection; clean & explicit
    use_rag: bool = True
    tags: Optional[List[str]] = None


class GradeMCQRequest(BaseModel):
    question: MCQQuestion
    answer_id: Literal["A", "B", "C", "D"]


class GradeShortRequest(BaseModel):
    question: ShortQuestion
    answer_text: str


class GradeResult(BaseModel):
    correct: bool
    score: float
    feedback: str
