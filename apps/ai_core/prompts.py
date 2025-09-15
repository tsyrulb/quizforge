GENERIC_SYSTEM = (
  "You are a precise item writer. Always output valid JSON matching the schema. "
  "No code fences or commentary."
)

MCQ_USER_TMPL = """Create ONE multiple-choice question for topic: {topic}.
Difficulty: {difficulty}. EXACTLY 4 options (A–D), ONE correct.

Return a SINGLE JSON object with EXACTLY these keys:
- type: "mcq"            # literal string 'mcq'
- topic: string
- difficulty: "easy" | "medium" | "hard"
- question: string
- choices: [             # array of 4 objects, in order A,B,C,D
    {{"id":"A","text":"...", "correct":false}},
    {{"id":"B","text":"...", "correct":false}},
    {{"id":"C","text":"...", "correct":false}},
    {{"id":"D","text":"...", "correct":false}}
  ]
- correct_id: "A" | "B" | "C" | "D"
- explanation: string
- citations: [string]?   # optional

STRICT RULES:
- Use type="mcq" exactly (do NOT use "multiple_choice").
- 'choices' MUST be objects with id/text/correct (do NOT use plain strings).
- Set 'correct': true only on the correct choice AND make 'correct_id' match it.
- No code fences, no extra commentary, JSON only.

Context (may be empty):
{context}
"""

SHORT_USER_TMPL = """Create ONE short-answer question for topic: {topic}.
Difficulty: {difficulty}.

Return a SINGLE JSON object with EXACTLY these keys:
- type: "short"          # literal 'short'
- topic: string
- difficulty: "easy" | "medium" | "hard"
- prompt: string
- rubric_points: [string, string, string]  # 3–6 short bullet points
- citations: [string]?   # optional

STRICT RULES:
- Use type="short" (NOT "short-answer").
- rubric_points MUST be an array of 3–6 SHORT strings.
- No code fences, no prose around the JSON, JSON only.

Context (may be empty):
{context}
"""

CODING_USER_TMPL = """Create ONE original coding task (not from public sites).
Language: {language}. Topic tags: {tags}. Difficulty: {difficulty}.
Return JSON with keys: type,title,language,difficulty,tags,prompt,signature,starter_code,tests[],constraints,explanation.
Each test has name,input,expected. Deterministic only.
"""

SQL_USER_TMPL = """Create ONE SQL task grounded in the given dataset domain.
Dataset: {dataset}. Difficulty: {difficulty}.
Return JSON with keys: type,title,dataset,difficulty,prompt,canonical_query,expected_result_hash,hints.

Context:
{context}
"""

GRADE_SHORT_SYSTEM = (
  "You grade short answers strictly by rubric. "
  "Return JSON {correct: bool, score: float (0..1), feedback: string}."
)

GRADE_SHORT_USER_TMPL = """Question JSON:
{question_json}

Student answer:
{answer}

Score=1.0 if all rubric_points satisfied; partial otherwise. Be terse; list missing points."""
