import re
import json
import random

# ---------- Optional deps (guarded) ----------
# PyMuPDF (fitz) for PDF text
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# Ollama chat (LLM)
try:
    from ollama import chat as _ollama_chat
except Exception:
    _ollama_chat = None

# scikit-learn for TF-IDF cosine
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# ============================================================
# PDF Extraction
# ============================================================
def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract text from an uploaded PDF using PyMuPDF.
    Returns empty string if PyMuPDF is unavailable.
    """
    try:
        data = uploaded_file.read()
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        if not fitz:
            return ""
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            text = []
            for page in doc:
                text.append(page.get_text() or "")
            return "".join(text)
        finally:
            doc.close()
    except Exception:
        return ""


# ============================================================
# Ollama helpers
# ============================================================
def _ollama_chat_or_raise(model: str, messages: list[dict]) -> dict:
    """Call Ollama chat if available; raise with clear message otherwise."""
    if _ollama_chat is None:
        raise RuntimeError(
            "Ollama is not installed or not running. Install/start Ollama or remove LLM features."
        )
    return _ollama_chat(model=model, messages=messages)


# ============================================================
# Resume analysis (LLM)
# ============================================================
def analyze_resume_with_gemma(resume_text: str) -> str:
    """
    Analyze resume text using Gemma 2B (Ollama).
    Returns string (or error message).
    """
    prompt = f"""
You are a professional resume reviewer.
Analyze the following resume and return EXACTLY these sections with clear headings:

Strengths:
- (bulleted points)

Weaknesses:
- (bulleted points)

Suggestions for improvement:
- (bulleted points)

Overall score: X/10

Resume:
{resume_text}
"""
    try:
        response = _ollama_chat_or_raise(
            model="gemma:2b",
            messages=[
                {"role": "system", "content": "You are an AI resume advisor."},
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]
    except Exception as e:
        return f"(Ollama error: {e})"


def analyze_resume_vs_jd(resume_text: str, jd_text: str) -> str:
    """
    Compare resume vs JD using Gemma 2B (Ollama).
    We enforce headings that the Streamlit parser expects.
    """
    prompt = f"""
You are a professional career advisor.
Compare the Resume with the Job Description and return EXACTLY these sections:

Match score: X/10

Matched skills/Experience:
- (bulleted list of overlaps)

Missing or weak areas:
- (bulleted list of gaps)

Suggestions:
- (bulleted, concrete tailoring suggestions)

Resume:
{resume_text}

Job Description:
{jd_text}
"""
    try:
        response = _ollama_chat_or_raise(
            model="gemma:2b",
            messages=[
                {"role": "system", "content": "You are an expert in job fit analysis."},
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]
    except Exception as e:
        return f"(Ollama error: {e})"


# ============================================================
# Deterministic KPI helpers (Cosine / Skills Overlap / Hybrid)
# ============================================================
DEFAULT_SKILLS = [
    "python", "java", "c", "c++", "html", "css", "javascript", "typescript", "react", "node",
    "sql", "postgres", "postgresql", "mysql", "mongodb",
    "aws", "gcp", "azure", "docker", "kubernetes",
    "linux", "git", "bash", "powershell",
    "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "pytorch",
    "tableau", "power bi", "excel",
    "airflow", "dbt", "spark", "hadoop", "kafka", "etl", "mlops", "ci/cd",
]


def _clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9+\s]", " ", t)
    return " ".join(t.split())


def _extract_skills(text: str, extra: list[str] | None = None) -> set[str]:
    corpus = text.lower()
    keys = set(DEFAULT_SKILLS)
    if extra:
        keys.update([k.lower() for k in extra])
    found = {kw for kw in keys if re.search(rf"\b{re.escape(kw)}\b", corpus)}
    return found


def compute_nlp_scores(resume_text: str, jd_text: str, extra_skill_keywords: list[str] | None = None) -> dict:
    """
    Returns:
      {
        "cosine": float(0..1),
        "skills_overlap": float(0..1),   # Jaccard of detected skills
        "hybrid": float(0..1),           # 0.65*cosine + 0.35*skills_overlap
        "resume_skills": set[str],
        "jd_skills": set[str]
      }

    If scikit-learn is not available, cosine=0.0 and hybrid uses only skills_overlap weight.
    """
    # Cosine similarity (TF-IDF) if available
    if _HAS_SK:
        try:
            vec = TfidfVectorizer(stop_words="english", min_df=1)
            tfidf = vec.fit_transform([_clean_text(resume_text), _clean_text(jd_text)])
            cos = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])  # 0..1
        except Exception:
            cos = 0.0
    else:
        cos = 0.0

    # Skills overlap (Jaccard)
    rs = _extract_skills(resume_text, extra_skill_keywords)
    js = _extract_skills(jd_text, extra_skill_keywords)
    union = max(1, len(rs | js))
    jacc = len(rs & js) / union  # 0..1

    # Weighted hybrid
    if _HAS_SK:
        hybrid = 0.65 * cos + 0.35 * jacc
    else:
        # degrade gracefully if sklearn missing
        hybrid = 0.35 * jacc

    return {
        "cosine": cos,
        "skills_overlap": jacc,
        "hybrid": hybrid,
        "resume_skills": rs,
        "jd_skills": js,
    }


# ============================================================
# Interview Coach helpers
# ============================================================
_BEHAVIORAL_QS = [
    "Tell me about a time you had to learn something quickly.",
    "Describe a conflict on a team and how you resolved it.",
    "Tell me about a time you made a mistake. What did you learn?",
    "Give an example of a time you worked under a tight deadline.",
    "Tell me about a time you influenced a decision without authority.",
    "Describe a time you handled ambiguous requirements.",
    "Tell me about a time you prioritized tasks with limited resources.",
]

_TECH_QS_GENERAL = [
    "Explain Big-O of an algorithm you recently optimized.",
    "What data structure would you use to implement an LRU cache and why?",
    "How would you design a rate limiter for an API? Outline components and trade-offs.",
    "Explain ACID vs BASE and when eventual consistency is acceptable.",
    "Given a large log file, how would you find the top K frequent entries?",
    "How do you handle memory leaks in Python? Give examples/tools.",
    "What is the difference between multiprocessing and multithreading in Python?",
]


def _safe_json_extract_list(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        s = text.find("[")
        e = text.rfind("]")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e+1])
    except Exception:
        return None
    return None


def generate_interview_questions(job_title: str = "",
                                 jd_text: str = "",
                                 mode: str = "Behavioral",
                                 level: str = "Entry",
                                 n: int = 5) -> list[str]:
    sys = "You are an expert interview coach. Respond in English."
    prompt = f"""
Generate {n} {mode} interview questions for a {level}-level candidate.
If a Job Description is provided, align topics and keywords to it. 
Return ONLY a JSON list of strings, no extra text or formatting.

Job Title: {job_title or "N/A"}

Job Description:
{jd_text.strip() or "N/A"}

Examples of the expected output:
["Question 1...", "Question 2...", "Question 3..."]
"""
    # Try LLM
    try:
        response = _ollama_chat_or_raise(
            model="gemma:2b",
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": prompt}],
        )
        raw = response["message"]["content"]
        parsed = _safe_json_extract_list(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed[:n]
    except Exception:
        pass

    # Fallback
    bank = _BEHAVIORAL_QS if mode.lower().startswith("behav") else _TECH_QS_GENERAL
    random.shuffle(bank)
    return bank[:n]


def critique_interview_answer(question: str,
                              answer: str,
                              mode: str = "Behavioral",
                              jd_text: str | None = None) -> dict:
    sys = "You are a rigorous interview evaluator. Be concise and actionable."
    rubric = """
Return ONLY JSON with this schema:
{
  "scores": {
    "clarity": 1-5,
    "structure": 1-5,
    "technical_depth": 1-5,
    "impact": 1-5,
    "conciseness": 1-5
  },
  "summary": "2-3 sentences overall feedback",
  "suggestions": ["bullet", "bullet", "bullet"],
  "improved_answer": "an improved answer (use STAR if behavioral)"
}
Rules:
- Use STAR (Situation, Task, Action, Result) if behavioral.
- If Job Description hints are present, align suggestions to it.
- Keep 'improved_answer' under 200-250 words.
"""
    user = f"""
Question: {question}

Candidate Answer:
{answer}

Mode: {mode}
Job Description (optional):
{jd_text or "N/A"}
"""
    # Try LLM
    try:
        response = _ollama_chat_or_raise(
            model="gemma:2b",
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": rubric + "\n" + user}],
        )
        raw = response["message"]["content"]
        data = json.loads(raw)
        if "scores" in data and "improved_answer" in data:
            return data
    except Exception:
        pass

    # Fallback heuristic
    length = len(answer.split())
    has_numbers = any(ch.isdigit() for ch in answer)
    star_hits = sum(1 for k in ["situation", "task", "action", "result"] if k in answer.lower())
    scores = {
        "clarity": 3 if length >= 40 else 2,
        "structure": min(5, 2 + star_hits),
        "technical_depth": 3 + (1 if has_numbers else 0),
        "impact": 3 + (1 if has_numbers else 0),
        "conciseness": 4 if 60 <= length <= 220 else (3 if length < 60 else 2),
    }
    summary = "Decent start. Sharpen structure and make outcomes measurable."
    suggestions = [
        "Use STAR: briefly set context, then focus on actions and results.",
        "Quantify impact (e.g., %, time saved, errors reduced).",
        "Explain trade-offs and tools used; keep it within 2 minutes.",
    ]
    improved = (
        "Situation: Briefly describe the context and goal.\n"
        "Task: Your specific responsibility.\n"
        "Action: 2–3 concrete steps you took, including tools and trade-offs.\n"
        "Result: Quantified outcome (e.g., 23% faster, 2 bugs/week → 0.3).\n"
        "Reflection: One learning or improvement you’d make."
    )
    return {"scores": scores, "summary": summary, "suggestions": suggestions, "improved_answer": improved}
