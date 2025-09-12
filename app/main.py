# main.py — CareerWise AI Assistant (no-lottie version)
import re
import json
from pathlib import Path
from resume_analysis import generate_interview_questions, critique_interview_answer

import streamlit as st

from resume_analysis import (
    extract_text_from_pdf,
    analyze_resume_with_gemma,
    analyze_resume_vs_jd,
    compute_nlp_scores,   # deterministic KPIs (cosine, skills overlap, hybrid)
)
#For page global styles
st.set_page_config(page_title="CareerWise AI Assistant", layout="wide")

st.markdown(
    """
<style>
.card{
  background:#fff;border:1px solid #e5f3e9;border-radius:16px;
  padding:1rem 1.2rem;margin:.75rem 0;box-shadow:0 6px 18px rgba(60,179,113,.10)
}
.card h4{margin:.2rem 0 .6rem 0;color:#3CB371}
.kpi{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;margin:.5rem 0 1rem}
.kpi .tile{
  background:#fff;border:1px solid #e5f3e9;border-radius:14px;padding:.9rem;text-align:center;
  box-shadow:0 4px 12px rgba(60,179,113,.08)
}
.kpi .big{font-size:1.8rem;font-weight:800;color:#1C1C1C;margin:.2rem 0}
.pills{display:flex;flex-wrap:wrap;gap:.4rem}
.pill{background:#e9f8f0;color:#2E8B57;border-radius:999px;padding:.25rem .6rem;font-size:.85rem}
.muted{opacity:.9}
.center{display:flex;justify-content:center;align-items:center;margin:.5rem 0 1rem;}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Simple inline SVG icons (no dependencies)
# ------------------------------------------------------------
SVG_UPLOAD = """
<svg viewBox="0 0 24 24" width="120" height="120" fill="none" stroke="#3CB371" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M20 16.58A5 5 0 0 0 18 7h-1.26A8 8 0 1 0 4 15.25"/>
  <path d="M12 12v9"/>
  <path d="m8 16 4-4 4 4"/>
</svg>
"""

SVG_SEARCH = """
<svg viewBox="0 0 24 24" width="120" height="120" fill="none" stroke="#3CB371" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="11" cy="11" r="8"/>
  <line x1="21" y1="21" x2="16.65" y2="16.65"/>
</svg>
"""

SVG_MIC = """
<svg viewBox="0 0 24 24" width="120" height="120" fill="none" stroke="#3CB371" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <rect x="9" y="2" width="6" height="11" rx="3"/>
  <path d="M5 10v2a7 7 0 0 0 14 0v-2"/>
  <line x1="12" y1="19" x2="12" y2="22"/>
</svg>
"""

def show_svg(svg_str: str):
    st.markdown(f'<div class="center">{svg_str}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# Hero Header
# ------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding:2rem 0; background:linear-gradient(135deg, #F8FFF4, #E9F8F0); border-radius:24px; box-shadow:0 12px 28px rgba(60,179,113,0.15);">
        <h1 style="color:#3CB371; font-size:2.6rem; font-weight:800; margin-bottom:0.4rem;">
            CareerWise AI Assistant
        </h1>
        <h3 style="color:#1C1C1C; font-weight:400; margin-top:0;">
            Your AI-powered career partner for resumes, job matching, and interviews.
        </h3>
    </div>
    """,
    unsafe_allow_html=True,
)

# Small hero icon
show_svg(SVG_UPLOAD)

# ------------------------------------------------------------
# Parser for LLM output (Resume vs JD)
# ------------------------------------------------------------
def parse_match_text(txt: str):
    out = {"score": None, "matched": "", "missing": "", "suggestions": "", "strengths": ""}
    if not txt:
        return out

    # "Match score: 8/10" (optionally with markdown ##)
    m = re.search(r"(?im)^\s*(?:#+\s*)?match\s*score\s*[:\-]?\s*(\d+)\s*/\s*10", txt)
    if m:
        try:
            out["score"] = int(m.group(1))
        except Exception:
            out["score"] = None

    # Grab body after label (label may or may not end with ':' or '-')
    def section(label_regex):
        pat = rf"(?is)(?:{label_regex})\s*[:\-]?\s*([\s\S]*?)(?=(?:\n\s*\n|\r?\n)\s*[A-Z][^\n]{{0,80}}[:\-]|\Z)"
        m2 = re.search(pat, txt)
        return (m2.group(1).strip() if m2 else "")

    out["matched"] = section(r"Matched\s+skills\/?Experience|Matched\s+skills|Matched\s+experience")
    out["missing"] = section(r"Missing\s+or\s+weak\s+areas|Missing|Weak\s+areas|Gaps")
    out["suggestions"] = section(r"Suggestions(?:\s+for\s+tailoring\s+the\s+resume)?|Improvements|Recommendations")

    strengths = section(r"Strengths")
    out["strengths"] = strengths if strengths else out["matched"]
    return out

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tabs = st.tabs(["📄 Resume Feedback", "📌 JD Match", "🎤 Interview Coach"])

# ===================== Resume Feedback =====================
with tabs[0]:
    st.subheader("Resume Feedback")

    uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")
    if uploaded_file is not None:
        with st.spinner("Extracting resume text..."):
            resume_text = extract_text_from_pdf(uploaded_file)
        st.success(f"✅ Extracted {len(resume_text)} characters of text")

        with st.spinner("Analyzing resume with Gemma 2B..."):
            feedback = analyze_resume_with_gemma(resume_text)

        st.markdown('<div class="card"><h4>🧠 AI Feedback</h4></div>', unsafe_allow_html=True)
        st.write(feedback)
        st.download_button("📄 Download Feedback", feedback, "resume_feedback.txt")

# ======================== JD Match ========================
with tabs[1]:
    st.subheader("JD Match")

    # Small search icon
    show_svg(SVG_SEARCH)

    uploaded_resume = st.file_uploader("Upload your Resume (PDF)", type="pdf", key="resume")
    uploaded_jd = st.file_uploader("Upload Job Description (PDF or TXT)", type=["pdf", "txt"], key="jd")

    st.markdown("### Or Paste Job Description (Optional)")
    pasted_jd = st.text_area("Paste Job Description here (if not uploading)", height=200)

    if uploaded_resume and (uploaded_jd or pasted_jd.strip()):
        with st.spinner("Extracting texts..."):
            resume_text = extract_text_from_pdf(uploaded_resume)
            if uploaded_jd:
                if uploaded_jd.name.lower().endswith(".pdf"):
                    jd_text = extract_text_from_pdf(uploaded_jd)
                else:
                    jd_text = uploaded_jd.read().decode("utf-8", errors="ignore")
            else:
                jd_text = pasted_jd

        with st.spinner("Comparing resume and JD..."):
            result = analyze_resume_vs_jd(resume_text, jd_text)

        parts = parse_match_text(result)
        llm_score = parts["score"] if parts["score"] is not None else None

        # ---------- Visual KPIs (deterministic) ----------
        nlp = compute_nlp_scores(resume_text, jd_text)
        cos_pct = int(round(nlp["cosine"] * 100))
        ovl_pct = int(round(nlp["skills_overlap"] * 100))
        hyb_pct = int(round(nlp["hybrid"] * 100))

        st.markdown("#### 🔍 Match Result")
        st.progress(hyb_pct)
        if llm_score is not None:
            st.caption(f"Hybrid match: {hyb_pct}%  •  LLM score: {llm_score}/10")
        else:
            st.caption(f"Hybrid match: {hyb_pct}%")

        st.markdown('<div class="kpi">', unsafe_allow_html=True)
        st.markdown(f'<div class="tile"><div class="muted">Cosine Similarity</div><div class="big">{cos_pct}%</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="tile"><div class="muted">Skills Overlap</div><div class="big">{ovl_pct}%</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="tile"><div class="muted">Final (Hybrid)</div><div class="big">{hyb_pct}%</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        # ---------- /Visual KPIs ----------

        # Cards: Strengths / Gaps
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                '<div class="card"><h4>✅ Strengths / Matches</h4>' + parts["strengths"].replace("\n", "<br>") + "</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                '<div class="card"><h4>⚡ Gaps / Missing</h4>' + parts["missing"].replace("\n", "<br>") + "</div>",
                unsafe_allow_html=True,
            )

        # Suggestions card
        st.markdown(
            '<div class="card"><h4>🛠️ Suggestions</h4>' + parts["suggestions"].replace("\n", "<br>") + "</div>",
            unsafe_allow_html=True,
        )

        # Skill pills (from matched or strengths)
        skills_line = parts["matched"] or parts["strengths"]
        skills = [s.strip(" -•\n\t") for s in re.split(r",|;|\n|- |\u2022|\* ", skills_line) if s.strip()]
        if skills:
            st.markdown(
                '<div class="card"><h4>🏷️ Skill Highlights</h4><div class="pills">'
                + "".join([f'<span class="pill">{s}</span>' for s in skills[:20]])
                + "</div></div>",
                unsafe_allow_html=True,
            )

        # Raw text (collapsible)
        with st.expander("🔎 Show raw analysis text"):
            st.write(result)

        # Download
        st.download_button("📄 Download Match Analysis", result, "resume_vs_jd.txt")
    else:
        st.info("Upload a resume and either a JD file or paste the JD to see the match analysis.")

# ===================== Interview Coach =====================
# ===================== Interview Coach =====================
# ===================== Interview Coach =====================
with tabs[2]:
    st.subheader("Interview Coach")

    # --- Session state init ---
    if "coach" not in st.session_state:
        st.session_state.coach = {"questions": [], "idx": 0, "records": []}

    # --- Controls ---
    c1, c2, c3 = st.columns(3)
    with c1:
        role = st.text_input("Job Title / Role", value=" ")
    with c2:
        level = st.selectbox("Seniority", ["Entry", "Mid", "Senior"], key="coach_level")
    with c3:
        mode = st.selectbox("Mode", ["Behavioral", "Technical", "Aptitude"], key="coach_mode")
    

    jd_hint = st.text_area("Paste Job Description (optional, used to tailor questions)", height=120, key="coach_jd")
    num_q = st.slider("Number of questions", 1, 8, 5, key="coach_n")

    gen = st.button("🎯 Generate Questions", type="primary", key="coach_generate")

    if gen:
        qs = generate_interview_questions(role, jd_hint, mode, level, n=num_q) or []
        # extra hard fallback just in case
        if not qs:
            qs = [
                "Tell me about yourself.",
                "Describe a conflict on a team and how you handled it.",
                "Walk me through a project you’re proud of.",
                "How do you handle ambiguous requirements?",
                "What’s a time you learned something quickly?"
            ][:num_q]

        st.session_state.coach["questions"] = list(qs)
        st.session_state.coach["idx"] = 0
        st.session_state.coach["records"] = []
        st.rerun()  # ensure the UI jumps straight to Question 1

    # --- Current question view ---
    qs = st.session_state.coach.get("questions", [])
    idx = st.session_state.coach.get("idx", 0)

    if qs:
        st.markdown(f"#### Question {idx+1} of {len(qs)}")
        st.markdown(f"**{qs[idx]}**")

        answer = st.text_area("Your answer (aim for 60–200 words)",
                              key=f"coach_answer_{idx}", height=180)

        # define flags to avoid NameError on reruns
        get_fb = next_q = restart = False

        cA, cB, cC = st.columns([1, 1, 1])
        with cA:
            get_fb = st.button("✅ Get Feedback", key=f"btn_getfb_{idx}")
        with cB:
            next_q = st.button("➡️ Next Question", key=f"btn_next_{idx}")
        with cC:
            restart = st.button("🔁 Restart Session", key="btn_restart_session")

        # restart
        if restart:
            st.session_state.coach = {"questions": [], "idx": 0, "records": []}
            st.rerun()

        # feedback
        if get_fb and answer.strip():
            with st.spinner("Evaluating your answer..."):
                fb = critique_interview_answer(qs[idx], answer, mode=mode, jd_text=jd_hint)

            st.session_state.coach["records"].append(
                {"question": qs[idx], "answer": answer, "feedback": fb}
            )

            sc = fb.get("scores", {})
            def tile(label, val):
                st.markdown(
                    f'<div class="tile"><div class="muted">{label}</div><div class="big">{int(val)}/5</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("#### 🔎 Feedback")
            st.markdown('<div class="kpi">', unsafe_allow_html=True)
            tile("Clarity", sc.get("clarity", 3))
            tile("Structure", sc.get("structure", 3))
            tile("Depth", sc.get("technical_depth", 3))
            tile("Impact", sc.get("impact", 3))
            tile("Conciseness", sc.get("conciseness", 3))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card"><h4>🧠 Summary</h4>' + (fb.get("summary","")).replace("\n","<br>") + "</div>", unsafe_allow_html=True)
            sugg = fb.get("suggestions", [])
            if sugg:
                st.markdown('<div class="card"><h4>🛠️ Suggestions</h4><ul>' + "".join([f"<li>{s}</li>" for s in sugg]) + "</ul></div>", unsafe_allow_html=True)
            st.markdown('<div class="card"><h4>✨ Improved Answer (Example)</h4>' + fb.get("improved_answer","").replace("\n","<br>") + "</div>", unsafe_allow_html=True)

        # next
        if next_q:
            if idx < len(qs) - 1:
                st.session_state.coach["idx"] += 1
                st.rerun()
            else:
                st.info("Session complete. Scroll down to export your practice log.")

    else:
        st.info("Click **Generate Questions** to start a practice session.")
