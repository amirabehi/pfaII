# =============================================================================
# score_agent.py  —  Candidate Scoring Agent  (Groq + Instructor + RAG + Reflection)
# =============================================================================
#
# ARCHITECTURE — Agentic Reflection Pattern with 2 specialized agents:
#
#   ┌─────────────┐    RAG query     ┌──────────────┐
#   │  ChromaDB   │ ──────────────── │              │
#   │  (CV text   │                  │   ANALYST    │ → draft score + pros + cons
#   │   vectors)  │ ◄── evidence ─── │   Agent      │
#   └─────────────┘                  └──────┬───────┘
#                                           │ draft ScoreAnalysis
#                                           ▼
#                                    ┌──────────────┐
#                                    │    CRITIC    │ → challenges the draft
#                                    │    Agent     │ → outputs final verified score
#                                    └──────┬───────┘
#                                           │ FinalScore (saved to DB)
#                                           ▼
#                                       SQLite DB
#
# WHY REFLECTION WORKS BETTER:
#   The Analyst is optimistic — it reads the CV and tries to match it.
#   The Critic is skeptical — it re-reads the evidence and challenges over-scoring.
#   Together they produce scores that are accurate and well-justified.
#
# GROQ + INSTRUCTOR:
#   Both agents use Groq (fast) + instructor (Pydantic output — no parse errors).
#
# RAG:
#   Before the Analyst scores, it queries ChromaDB with the job description
#   as the search query. ChromaDB returns the most relevant passages from the
#   candidate's CV. The Analyst only reads those relevant passages — not the
#   entire CV — which makes scoring faster and more targeted.
# =============================================================================

import re
import sqlite3
import os
from typing import List

# ── load .env ─────────────────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from pydantic import BaseModel, Field, field_validator

DB_PATH    = "data/cv_system.db"
GROQ_MODEL = "llama-3.3-70b-versatile"


# =============================================================================
#  PYDANTIC MODELS  —  one per agent output
# =============================================================================

class ScoreAnalysis(BaseModel):
    """
    Output of the ANALYST agent.
    The analyst reads the evidence and drafts an initial assessment.
    """
    draft_score: int = Field(
        description="Initial score 0-100 based on how well the candidate matches the job",
        ge=0, le=100
    )
    matched_skills: List[str] = Field(
        description="Skills or experiences from the CV that directly match job requirements"
    )
    missing_skills: List[str] = Field(
        description="Key requirements from the job that are absent or unclear in the CV"
    )
    supporting_evidence: str = Field(
        description="2-3 specific quotes or facts from the CV that justify the draft score"
    )
    analyst_reasoning: str = Field(
        description="One paragraph explaining the draft score"
    )

    @field_validator("draft_score", mode="before")
    @classmethod
    def clamp(cls, v):
        try: return max(0, min(100, int(v)))
        except Exception: return 50


class FinalScore(BaseModel):
    """
    Output of the CRITIC agent.
    The critic reviews the analyst's draft, challenges it, and outputs the final score.
    """
    score: int = Field(
        description="Final verified score 0-100 after critical review",
        ge=0, le=100
    )
    pros: List[str] = Field(
        description="2-4 specific strengths of this candidate for this role. Each item is a complete sentence."
    )
    cons: List[str] = Field(
        description="2-4 specific gaps or weaknesses of this candidate for this role. Each item is a complete sentence."
    )
    reason: str = Field(
        description="One clear sentence summarising why the candidate received this score"
    )
    confidence: str = Field(
        default="medium",
        description="Confidence level: 'high', 'medium', or 'low' — based on quality of evidence found"
    )

    @field_validator("score", mode="before")
    @classmethod
    def clamp(cls, v):
        try: return max(0, min(100, int(v)))
        except Exception: return 50

    @field_validator("pros", "cons", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if isinstance(v, list): return v
        if isinstance(v, str):  return [v]
        return []


# =============================================================================
#  GROQ + INSTRUCTOR CLIENT
# =============================================================================

def _get_client():
    try:
        import groq
        import instructor
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set. Get a free key at https://console.groq.com")
        return instructor.from_groq(groq.Groq(api_key=api_key), mode=instructor.Mode.JSON)
    except ImportError as e:
        raise ImportError(f"Missing library: {e}\n  pip install groq instructor") from e


# =============================================================================
#  RAG RETRIEVAL  (imported function from cv_agent)
# =============================================================================

def _retrieve_evidence(candidate_id: int, job_description: str) -> str:
    """
    Query ChromaDB for the most relevant CV passages for the given job.
    Falls back to empty string if ChromaDB is unavailable.
    """
    try:
        # Import from cv_agent — both files live in the same folder
        from cv_agent import retrieve_cv_evidence
        return retrieve_cv_evidence(candidate_id, job_description, n_results=5)
    except Exception:
        return ""


# =============================================================================
#  SCORE AGENT CLASS
# =============================================================================

class ScoreAgent:
    """
    Two-agent Reflection scoring pipeline:
      1. Analyst  — retrieves RAG evidence, drafts initial score
      2. Critic   — challenges the draft, outputs final verified score
    Both use Groq + instructor for fast, structured, reliable output.
    """

    def __init__(self):
        self.name   = "ScoreAgent"
        self.client = _get_client()
        print(f"🤖 {self.name} ready  |  Groq model: {GROQ_MODEL}")
        print("   🔄 Reflection pattern: Analyst → Critic")

    # ─────────────────────────────────────────────────────────────────────────
    #  AGENT 1: ANALYST
    # ─────────────────────────────────────────────────────────────────────────
    def _run_analyst(self,
                     candidate_profile: str,
                     job_description:   str,
                     rag_evidence:      str) -> ScoreAnalysis:
        """
        Analyst agent: reads the candidate profile + RAG evidence,
        drafts a score and justification.
        """
        evidence_section = (
            f"\n\nEVIDENCE FROM CV (most relevant passages to this job):\n{rag_evidence}"
            if rag_evidence else
            "\n\n(No RAG evidence available — scoring from profile summary only)"
        )

        system_prompt = """You are an expert HR Analyst.
Your job is to carefully evaluate a candidate's CV against a job description.
Read every piece of evidence, identify specific matches and gaps, and draft an honest initial score.
Be precise — cite actual evidence, not assumptions.
Scoring guide:
  85-100: Excellent — has all required skills, right experience level, strong relevant background
  65-84:  Good      — has most requirements, minor gaps easily bridged
  45-64:  Partial   — relevant background but notable skill or experience gaps
  25-44:  Weak      — some transferable skills but significant gaps
  0-24:   Poor      — does not match core requirements"""

        user_prompt = f"""JOB OFFER:
{job_description}

CANDIDATE PROFILE:
{candidate_profile}
{evidence_section}

Draft your initial analysis and score."""

        return self.client.chat.completions.create(
            model          = GROQ_MODEL,
            messages       = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_model = ScoreAnalysis,
            max_retries    = 3,
            temperature    = 0.1,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  AGENT 2: CRITIC
    # ─────────────────────────────────────────────────────────────────────────
    def _run_critic(self,
                    analyst_output:    ScoreAnalysis,
                    candidate_profile: str,
                    job_description:   str,
                    rag_evidence:      str) -> FinalScore:
        """
        Critic agent: receives the Analyst's draft and challenges it.
        Asks: "Is the score accurate? Are the pros/cons specific enough?
               Is the evidence actually there in the CV?"
        Outputs the final, verified score.
        """
        evidence_section = (
            f"\nCV EVIDENCE:\n{rag_evidence}"
            if rag_evidence else ""
        )

        system_prompt = """You are a rigorous HR Critic with high standards.
You receive a draft analysis from an Analyst and your job is to:
1. Challenge any over-scoring — if the evidence is weak, lower the score
2. Challenge any under-scoring — if the candidate is clearly strong, raise the score
3. Make pros and cons SPECIFIC and useful to an HR manager (not generic)
4. Ensure the final score is accurate and defensible
Output only your final verified assessment."""

        user_prompt = f"""ANALYST'S DRAFT ASSESSMENT:
- Draft Score: {analyst_output.draft_score}/100
- Matched Skills: {', '.join(analyst_output.matched_skills)}
- Missing Skills: {', '.join(analyst_output.missing_skills)}
- Evidence Used: {analyst_output.supporting_evidence}
- Analyst Reasoning: {analyst_output.analyst_reasoning}

JOB OFFER:
{job_description}

CANDIDATE PROFILE:
{candidate_profile}
{evidence_section}

Review the draft. Challenge it if needed. Output the final accurate score."""

        return self.client.chat.completions.create(
            model          = GROQ_MODEL,
            messages       = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_model = FinalScore,
            max_retries    = 3,
            temperature    = 0.15,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Score ONE candidate  (full Analyst → Critic pipeline)
    # ─────────────────────────────────────────────────────────────────────────
    def _score_one(self, candidate_row: tuple, job_row: tuple) -> dict:
        """
        candidate_row: (id, name, skills, job_id, years_experience,
                        education, work_experience, languages,
                        certifications, summary, current_job)
        job_row:       (id, title, description, requirements)
        """
        (cid, name, skills, job_id, years, edu,
         work, langs, certs, summary, curr_job) = candidate_row

        jid, jtitle, jdesc, jreq = job_row

        # ── build text representations ──────────────────────────────────────
        job_description = f"""Title: {jtitle}
Description: {(jdesc or '')[:600]}
Requirements: {(jreq or '')[:400]}"""

        candidate_profile = f"""Name: {name or 'Unknown'}
Current Title: {curr_job or 'Not specified'}
Years of Experience: {years or 0}
Skills: {(skills or 'Not specified')[:400]}
Education: {(edu or 'Not specified')[:300]}
Work Experience: {(work or 'Not specified')[:500]}
Languages: {(langs or 'Not specified')[:150]}
Certifications: {(certs or 'None')[:200]}
Professional Summary: {(summary or 'Not provided')[:300]}"""

        # ── RAG: retrieve relevant CV evidence ──────────────────────────────
        print(f"      🔍 RAG retrieval for candidate {name}…")
        rag_evidence = _retrieve_evidence(cid, job_description)
        if rag_evidence:
            print(f"         Found {len(rag_evidence.split(chr(10)))} relevant passages")
        else:
            print("         ChromaDB not available — using profile summary")

        # ── Agent 1: Analyst ─────────────────────────────────────────────────
        print(f"      🧠 Analyst drafting score…")
        try:
            analyst_output = self._run_analyst(candidate_profile, job_description, rag_evidence)
            print(f"         Draft score: {analyst_output.draft_score}")
        except Exception as e:
            print(f"      ⚠  Analyst error: {e}")
            # fallback — skip reflection, return a basic score
            return {"score": 50, "pros": [], "cons": [], "reason": f"Analyst error: {e}", "confidence": "low"}

        # ── Agent 2: Critic ──────────────────────────────────────────────────
        print(f"      🔍 Critic reviewing draft…")
        try:
            final = self._run_critic(analyst_output, candidate_profile, job_description, rag_evidence)
            print(f"         Final score: {final.score} (confidence: {final.confidence})")
        except Exception as e:
            print(f"      ⚠  Critic error: {e} — using analyst draft")
            # fallback to analyst draft if critic fails
            final = FinalScore(
                score      = analyst_output.draft_score,
                pros       = analyst_output.matched_skills[:3],
                cons       = analyst_output.missing_skills[:3],
                reason     = analyst_output.analyst_reasoning[:200],
                confidence = "low",
            )

        return {
            "score":      final.score,
            "pros":       final.pros,
            "cons":       final.cons,
            "reason":     final.reason,
            "confidence": final.confidence,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Save scored result to DB
    # ─────────────────────────────────────────────────────────────────────────
    def _save_score(self, candidate_id: int, result: dict):
        # pros and cons are lists — join with " | " for DB storage
        # dashboard.py will split them back on " | " for display
        pros_str = " | ".join(result["pros"])[:500]   if result["pros"] else ""
        cons_str = " | ".join(result["cons"])[:500]   if result["cons"] else ""
        reason   = (result["reason"] or "")[:300]
        score    = result["score"]

        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute("""
            UPDATE candidates
            SET score=?, pros=?, cons=?, reason=?, status='scored'
            WHERE id=?
        """, (score, pros_str, cons_str, reason, candidate_id))
        conn.commit()
        conn.close()

    # ─────────────────────────────────────────────────────────────────────────
    #  Score ALL pending candidates  (main public method)
    # ─────────────────────────────────────────────────────────────────────────
    def score_pending_candidates(self, job_id: int = None) -> int:
        """
        Finds all candidates with status='pending' or score=0,
        runs the full Analyst → Critic pipeline on each,
        saves final scores to the DB.
        Returns the count of candidates scored.
        """
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()

        if job_id:
            c.execute("""
                SELECT id, name, skills, job_id,
                       years_experience, education, work_experience,
                       languages, certifications, summary, current_job
                FROM candidates
                WHERE job_id=? AND (status='pending' OR score=0)
            """, (job_id,))
        else:
            c.execute("""
                SELECT id, name, skills, job_id,
                       years_experience, education, work_experience,
                       languages, certifications, summary, current_job
                FROM candidates
                WHERE status='pending' OR score=0
            """)

        pending = c.fetchall()
        conn.close()

        if not pending:
            print("📭 No pending candidates to score.")
            return 0

        print(f"\n🎯 {self.name} — {len(pending)} candidate(s) to score (Analyst → Critic)")
        print("=" * 60)

        scored = 0
        for row in pending:
            cid  = row[0]
            name = row[1]
            jid  = row[3]

            # fetch job details
            conn = sqlite3.connect(DB_PATH)
            c    = conn.cursor()
            c.execute("SELECT id, title, description, requirements FROM jobs WHERE id=?", (jid,))
            job_row = c.fetchone()
            conn.close()

            if not job_row:
                print(f"   ⚠  Job not found for {name} (job_id={jid}) — skipping")
                continue

            print(f"\n   → {name}  |  Job: {job_row[1]}")

            result = self._score_one(row, job_row)
            self._save_score(cid, result)

            tag = "✅" if result["score"] >= 70 else "📋" if result["score"] >= 50 else "❌"
            print(f"      {tag} Final Score: {result['score']}%")
            if result["pros"]:
                print(f"      + {result['pros'][0][:80]}")
            if result["cons"]:
                print(f"      - {result['cons'][0][:80]}")

            scored += 1

        print(f"\n🏁 Scored {scored} candidate(s)")
        return scored

    # ─────────────────────────────────────────────────────────────────────────
    #  Print top-3 to terminal  (used by orchestrator.py)
    # ─────────────────────────────────────────────────────────────────────────
    def print_top3(self, job_id: int):
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute("""
            SELECT name, email, score, pros, cons, reason
            FROM candidates
            WHERE job_id=? AND score>0
            ORDER BY score DESC LIMIT 3
        """, (job_id,))
        rows = c.fetchall()
        conn.close()

        if not rows:
            return

        medals = ["🥇","🥈","🥉"]
        print("\n" + "═" * 60)
        print(f"  TOP 3 — Job #{job_id}")
        print("═" * 60)
        for i, (name, email, score, pros, cons, reason) in enumerate(rows):
            m = medals[i] if i < 3 else "  "
            print(f"\n  {m} {name}  —  {score}%")
            print(f"     📧 {email}")
            if pros:
                print(f"     + {pros.split(' | ')[0][:100]}")
            if cons:
                print(f"     - {cons.split(' | ')[0][:100]}")
            print(f"     → {(reason or '')[:100]}")
        print("═" * 60)


# =============================================================================
#  STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SCORE AGENT — STANDALONE TEST")
    print("=" * 60)

    if not os.environ.get("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not set.  Get a free key at: https://console.groq.com")
        exit(1)

    agent = ScoreAgent()
    n     = agent.score_pending_candidates()
    print(f"\n{'✅' if n else '📭'} Scored {n} candidate(s)")

    # show top-3 for all jobs
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT id, title FROM jobs")
    jobs = c.fetchall()
    conn.close()
    for jid, jtitle in jobs:
        print(f"\nJob: {jtitle}")
        agent.print_top3(jid)
