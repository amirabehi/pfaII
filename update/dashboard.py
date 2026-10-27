# =============================================================================
# dashboard.py  —  HR Recruitment Dashboard  (Fully Integrated Pipeline)
# =============================================================================
#
# WHAT'S NEW vs previous version:
#   • NO more separate orchestrator terminal — processing runs INSIDE this app
#   • "🚀 Process & Score" button triggers cv_agent + score_agent directly
#   • st.spinner + live step-by-step status during processing
#   • Pros/Cons display fixed — no more raw HTML div tags showing as text
#   • Pros and cons render as clean bullet lists inside styled panels
#   • Top-3 cards show full detail: score, pros, cons, contact, AI verdict
#
# HOW TO RUN:
#   1. Add your Groq API key to a .env file:   GROQ_API_KEY=your_key_here
#   2. streamlit run dashboard.py
#   That's it. No other terminals needed.
#
# INSTALL:
#   pip install streamlit groq instructor pydantic chromadb sentence-transformers
#   pip install pypdf python-docx
# =============================================================================

import os
import time
import sqlite3
import streamlit as st
import pandas as pd
from datetime import datetime

# ── load .env ──────────────────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ── MUST be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title = "RecruitAI",
    page_icon  = "🎯",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

DB_PATH = "data/cv_system.db"

# =============================================================================
#  CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Instrument+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif; }

/* sidebar */
[data-testid="stSidebar"] { background: #0a0a0f !important; border-right: 1px solid #1a1a2e; }
[data-testid="stSidebar"] * { color: #b0b0c8 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #fff !important; }

/* layout */
.main .block-container { padding-top: 2rem; max-width: 1240px; }

/* masthead */
.masthead-title { font-family:'Syne',sans-serif; font-weight:800; font-size:3rem;
                  color:#0a0a0f; letter-spacing:-0.03em; line-height:1; }
.masthead-sub   { font-size:.8rem; font-weight:500; letter-spacing:.12em;
                  text-transform:uppercase; color:#888; margin-bottom:2rem; }

/* section title */
.section-title  { font-family:'Syne',sans-serif; font-weight:700; font-size:1.35rem;
                  color:#0a0a0f; border-left:4px solid #2563eb;
                  padding-left:.7rem; margin:1.6rem 0 1rem 0; }

/* stat metric cards */
div[data-testid="stMetric"] { background:#fff; border:1px solid #e8e8f0;
    border-radius:14px; padding:1rem 1.3rem; box-shadow:0 1px 3px rgba(0,0,0,.06); }
div[data-testid="stMetricValue"] { font-family:'Syne',sans-serif;
    font-size:2.2rem !important; font-weight:700; color:#0a0a0f; }

/* process button — make it prominent */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: #fff !important;
    font-size: 1rem !important;
    padding: .75rem 2rem !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: .02em;
    box-shadow: 0 4px 14px rgba(37,99,235,.35);
    transition: transform .15s, box-shadow .15s;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(37,99,235,.45);
}

/* top-3 candidate card */
.top-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.5rem 1.6rem;
    margin-bottom: .8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
    position: relative;
    overflow: hidden;
}
.top-card::before { content:''; position:absolute; top:0; left:0; right:0; height:4px; }
.top-card-1::before { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.top-card-2::before { background: linear-gradient(90deg,#94a3b8,#cbd5e1); }
.top-card-3::before { background: linear-gradient(90deg,#c2410c,#fb923c); }

.card-rank   { font-size:1rem; font-weight:700; color:#94a3b8; margin-bottom:.2rem; }
.card-name   { font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700;
               color:#0a0a0f; margin-bottom:.1rem; }
.card-role   { font-size:.85rem; color:#64748b; margin-bottom:.8rem; }

.score-badge { display:inline-block; font-family:'Syne',sans-serif; font-size:1.75rem;
               font-weight:800; padding:5px 16px; border-radius:12px; margin-bottom:.9rem; }
.sb-hi  { background:#dcfce7; color:#14532d; }
.sb-mid { background:#fef9c3; color:#713f12; }
.sb-lo  { background:#fee2e2; color:#7f1d1d; }
.sb-pnd { background:#f1f5f9; color:#475569; }

/* pros / cons panels — clean, no raw HTML bleed */
.pros-panel {
    background: #f0fdf4;
    border-left: 3px solid #22c55e;
    border-radius: 0 10px 10px 0;
    padding: .7rem 1rem;
    margin: .45rem 0;
}
.pros-panel .panel-header { font-size:.75rem; font-weight:700; letter-spacing:.05em;
    text-transform:uppercase; color:#15803d; margin-bottom:.4rem; }
.pros-panel .panel-item { font-size:.84rem; color:#166534;
    padding:.15rem 0; display:flex; align-items:flex-start; gap:.35rem; }
.pros-panel .panel-item::before { content:"✓"; font-weight:700;
    color:#22c55e; flex-shrink:0; margin-top:.05rem; }

.cons-panel {
    background: #fff7ed;
    border-left: 3px solid #f97316;
    border-radius: 0 10px 10px 0;
    padding: .7rem 1rem;
    margin: .45rem 0;
}
.cons-panel .panel-header { font-size:.75rem; font-weight:700; letter-spacing:.05em;
    text-transform:uppercase; color:#c2410c; margin-bottom:.4rem; }
.cons-panel .panel-item { font-size:.84rem; color:#7c2d12;
    padding:.15rem 0; display:flex; align-items:flex-start; gap:.35rem; }
.cons-panel .panel-item::before { content:"✗"; font-weight:700;
    color:#f97316; flex-shrink:0; margin-top:.05rem; }

.contact-block { font-size:.83rem; color:#475569; margin-top:.7rem; line-height:1.7; }
.verdict-block { font-size:.83rem; font-style:italic; color:#64748b;
                 margin-top:.5rem; padding-top:.5rem;
                 border-top:1px solid #f1f5f9; }

/* processing banner */
.proc-banner { background:linear-gradient(135deg,#fefce8,#fef9c3);
               border:2px solid #eab308; border-radius:16px;
               padding:1.3rem 1.7rem; margin:1rem 0; }
.proc-title  { font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:700;
               color:#713f12; margin-bottom:.3rem; }
.proc-hint   { font-size:.82rem; color:#92400e; margin-top:.4rem; }

/* done banner */
.done-banner { background:linear-gradient(135deg,#f0fdf4,#dcfce7);
               border:2px solid #22c55e; border-radius:16px;
               padding:1.8rem 2rem; text-align:center; margin:1rem 0; }
.done-title  { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800;
               color:#14532d; }
.done-sub    { font-size:.95rem; color:#166534; }

/* score pill (ranked list) */
.sp { display:inline-block; font-family:'Syne',sans-serif; font-size:.88rem;
      font-weight:700; padding:2px 11px; border-radius:20px; }
.sp-hi  { background:#dcfce7; color:#14532d; }
.sp-mid { background:#fef9c3; color:#713f12; }
.sp-lo  { background:#fee2e2; color:#7f1d1d; }
.sp-pnd { background:#f1f5f9; color:#475569; }

/* match label */
.ml { font-size:.7rem; font-weight:700; letter-spacing:.05em;
      text-transform:uppercase; padding:2px 9px; border-radius:20px; }
.ml-hi  { background:#dcfce7; color:#14532d; }
.ml-mid { background:#fef9c3; color:#713f12; }
.ml-lo  { background:#fee2e2; color:#7f1d1d; }
.ml-pnd { background:#f1f5f9; color:#475569; }

/* danger zone */
.danger-box   { background:#fff5f5; border:2px solid #ef4444;
                border-radius:16px; padding:1.5rem 2rem; margin-top:1rem; }
.danger-title { font-family:'Syne',sans-serif; font-size:1.25rem; font-weight:700;
                color:#b91c1c; margin-bottom:.5rem; }

/* misc */
.stButton > button { border-radius:10px; font-family:'Instrument Sans',sans-serif;
                     font-weight:600; }
hr { border:none; border-top:1px solid #e8e8f0; margin:1.5rem 0; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  DATABASE HELPERS
# =============================================================================

def _ensure_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL, description TEXT DEFAULT '',
            requirements TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
    c.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER, name TEXT DEFAULT 'Unknown',
            email TEXT DEFAULT '', phone TEXT DEFAULT '',
            location TEXT DEFAULT '', years_experience INTEGER DEFAULT 0,
            current_job TEXT DEFAULT '', skills TEXT DEFAULT '',
            education TEXT DEFAULT '', work_experience TEXT DEFAULT '',
            languages TEXT DEFAULT '', certifications TEXT DEFAULT '',
            summary TEXT DEFAULT '', source TEXT DEFAULT 'manual',
            raw_text TEXT DEFAULT '', filename TEXT DEFAULT '',
            score INTEGER DEFAULT 0, score_detail TEXT DEFAULT '',
            pros TEXT DEFAULT '', cons TEXT DEFAULT '',
            reason TEXT DEFAULT '', status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES jobs(id)
        )""")
    conn.commit()
    # migrations
    c.execute("PRAGMA table_info(jobs)");       jc = {r[1] for r in c.fetchall()}
    c.execute("PRAGMA table_info(candidates)"); cc = {r[1] for r in c.fetchall()}
    for need, sql in [
        ("requirements"   not in jc,  "ALTER TABLE jobs ADD COLUMN requirements TEXT DEFAULT ''"),
        ("score_detail"   not in cc,  "ALTER TABLE candidates ADD COLUMN score_detail TEXT DEFAULT ''"),
        ("pros"           not in cc,  "ALTER TABLE candidates ADD COLUMN pros TEXT DEFAULT ''"),
        ("cons"           not in cc,  "ALTER TABLE candidates ADD COLUMN cons TEXT DEFAULT ''"),
        ("raw_text"       not in cc,  "ALTER TABLE candidates ADD COLUMN raw_text TEXT DEFAULT ''"),
        ("filename"       not in cc,  "ALTER TABLE candidates ADD COLUMN filename TEXT DEFAULT ''"),
        ("work_experience" not in cc, "ALTER TABLE candidates ADD COLUMN work_experience TEXT DEFAULT ''"),
        ("languages"      not in cc,  "ALTER TABLE candidates ADD COLUMN languages TEXT DEFAULT ''"),
        ("certifications" not in cc,  "ALTER TABLE candidates ADD COLUMN certifications TEXT DEFAULT ''"),
        ("summary"        not in cc,  "ALTER TABLE candidates ADD COLUMN summary TEXT DEFAULT ''"),
    ]:
        if need:
            try: c.execute(sql)
            except Exception: pass
    conn.commit()
    conn.close()


def _q(sql, params=(), one=False, many=False, write=False):
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute(sql, params)
        r = c.fetchone() if one else (c.fetchall() if many else None)
        if write: conn.commit()
        conn.close()
        return r
    except Exception:
        return None

def _cnt(sql, params=()):
    r = _q(sql, params, one=True)
    return r[0] if r else 0

def db_jobs():      return _q("SELECT id,title,description,requirements,created_at FROM jobs ORDER BY created_at DESC", many=True) or []
def db_job(jid):    return _q("SELECT id,title,description,requirements FROM jobs WHERE id=?", (jid,), one=True)
def db_create_job(t,d,r):
    conn = sqlite3.connect(DB_PATH); c = conn.cursor()
    c.execute("INSERT INTO jobs(title,description,requirements) VALUES(?,?,?)",(t,d,r))
    jid = c.lastrowid; conn.commit(); conn.close(); return jid
def db_delete_job(jid):
    _q("DELETE FROM candidates WHERE job_id=?",(jid,),write=True)
    _q("DELETE FROM jobs WHERE id=?",(jid,),write=True)
def db_candidates(jid):
    return _q("""SELECT id,name,email,phone,location,skills,education,
               years_experience,current_job,score,pros,cons,reason,source,status,created_at
               FROM candidates WHERE job_id=? ORDER BY score DESC,created_at DESC""",
               (jid,), many=True) or []
def db_total(jid):   return _cnt("SELECT COUNT(*) FROM candidates WHERE job_id=?",(jid,))
def db_scored(jid):  return _cnt("SELECT COUNT(*) FROM candidates WHERE job_id=? AND score>0",(jid,))
def db_pending(jid): return _cnt("SELECT COUNT(*) FROM candidates WHERE job_id=? AND (status='pending' OR score=0)",(jid,))
def db_njobs():      return _cnt("SELECT COUNT(*) FROM jobs")
def db_ncvs():       return _cnt("SELECT COUNT(*) FROM candidates")


# =============================================================================
#  FILE HELPERS
# =============================================================================

def fs_incoming():
    d = "incoming"
    if not os.path.exists(d): return []
    return [f for f in os.listdir(d) if f!="processed" and os.path.isfile(os.path.join(d,f))]

def fs_processed():
    d = os.path.join("incoming","processed")
    if not os.path.exists(d): return []
    return [f for f in os.listdir(d) if os.path.isfile(os.path.join(d,f))]

def fs_clear_incoming():
    n=0
    for f in list(fs_incoming()):
        try: os.remove(os.path.join("incoming",f)); n+=1
        except Exception: pass
    return n

def fs_clear_processed():
    n=0
    for f in list(fs_processed()):
        try: os.remove(os.path.join("incoming","processed",f)); n+=1
        except Exception: pass
    return n

def fs_save(uf):
    os.makedirs("incoming", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fp = os.path.join("incoming", f"{ts}_{uf.name}")
    with open(fp,"wb") as fh: fh.write(uf.getbuffer())
    return fp

def fs_hard_reset():
    errors = []
    try:
        conn = sqlite3.connect(DB_PATH); c = conn.cursor()
        c.execute("DELETE FROM candidates"); c.execute("DELETE FROM jobs")
        for t in ("candidates","jobs"):
            try: c.execute(f"DELETE FROM sqlite_sequence WHERE name='{t}'")
            except Exception: pass
        conn.commit(); conn.close()
    except Exception as e: errors.append(str(e))
    for folder in ("incoming", os.path.join("incoming","processed")):
        if os.path.exists(folder):
            for f in list(os.listdir(folder)):
                fp = os.path.join(folder,f)
                if os.path.isfile(fp):
                    try: os.remove(fp)
                    except Exception as e: errors.append(str(e))
    # also wipe ChromaDB vectors
    chroma_path = "data/chroma_db"
    if os.path.exists(chroma_path):
        try:
            import shutil
            shutil.rmtree(chroma_path)
        except Exception as e:
            errors.append(str(e))
    for folder in ("incoming", os.path.join("incoming","processed"), "data"):
        os.makedirs(folder, exist_ok=True)
    _ensure_db()
    return (not errors), ("; ".join(errors) if errors else "System fully reset.")


# =============================================================================
#  DISPLAY HELPERS
# =============================================================================

def _cls(s):
    if s >= 75: return "hi"
    if s >= 50: return "mid"
    if s >  0:  return "lo"
    return "pnd"

def _label(s):
    if s >= 75: return "Strong Match"
    if s >= 50: return "Potential"
    if s >  0:  return "Low Match"
    return "Pending"

def _medal(i): return ["🥇","🥈","🥉"][i] if i < 3 else f"#{i+1}"

def _pros_html(pros_str: str) -> str:
    """
    Convert 'item1 | item2 | item3' string into a styled HTML pros panel.
    This is the fix for the 'raw div tags showing as text' bug:
    we build the HTML here in Python, never inside f-strings with user data.
    """
    if not pros_str or not pros_str.strip():
        items = ["No specific strengths identified"]
    else:
        items = [x.strip() for x in pros_str.split("|") if x.strip()]
    items_html = "".join(f'<div class="panel-item">{item}</div>' for item in items)
    return f"""<div class="pros-panel">
  <div class="panel-header">✅ Strengths</div>
  {items_html}
</div>"""

def _cons_html(cons_str: str) -> str:
    """Convert 'item1 | item2 | item3' string into a styled HTML cons panel."""
    if not cons_str or not cons_str.strip():
        items = ["No specific gaps identified"]
    else:
        items = [x.strip() for x in cons_str.split("|") if x.strip()]
    items_html = "".join(f'<div class="panel-item">{item}</div>' for item in items)
    return f"""<div class="cons-panel">
  <div class="panel-header">⚠️ Gaps</div>
  {items_html}
</div>"""


# =============================================================================
#  INTEGRATED PIPELINE  — runs directly inside Streamlit (no separate terminal)
# =============================================================================

def run_pipeline(job_id: int) -> dict:
    """
    Runs the full extraction + scoring pipeline for a job inside Streamlit.
    Called when HR clicks "🚀 Process & Score" button.

    Returns a summary dict with counts for the status message.
    """
    result = {"extracted": 0, "scored": 0, "errors": []}

    # Step 1 — CV extraction
    st.write("**Step 1/2 — Extracting CVs…**")
    try:
        from cv_agent import CVAgent
        agent    = CVAgent()
        ext_result = agent.process_incoming_files(
            folder     = "incoming",
            job_id     = job_id,
            max_workers = 3,   # 3 parallel workers — good balance for Groq rate limits
        )
        result["extracted"] = ext_result.get("done", 0)
        st.write(f"   ✅ Extracted {result['extracted']} CV(s)")
    except Exception as e:
        err = f"Extraction error: {e}"
        result["errors"].append(err)
        st.error(err)
        return result

    # Step 2 — Scoring
    st.write("**Step 2/2 — Scoring candidates (Analyst → Critic)…**")
    try:
        from score_agent import ScoreAgent
        scorer = ScoreAgent()
        result["scored"] = scorer.score_pending_candidates(job_id=job_id)
        st.write(f"   ✅ Scored {result['scored']} candidate(s)")
    except Exception as e:
        err = f"Scoring error: {e}"
        result["errors"].append(err)
        st.error(err)

    return result


# =============================================================================
#  SESSION STATE
# =============================================================================

def _ss(k, v):
    if k not in st.session_state: st.session_state[k] = v

def init_ss():
    _ss("confirm_delete", False)
    _ss("auto_refresh",   False)
    _ss("pipeline_ran",   False)


# =============================================================================
#  MAIN
# =============================================================================

def main():
    init_ss()
    _ensure_db()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎯 RecruitAI")
        st.markdown("---")
        page = st.radio("Navigate",
            ["🏠 Dashboard","➕ New Job Offer","📤 Upload CVs","⚙️ Settings"],
            label_visibility="collapsed")
        st.markdown("---")
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
        jc = db_njobs(); cc = db_ncvs(); wf = len(fs_incoming())
        st.markdown(f"**Jobs:** {jc} &nbsp;|&nbsp; **CVs:** {cc}")
        if wf: st.warning(f"📁 {wf} file(s) in queue")
        st.markdown("---")
        auto = st.toggle("⚡ Auto-refresh (8 s)", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto
        st.markdown("---")
        st.markdown("**Quick Start**")
        st.caption("1. New Job Offer")
        st.caption("2. Upload CVs")
        st.caption("3. Click 🚀 Process & Score")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown('<div class="masthead-title">RecruitAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="masthead-sub">AI-Powered Recruitment Platform — Groq · Instructor · RAG</div>', unsafe_allow_html=True)

    # ── Route ─────────────────────────────────────────────────────────────────
    if   page == "🏠 Dashboard":     pg_dashboard()
    elif page == "➕ New Job Offer": pg_new_job()
    elif page == "📤 Upload CVs":    pg_upload()
    elif page == "⚙️ Settings":      pg_settings()

    if st.session_state.auto_refresh:
        time.sleep(8)
        st.rerun()


# =============================================================================
#  PAGE: DASHBOARD
# =============================================================================

def pg_dashboard():
    jobs = db_jobs()
    if not jobs:
        st.info("👋 No job offers yet — go to **New Job Offer** to create one.")
        return

    job_map   = {f"{j[1]}  (#{j[0]})": j[0] for j in jobs}
    sel_label = st.selectbox("Select Job Offer", list(job_map.keys()))
    job_id    = job_map[sel_label]

    job = db_job(job_id)
    if job and (job[2] or job[3]):
        with st.expander("📄 Job Details"):
            if job[2]: st.markdown(f"**Description:** {job[2]}")
            if job[3]: st.markdown(f"**Requirements:** {job[3]}")

    st.markdown("---")

    # ── Stats ─────────────────────────────────────────────────────────────────
    total   = db_total(job_id)
    scored  = db_scored(job_id)
    pending = db_pending(job_id)
    waiting = len(fs_incoming())

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total CVs",    total)
    c2.metric("Scored",       scored)
    c3.metric("Pending",      pending)
    c4.metric("Files Queued", waiting)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  🚀  PROCESS & SCORE BUTTON  — the key UX upgrade
    #  Triggers the full pipeline inside this Streamlit session.
    #  No separate terminal, no orchestrator.py.
    # ══════════════════════════════════════════════════════════════════════════
    api_key_set = bool(os.environ.get("GROQ_API_KEY", "").strip())

    if waiting > 0 or pending > 0:
        st.markdown('<div class="proc-banner">', unsafe_allow_html=True)
        st.markdown('<div class="proc-title">📋 Files Ready to Process</div>', unsafe_allow_html=True)

        if total > 0:
            pct = scored / total
            st.progress(pct, text=f"Scoring: {scored}/{total} ({int(pct*100)}%)")
        else:
            st.progress(0.0, text="Awaiting extraction…")

        d1,d2,d3 = st.columns(3)
        d1.markdown(f"📁 **{waiting}** file(s) in queue")
        d2.markdown(f"⏳ **{pending}** awaiting score")
        d3.markdown(f"✅ **{scored}** scored")

        if not api_key_set:
            st.markdown('<div class="proc-hint">⚠️ GROQ_API_KEY not set — add it to your .env file to enable processing.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")

        if api_key_set:
            if st.button("🚀 Process New Uploads & Score Candidates",
                         type="primary", use_container_width=True):
                with st.spinner("Running pipeline… this may take 1-3 minutes for large batches"):
                    result = run_pipeline(job_id)
                if not result["errors"]:
                    st.success(
                        f"✅ Pipeline complete — "
                        f"Extracted: {result['extracted']}, Scored: {result['scored']}"
                    )
                    st.session_state.pipeline_ran = True
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Pipeline finished with errors. See details above.")
        st.markdown("---")

    # ── Candidates ────────────────────────────────────────────────────────────
    candidates = db_candidates(job_id)
    if not candidates:
        st.info("No candidates yet. Upload CVs and click 🚀 Process & Score.")
        return

    # ── TOP 3 CARDS ───────────────────────────────────────────────────────────
    scored_cands = [c for c in candidates if (c[9] or 0) > 0]
    if scored_cands:
        st.markdown('<div class="section-title">🏆 Top 3 Candidates</div>', unsafe_allow_html=True)
        top3 = scored_cands[:3]
        cols = st.columns(min(len(top3), 3))
        card_cls = ["top-card top-card-1","top-card top-card-2","top-card top-card-3"]
        ranks    = ["🥇 #1","🥈 #2","🥉 #3"]

        for i, (col, cand) in enumerate(zip(cols, top3)):
            (cid, name, email, phone, location, skills, education,
             yrs, curr_job, score, pros, cons, reason, source, status, created) = cand
            sc   = score or 0
            cls  = _cls(sc)

            with col:
                # ── pros and cons are rendered via Python helper functions ──
                # This is the fix: _pros_html() and _cons_html() build clean HTML
                # from the " | "-separated strings stored in the DB.
                # No raw user data goes directly into the HTML template.
                pros_html = _pros_html(pros or "")
                cons_html = _cons_html(cons or "")

                st.markdown(f"""
<div class="{card_cls[i]}">
  <div class="card-rank">{ranks[i]}</div>
  <div class="card-name">{name or 'Unknown'}</div>
  <div class="card-role">{curr_job or education or '—'}</div>
  <div class="score-badge sb-{cls}">{sc}%</div>
  {pros_html}
  {cons_html}
  <div class="contact-block">
    📧 {email or '—'}<br>
    📞 {phone or '—'} &nbsp;·&nbsp; 📍 {location or '—'}<br>
    🗓 {yrs or 0} years experience
  </div>
  <div class="verdict-block">💬 &ldquo;{reason or 'No verdict available'}&rdquo;</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── FULL RANKED LIST ──────────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">📋 All Candidates ({len(candidates)})</div>', unsafe_allow_html=True)

    for i, cand in enumerate(candidates):
        (cid, name, email, phone, location, skills, education,
         yrs, curr_job, score, pros, cons, reason, source, status, created) = cand

        sc  = score or 0
        cls = _cls(sc)

        ca, cb, cc_, cd = st.columns([3,1,1,.5])
        with ca:
            st.markdown(f"**{_medal(i)} {name or 'Unknown'}**")
            st.caption(f"📧 {email or '—'}  ·  📍 {location or '—'}  ·  {yrs or 0} yrs")
        with cb:
            st.markdown(f'<span class="sp sp-{cls}">{sc}%</span>' if sc else
                        f'<span class="sp sp-pnd">Pending</span>', unsafe_allow_html=True)
        with cc_:
            st.markdown(f'<span class="ml ml-{cls}">{_label(sc)}</span>', unsafe_allow_html=True)

        with st.expander("Full Profile"):
            # Use st.columns inside expander for clean layout — no raw HTML needed here
            pl, pr = st.columns(2)
            with pl:
                st.markdown(f"**🔧 Skills:** {skills or '—'}")
                st.markdown(f"**🎓 Education:** {education or '—'}")
                st.markdown(f"**💼 Current Role:** {curr_job or '—'}")
                st.markdown(f"**📅 Experience:** {yrs or 0} years")
            with pr:
                st.markdown(f"**📞 Phone:** {phone or '—'}")
                st.markdown(f"**📍 Location:** {location or '—'}")
                st.markdown(f"**📂 File:** {source or '—'}")

            # Pros and cons in expander — use native Streamlit for simplicity
            if pros:
                st.markdown("**✅ Strengths:**")
                for item in pros.split("|"):
                    if item.strip():
                        st.markdown(f"- {item.strip()}")
            if cons:
                st.markdown("**⚠️ Gaps:**")
                for item in cons.split("|"):
                    if item.strip():
                        st.markdown(f"- {item.strip()}")
            if reason:
                st.markdown(f"**💬 AI Verdict:** *{reason}*")

        st.divider()

    # ── Completion banner ─────────────────────────────────────────────────────
    if total > 0 and scored == total and waiting == 0 and pending == 0:
        top   = candidates[0] if candidates else None
        tname = top[1] if top else "—"
        tsc   = top[9] if top else 0
        st.markdown(f"""
<div class="done-banner">
  <div class="done-title">🎉 Ranking Complete!</div>
  <div class="done-sub">All {total} CVs processed and scored.</div>
  <div style="margin-top:.8rem;font-size:1.05rem;color:#166534">
    🏆 Best Candidate: <strong>{tname}</strong> — <strong>{tsc}%</strong>
  </div>
</div>""", unsafe_allow_html=True)
        st.balloons()

    # ── CSV export ────────────────────────────────────────────────────────────
    df = pd.DataFrame([{
        "Rank": i+1, "Name": c[1], "Email": c[2], "Phone": c[3],
        "Location": c[4], "Score": c[9] if c[9] else "Pending",
        "Match": _label(c[9] or 0),
        "Pros": (c[10] or "").replace("|",";"),
        "Cons": (c[11] or "").replace("|",";"),
        "Reason": c[12] or "", "Skills": c[5] or "",
    } for i,c in enumerate(candidates)])
    st.download_button("📥 Export Results (CSV)",
                       df.to_csv(index=False), "results.csv", "text/csv")


# =============================================================================
#  PAGE: NEW JOB OFFER
# =============================================================================

def pg_new_job():
    st.markdown('<div class="section-title">➕ Create New Job Offer</div>', unsafe_allow_html=True)
    with st.form("new_job", clear_on_submit=True):
        title = st.text_input("Job Title *", placeholder="e.g. Senior Python Developer")
        desc  = st.text_area("Job Description *", height=180,
                             placeholder="Describe the role, team, responsibilities…")
        req   = st.text_area("Requirements", height=120,
                             placeholder="Required skills, years of experience, education…")
        sub   = st.form_submit_button("✅ Create Job Offer", type="primary")
    if sub:
        if title.strip() and desc.strip():
            jid = db_create_job(title.strip(), desc.strip(), req.strip())
            st.success(f"✅ Job offer **{title}** created (ID #{jid}). "
                       "Now go to Upload CVs.")
            st.balloons()
        else:
            st.error("Title and Description are required.")


# =============================================================================
#  PAGE: UPLOAD CVs
# =============================================================================

def pg_upload():
    st.markdown('<div class="section-title">📤 Upload Candidate CVs</div>', unsafe_allow_html=True)

    jobs = db_jobs()
    if not jobs:
        st.warning("Create a job offer first.")
        return

    job_map = {j[1]: j[0] for j in jobs}
    st.selectbox("These CVs belong to job:", list(job_map.keys()))
    st.caption("After uploading, go to Dashboard and click **🚀 Process & Score**.")

    wf = fs_incoming(); pf = fs_processed()
    cw, cp = st.columns(2)
    cw.metric("Files Waiting",   len(wf))
    cp.metric("Files Processed", len(pf))

    if wf:
        st.markdown("---")
        st.warning(f"**{len(wf)}** file(s) waiting to be processed.")
        with st.expander("View queued files"):
            for f in wf:
                fa, fb = st.columns([5,1])
                fa.write(f"📄 {f}")
                if fb.button("🗑️", key=f"qd_{f}"):
                    try: os.remove(os.path.join("incoming",f)); st.rerun()
                    except Exception as e: st.error(str(e))
        if st.button("🗑️ Clear All Queued Files"):
            st.success(f"Removed {fs_clear_incoming()} file(s)."); st.rerun()

    if pf:
        st.markdown("---")
        st.info(f"📦 **{len(pf)}** file(s) already processed.")
        with st.expander("View processed files"):
            for f in pf[:40]: st.write(f"✅ {f}")
        if st.button("🗑️ Clear Processed Archive"):
            st.success(f"Cleared {fs_clear_processed()} file(s)."); st.rerun()

    st.markdown("---")
    uploads = st.file_uploader(
        "Choose CV files",
        type=["pdf","docx","txt","jpg","jpeg","png","bmp","tiff","webp"],
        accept_multiple_files=True,
        help="PDF, Word, plain text, or image files (OCR applied to images if Tesseract installed)"
    )
    if uploads:
        if st.button("🚀 Add to Queue", type="primary"):
            for uf in uploads: fs_save(uf)
            st.success(f"✅ {len(uploads)} file(s) added. Go to Dashboard → 🚀 Process & Score.")
            st.rerun()


# =============================================================================
#  PAGE: SETTINGS
# =============================================================================

def pg_settings():
    st.markdown('<div class="section-title">⚙️ Settings & Administration</div>', unsafe_allow_html=True)

    # API key status
    api_key = os.environ.get("GROQ_API_KEY","")
    if api_key:
        masked = api_key[:8] + "..." + api_key[-4:]
        st.success(f"✅ GROQ_API_KEY configured ({masked})")
    else:
        st.error("❌ GROQ_API_KEY not set — processing will not work.")
        st.code("# Create a file called .env in your project folder:\nGROQ_API_KEY=your_key_here")
        st.markdown("Get a free key at [console.groq.com](https://console.groq.com)")

    st.markdown("---")

    # System stats
    st.subheader("System Overview")
    jc=db_njobs(); cc=db_ncvs(); wf=fs_incoming(); pf=fs_processed()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Jobs",         jc)
    c2.metric("Candidates",   cc)
    c3.metric("Queued",       len(wf))
    c4.metric("Processed",    len(pf))

    # ChromaDB status
    chroma_path = "data/chroma_db"
    if os.path.exists(chroma_path):
        st.info("🔍 ChromaDB RAG index present at data/chroma_db/")
    else:
        st.warning("ℹ ChromaDB index not yet created — will be built on first processing run.")

    # Manage jobs
    st.markdown("---")
    st.subheader("Manage Job Offers")
    jobs = db_jobs()
    if jobs:
        for j in jobs:
            jid,title,desc,req,created = j
            cnt = db_total(jid)
            with st.expander(f"#{jid} — {title}  ({cnt} candidate(s))"):
                st.markdown(f"**Description:** {desc or '—'}")
                if req: st.markdown(f"**Requirements:** {req}")
                st.caption(f"Created: {created}")
                _, dc = st.columns([4,1])
                if dc.button("🗑️ Delete", key=f"dj_{jid}"):
                    db_delete_job(jid)
                    st.success(f"Deleted job #{jid}."); st.rerun()
    else:
        st.info("No jobs in the database.")

    # File management
    st.markdown("---")
    st.subheader("File Management")
    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown(f"**Queue: {len(wf)} file(s)**")
        if wf:
            with st.expander("View"):
                for f in wf:
                    ra,rb = st.columns([5,1])
                    ra.write(f"📄 {f}")
                    if rb.button("🗑️", key=f"sq_{f}"):
                        try: os.remove(os.path.join("incoming",f)); st.rerun()
                        except Exception as e: st.error(str(e))
            if st.button("Clear Queue", key="cq"):
                st.success(f"Removed {fs_clear_incoming()} file(s)."); st.rerun()
    with fc2:
        st.markdown(f"**Archive: {len(pf)} processed file(s)**")
        if pf and st.button("Clear Archive", key="ca"):
            st.success(f"Removed {fs_clear_processed()} file(s)."); st.rerun()

    # Danger zone
    st.markdown("---")
    st.markdown('<div class="danger-box">', unsafe_allow_html=True)
    st.markdown('<div class="danger-title">⚠️ Danger Zone</div>', unsafe_allow_html=True)
    st.write("Deletes all jobs, candidates, scores, files, and ChromaDB index. Irreversible.")

    if not st.session_state.confirm_delete:
        if st.button("🗑️ Reset Everything", type="primary", key="open_reset"):
            st.session_state.confirm_delete = True; st.rerun()
    else:
        st.error("This will wipe everything.")
        typed = st.text_input("Type DELETE to confirm", placeholder="DELETE", key="del_typed")
        cy, cn = st.columns(2)
        with cy:
            if st.button("✅ Confirm Wipe", type="primary", key="do_wipe"):
                if typed.strip() == "DELETE":
                    with st.spinner("Wiping…"):
                        ok, msg = fs_hard_reset()
                    for k in list(st.session_state.keys()): del st.session_state[k]
                    st.success("✅ Done." if ok else f"⚠️ {msg}")
                    time.sleep(1); st.rerun()
                else:
                    st.error("Type DELETE exactly.")
        with cn:
            if st.button("❌ Cancel", key="cancel_wipe"):
                st.session_state.confirm_delete = False; st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
#  ENTRY
# =============================================================================
if __name__ == "__main__":
    main()
