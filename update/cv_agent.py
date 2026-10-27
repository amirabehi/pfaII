# =============================================================================
# cv_agent.py  —  CV Extraction Agent  (Groq + Instructor + ChromaDB RAG)
# =============================================================================
#
# ARCHITECTURE:
#   1. Extract raw text from the file  (pypdf / python-docx / OCR)
#   2. Send raw text → Groq API (llama-3.3-70b-versatile) via the `instructor`
#      library, which wraps the LLM call and forces the response to match a
#      Pydantic model — so we ALWAYS get perfectly structured JSON, never garbage.
#   3. Store the raw CV text as a vector embedding in ChromaDB so the
#      ScoreAgent can later do RAG (search for relevant evidence from the CV
#      before scoring the candidate against a job description).
#   4. Save every parsed field into SQLite.
#
# KEY UPGRADE over previous version:
#   • Groq API  → ~10x faster than local Ollama, completely free tier
#   • instructor → guaranteed valid Pydantic objects, no more JSON parse errors
#   • ChromaDB  → CV text is indexed so ScoreAgent can retrieve evidence by
#                 semantic similarity instead of sending the full CV text
#
# INSTALL REQUIREMENTS:
#   pip install groq instructor pydantic chromadb sentence-transformers
#   pip install pypdf python-docx pytesseract pillow pdf2image
#
# SETUP:
#   1. Get a free Groq API key at https://console.groq.com
#   2. Set environment variable:  GROQ_API_KEY=your_key_here
#      Or create a file called  .env  in the project folder with:
#      GROQ_API_KEY=your_key_here
# =============================================================================

import os
import re
import json
import sqlite3
import shutil
import threading
from datetime import datetime
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── load .env file if present (simple manual loader, no python-dotenv needed) ─
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ── folder bootstrap ──────────────────────────────────────────────────────────
os.makedirs("data",               exist_ok=True)
os.makedirs("incoming",           exist_ok=True)
os.makedirs("incoming/processed", exist_ok=True)

DB_PATH     = "data/cv_system.db"
CHROMA_PATH = "data/chroma_db"   # ChromaDB stores vectors here on disk

# =============================================================================
#  PYDANTIC MODEL  —  defines the exact structure the LLM MUST return
#  instructor enforces this at the API call level, retrying automatically
#  if the LLM output doesn't match.
# =============================================================================

from pydantic import BaseModel, Field, field_validator

class CandidateProfile(BaseModel):
    """
    Structured candidate profile extracted from a CV.
    Every field has a description — instructor passes these as hints to the LLM.
    """
    full_name:         str          = Field(description="Full name of the candidate")
    email:             str          = Field(description="Email address, lowercase")
    phone:             str          = Field(default="", description="Phone number with country code if present")
    location:          str          = Field(default="", description="City and country e.g. 'Paris, France'")
    years_experience:  int          = Field(default=0,  description="Total years of professional experience as integer")
    current_job_title: str          = Field(default="", description="Most recent job title")
    skills:            List[str]    = Field(default=[], description="Technical and soft skills as a flat list of strings")
    education:         List[str]    = Field(default=[], description="Education entries as plain strings: 'Degree — Institution — Year'")
    work_experience:   List[str]    = Field(default=[], description="Work entries as plain strings: 'Title — Company — Start-End — Key achievement'")
    languages:         List[str]    = Field(default=[], description="Languages spoken e.g. ['English (C2)', 'French (B2)']")
    certifications:    List[str]    = Field(default=[], description="Professional certifications e.g. ['AWS Solutions Architect', 'PMP']")
    summary:           str          = Field(default="", description="3-sentence professional summary written in third person")

    # ── validators: clean common LLM mistakes ────────────────────────────────

    @field_validator("email")
    @classmethod
    def clean_email(cls, v):
        # strip whitespace and lowercase
        return v.strip().lower() if v else "no-email@unknown.com"

    @field_validator("years_experience", mode="before")
    @classmethod
    def parse_years(cls, v):
        # LLM sometimes returns "5 years" — extract the number
        if isinstance(v, int):
            return max(0, v)
        try:
            return max(0, int(re.search(r"\d+", str(v)).group()))
        except Exception:
            return 0

    @field_validator("skills", "education", "work_experience", "languages", "certifications", mode="before")
    @classmethod
    def ensure_list_of_strings(cls, v):
        # LLM sometimes returns dicts inside lists — flatten them
        if not isinstance(v, list):
            return []
        result = []
        for item in v:
            if isinstance(item, dict):
                result.append(", ".join(str(x) for x in item.values() if x))
            elif item:
                result.append(str(item))
        return result


# =============================================================================
#  GROQ + INSTRUCTOR CLIENT SETUP
# =============================================================================

def _get_groq_client():
    """
    Returns an instructor-patched Groq client.
    instructor.from_groq() wraps every chat.completions.create() call so that:
      - the response is validated against a Pydantic model
      - if validation fails, instructor automatically retries (up to 3 times)
      - you always get back a proper Python object, never raw JSON strings
    """
    try:
        import groq
        import instructor
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set.\n"
                "  1. Get a free key at https://console.groq.com\n"
                "  2. Set it:  export GROQ_API_KEY=your_key\n"
                "     Or add  GROQ_API_KEY=your_key  to a .env file"
            )
        raw_client = groq.Groq(api_key=api_key)
        # instructor.Mode.JSON tells instructor to use JSON mode (most reliable)
        return instructor.from_groq(raw_client, mode=instructor.Mode.JSON)
    except ImportError as e:
        raise ImportError(f"Missing library: {e}\n  pip install groq instructor") from e


# Groq model to use — llama-3.3-70b-versatile is free, fast, and highly capable
GROQ_MODEL = "llama-3.3-70b-versatile"


# =============================================================================
#  CHROMADB  —  Vector store for RAG
#  We store each CV as a document in ChromaDB, indexed by candidate_id.
#  ScoreAgent can then query "find CV passages relevant to this job description"
#  and get back the most relevant sentences — this is Retrieval-Augmented
#  Generation (RAG): we retrieve evidence BEFORE asking the LLM to score.
# =============================================================================

def _get_chroma_collection():
    """
    Returns the ChromaDB collection, creating it if needed.
    Uses a local persistent store at data/chroma_db/.
    The embedding function uses sentence-transformers (runs locally, free).
    """
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.PersistentClient(path=CHROMA_PATH)

        # all-MiniLM-L6-v2 is small, fast, and excellent for semantic search
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # get_or_create_collection — safe to call multiple times
        collection = client.get_or_create_collection(
            name="cv_texts",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        return collection

    except ImportError:
        # ChromaDB not installed — RAG will be skipped, scoring still works
        return None
    except Exception as e:
        print(f"   ⚠  ChromaDB init error (RAG disabled): {e}")
        return None


def store_cv_in_chroma(candidate_id: int, raw_text: str, metadata: dict):
    """
    Stores the raw CV text in ChromaDB for later RAG retrieval.
    If ChromaDB is not installed, this is a silent no-op.

    We chunk the text into ~500-char paragraphs so the embeddings are
    more precise (short chunks match job descriptions better than
    embedding an entire CV as one blob).
    """
    collection = _get_chroma_collection()
    if collection is None or not raw_text.strip():
        return

    try:
        # Split into paragraphs, skip empty ones
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > 50]

        # If no paragraphs found, treat full text as single chunk
        if not paragraphs:
            paragraphs = [raw_text[:3000]]

        ids       = [f"cand_{candidate_id}_chunk_{i}" for i in range(len(paragraphs))]
        metadatas = [{**metadata, "chunk_index": i} for i in range(len(paragraphs))]

        # upsert = insert or update (safe to call multiple times for same candidate)
        collection.upsert(
            ids       = ids,
            documents = paragraphs,
            metadatas = metadatas,
        )
        print(f"   🔍 Indexed {len(paragraphs)} chunks in ChromaDB (id={candidate_id})")

    except Exception as e:
        print(f"   ⚠  ChromaDB store error: {e}")


def retrieve_cv_evidence(candidate_id: int, query: str, n_results: int = 4) -> str:
    """
    Query ChromaDB for the most relevant CV passages for a given job description.
    Called by ScoreAgent before scoring — this is the RAG retrieval step.

    Returns: a string of the most relevant CV passages, or "" if unavailable.
    """
    collection = _get_chroma_collection()
    if collection is None:
        return ""

    try:
        results = collection.query(
            query_texts = [query],
            n_results   = n_results,
            where       = {"candidate_id": candidate_id},  # only this candidate's chunks
        )
        passages = results.get("documents", [[]])[0]
        return "\n---\n".join(passages) if passages else ""

    except Exception as e:
        print(f"   ⚠  ChromaDB retrieve error: {e}")
        return ""


# =============================================================================
#  OPTIONAL FILE EXTRACTION LIBRARIES
# =============================================================================

try:
    from pypdf import PdfReader
    PDF_OK = True
except ImportError:
    PDF_OK = False
    print("⚠  pypdf not installed.  pip install pypdf")

try:
    from docx import Document as DocxDoc
    DOCX_OK = True
except ImportError:
    DOCX_OK = False
    print("⚠  python-docx not installed.  pip install python-docx")

# OCR: two-step check — Python library + system binary
_PIL_OK = False
OCR_OK  = False
try:
    import pytesseract
    from PIL import Image as PilImage
    _PIL_OK = True
    pytesseract.get_tesseract_version()   # fails if binary not on PATH
    OCR_OK = True
    print(f"   📸 OCR ready (Tesseract {pytesseract.get_tesseract_version()})")
except Exception:
    if _PIL_OK:
        print("⚠  Tesseract binary missing — OCR off. Install from https://github.com/UB-Mannheim/tesseract/wiki")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_OK = True
except ImportError:
    PDF2IMAGE_OK = False

# ── Poppler path (needed on Windows for pdf2image) ────────────────────────────
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"


# =============================================================================
#  RAW TEXT EXTRACTION  (unchanged from previous version — already solid)
# =============================================================================

def _pdf_to_text(path: str) -> str:
    if not PDF_OK: return ""
    try:
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    except Exception as e:
        print(f"   PDF error: {e}"); return ""

def _docx_to_text(path: str) -> str:
    if not DOCX_OK: return ""
    try:
        return "\n".join(p.text for p in DocxDoc(path).paragraphs if p.text)
    except Exception as e:
        print(f"   DOCX error: {e}"); return ""

def _txt_to_text(path: str) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f: return f.read()
        except Exception: pass
    return ""

def _ocr_image(path: str) -> str:
    if not OCR_OK: return ""
    try:
        try:
            return pytesseract.image_to_string(PilImage.open(path), lang="eng+fra+ara")
        except pytesseract.TesseractError:
            return pytesseract.image_to_string(PilImage.open(path), lang="eng")
    except Exception as e:
        print(f"   OCR error: {e}"); return ""

def _scanned_pdf_to_text(path: str) -> str:
    """Convert scanned PDF pages to images, then OCR each page."""
    if not OCR_OK: return ""
    if not PDF2IMAGE_OK:
        return _ocr_image(path)
    try:
        pages = convert_from_path(path, dpi=200, poppler_path=POPPLER_PATH)
        return "\n".join(pytesseract.image_to_string(p, lang="eng") for p in pages)
    except Exception as e:
        print(f"   Scanned-PDF OCR error: {e}"); return ""

def extract_raw_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = _pdf_to_text(path)
        if text.strip():
            return text
        if OCR_OK:
            print("   📸 Scanned PDF — running OCR on pages…")
            return _scanned_pdf_to_text(path)
        print("   ⚠  Scanned PDF detected but Tesseract not installed — skipping.")
        return ""
    elif ext == ".docx":
        return _docx_to_text(path)
    elif ext == ".txt":
        return _txt_to_text(path)
    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        if OCR_OK:
            return _ocr_image(path)
        print("   ⚠  Image file but Tesseract not installed — skipping.")
        return ""
    return ""


# =============================================================================
#  DATABASE  —  schema + migrations (same as before)
# =============================================================================

def _ensure_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            requirements TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER,
            name TEXT DEFAULT 'Unknown',
            email TEXT DEFAULT '',
            phone TEXT DEFAULT '',
            location TEXT DEFAULT '',
            years_experience INTEGER DEFAULT 0,
            current_job TEXT DEFAULT '',
            skills TEXT DEFAULT '',
            education TEXT DEFAULT '',
            work_experience TEXT DEFAULT '',
            languages TEXT DEFAULT '',
            certifications TEXT DEFAULT '',
            summary TEXT DEFAULT '',
            source TEXT DEFAULT 'manual',
            raw_text TEXT DEFAULT '',
            filename TEXT DEFAULT '',
            score INTEGER DEFAULT 0,
            score_detail TEXT DEFAULT '',
            pros TEXT DEFAULT '',
            cons TEXT DEFAULT '',
            reason TEXT DEFAULT '',
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES jobs(id)
        )
    """)
    conn.commit()
    # migrations for older DB files
    c.execute("PRAGMA table_info(jobs)");      jc = {r[1] for r in c.fetchall()}
    c.execute("PRAGMA table_info(candidates)"); cc = {r[1] for r in c.fetchall()}
    for need, sql in [
        ("requirements"  not in jc,  "ALTER TABLE jobs ADD COLUMN requirements TEXT DEFAULT ''"),
        ("score_detail"  not in cc,  "ALTER TABLE candidates ADD COLUMN score_detail TEXT DEFAULT ''"),
        ("pros"          not in cc,  "ALTER TABLE candidates ADD COLUMN pros TEXT DEFAULT ''"),
        ("cons"          not in cc,  "ALTER TABLE candidates ADD COLUMN cons TEXT DEFAULT ''"),
        ("raw_text"      not in cc,  "ALTER TABLE candidates ADD COLUMN raw_text TEXT DEFAULT ''"),
        ("filename"      not in cc,  "ALTER TABLE candidates ADD COLUMN filename TEXT DEFAULT ''"),
        ("work_experience" not in cc,"ALTER TABLE candidates ADD COLUMN work_experience TEXT DEFAULT ''"),
        ("languages"     not in cc,  "ALTER TABLE candidates ADD COLUMN languages TEXT DEFAULT ''"),
        ("certifications" not in cc, "ALTER TABLE candidates ADD COLUMN certifications TEXT DEFAULT ''"),
        ("summary"       not in cc,  "ALTER TABLE candidates ADD COLUMN summary TEXT DEFAULT ''"),
    ]:
        if need:
            try: c.execute(sql)
            except Exception: pass
    conn.commit()
    conn.close()


def _to_str(val, max_len: int = 600) -> str:
    """Convert list / dict / anything to a clean, truncated string."""
    if val is None: return ""
    if isinstance(val, list):
        return ", ".join(str(x) for x in val if x)[:max_len]
    if isinstance(val, dict):
        return ", ".join(f"{k}: {v}" for k, v in val.items() if v)[:max_len]
    return str(val)[:max_len]


def _save_candidate(profile: CandidateProfile, raw_text: str,
                    job_id: int, filename: str) -> int:
    """Upsert a parsed CandidateProfile into SQLite. Returns candidate id."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    name  = (profile.full_name or "Unknown")[:200]
    email = (profile.email     or "no-email@unknown.com")[:200]
    phone = (profile.phone     or "")[:80]
    loc   = (profile.location  or "")[:120]
    yrs   = profile.years_experience
    curr  = (profile.current_job_title or "")[:200]
    skills= _to_str(profile.skills,        600)
    edu   = _to_str(profile.education,     600)
    work  = _to_str(profile.work_experience, 800)
    langs = _to_str(profile.languages,     200)
    certs = _to_str(profile.certifications, 300)
    summ  = (profile.summary or "")[:600]

    c.execute("SELECT id FROM candidates WHERE job_id=? AND email=?", (job_id, email))
    existing = c.fetchone()

    if existing:
        c.execute("""
            UPDATE candidates SET
                name=?, phone=?, location=?, years_experience=?,
                current_job=?, skills=?, education=?, work_experience=?,
                languages=?, certifications=?, summary=?,
                raw_text=?, filename=?,
                score=0, score_detail='', pros='', cons='', reason='', status='pending'
            WHERE id=?
        """, (name, phone, loc, yrs, curr, skills, edu, work,
              langs, certs, summ, raw_text[:6000], filename, existing[0]))
        cid = existing[0]
        print(f"   ↺  Updated: {name}")
    else:
        c.execute("""
            INSERT INTO candidates
                (job_id, name, email, phone, location, years_experience,
                 current_job, skills, education, work_experience,
                 languages, certifications, summary, source,
                 raw_text, filename, status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'manual',?,?,'pending')
        """, (job_id, name, email, phone, loc, yrs, curr,
              skills, edu, work, langs, certs, summ,
              raw_text[:6000], filename))
        cid = c.lastrowid
        print(f"   ✓  Inserted: {name}")

    conn.commit()
    conn.close()
    return cid


# =============================================================================
#  CV AGENT CLASS
# =============================================================================

class CVAgent:
    """
    Parses CV files into structured candidate profiles using:
      • Groq API (fast cloud LLM)
      • instructor (Pydantic-validated JSON — no parse errors)
      • ChromaDB (RAG vector store for ScoreAgent)
    """

    SUPPORTED_EXT = {".pdf", ".docx", ".txt",
                     ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self):
        self.name   = "CVAgent"
        self.client = _get_groq_client()   # raises if GROQ_API_KEY missing
        _ensure_db()
        print(f"🤖 {self.name} ready  |  Groq model: {GROQ_MODEL}")
        if OCR_OK:
            print("   📸 OCR ready")
        chroma = _get_chroma_collection()
        if chroma is not None:
            print(f"   🔍 ChromaDB RAG ready  ({CHROMA_PATH})")
        else:
            print("   ℹ  ChromaDB not available — RAG disabled (scoring still works)")

    # ─────────────────────────────────────────────────────────────────────────
    #  Parse ONE file
    # ─────────────────────────────────────────────────────────────────────────
    def parse_cv(self, filepath: str, job_id: int, source: str = "manual") -> dict:
        fname = os.path.basename(filepath)
        print(f"\n📄 [{self.name}] Parsing: {fname}")

        # Step 1 — extract raw text from file
        raw = extract_raw_text(filepath)
        if not raw.strip():
            print(f"   ❌ No text extracted from {fname}")
            return {"success": False, "error": "empty text", "name": fname}
        print(f"   ✓ Extracted {len(raw):,} chars")

        # Step 2 — send to Groq via instructor
        # instructor guarantees the response matches CandidateProfile exactly.
        # If the LLM returns bad JSON, instructor retries automatically.
        print("   🤖 Groq extracting structured profile…")
        try:
            profile: CandidateProfile = self.client.chat.completions.create(
                model    = GROQ_MODEL,
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert CV parser. "
                            "Extract every piece of information from the CV text provided. "
                            "Be thorough — include all skills, all work experiences, all education. "
                            "For years_experience, count only professional work years (not education). "
                            "Return only valid structured data matching the schema exactly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Parse this CV:\n\n{raw[:5000]}",
                    },
                ],
                response_model = CandidateProfile,   # ← this is what instructor adds
                max_retries    = 3,                  # retry up to 3 times on bad output
                temperature    = 0.05,               # low temp = consistent structured output
            )
        except Exception as e:
            print(f"   ❌ Groq/instructor error: {e}")
            return {"success": False, "error": str(e), "name": fname}

        print(f"   ✓ Parsed: {profile.full_name}  <{profile.email}>")

        # Step 3 — save to SQLite
        cid = _save_candidate(profile, raw, job_id, fname)

        # Step 4 — store in ChromaDB for RAG (non-blocking, best-effort)
        store_cv_in_chroma(
            candidate_id = cid,
            raw_text     = raw,
            metadata     = {
                "candidate_id": cid,
                "job_id":       job_id,
                "name":         profile.full_name,
                "filename":     fname,
            }
        )

        return {
            "success":      True,
            "candidate_id": cid,
            "name":         profile.full_name,
            "email":        profile.email,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Process ALL files in a folder — PARALLEL
    # ─────────────────────────────────────────────────────────────────────────
    def process_incoming_files(self, folder: str = "incoming",
                               job_id: int = None,
                               max_workers: int = 4) -> dict:
        """
        Processes every CV in the folder in parallel.
        Returns a dict with counts: {total, done, failed, names}
        """
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            return {"total": 0, "done": 0, "failed": 0, "names": []}

        files = [
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
            and os.path.splitext(f)[1].lower() in self.SUPPORTED_EXT
        ]
        if not files:
            return {"total": 0, "done": 0, "failed": 0, "names": []}

        processed_dir = os.path.join(folder, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        print(f"\n📁 {len(files)} file(s) — parallel extraction (workers={max_workers})…")

        done_names : list  = []
        failed_names: list = []
        lock = threading.Lock()

        def _process_one(filename: str):
            src    = os.path.join(folder, filename)
            result = self.parse_cv(src, job_id, source="manual")
            if result["success"]:
                dst = os.path.join(processed_dir, filename)
                try:
                    os.rename(src, dst)
                except Exception:
                    shutil.copy2(src, dst)
                    os.remove(src)
                with lock:
                    done_names.append(result["name"])
                print(f"   ✅ {filename}  →  {result['name']}")
            else:
                with lock:
                    failed_names.append(filename)
                print(f"   ❌ {filename}  ({result.get('error', '?')})")
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_one, f): f for f in files}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    fname = futures[future]
                    print(f"   ❌ Exception on {fname}: {e}")
                    with lock:
                        failed_names.append(fname)

        total = len(files)
        done  = len(done_names)
        print(f"\n✅ CVAgent done: {done}/{total} extracted")
        return {"total": total, "done": done, "failed": len(failed_names), "names": done_names}


# =============================================================================
#  STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("CV AGENT — STANDALONE TEST")
    print("=" * 60)

    if not os.environ.get("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not set.")
        print("   Get a free key at: https://console.groq.com")
        exit(1)

    agent = CVAgent()

    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT id, title FROM jobs LIMIT 1")
    row = c.fetchone()
    conn.close()

    if not row:
        print("No jobs found. Create one via the dashboard first.")
        exit(0)

    job_id = row[0]
    print(f"Using job: [{job_id}] {row[1]}\n")
    result = agent.process_incoming_files("incoming", job_id=job_id)
    print(f"\nResult: {result}")
