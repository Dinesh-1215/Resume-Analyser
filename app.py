"""
Flask Resume Analyzer (single-file)
Enhanced with:
- URL support for fetching job description text (via requests)
- Detailed console logging for every major operation (embedding model loading, text extraction, scoring, etc.)
"""

import os
import re
import tempfile
import shutil
import requests
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from io import BytesIO
from google import genai
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

Client = genai.Client()

# Text extraction libs
import pdfplumber
import docx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    types = None
    GENAI_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev_secret')
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# ----------------------- Utilities -----------------------

def log(msg):
    print(f"[LOG] {msg}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    log(f"Extracting text from PDF: {file_path}")
    text = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ''
            log(f"  Extracted page {i+1} ({len(page_text)} chars)")
            text.append(page_text)
    return '\n'.join(text)

def extract_text_from_docx(file_path):
    log(f"Extracting text from DOCX: {file_path}")
    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs]
    log(f"  Extracted {len(paragraphs)} paragraphs")
    return '\n'.join(paragraphs)

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

# ------------------ Embeddings ------------------

def get_embeddings_genai(texts, model_name='embed-model'):
    log("Attempting to create embeddings using Google GenAI SDK")
    if not GENAI_AVAILABLE:
        raise RuntimeError('google.genai SDK not available')
    client = genai.Client()
    try:
        resp = client.models.embed_content(model="gemini-embedding-001", contents=texts)
        embs = resp.embeddings
        log(f"  Embeddings created for {len(texts)} texts")
        return np.array(embs)
    except Exception as e:
        log(f"Embedding creation failed: {e}")
        raise

def get_embeddings_fallback(texts):
    log("Using fallback TF-IDF embeddings")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    X = vectorizer.fit_transform(texts)
    log(f"  TF-IDF matrix shape: {X.shape}")
    return X.toarray(), vectorizer

# ------------------ Scoring ------------------

def compute_keyword_score(resume_text, jd_text, top_n_keywords=40):
    """Extract top n keywords from the JD and compute which of those appear in the resume.
    Uses TF-IDF to pick the most important ngrams from the JD, then checks presence in resume
    using a robust word-boundary check that also supports multi-word tokens.
    Returns: (score_float_0_1, list_of_keywords, present_list)
    """
    log("Computing keyword score")
    # ensure strings
    r = (resume_text or "").lower()
    j = (jd_text or "").lower()

    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=top_n_keywords)
    try:
        X = vec.fit_transform([j, r])
    except Exception as e:
        log(f"TF-IDF vectorization failed: {e}")
        return 0.0, [], []

    features = [f.lower() for f in vec.get_feature_names_out()]

    # Robust presence check: match tokens or phrases where they occur as whole words
    present = []
    for feat in features:
        # escape and create a regex that allows punctuation boundaries for multi-word features
        pattern = r'(?<!\w)' + re.escape(feat) + r'(?!\w)'
        if re.search(pattern, r, flags=re.IGNORECASE):
            present.append(feat)

    score = (len(present) / len(features)) if len(features) > 0 else 0.0
    log(f"  Found {len(present)}/{len(features)} JD keywords in resume")
    return score, features, present

def compute_ats_score(resume_text, jd_text, use_embeddings=True):
    log("Starting ATS computation")
    resume_text = clean_text(resume_text).lower() if resume_text else ""
    jd_text = clean_text(jd_text).lower() if jd_text else ""

    kw_score, features, present = compute_keyword_score(resume_text, jd_text, 60)

    sem_score = 0.0
    semantic_method = 'unknown'
    try:
        if use_embeddings and GENAI_AVAILABLE:
            embs = get_embeddings_genai([jd_text, resume_text])
            # ensure embeddings are 2-D arrays
            if hasattr(embs, 'shape') and embs.shape[0] >= 2:
                sem_score = float(cosine_similarity([embs[0]], [embs[1]])[0][0])
            else:
                # fallback to tfidf similarity if embeddings shape unexpected
                raise Exception('invalid embeddings shape')
            semantic_method = 'genai'
            log(f"  Semantic similarity via GenAI: {sem_score:.3f}")
        else:
            raise Exception('fallback')
    except Exception:
        X = TfidfVectorizer(stop_words='english', max_features=400).fit_transform([jd_text, resume_text])
        sem_score = float(cosine_similarity(X[0], X[1])[0][0])
        semantic_method = 'tfidf'
        log(f"  Semantic similarity via TF-IDF fallback: {sem_score:.3f}")

    # combine weights: keywords 60%, semantic 40%
    combined = (0.6 * kw_score) + (0.4 * sem_score)
    ats_percent = round(combined * 100, 1)

    # compute missing keywords (jd keywords not in resume)
    missing = [k for k in features if k not in present]

    return {
        'ats_percent': ats_percent,
        'keyword_score': round(kw_score * 100, 1),
        'semantic_score': round(sem_score * 100, 1),
        'semantic_method': semantic_method,
        'keywords': features,
        'present_keywords': present,
        'missing_keywords': missing
    }

# ------------------ Suggestions ------------------

def generate_suggestion_bullets(resume_text, jd_text):
    """Generate cleaned, concise suggestions for resume bullets.
    - Uses GenAI when available.
    - Sanitizes output: removes markdown, horizontal rules, large noisy paragraphs, and conversational lead-ins like "Let's" or "Okay".
    - Returns a short, clean multiline string suitable for display in the UI.
    """
    def _sanitize(raw: str) -> str:
        # remove markdown emphasis / bold / code markers
        s = re.sub(r"(``?\\w*`|\*\*|__|\*|_)", "", raw)
        # remove horizontal rules and repeated punctuation lines
        s = re.sub(r"(^[-_*]{2,}|^\s*[-_*]{3,}\s*$)", "", s, flags=re.M)
        # remove common conversational lead-ins and headings
        s = re.sub(r"(?im)^(ok|okay|let\'s|lets|here\'s|here|the following|overview)[:\-\s].*\\n?", "", s)
        # collapse > quoted lines
        s = re.sub(r"^> .*\\n?", "", s, flags=re.M)
        # collapse multiple newlines to two
        s = re.sub(r"\n{3,}", "\n\n", s)
        # trim whitespace
        s = s.strip()
        # if still large, keep top 12 lines
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if len(lines) > 12:
            lines = lines[:12]
            # add indicator
            lines.append('... (truncated)')
        # remove any leading numbering like '1.' or '-' from lines
        clean_lines = [re.sub(r"^\s*\d+\.|^\s*[-–—]\s*", "", ln).strip() for ln in lines]
        return "\n".join(clean_lines)

    # Try GenAI first
    if GENAI_AVAILABLE:
        try:
            client = genai.Client()
            prompt = (
                "Produce up to 8 concise, action-oriented resume bullets (one per line) that align the candidate's experience to the Job Description. "
                "Respond with bullets only (no preamble, no headings).\n\n"
                f"Job Description:\n{jd_text[:2000]}\n\nResume:\n{resume_text[:2000]}\n\n"
            )
            log("Requesting suggestions from Gemma")
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0))
            )
            raw = getattr(resp, 'text', '') or str(resp)
            cleaned = _sanitize(raw)
            if cleaned:
                return cleaned
        except Exception as e:
            log(f"Gemma suggestions failed: {e}")

    # Fallback heuristic: extract top lines from resume and prefix with action verbs
    lines = [l.strip() for l in resume_text.splitlines() if l.strip()]
    verbs = ['Led', 'Designed', 'Implemented', 'Optimized', 'Automated', 'Improved', 'Developed', 'Built']
    bullets = []
    for i, line in enumerate(lines[:8]):
        verb = verbs[i % len(verbs)]
        snippet = line if len(line) <= 140 else line[:137] + '...'
        # remove internal markdown and odd characters
        snippet = re.sub(r"[`*_]{1,}", "", snippet)
        bullets.append(f"{verb} {snippet}")
    return "\n".join(bullets) if bullets else 'No actionable bullets found; add achievement-focused statements with metrics.'

# ------------------ Flask ------------------


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result=None)

@app.route('/analyze', methods=['POST'])
def analyze():
    # single POST handler that returns the same page with inline errors (no redirects)
    file = request.files.get('resume_file')
    resume_text = request.form.get('resume_text','').strip()
    jd_text = request.form.get('jd_text','').strip()
    jd_url = request.form.get('jd_url','').strip()

    error = None

    # Handle file upload (extract then delete)
    if file and file.filename:
        tmpdir = tempfile.mkdtemp()
        filename = secure_filename(file.filename)
        path = os.path.join(tmpdir, filename)
        file.save(path)
        try:
            if filename.lower().endswith('.pdf'):
                resume_text = extract_text_from_pdf(path)
            elif filename.lower().endswith('.docx'):
                resume_text = extract_text_from_docx(path)
            else:
                with open(path,'r',encoding='utf-8',errors='ignore') as f:
                    resume_text = f.read()
            log(f"Resume text length: {len(resume_text)} chars")
        except Exception as e:
            error = f'Failed to extract resume: {e}'
            log(error)
            shutil.rmtree(tmpdir)
            # render page with error
            return render_template('index.html', error=error, result=None, resume_text=resume_text, jd_text=jd_text, jd_url=jd_url, use_embeddings=bool(request.form.get('use_embeddings')))
        shutil.rmtree(tmpdir)

    # If JD URL given and no JD text, attempt to fetch and convert HTML to plain text (defensive)
    if jd_url and not jd_text:
        try:
            log(f"Fetching JD from URL: {jd_url}")
            r = requests.get(jd_url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
            r.raise_for_status()
            content = r.text or ''
            # crude HTML-to-text fallback (avoids requiring BeautifulSoup)
            if '<' in content and '>' in content:
                try:
                    # remove script/style blocks first
                    content = re.sub(r'<(script|style).*?>.*?</\1>', ' ', content, flags=re.S|re.I)
                    # strip tags
                    text = re.sub(r'<[^>]+>', ' ', content)
                    text = re.sub(r'\s+', ' ', text).strip()
                except Exception:
                    text = re.sub(r'<[^>]+>', ' ', content)
                    text = re.sub(r'\s+', ' ', text).strip()
                jd_text = text[:15000]
            else:
                jd_text = content[:15000]
            log(f"Fetched JD text length: {len(jd_text)} chars")
        except Exception as e:
            error = f'Failed to fetch JD URL: {e}'
            log(error)

    # Validate presence of both resume and JD; collect errors into one message
    if not resume_text:
        error = (error + ' ' if error else '') + 'Please provide a resume (upload or paste).'
    if not jd_text:
        error = (error + ' ' if error else '') + 'Please provide a job description (paste or URL).'

    if error:
        # return the same page with an inline error message and preserve form contents
        return render_template(
            'index.html',
            error=error,
            result=None,
            resume_text=resume_text,
            jd_text=jd_text,
            jd_url=jd_url,
            use_embeddings=bool(request.form.get('use_embeddings')),
            suggestions=None
        )

    use_embeddings = bool(request.form.get('use_embeddings'))

    try:
        result = compute_ats_score(resume_text, jd_text, use_embeddings)
        suggestions = generate_suggestion_bullets(resume_text, jd_text)
    except Exception as e:
        log(f"Error during analysis: {e}")
        error = f"Error during analysis: {e}"
        return render_template('index.html', error=error, result=None, resume_text=resume_text, jd_text=jd_text, jd_url=jd_url, use_embeddings=use_embeddings)

    # success — render results inline and preserve inputs
    return render_template('index.html',
        error=None,
        result=result,
        suggestions=suggestions,
        resume_text=resume_text,
        jd_text=jd_text,
        jd_url=jd_url,
        use_embeddings=use_embeddings
    )


# --- AJAX endpoint for regenerating suggestions ---
@app.route('/suggest', methods=['POST'])
def suggest():
    """AJAX endpoint to return regenerated suggestions without a full page reload.
    Expects JSON: { resume_text, jd_text, jd_url, use_embeddings }
    Returns JSON: { suggestions: str } or { error: str }
    """
    try:
        payload = request.get_json(force=True)
        resume_text = (payload.get('resume_text') or '').strip()
        jd_text = (payload.get('jd_text') or '').strip()
        jd_url = (payload.get('jd_url') or '').strip()
        use_embeddings = bool(payload.get('use_embeddings'))

        # If jd_url provided and jd_text empty, try to fetch (defensive, same logic as /analyze)
        if jd_url and not jd_text:
            try:
                r = requests.get(jd_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                content = r.text or ''
                if '<' in content and '>' in content:
                    content = re.sub(r'<(script|style).*?>.*?</\1>', ' ', content, flags=re.S|re.I)
                    jd_text = re.sub(r'<[^>]+>', ' ', content)
                    jd_text = re.sub(r'\s+', ' ', jd_text).strip()[:15000]
                else:
                    jd_text = content[:15000]
            except Exception as e:
                return { 'error': f'Failed to fetch JD URL: {e}' }

        if not resume_text:
            return { 'error': 'Please provide resume text (paste or upload first).' }
        if not jd_text:
            return { 'error': 'Please provide JD text or a valid JD URL.' }

        suggestions = generate_suggestion_bullets(resume_text, jd_text)
        return { 'suggestions': suggestions }
    except Exception as e:
        log(f"/suggest error: {e}")
        return { 'error': str(e) }

if __name__ == '__main__':
    log('Starting Flask app on http://127.0.0.1:5000')
    app.run(debug=True)
