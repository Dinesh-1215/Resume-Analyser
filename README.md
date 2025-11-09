# Resume-Matcher — Flask Resume Analyzer

A single-file Flask app that analyzes a resume against a job description (JD), computes an ATS-style score, highlights matched / missing keywords, and produces cleaned, actionable resume bullet suggestions (optionally using Google Gemini/GenAI).

---

## Features

- Accepts resume input via **file upload** (PDF / DOCX / TXT) or **pasted text**.
- Accepts JD input via **pasted text** or **JD URL** (server fetch + HTML→text fallback).
- Keyword extraction (TF-IDF) and presence checking for single & multi-word tokens.
- Semantic similarity via Gemini embeddings (optional) or TF-IDF fallback.
- ATS score combining keyword + semantic signals.
- Cleaned suggestions using Gemini (if available) with robust sanitization and a fallback heuristic.
- Interactive UI with:
  - Side-by-side resume & JD inputs (Upload / Paste tabs).
  - Results card with colored ATS header, matched/missing keyword badges.
  - Suggestions list with per-line Edit / Copy buttons and a Regenerate (AJAX) action.
- Console logging for traceability (extraction, fetch, embedding, scoring).

---

## Repo layout

```
Resume-Matcher/
├─ app.py                 # main Flask application (single-file)
├─ requirements.txt
├─ README.md
└─ .env (optional)        # GEMINI_API_KEY, FLASK_SECRET
```

---

## Quick start (local)

1. (Optional) Create & activate a virtual env:
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate    # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Configure environment variables:
```bash
# .env file or exported environment variables
GEMINI_API_KEY="your_gemini_api_key_here"
FLASK_SECRET="replace_with_a_secure_value"
```

4. Run the app:
```bash
python app.py
```

Open: `http://127.0.0.1:5000`

---

## Usage (UI)

1. Upload or paste your resume (left column).
2. Paste JD text or provide JD URL (right column).
3. Toggle **Use embeddings** to enable Gemini-based semantic checks and suggestions (requires `google-genai`).
4. Click **Analyze**.
5. Review:
   - ATS score and breakdown,
   - Matched / missing keywords (badges),
   - Suggestions (bulleted list). Use **Regenerate** to re-run suggestions, **Edit** to modify bullets inline, and **Copy** to copy a bullet or all bullets.

---

## API endpoints

### `POST /analyze` (form)
Used by the UI. Returns the rendered HTML with results.

Form fields:
- `resume_file` (file, optional)
- `resume_text` (text, optional)
- `jd_text` (text, optional)
- `jd_url` (text, optional)
- `use_embeddings` (checkbox, optional)

### `POST /suggest` (AJAX)
Regenerates suggestions without full page reload.

Request JSON:
```json
{
  "resume_text": "...",
  "jd_text": "...",
  "jd_url": "...",
  "use_embeddings": true
}
```

Response:
```json
{"suggestions": "bullet1\nbullet2\n..."}
# or {"error": "message"}
```

---

## Docker (optional)

```bash
docker build -t resume-matcher .
docker run -p 5000:5000 resume-matcher
```

---

## License

MIT License
