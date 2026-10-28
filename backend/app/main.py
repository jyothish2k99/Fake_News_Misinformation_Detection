from __future__ import annotations
from pathlib import Path
import ipaddress
import io, re
import socket
from urllib.parse import urljoin, urlsplit
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from .services.model_service import ModelService
from .services.analytics_service import summary
from .services.evidence_service import EvidenceService
from .services.source_service import leaderboard, score_for_url

MAX_EXTRACT_BYTES = 2 * 1024 * 1024
MAX_CSV_BYTES = 5 * 1024 * 1024
MAX_CSV_ROWS = 5000
MAX_REDIRECTS = 5

app=FastAPI(title="Misinformation Intelligence API",version="1.0.0",description="Baseline news classification, dataset analytics, related-article retrieval and source-history scoring.")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
model=ModelService(); evidence=EvidenceService()

class ArticleRequest(BaseModel):
    title: str = Field(default="",max_length=1000)
    text: str = Field(default="",max_length=100000)
    url: str|None = Field(default=None,max_length=2048)


def _validate_public_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError
        if parsed.username or parsed.password:
            raise ValueError
        host = parsed.hostname.rstrip(".")
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
        try:
            literal = ipaddress.ip_address(host)
            addresses = [literal]
        except ValueError:
            addresses = [
                ipaddress.ip_address(record[4][0])
                for record in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            ]
        if not addresses or any(not address.is_global for address in addresses):
            raise ValueError
    except (ValueError, UnicodeError, OSError):
        raise HTTPException(status_code=400, detail="Article URL must use a public HTTP(S) address.") from None


def _fetch_article(url: str) -> tuple[str, str, str]:
    current_url = url
    session = requests.Session()
    session.trust_env = False
    try:
        for redirect_count in range(MAX_REDIRECTS + 1):
            _validate_public_url(current_url)
            with session.get(
                current_url,
                timeout=(3, 8),
                headers={"User-Agent": "Mozilla/5.0 (compatible; MisinformationResearch/1.0)"},
                allow_redirects=False,
                stream=True,
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location or redirect_count >= MAX_REDIRECTS:
                        raise HTTPException(status_code=400, detail="Article URL redirected too many times or to an invalid destination.")
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_EXTRACT_BYTES:
                    raise HTTPException(status_code=413, detail="The article page exceeds the extraction size limit.")
                content = bytearray()
                for chunk in response.iter_content(chunk_size=65536):
                    content.extend(chunk)
                    if len(content) > MAX_EXTRACT_BYTES:
                        raise HTTPException(status_code=413, detail="The article page exceeds the extraction size limit.")
                page = BeautifulSoup(bytes(content), "html.parser")
                for node in page(["script", "style", "nav", "footer", "header", "noscript"]):
                    node.decompose()
                title = page.title.get_text(" ", strip=True) if page.title else ""
                text = " ".join(page.get_text(" ", strip=True).split())[:100000]
                return title, text, current_url
    except HTTPException:
        raise
    except (requests.RequestException, OSError, ValueError, UnicodeError):
        raise HTTPException(status_code=502, detail="Could not retrieve readable article text from that URL.") from None
    finally:
        session.close()
    raise HTTPException(status_code=400, detail="Article URL redirected too many times.")

@app.get("/api/health")
def health():
    return {"status":"ok","model_ready":model.ready,"model_metadata":model.metadata,"evidence_ready":evidence.vectorizer is not None}

@app.post("/api/predict")
def predict(req: ArticleRequest):
    title = req.title.strip()
    text = req.text.strip()
    extracted_url = None
    source_url = req.url
    if source_url and not text:
        title, text, extracted_url = _fetch_article(source_url)
    if len((title + " " + text).strip()) < 30:
        raise HTTPException(status_code=422, detail="Provide a headline and article text (at least 30 characters), or a URL that exposes readable article text.")
    try:
        result = model.predict(title, text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    result["title"] = title[:240]
    result["url"] = extracted_url or source_url
    result["analyzed_text"] = (title + " " + text).strip()
    result["evidence"] = evidence.search(title + " " + text, 3)
    result["source_credibility"] = score_for_url(extracted_url or source_url)
    return result

@app.get("/api/analytics")
def analytics(): return summary()

@app.get("/api/sources")
def sources(): return leaderboard()

@app.post("/api/batch")
async def batch(file: UploadFile=File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")
    content = await file.read(MAX_CSV_BYTES + 1)
    if len(content) > MAX_CSV_BYTES:
        raise HTTPException(status_code=413, detail="CSV files must be 5 MB or smaller.")
    if not content.strip():
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")
    try:
        df = pd.read_csv(io.BytesIO(content), on_bad_lines="error")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="Could not read the CSV. Check its encoding and row structure.") from None
    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded CSV contains no article rows.")
    if len(df) > MAX_CSV_ROWS:
        raise HTTPException(status_code=413, detail=f"CSV files are limited to {MAX_CSV_ROWS} rows.")
    lower = {str(column).lower().strip(): column for column in df.columns}
    title_col = lower.get("title") or lower.get("headline")
    text_col = lower.get("text") or lower.get("article") or lower.get("content")
    if not title_col and not text_col:
        raise HTTPException(status_code=422, detail="CSV must contain a title/headline and/or text/article/content column.")
    if not model.ready:
        raise HTTPException(status_code=503, detail="Model is not trained. Run python train_model.py first.")
    rows = []
    for _, row in df.iterrows():
        title_value = row.get(title_col, "") if title_col else ""
        text_value = row.get(text_col, "") if text_col else ""
        title = "" if pd.isna(title_value) else str(title_value).strip()
        text = "" if pd.isna(text_value) else str(text_value).strip()
        if not (title + text).strip():
            rows.append({"title": title, "text": text, "prediction": "UNABLE_TO_ANALYZE", "fake_probability": None, "real_probability": None, "confidence": None})
            continue
        prediction = model.predict(title, text)
        rows.append({"title": title, "text": text, "prediction": prediction["label"], "fake_probability": prediction["fake_probability"], "real_probability": prediction["real_probability"], "confidence": prediction["confidence"]})
    output = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    return StreamingResponse(io.BytesIO(output), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=misinformation_batch_report.csv"})
