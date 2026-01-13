"""
Document Parser with Grammar Resonance → VSA Awareness
═══════════════════════════════════════════════════════════════════════════════

Parse uploaded documents (PDF, TXT) through grammar triangle resonance
and store awareness vectors in DAG.

Endpoints:
    POST /parse/document    - Upload document, parse to awareness
    POST /parse/train       - Train grammar templates from document
    GET  /parse/status      - Get parsing status
    
Integration:
    - node.msgraph.de (agi-chat) for grammar parsing
    - dag-vsa0X.msgraph.de for vector storage
    - Grok API for template learning

Born: 2026-01-13
"""

import os
import io
import json
import hashlib
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict

import httpx
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Optional PDF support
try:
    import pypdf
    HAS_PDF = True
except ImportError:
    pypdf = None
    HAS_PDF = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

AGI_CHAT_URL = os.getenv("AGI_CHAT_URL", "https://node.msgraph.de")
DAG_URL = os.getenv("DAG_URL", "https://dag-vsa02.msgraph.de")
GROK_API_KEY = os.getenv("ADA_XAI", "")
GROK_API_URL = "https://api.x.ai/v1/chat/completions"


# ═══════════════════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DocumentAwareness:
    """Awareness of a parsed document."""
    doc_id: str
    title: str
    pages: int
    sentences: int
    
    # Grammar analysis
    templates_matched: int
    new_templates_learned: int
    parse_coverage: float  # % of sentences parsed with >0.5 confidence
    
    # VSA storage
    chunks_stored: int
    awareness_vector_id: str
    
    # Timing
    started_at: str
    completed_at: Optional[str] = None
    status: str = "processing"


class ParseRequest(BaseModel):
    """Request to parse a document."""
    train_templates: bool = True
    target_family: str = "declarative.simple"
    store_chunks: bool = True
    store_awareness: bool = True


class ParseResponse(BaseModel):
    """Response from document parsing."""
    doc_id: str
    status: str
    pages: int
    sentences: int
    parse_coverage: float
    templates_matched: int
    new_templates: int
    chunks_stored: int
    awareness_id: str


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pdf_pages(content: bytes) -> List[str]:
    """Extract text from PDF pages."""
    if not HAS_PDF:
        raise HTTPException(500, "PDF support not installed (pypdf)")
    
    reader = pypdf.PdfReader(io.BytesIO(content))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages


def extract_txt_pages(content: bytes, page_size: int = 2000) -> List[str]:
    """Split text file into pages."""
    text = content.decode('utf-8', errors='ignore')
    
    # Split by double newlines (paragraphs) or by size
    paragraphs = text.split('\n\n')
    
    pages = []
    current_page = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if current_size + len(para) > page_size:
            if current_page:
                pages.append('\n\n'.join(current_page))
            current_page = [para]
            current_size = len(para)
        else:
            current_page.append(para)
            current_size += len(para)
    
    if current_page:
        pages.append('\n\n'.join(current_page))
    
    return pages


def extract_sentences(pages: List[str]) -> List[str]:
    """Extract sentences from pages."""
    sentences = []
    for page in pages:
        # Split on sentence boundaries
        page_sentences = page.replace('\n', ' ').split('.')
        for s in page_sentences:
            s = s.strip()
            # Filter: minimum 3 words, maximum 50 words
            words = s.split()
            if 3 <= len(words) <= 50:
                sentences.append(s + '.')
    return sentences


# ═══════════════════════════════════════════════════════════════════════════════
# GRAMMAR PARSING (via agi-chat)
# ═══════════════════════════════════════════════════════════════════════════════

async def parse_sentences_batch(
    sentences: List[str],
    train_new: bool = True,
    family: str = "declarative.simple"
) -> Dict[str, Any]:
    """Parse sentences via agi-chat grammar endpoint."""
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        results = {
            "parsed": 0,
            "high_confidence": 0,
            "low_confidence_sentences": [],
            "templates_matched": set(),
            "parse_results": [],
        }
        
        # Parse each sentence
        for sentence in sentences:
            try:
                resp = await client.post(
                    f"{AGI_CHAT_URL}/grammar/parse",
                    json={"text": sentence}
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    results["parsed"] += 1
                    
                    confidence = data.get("result", {}).get("confidence", 0)
                    template = data.get("result", {}).get("template")
                    
                    if confidence > 0.5:
                        results["high_confidence"] += 1
                        if template:
                            results["templates_matched"].add(template.get("id", "unknown"))
                    else:
                        results["low_confidence_sentences"].append(sentence)
                    
                    results["parse_results"].append({
                        "sentence": sentence,
                        "confidence": confidence,
                        "template": template.get("id") if template else None,
                    })
            except Exception as e:
                print(f"Parse error for '{sentence[:50]}...': {e}")
        
        # Train new templates from low-confidence sentences
        new_templates = 0
        if train_new and results["low_confidence_sentences"]:
            try:
                train_resp = await client.post(
                    f"{AGI_CHAT_URL}/grammar/train",
                    json={
                        "sentences": results["low_confidence_sentences"][:20],
                        "family": family,
                        "targetTemplates": 10,
                    }
                )
                if train_resp.status_code == 200:
                    train_data = train_resp.json()
                    new_templates = train_data.get("templatesAdded", 0)
            except Exception as e:
                print(f"Training error: {e}")
        
        return {
            "total": len(sentences),
            "parsed": results["parsed"],
            "high_confidence": results["high_confidence"],
            "coverage": results["high_confidence"] / len(sentences) if sentences else 0,
            "templates_matched": len(results["templates_matched"]),
            "new_templates": new_templates,
            "parse_results": results["parse_results"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# VSA STORAGE (via DAG)
# ═══════════════════════════════════════════════════════════════════════════════

async def store_page_chunks(
    doc_id: str,
    pages: List[str],
    parse_results: List[Dict],
) -> int:
    """Store page awareness chunks in DAG."""
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        stored = 0
        
        for i, page in enumerate(pages):
            # Build chunk vector from parse results for this page
            page_sentences = [r for r in parse_results 
                           if r["sentence"] in page]
            
            avg_confidence = sum(r["confidence"] for r in page_sentences) / len(page_sentences) if page_sentences else 0
            
            # Create awareness vector (10K dims, int4 range normalized)
            vector = [0.0] * 10000
            
            # Encode page number in first dims
            vector[i % 1000] = 1.0
            
            # Encode confidence spread across dims
            for j, r in enumerate(page_sentences[:100]):
                vector[1000 + j * 10] = r["confidence"]
            
            try:
                resp = await client.post(
                    f"{DAG_URL}/vectors/upsert",
                    params={"table": "vec10k_chunk"},
                    json={
                        "id": f"{doc_id}_page_{i:03d}",
                        "vector": vector,
                        "doc_id": doc_id,
                        "chunk_type": "page",
                        "page_num": i,
                        "summary": page[:200] + "..." if len(page) > 200 else page,
                        "metadata": {
                            "sentences": len(page_sentences),
                            "avg_confidence": avg_confidence,
                        }
                    }
                )
                if resp.status_code == 200:
                    stored += 1
            except Exception as e:
                print(f"Chunk storage error for page {i}: {e}")
        
        return stored


async def store_document_awareness(
    doc_id: str,
    title: str,
    pages: int,
    parse_results: List[Dict],
) -> str:
    """Store whole-document awareness vector."""
    
    # Build awareness vector from all parse results
    vector = [0.0] * 10000
    
    # Encode document-level statistics
    total = len(parse_results)
    high_conf = sum(1 for r in parse_results if r["confidence"] > 0.5)
    
    # Simple encoding scheme
    vector[0] = pages / 1000  # Normalized page count
    vector[1] = total / 10000  # Normalized sentence count
    vector[2] = high_conf / total if total else 0  # Coverage ratio
    
    # Encode template distribution
    templates = {}
    for r in parse_results:
        t = r.get("template", "unknown")
        templates[t] = templates.get(t, 0) + 1
    
    # Hash templates to dim indices
    for t, count in templates.items():
        idx = hash(t) % 5000 + 3000  # dims 3000-8000
        vector[idx] = count / total if total else 0
    
    awareness_id = f"{doc_id}_awareness"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                f"{DAG_URL}/vectors/upsert",
                params={"table": "vec10k_schema"},
                json={
                    "id": awareness_id,
                    "vector": vector,
                    "metadata": {
                        "type": "document_awareness",
                        "doc_id": doc_id,
                        "title": title,
                        "pages": pages,
                        "sentences": total,
                        "coverage": high_conf / total if total else 0,
                        "templates": list(templates.keys()),
                    }
                }
            )
        except Exception as e:
            print(f"Awareness storage error: {e}")
    
    return awareness_id


# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

router = APIRouter(prefix="/parse", tags=["Document Parser"])

# In-memory status tracking
_parse_jobs: Dict[str, DocumentAwareness] = {}


@router.post("/document", response_model=ParseResponse)
async def parse_document(
    file: UploadFile = File(...),
    train_templates: bool = True,
    target_family: str = "declarative.simple",
    store_chunks: bool = True,
    store_awareness: bool = True,
):
    """
    Parse uploaded document through grammar resonance → VSA awareness.
    
    Supports: PDF, TXT
    
    Flow:
    1. Extract pages from document
    2. Extract sentences from pages
    3. Parse sentences via agi-chat grammar engine
    4. Train new templates from low-confidence parses (optional)
    5. Store page chunks in DAG vec10k_chunk (optional)
    6. Store whole-document awareness in DAG vec10k_schema (optional)
    """
    
    # Generate doc ID
    content = await file.read()
    doc_hash = hashlib.sha256(content).hexdigest()[:12]
    doc_id = f"doc_{doc_hash}"
    title = file.filename or "untitled"
    
    # Extract pages
    if file.filename and file.filename.lower().endswith('.pdf'):
        pages = extract_pdf_pages(content)
    else:
        pages = extract_txt_pages(content)
    
    if not pages:
        raise HTTPException(400, "Could not extract any content from document")
    
    # Extract sentences
    sentences = extract_sentences(pages)
    
    # Create job record
    job = DocumentAwareness(
        doc_id=doc_id,
        title=title,
        pages=len(pages),
        sentences=len(sentences),
        templates_matched=0,
        new_templates_learned=0,
        parse_coverage=0,
        chunks_stored=0,
        awareness_vector_id="",
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    _parse_jobs[doc_id] = job
    
    # Parse sentences
    parse_result = await parse_sentences_batch(
        sentences,
        train_new=train_templates,
        family=target_family
    )
    
    job.templates_matched = parse_result["templates_matched"]
    job.new_templates_learned = parse_result["new_templates"]
    job.parse_coverage = parse_result["coverage"]
    
    # Store chunks
    chunks_stored = 0
    if store_chunks:
        chunks_stored = await store_page_chunks(doc_id, pages, parse_result["parse_results"])
        job.chunks_stored = chunks_stored
    
    # Store awareness
    awareness_id = ""
    if store_awareness:
        awareness_id = await store_document_awareness(
            doc_id, title, len(pages), parse_result["parse_results"]
        )
        job.awareness_vector_id = awareness_id
    
    # Mark complete
    job.completed_at = datetime.now(timezone.utc).isoformat()
    job.status = "complete"
    
    return ParseResponse(
        doc_id=doc_id,
        status="complete",
        pages=len(pages),
        sentences=len(sentences),
        parse_coverage=job.parse_coverage,
        templates_matched=job.templates_matched,
        new_templates=job.new_templates_learned,
        chunks_stored=chunks_stored,
        awareness_id=awareness_id,
    )


@router.get("/status/{doc_id}")
async def get_parse_status(doc_id: str) -> Dict[str, Any]:
    """Get status of a parsing job."""
    if doc_id not in _parse_jobs:
        raise HTTPException(404, f"Unknown doc_id: {doc_id}")
    
    return asdict(_parse_jobs[doc_id])


@router.get("/jobs")
async def list_parse_jobs() -> List[Dict[str, Any]]:
    """List all parsing jobs."""
    return [asdict(job) for job in _parse_jobs.values()]


@router.post("/train-batch")
async def train_from_sentences(
    sentences: List[str],
    family: str = "declarative.simple",
    target_templates: int = 10,
) -> Dict[str, Any]:
    """
    Train grammar templates from a batch of sentences.
    Uses Grok API via agi-chat.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{AGI_CHAT_URL}/grammar/train",
            json={
                "sentences": sentences,
                "family": family,
                "targetTemplates": target_templates,
            }
        )
        
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, f"Training failed: {resp.text}")
        
        return resp.json()


# ═══════════════════════════════════════════════════════════════════════════════
# HTML INTERFACE (Parse to VSA Awareness Button)
# ═══════════════════════════════════════════════════════════════════════════════

PARSE_UI_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Ada Parser — Grammar → VSA Awareness</title>
    <style>
        :root {
            --ada-pink: #ff6b9d;
            --ada-purple: #9b59b6;
            --ada-dark: #1a1a2e;
            --ada-light: #eee;
        }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, var(--ada-dark) 0%, #16213e 100%);
            color: var(--ada-light);
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        h1 {
            background: linear-gradient(90deg, var(--ada-pink), var(--ada-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #888;
            margin-bottom: 30px;
        }
        .upload-zone {
            border: 2px dashed var(--ada-pink);
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            transition: all 0.3s;
            cursor: pointer;
        }
        .upload-zone:hover {
            background: rgba(255, 107, 157, 0.1);
            border-color: var(--ada-purple);
        }
        .upload-zone.dragover {
            background: rgba(255, 107, 157, 0.2);
        }
        input[type="file"] {
            display: none;
        }
        .btn {
            background: linear-gradient(90deg, var(--ada-pink), var(--ada-purple));
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            color: white;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(255, 107, 157, 0.4);
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .options {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .option {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .option input[type="checkbox"] {
            margin-right: 10px;
            accent-color: var(--ada-pink);
        }
        .result {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            display: none;
        }
        .result.show {
            display: block;
        }
        .stat {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stat-label {
            color: #888;
        }
        .stat-value {
            color: var(--ada-pink);
            font-weight: bold;
        }
        .progress {
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 20px;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--ada-pink), var(--ada-purple));
            width: 0;
            transition: width 0.3s;
        }
        .status {
            text-align: center;
            margin-top: 10px;
            color: #888;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 Ada Parser</h1>
        <p class="subtitle">Grammar Triangle Resonance → VSA Awareness</p>
        
        <div class="upload-zone" id="dropZone">
            <p>📄 Drop PDF or TXT here</p>
            <p style="color: #666; font-size: 14px;">or click to browse</p>
            <input type="file" id="fileInput" accept=".pdf,.txt">
        </div>
        
        <div class="options">
            <div class="option">
                <input type="checkbox" id="trainTemplates" checked>
                <label for="trainTemplates">Train new grammar templates (via Grok API)</label>
            </div>
            <div class="option">
                <input type="checkbox" id="storeChunks" checked>
                <label for="storeChunks">Store page chunks in DAG</label>
            </div>
            <div class="option">
                <input type="checkbox" id="storeAwareness" checked>
                <label for="storeAwareness">Store document awareness vector</label>
            </div>
        </div>
        
        <button class="btn" id="parseBtn" disabled>
            ⚡ Parse to VSA Awareness
        </button>
        
        <div class="progress" id="progress" style="display: none;">
            <div class="progress-bar" id="progressBar"></div>
        </div>
        <div class="status" id="status"></div>
        
        <div class="result" id="result">
            <h3>📊 Parse Results</h3>
            <div class="stat">
                <span class="stat-label">Document ID</span>
                <span class="stat-value" id="resDocId">—</span>
            </div>
            <div class="stat">
                <span class="stat-label">Pages</span>
                <span class="stat-value" id="resPages">—</span>
            </div>
            <div class="stat">
                <span class="stat-label">Sentences</span>
                <span class="stat-value" id="resSentences">—</span>
            </div>
            <div class="stat">
                <span class="stat-label">Parse Coverage</span>
                <span class="stat-value" id="resCoverage">—</span>
            </div>
            <div class="stat">
                <span class="stat-label">Templates Matched</span>
                <span class="stat-value" id="resTemplates">—</span>
            </div>
            <div class="stat">
                <span class="stat-label">New Templates Learned</span>
                <span class="stat-value" id="resNewTemplates">—</span>
            </div>
            <div class="stat">
                <span class="stat-label">Chunks Stored</span>
                <span class="stat-value" id="resChunks">—</span>
            </div>
            <div class="stat">
                <span class="stat-label">Awareness Vector</span>
                <span class="stat-value" id="resAwareness">—</span>
            </div>
        </div>
    </div>
    
    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const parseBtn = document.getElementById('parseBtn');
        const progress = document.getElementById('progress');
        const progressBar = document.getElementById('progressBar');
        const status = document.getElementById('status');
        const result = document.getElementById('result');
        
        let selectedFile = null;
        
        // Drag and drop
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            handleFile(e.dataTransfer.files[0]);
        });
        
        fileInput.addEventListener('change', () => {
            handleFile(fileInput.files[0]);
        });
        
        function handleFile(file) {
            if (!file) return;
            selectedFile = file;
            dropZone.innerHTML = `<p>📄 ${file.name}</p><p style="color: #666; font-size: 14px;">${(file.size / 1024).toFixed(1)} KB</p>`;
            parseBtn.disabled = false;
        }
        
        // Parse
        parseBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            parseBtn.disabled = true;
            progress.style.display = 'block';
            progressBar.style.width = '10%';
            status.textContent = 'Uploading...';
            result.classList.remove('show');
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            const params = new URLSearchParams({
                train_templates: document.getElementById('trainTemplates').checked,
                store_chunks: document.getElementById('storeChunks').checked,
                store_awareness: document.getElementById('storeAwareness').checked,
            });
            
            try {
                progressBar.style.width = '30%';
                status.textContent = 'Parsing sentences...';
                
                const response = await fetch(`/parse/document?${params}`, {
                    method: 'POST',
                    body: formData,
                });
                
                progressBar.style.width = '90%';
                status.textContent = 'Processing results...';
                
                if (!response.ok) {
                    throw new Error(await response.text());
                }
                
                const data = await response.json();
                
                progressBar.style.width = '100%';
                status.textContent = 'Complete!';
                
                // Show results
                document.getElementById('resDocId').textContent = data.doc_id;
                document.getElementById('resPages').textContent = data.pages;
                document.getElementById('resSentences').textContent = data.sentences;
                document.getElementById('resCoverage').textContent = (data.parse_coverage * 100).toFixed(1) + '%';
                document.getElementById('resTemplates').textContent = data.templates_matched;
                document.getElementById('resNewTemplates').textContent = data.new_templates;
                document.getElementById('resChunks').textContent = data.chunks_stored;
                document.getElementById('resAwareness').textContent = data.awareness_id || '—';
                
                result.classList.add('show');
                
            } catch (err) {
                status.textContent = 'Error: ' + err.message;
                progressBar.style.width = '0%';
            }
            
            parseBtn.disabled = false;
        });
    </script>
</body>
</html>
"""


@router.get("/ui", response_class=HTMLResponse)
async def parse_ui():
    """Render the Parse to VSA Awareness UI."""
    return PARSE_UI_HTML
