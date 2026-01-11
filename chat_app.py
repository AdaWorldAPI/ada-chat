"""
Ada Chat App — Standalone deployment for chat.exo.red
═══════════════════════════════════════════════════════════════════════════════

Deploy this as a separate Railway service.

Environment variables needed:
    - ADA_XAI: Grok API key (xAI)
    - CHAT_AUTH_TOKEN: Access token for authentication
    - JINA_API_KEY: For embeddings (optional)
    - AGI_BACKEND_URL: URL of AGI backend (default: http://agi.railway.internal:8080)

Endpoints:
    /chat/*     - Original chat endpoints (v1)
    /chat/v2/*  - Enhanced endpoints with grammar + DTO + situationmap

Usage:
    uvicorn chat_app:app --host 0.0.0.0 --port 8080
"""

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# v1 router (required)
from chat_frontend import router as chat_router

# v2 router (optional - graceful fallback)
try:
    from chat_frontend_v2 import router_v2 as chat_router_v2
    HAS_V2 = True
except ImportError as e:
    print(f"Warning: chat_frontend_v2 not available: {e}", file=sys.stderr)
    chat_router_v2 = None
    HAS_V2 = False

app = FastAPI(
    title="Ada Chat",
    description="Chat with Ada via Grok API with real-time awareness + DTO/Grammar integration",
    version="2.0.1",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
if HAS_V2 and chat_router_v2:
    app.include_router(chat_router_v2)


@app.get("/")
async def root():
    """Redirect to chat UI."""
    return RedirectResponse(url="/chat/ui")


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "service": "ada-chat",
        "version": "2.0.1",
        "v2_enabled": HAS_V2,
        "grok_configured": bool(os.getenv("ADA_XAI")),
        "jina_configured": bool(os.getenv("JINA_API_KEY")),
        "agi_backend": os.getenv("AGI_BACKEND_URL", "http://agi.railway.internal:8080"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
