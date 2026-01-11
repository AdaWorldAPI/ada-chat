"""
Ada Chat App — Standalone deployment for chat.msgraph.de
═══════════════════════════════════════════════════════════════════════════════

Deploy this as a separate Railway service.

Environment variables needed:
    - ADA_XAI: Grok API key (xAI)
    - CHAT_AUTH_TOKEN: Access token for authentication
    - CHAT_AUTH_SEED: Alternative - seed phrase for auth
    - AGI_BACKEND_URL: URL of AGI backend (default: https://agi.msgraph.de)
    - CHAT_LANCE_PATH: Path for chat history LanceDB (default: /data/chat_lancedb)

Usage:
    uvicorn extension.agi_stack.chat_app:app --host 0.0.0.0 --port 8080
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from chat_frontend import router as chat_router

app = FastAPI(
    title="Ada Chat",
    description="Chat with Ada via Grok API with real-time awareness",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include chat router
app.include_router(chat_router)


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
        "grok_configured": bool(os.getenv("ADA_XAI")),
        "auth_configured": bool(os.getenv("CHAT_AUTH_TOKEN") or os.getenv("CHAT_AUTH_SEED")),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))

