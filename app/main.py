"""FastAPI app skeleton for SellSmart AI RAG service."""

from __future__ import annotations

import os
from fastapi import FastAPI, Form, HTTPException, Response
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse

from app.config import get_settings
from app.rag_pipeline import get_vector_store, run_sales_agent


app = FastAPI(title="SellSmart AI", version="0.1.0")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.get("/health")
def health_check() -> dict:
    """Basic health check endpoint."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_catalog(payload: QueryRequest) -> QueryResponse:
    """
    Query endpoint for future channel integrations (e.g., WhatsApp webhook handler).
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        settings = get_settings()
        vector_store = get_vector_store(settings)
        answer = run_sales_agent(
            user_phone="api_user",
            user_message=question,
            vector_store=vector_store,
            settings=settings,
        )
        return QueryResponse(answer=answer)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@app.post("/whatsapp")
def whatsapp_webhook(
    From: str = Form(default=""),  # noqa: N803 (Twilio's field name)
    Body: str = Form(default=""),  # noqa: N803 (Twilio's field name)
) -> Response:
    """
    Twilio WhatsApp webhook endpoint.

    Twilio sends message payloads as form-data with fields like:
    - From: sender number, e.g., "whatsapp:+9198xxxxxx"
    - Body: user text message
    """
    sender = From.strip()
    user_message = Body.strip()

    twiml = MessagingResponse()

    if not user_message:
        twiml.message("Please send a valid message so I can help you.")
        return Response(content=str(twiml), media_type="application/xml")

    try:
        settings = get_settings()
        vector_store = get_vector_store(settings)
        answer = run_sales_agent(
            user_phone=sender or "unknown_user",
            user_message=user_message,
            vector_store=vector_store,
            settings=settings,
        )

        # Optional lightweight tracing in server logs.
        print(f"WhatsApp query from {sender or 'unknown'}: {user_message}")
        twiml.message(answer)
    except Exception:  # noqa: BLE001
        twiml.message(
            "We are currently busy processing requests. "
            "Please try again in a moment."
        )

    return Response(content=str(twiml), media_type="application/xml")


import uvicorn

if __name__ == "__main__":
    # Render.com provides PORT env var, default to 10000 if not set
    port = int(os.environ.get("PORT", 10000))
    
    # Run the FastAPI app - app.main:app means app/main.py file with app object
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
