<![CDATA[# 🧵 SellSmart AI

> **AI-powered B2B textile sales assistant** — a WhatsApp-integrated chatbot that helps saree wholesalers handle product inquiries, catalog search, and order booking automatically using Retrieval-Augmented Generation (RAG).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Catalog Ingestion](#catalog-ingestion)
  - [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [WhatsApp Integration](#whatsapp-integration)
- [Order Management](#order-management)
- [Terminal Mode](#terminal-mode)
- [Deployment](#deployment)
- [Configuration Reference](#configuration-reference)
- [License](#license)

---

## Overview

SellSmart AI is a production-ready sales assistant designed for textile wholesalers. It combines a **RAG pipeline** (LangChain + Pinecone + Groq) with a **ReAct-style agentic loop** that can search a product catalog, answer customer queries in friendly Hinglish/Gujarati-English, and autonomously place orders — all through WhatsApp or a REST API.

---

## Features

| Feature | Description |
|---|---|
| 🔍 **Semantic Catalog Search** | Vector-based product search using Pinecone and HuggingFace embeddings |
| 🤖 **Agentic Sales Loop** | ReAct-style agent with tool calling (search + order booking) |
| 💬 **WhatsApp Integration** | Twilio webhook for WhatsApp Business messaging |
| 📦 **Automated Order Booking** | Orders saved to Google Sheets (primary) with local CSV fallback |
| 🧠 **Per-User Memory** | Conversation history per phone number for contextual multi-turn chat |
| 🔄 **Model Fallback** | Automatic retry across multiple Groq LLM models on failure |
| 🗣️ **Bilingual Personality** | Friendly Hinglish/Gujarati-English sales persona |
| 📄 **Multi-Format Ingestion** | Supports `.txt` and `.pdf` catalog files |
| 🏥 **Health Checks** | Built-in health and root status endpoints |

---

## Architecture

```
┌──────────────┐      ┌──────────────────┐      ┌────────────────┐
│   WhatsApp   │─────▶│  FastAPI Server   │─────▶│  Groq LLM      │
│   (Twilio)   │      │  /whatsapp        │      │  (Llama 3.x)   │
└──────────────┘      │  /query           │      └───────┬────────┘
                      │  /health          │              │
┌──────────────┐      └────────┬─────────┘              │
│  REST Client │───────────────┘                        │
└──────────────┘                            ┌───────────▼─────────┐
                                            │  ReAct Agent Loop   │
                                            │  ┌───────────────┐  │
                                            │  │ search_catalog │  │
                                            │  │ book_order     │  │
                                            │  └───────────────┘  │
                                            └───────────┬─────────┘
                                                        │
                              ┌──────────────────────────┼──────────────────┐
                              │                          │                  │
                    ┌─────────▼────────┐    ┌────────────▼───┐   ┌─────────▼──────┐
                    │   Pinecone       │    │ Google Sheets  │   │  Local CSV     │
                    │   Vector Store   │    │ (Orders)       │   │  (Fallback)    │
                    │   (Serverless)   │    └────────────────┘   └────────────────┘
                    └──────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **LLM Provider** | [Groq](https://groq.com) — Llama 3.1 / 3.3 models |
| **Orchestration** | [LangChain](https://langchain.com) + LangGraph |
| **Vector Database** | [Pinecone](https://pinecone.io) (Serverless, AWS us-east-1) |
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **Web Framework** | [FastAPI](https://fastapi.tiangolo.com) + Uvicorn |
| **Messaging** | [Twilio](https://twilio.com) WhatsApp API |
| **Order Storage** | Google Sheets API (primary) + CSV (fallback) |
| **Language** | Python 3.10+ |

---

## Project Structure

```
SellSmart-AI/
├── app/
│   ├── __init__.py          # Package marker
│   ├── config.py            # Centralized settings from .env
│   ├── ingest.py            # CLI tool to ingest catalog into Pinecone
│   ├── main.py              # FastAPI app with /query, /whatsapp, /health endpoints
│   ├── query.py             # Interactive terminal Q&A loop
│   └── rag_pipeline.py      # Core RAG: load, split, embed, retrieve, agent loop
├── .agents/                 # Pinecone agent instruction files
│   ├── PINECONE.md
│   ├── PINECONE-python.md
│   └── ...
├── catalog.txt              # Sample saree product catalog (10 items)
├── orders.csv               # Local order log (CSV fallback)
├── credentials.json         # Google service account credentials (⚠️ do not commit)
├── .env                     # Environment variables (⚠️ do not commit)
├── .env.example             # Template for environment configuration
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies
├── run.py                   # Terminal QA launcher
├── server.py                # Uvicorn production server launcher
├── AGENTS.md                # AI agent coding instructions
├── LICENSE                  # MIT License
└── README.md                # This file
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Groq API key** — [Get one free at groq.com](https://console.groq.com)
- **Pinecone API key** — [Get one free at pinecone.io](https://app.pinecone.io)
- **Twilio account** *(optional, for WhatsApp)* — [twilio.com](https://twilio.com)
- **Google Cloud service account** *(optional, for Google Sheets orders)* — [console.cloud.google.com](https://console.cloud.google.com)

### Installation

```bash
# Clone the repository
git clone https://github.com/renish11/SellSmart-AI.git
cd SellSmart-AI

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ | — | Groq API key for LLM inference |
| `GROQ_MODEL` | ❌ | `llama-3.1-8b-instant` | Primary Groq model name |
| `GROQ_FALLBACK_MODELS` | ❌ | `llama-3.3-70b-versatile` | Comma-separated fallback models |
| `PINECONE_API_KEY` | ✅ | — | Pinecone API key |
| `PINECONE_INDEX_NAME` | ❌ | `sellsmart-catalog` | Pinecone index name |
| `PINECONE_CLOUD` | ❌ | `aws` | Pinecone cloud provider |
| `PINECONE_REGION` | ❌ | `us-east-1` | Pinecone region |
| `EMBEDDING_PROVIDER` | ❌ | `huggingface` | Embedding provider (must be `huggingface`) |
| `HF_EMBEDDING_MODEL` | ❌ | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model |
| `TWILIO_ACCOUNT_SID` | ❌ | — | Twilio account SID |
| `TWILIO_AUTH_TOKEN` | ❌ | — | Twilio auth token |
| `TWILIO_WHATSAPP_NUMBER` | ❌ | `whatsapp:+14155238886` | Twilio WhatsApp sender |
| `GOOGLE_CREDENTIALS_JSON` | ❌ | — | Service account JSON string (for production) |

### Catalog Ingestion

Before querying, you must ingest your product catalog into Pinecone:

```bash
# Ingest the default catalog.txt
python -m app.ingest

# Or specify a custom catalog file
python -m app.ingest --file path/to/your_catalog.pdf
```

The catalog file uses a pipe-delimited format:
```
Royal Red Banarasi Silk Saree | Color: Red | Price: 499 Rs
Crimson Wedding Kanjivaram Saree | Color: Red | Price: 899 Rs
...
```

### Running the Server

```bash
# Development (with auto-reload)
uvicorn app.main:app --reload --port 8000

# Production (via server.py — reads PORT from environment, defaults to 10000)
python server.py
```

---

## API Endpoints

### `GET /` — Root Status
```bash
curl http://localhost:8000/
# {"status": "I am awake!"}
```

### `GET /health` — Health Check
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### `POST /query` — Catalog Query (JSON)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me red sarees under 500 Rs"}'

# {"answer": "Here are the red sarees under 500 Rs from our catalog..."}
```

### `POST /whatsapp` — Twilio WhatsApp Webhook
Receives form-data from Twilio with `From` and `Body` fields. Returns TwiML XML response. Configure this URL as your Twilio WhatsApp webhook.

---

## WhatsApp Integration

1. Set up a [Twilio WhatsApp Sandbox](https://www.twilio.com/docs/whatsapp/sandbox) or approved WhatsApp Business number.
2. Set the webhook URL to `https://your-domain.com/whatsapp` (POST).
3. Add your Twilio credentials to `.env`.
4. Messages from customers are processed through the agentic RAG pipeline and responses are returned as TwiML.

**Conversation flow:**
- Customer asks about products → Agent searches catalog via Pinecone
- Customer wants to order → Agent extracts details and calls `book_order` tool
- Agent resolves contextual references ("the red one") using conversation memory

---

## Order Management

Orders are persisted using a **dual-write strategy**:

| Priority | Storage | When |
|---|---|---|
| 1️⃣ | **Google Sheets** | Service account credentials available |
| 2️⃣ | **Local CSV** (`orders.csv`) | Fallback when Sheets is unavailable |

**Google Sheets Setup:**
1. Create a service account in Google Cloud Console.
2. Enable Google Sheets API and Google Drive API.
3. Download the JSON key file as `credentials.json` (for local dev) or set `GOOGLE_CREDENTIALS_JSON` env var (for production).
4. The app automatically creates an "Orders" spreadsheet on first use.

**CSV Format:**
```csv
timestamp_utc,customer_phone,item_name,quantity,total_price
2026-04-27T07:06:35+00:00,whatsapp:+919999888877,Royal Red Banarasi Silk Saree,2,998.0
```

---

## Terminal Mode

For testing without a server, use the interactive terminal Q&A loop:

```bash
python run.py
```

This starts a CLI session where you can ask catalog questions directly. Type `exit` or `quit` to stop.

---

## Deployment

### Render.com

The project is configured for [Render](https://render.com) deployment:

- **Start command:** `python server.py`
- `server.py` reads `PORT` from the environment (Render sets this automatically, defaults to `10000`).
- Set all required environment variables in Render's dashboard.
- For Google Sheets, set `GOOGLE_CREDENTIALS_JSON` as a Render environment variable containing the full JSON string of your service account key.

### Docker (Optional)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "server.py"]
```

---

## Configuration Reference

### Model Fallback Chain

SellSmart AI supports automatic model fallback. If the primary model fails (quota exhausted, model unavailable), it tries fallback models in order:

```
GROQ_MODEL → GROQ_FALLBACK_MODELS[0] → GROQ_FALLBACK_MODELS[1] → ...
```

### RAG Pipeline Parameters

| Parameter | Value | Description |
|---|---|---|
| Chunk size | 500 chars | Document splitting chunk size |
| Chunk overlap | 80 chars | Overlap between chunks |
| Retrieval top-k | 4 | Number of similar chunks retrieved |
| LLM temperature | 0.1 | Low temperature for factual responses |
| Vector metric | Cosine | Similarity metric for Pinecone |
| Agent max iterations | 4 | ReAct loop iteration limit |

### Catalog Format

The catalog supports pipe-delimited text files (`.txt`) and PDF files (`.pdf`). Each product entry should include:
- **Product name**
- **Color**
- **Price** (in Rs)

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2026 Renish Nakrani**
]]>