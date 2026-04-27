"""Configuration utilities for environment variables and settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    """Centralized runtime settings loaded from .env."""

    groq_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    embedding_provider: str
    hf_embedding_model: str
    groq_model: str
    groq_fallback_models: List[str]


def get_settings() -> Settings:
    """Load settings from environment and validate required values."""
    settings = Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "sellsmart-catalog"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "huggingface").lower(),
        hf_embedding_model=os.getenv(
            "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        groq_fallback_models=[
            model.strip()
            for model in os.getenv(
                "GROQ_FALLBACK_MODELS", "llama-3.3-70b-versatile"
            ).split(",")
            if model.strip()
        ],
    )

    missing = []
    if not settings.groq_api_key:
        missing.append("GROQ_API_KEY")
    if not settings.pinecone_api_key:
        missing.append("PINECONE_API_KEY")

    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Missing required environment variables: {missing_str}. "
            "Create a .env file using .env.example."
        )

    if settings.embedding_provider != "huggingface":
        raise ValueError("EMBEDDING_PROVIDER must be 'huggingface' when using Groq.")

    return settings
