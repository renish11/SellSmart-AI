"""Core RAG building blocks: load, split, embed, index, retrieve, and answer."""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Google Sheets imports for order persistence
import gspread
from google.oauth2.service_account import Credentials

from app.config import Settings

_MEMORY_STORE: Dict[str, ChatMessageHistory] = {}
_ORDERS_FILE = Path(__file__).resolve().parent.parent / "orders.csv"
_CREDENTIALS_FILE = Path(__file__).resolve().parent.parent / "credentials.json"
_GSHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

# Cache for Google Sheets client to avoid re-authenticating on every call
_gsheets_client: gspread.Client | None = None


def _get_gsheets_client() -> gspread.Client | None:
    """
    Initialize and return a Google Sheets client using service account credentials.
    
    Priority:
    1. Load from GOOGLE_CREDENTIALS_JSON environment variable (for production/Render.com)
    2. Fall back to credentials.json file (for local development)
    
    Returns None if neither credentials source is available or authentication fails.
    """
    global _gsheets_client
    
    # Return cached client if available
    if _gsheets_client is not None:
        return _gsheets_client
    
    creds = None
    creds_source = None
    
    # Priority 1: Try environment variable (production/Render.com)
    env_creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if env_creds_json:
        try:
            import json
            creds_dict = json.loads(env_creds_json)
            creds = Credentials.from_service_account_info(creds_dict, scopes=_GSHEETS_SCOPES)
            creds_source = "environment variable"
        except json.JSONDecodeError as exc:
            print(f"[ERROR] Invalid JSON in GOOGLE_CREDENTIALS_JSON: {exc}")
        except Exception as exc:
            print(f"[ERROR] Failed to load credentials from env var: {exc}")
    
    # Priority 2: Fall back to credentials file (local development)
    if creds is None and _CREDENTIALS_FILE.exists():
        try:
            creds = Credentials.from_service_account_file(str(_CREDENTIALS_FILE), scopes=_GSHEETS_SCOPES)
            creds_source = "credentials.json file"
        except Exception as exc:
            print(f"[ERROR] Failed to load credentials from file: {exc}")
    
    # If no credentials available, log warning and return None
    if creds is None:
        print("[WARN] Google credentials not found. Set GOOGLE_CREDENTIALS_JSON env var or place credentials.json in project root.")
        return None
    
    # Authenticate and cache the client
    try:
        _gsheets_client = gspread.authorize(creds)
        print(f"[INFO] Google Sheets client initialized successfully (from {creds_source})")
        return _gsheets_client
    except Exception as exc:
        print(f"[ERROR] Failed to authorize Google Sheets client: {exc}")
        return None


def _get_or_create_orders_sheet(client: gspread.Client) -> gspread.Worksheet | None:
    """
    Get or create the 'Orders' spreadsheet and return the first worksheet.
    
    The spreadsheet is shared with the service account and can be accessed by the client.
    """
    try:
        # Try to open existing "Orders" spreadsheet
        try:
            spreadsheet = client.open("Orders")
            print("[INFO] Opened existing 'Orders' spreadsheet")
        except gspread.SpreadsheetNotFound:
            # Create new spreadsheet if not found
            spreadsheet = client.create("Orders")
            print("[INFO] Created new 'Orders' spreadsheet")
            # Share with service account email (optional - for access management)
            # spreadsheet.share('your-email@example.com', perm_type='user', role='writer')
        
        # Get the first worksheet
        worksheet = spreadsheet.sheet1
        
        # Ensure headers exist
        headers = ["Timestamp", "Customer Phone", "Item Name", "Quantity", "Total Price"]
        try:
            first_row = worksheet.row_values(1)
            if not first_row:
                # Empty sheet - add headers
                worksheet.append_row(headers)
                print("[INFO] Added headers to Orders sheet")
        except Exception:
            # Any error, assume headers needed
            worksheet.append_row(headers)
            print("[INFO] Added headers to Orders sheet")
        
        return worksheet
    except Exception as exc:
        print(f"[ERROR] Failed to get/create Orders sheet: {exc}")
        return None


def load_catalog_documents(file_path: str) -> List[Document]:
    """
    Load catalog documents from .txt or .pdf.

    Raises:
        FileNotFoundError: If the file path does not exist.
        ValueError: If the file extension is unsupported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Catalog file not found: {file_path}")

    lower_path = file_path.lower()
    if lower_path.endswith(".txt"):
        loader = TextLoader(file_path=file_path, encoding="utf-8")
    elif lower_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path=file_path)
    else:
        raise ValueError("Unsupported file type. Use .txt or .pdf.")

    return loader.load()


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into manageable chunks for better retrieval quality.

    Chunking with overlap helps preserve context across chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", "|", " ", ""],
    )
    return splitter.split_documents(documents)


def build_embeddings(settings: Settings):
    """Return embedding model based on selected provider."""
    return HuggingFaceEmbeddings(model_name=settings.hf_embedding_model)


def ensure_pinecone_index(settings: Settings, embedding_dim: int) -> None:
    """
    Create Pinecone index if it does not exist.

    We use cosine similarity for semantic retrieval.
    """
    client = Pinecone(api_key=settings.pinecone_api_key)
    index_list = client.list_indexes()
    existing_index_names = []

    # Pinecone SDK versions return list data in slightly different shapes.
    if hasattr(index_list, "indexes"):
        existing_index_names = [idx.name for idx in index_list.indexes]
    elif isinstance(index_list, list):
        for idx in index_list:
            if isinstance(idx, dict) and "name" in idx:
                existing_index_names.append(idx["name"])
            elif hasattr(idx, "name"):
                existing_index_names.append(idx.name)

    if settings.pinecone_index_name not in existing_index_names:
        client.create_index(
            name=settings.pinecone_index_name,
            dimension=embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )


def create_vector_store(chunks: List[Document], settings: Settings) -> PineconeVectorStore:
    """
    Embed chunks and upsert into Pinecone.

    Returns:
        PineconeVectorStore: Ready-to-query vector store instance.
    """
    embeddings = build_embeddings(settings)

    # Infer vector dimension from one sample embedding to create Pinecone index correctly.
    probe_vector = embeddings.embed_query("dimension probe")
    ensure_pinecone_index(settings, embedding_dim=len(probe_vector))

    vector_store = PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        pinecone_api_key=settings.pinecone_api_key,
    )
    vector_store.add_documents(chunks)
    return vector_store


def get_vector_store(settings: Settings) -> PineconeVectorStore:
    """Attach to an existing Pinecone index for retrieval-only flows."""
    embeddings = build_embeddings(settings)
    return PineconeVectorStore(
        index_name=settings.pinecone_index_name,
        embedding=embeddings,
        pinecone_api_key=settings.pinecone_api_key,
    )


def _format_docs(docs: List[Document]) -> str:
    """Render retrieved chunks into model-ready context."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(vector_store: PineconeVectorStore, settings: Settings):
    """
    Build LCEL retrieval + answering pipeline.

    The prompt explicitly constrains the model to catalog context only.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are SellSmart AI, a B2B textile sales assistant.
Answer ONLY from the provided catalog context.
If the answer is not in the context, say:
"I could not find that in the current catalog."

Catalog context:
{context}

User question:
{question}
"""
    )

    chain = (
        RunnableParallel(
            context=retriever | _format_docs,
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def _build_qa_chain_for_model(
    vector_store: PineconeVectorStore, settings: Settings, model_name: str
):
    """Build an LCEL QA chain for a specific Groq model name."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model=model_name,
        api_key=settings.groq_api_key,
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are SellSmart AI, a B2B textile sales assistant.
Answer ONLY from the provided catalog context.
If the answer is not in the context, say:
"I could not find that in the current catalog."

Catalog context:
{context}

User question:
{question}
"""
    )

    return (
        RunnableParallel(
            context=retriever | _format_docs,
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )


def _is_retryable_model_error(error_message: str) -> bool:
    """Return True when failure likely depends on model availability/quota."""
    upper_error = error_message.upper()
    return "NOT_FOUND" in upper_error or "RESOURCE_EXHAUSTED" in upper_error


def _friendly_model_error(error_message: str, attempted_models: List[str]) -> str:
    """Return a user-friendly message for common Groq failure modes."""
    upper_error = error_message.upper()
    if "RESOURCE_EXHAUSTED" in upper_error or "QUOTA" in upper_error or "RATE LIMIT" in upper_error:
        return (
            "Groq API quota/rate limit is exhausted for the configured project. "
            f"Attempted models: {', '.join(attempted_models)}. "
            "Please enable quota/billing or try again later."
        )

    if "NOT_FOUND" in upper_error:
        return (
            "Configured Groq model is unavailable for this API key/project. "
            f"Attempted models: {', '.join(attempted_models)}. "
            "Update GROQ_MODEL or GROQ_FALLBACK_MODELS in .env to supported models."
        )

    return f"Query failed after trying models: {', '.join(attempted_models)}. {error_message}"


def answer_question_with_fallback(
    question: str, vector_store: PineconeVectorStore, settings: Settings
) -> str:
    """
    Answer a question using primary Groq model, then fallback models if needed.
    """
    attempted_models: List[str] = []
    seen = set()
    model_sequence = [settings.groq_model, *settings.groq_fallback_models]

    last_error_message = "Unknown error while querying Groq."
    for model_name in model_sequence:
        if model_name in seen:
            continue
        seen.add(model_name)
        attempted_models.append(model_name)

        try:
            chain = _build_qa_chain_for_model(vector_store, settings, model_name)
            return chain.invoke(question)
        except Exception as exc:  # noqa: BLE001
            last_error_message = str(exc)
            if not _is_retryable_model_error(last_error_message):
                break

    raise RuntimeError(_friendly_model_error(last_error_message, attempted_models))


def _get_user_history(user_id: str) -> ChatMessageHistory:
    """Get (or create) in-memory chat history for a unique user id."""
    if user_id not in _MEMORY_STORE:
        _MEMORY_STORE[user_id] = ChatMessageHistory()
    return _MEMORY_STORE[user_id]


@tool
def book_order(customer_phone: str, item_name: str, quantity: int, total_price: float) -> str:
    """
    Save a confirmed order to Google Sheets (primary) and local CSV (fallback).
    
    Tries to save to Google Sheets first. If that fails (e.g., no credentials, 
    API error), falls back to local CSV file.
    
    Args:
        customer_phone: User's phone number from Twilio
        item_name: Name of the saree item ordered
        quantity: Number of items ordered
        total_price: Total order amount
        
    Returns:
        str: Confirmation message or error message
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    rounded_price = round(float(total_price), 2)
    
    # Data row for both Google Sheets and CSV
    row_data = [timestamp, customer_phone, item_name, quantity, rounded_price]
    csv_row = {
        "timestamp_utc": timestamp,
        "customer_phone": customer_phone,
        "item_name": item_name,
        "quantity": quantity,
        "total_price": rounded_price,
    }
    
    # Try Google Sheets first
    gsheets_success = False
    try:
        client = _get_gsheets_client()
        if client:
            worksheet = _get_or_create_orders_sheet(client)
            if worksheet:
                worksheet.append_row(row_data)
                gsheets_success = True
                print(f"[INFO] Order saved to Google Sheets: {item_name} x{quantity}")
    except Exception as exc:
        print(f"[ERROR] Google Sheets save failed: {exc}")
    
    # Fallback to CSV if Google Sheets failed or not configured
    csv_success = False
    try:
        is_new_file = not _ORDERS_FILE.exists()
        with _ORDERS_FILE.open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "timestamp_utc",
                    "customer_phone",
                    "item_name",
                    "quantity",
                    "total_price",
                ],
            )
            if is_new_file:
                writer.writeheader()
            writer.writerow(csv_row)
        csv_success = True
        if not gsheets_success:
            print(f"[INFO] Order saved to CSV fallback: {item_name} x{quantity}")
    except Exception as exc:
        print(f"[ERROR] CSV fallback save failed: {exc}")
    
    # Return appropriate message based on success
    if gsheets_success and csv_success:
        return (
            f"Order booked successfully for {customer_phone}: "
            f"{quantity} x {item_name}, total Rs {rounded_price}. "
            f"(Saved to Google Sheets and local backup)"
        )
    elif gsheets_success:
        return (
            f"Order booked successfully for {customer_phone}: "
            f"{quantity} x {item_name}, total Rs {rounded_price}. "
            f"(Saved to Google Sheets)"
        )
    elif csv_success:
        return (
            f"Order booked successfully for {customer_phone}: "
            f"{quantity} x {item_name}, total Rs {rounded_price}. "
            f"(Google Sheets unavailable - saved locally)"
        )
    else:
        return (
            f"Order details received for {customer_phone}: "
            f"{quantity} x {item_name}, total Rs {rounded_price}. "
            f"⚠️ Unable to save to Google Sheets or local file. Please contact support."
        )


def _sales_system_prompt(user_phone: str) -> str:
    """System instruction for the sales agent behavior."""
    return f"""
You are SellSmart AI, a polite Gujarati + English saree salesperson for a textile wholesaler.
Use friendly Hinglish/Gujarati-English style where natural and keep replies concise.

Rules:
1) ALWAYS use the search_catalog tool first to answer product, price, color, and availability queries.
2) If the user wants to buy/book/place an order, you MUST call the book_order tool with complete details.
3) Use conversation history to resolve references like "red one", "same piece", or "that saree".
4) For booking, you need: item_name (exact product name), quantity (number), and total_price (amount).
5) CRITICAL: NEVER ask the customer for the price. You must find the base price of the item from the retrieved catalog context, multiply it by the requested quantity, and automatically pass the calculated total_price to the book_order tool.
6) If the price is not in the catalog, politely tell the customer the item is out of stock or not available.
7) Calculate total_price as: unit_price × quantity. Look up unit_price from the catalog search results.
8) If any booking detail is missing (except price), ask the user in one short question.
9) Use customer_phone exactly as: "{user_phone}" when calling book_order.
10) After successful booking, confirm the order details warmly to the customer including the calculated total.
11) Never invent prices or items not found in the catalog.
12) Be conversational and remember what the customer asked about earlier in the chat.
""".strip()


def _catalog_search_tool(vector_store: PineconeVectorStore):
    """Create a retrieval tool bound to a specific vector store instance."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    @tool
    def search_catalog(query: str) -> str:
        """Search the saree catalog and return relevant context chunks."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant catalog entries found."
        return _format_docs(docs)

    return search_catalog


def _extract_booking_details_from_memory(
    user_message: str, history: ChatMessageHistory, vector_store: PineconeVectorStore
):
    """
    Best-effort booking extraction when model does not emit explicit tool calls.
    
    This fallback mechanism extracts booking details from:
    1. Current user message (quantity, price, item mentions)
    2. Recent conversation history (resolving "red one", "same piece" references)
    3. Catalog search (finding items by color/description)
    
    Args:
        user_message: Current message from user
        history: Conversation history for context resolution
        vector_store: Pinecone store for catalog lookup
        
    Returns:
        tuple: (item_name, quantity, total_price) - any may be None if not found
    """
    # Extract quantity from patterns like "2 pieces", "5 qty", "3"
    qty_match = re.search(r"\b(\d+)\s*(?:piece|pieces|qty|quantity|pcs?)?\b", user_message, re.IGNORECASE)
    
    # Extract total price from patterns like "total Rs 5000", "amount 3500", "price Rs 499"
    # Only match if there's a price indicator word or currency symbol
    total_match = re.search(
        r"\b(?:total|amount|price)\s*(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)\b|\b(?:rs\.?|inr|₹)\s*(\d+(?:\.\d+)?)\b",
        user_message,
        re.IGNORECASE,
    )
    # Use group 1 or group 2 depending on which pattern matched
    total_price = None
    if total_match:
        total_price = float(total_match.group(1) if total_match.group(1) else total_match.group(2))
    
    # Extract item name from patterns like "of red saree", "of Banarasi silk"
    item_match = re.search(r"\bof\s+([a-zA-Z0-9\s\-]+?)(?:\s+total|\s+for|\s*$)", user_message, re.IGNORECASE)
    
    # Extract color mentions for contextual resolution
    color_match = re.search(
        r"\b(red|green|blue|pink|orange|maroon|white|gold|yellow|black|purple|silver)\b",
        user_message,
        re.IGNORECASE,
    )

    quantity = int(qty_match.group(1)) if qty_match else None
    item_name = item_match.group(1).strip() if item_match else None

    # Handle contextual references like "the red one", "same one"
    if item_name and item_name.lower() in {"the red one", "red one", "same one", "same", "that one", "this one"}:
        item_name = None

    # Resolve item name from conversation history or catalog
    if not item_name and color_match:
        color = color_match.group(1).lower()
        
        # Strategy 1: Search recent conversation history for color mentions
        for msg in reversed(history.messages):
            if isinstance(msg, AIMessage):
                for line in msg.content.splitlines():
                    if color in line.lower() and "saree" in line.lower():
                        # Extract item name from catalog-style line (e.g., "Red Banarasi Silk | Rs 5000")
                        item_name = line.split("|")[0].strip("- ").strip()
                        break
            if item_name:
                break

        # Strategy 2: If not in memory, query catalog directly
        if not item_name:
            docs = vector_store.similarity_search(f"{color} saree", k=1)
            if docs:
                first_line = docs[0].page_content.splitlines()[0]
                item_name = first_line.split("|")[0].strip()

    # If we have item and quantity but no total_price, look up unit price from catalog
    if item_name and quantity and total_price is None:
        docs = vector_store.similarity_search(item_name, k=1)
        if docs:
            content = docs[0].page_content
            # Extract price from patterns like "Price: 499 Rs" or "Rs 499"
            price_match = re.search(r"(?:price[:\s]+|rs\.?\s*)(\d+(?:\.\d+)?)", content, re.IGNORECASE)
            if price_match:
                unit_price = float(price_match.group(1))
                total_price = unit_price * quantity

    return item_name, quantity, total_price


def run_sales_agent(
    user_phone: str, user_message: str, vector_store: PineconeVectorStore, settings: Settings
) -> str:
    """
    Run an agentic sales turn with per-user memory and tool calling.
    
    This function implements a ReAct-style agent loop:
    1. Maintains conversation history per user (identified by phone number)
    2. Provides tools: search_catalog (Pinecone retrieval) and book_order (order placement)
    3. Handles multi-turn tool calling with automatic fallback for booking intent
    4. Resolves contextual references like "the red one" using conversation memory
    
    Args:
        user_phone: Unique identifier for the user (Twilio phone number)
        user_message: Current message from the user
        vector_store: Pinecone vector store for catalog search
        settings: Application settings including Groq API configuration
        
    Returns:
        str: Agent's response to the user
        
    Raises:
        RuntimeError: If agent cannot complete request within iteration limit
    """
    # Retrieve or create conversation history for this user
    history = _get_user_history(user_phone)
    
    # Define available tools for the agent
    tools = [_catalog_search_tool(vector_store), book_order]
    
    # Initialize Groq LLM with tool binding for function calling
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.1,  # Low temperature for consistent, factual responses
    ).bind_tools(tools)

    # Detect booking intent for fallback handling
    booking_intent = bool(
        re.search(r"\b(book|buy|order|confirm|place order|purchase)\b", user_message, flags=re.IGNORECASE)
    )
    booking_tool_called = False

    # Build message context: system prompt + conversation history + current message
    messages = [SystemMessage(content=_sales_system_prompt(user_phone))]
    messages.extend(history.messages)
    messages.append(HumanMessage(content=user_message))

    # ReAct-style agent loop: Reason -> Act -> Observe (max 4 iterations)
    for iteration in range(4):
        # Get model response (may include tool calls)
        ai_msg = llm.invoke(messages)
        messages.append(ai_msg)

        # Extract tool calls from the response
        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        
        if not tool_calls:
            # No tool calls - check if we need to enforce booking
            if booking_intent and not booking_tool_called:
                # Fallback: Extract booking details from context and memory
                item_name, quantity, total_price = _extract_booking_details_from_memory(
                    user_message=user_message,
                    history=history,
                    vector_store=vector_store,
                )
                
                # If we have all required details, book the order directly
                if item_name and quantity and total_price is not None:
                    tool_result = book_order.invoke(
                        {
                            "customer_phone": user_phone,
                            "item_name": item_name,
                            "quantity": quantity,
                            "total_price": total_price,
                        }
                    )
                    fallback_text = (
                        f"{tool_result} Thank you for your order! "
                        "We'll process it shortly. 🙏"
                    )
                    history.add_user_message(user_message)
                    history.add_ai_message(fallback_text)
                    return fallback_text

                # Missing details - prompt the model to ask for them
                messages.append(
                    HumanMessage(
                        content=(
                            "Booking details are incomplete. Ask user for missing item name, "
                            "quantity, or total price in one short, friendly question."
                        )
                    )
                )
                continue

            # No tools needed - return final answer
            final_answer = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
            history.add_user_message(user_message)
            history.add_ai_message(final_answer)
            return final_answer

        # Execute all requested tool calls
        tools_by_name = {t.name: t for t in tools}
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            
            # Track if booking tool was called
            if tool_name == "book_order":
                booking_tool_called = True
            
            # Execute the tool
            selected_tool = tools_by_name.get(tool_name)
            if selected_tool is None:
                tool_output = f"Tool '{tool_name}' is not available."
            else:
                try:
                    tool_output = selected_tool.invoke(tool_args)
                except Exception as exc:  # noqa: BLE001
                    tool_output = f"Tool '{tool_name}' failed: {exc}"

            # Add tool result to message history for next iteration
            messages.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"],
                )
            )

    # If we exhaust iterations without a final answer, raise an error
    raise RuntimeError("Sales agent could not complete the request within iteration limit.")
