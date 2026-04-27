"""Interactive terminal Q&A loop for catalog retrieval."""

from __future__ import annotations

from app.config import get_settings
from app.rag_pipeline import answer_question_with_fallback, get_vector_store


def run_terminal_qa_loop() -> None:
    """Start a simple terminal loop to ask catalog questions."""
    settings = get_settings()
    vector_store = get_vector_store(settings)

    print("SellSmart AI Catalog Assistant")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            print("Please enter a valid question.")
            continue

        try:
            answer = answer_question_with_fallback(question, vector_store, settings)
            print(f"Assistant: {answer}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Query failed: {exc}\n")


if __name__ == "__main__":
    run_terminal_qa_loop()
