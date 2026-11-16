"""Interactive CLI for the NIST AI RMF and GSDR RAG corpora."""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent import LLMController
from src.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from starter.agentic import run_agent


def bounded_int(value: str, minimum: int, maximum: int, label: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be an integer") from exc
    if not minimum <= number <= maximum:
        raise argparse.ArgumentTypeError(f"{label} must be between {minimum} and {maximum}")
    return number


def create_retriever(corpus: str):
    """Load the requested local corpus and return its retriever and dense client."""
    if corpus == "nist":
        chunks_path = ROOT_DIR / "data" / "converted" / "chunks.json"
        storage_path = ROOT_DIR / "data" / "day6_qdrant_storage"
        if not chunks_path.is_file():
            raise FileNotFoundError(f"NIST chunks were not found: {chunks_path}")
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        dense = DenseRetriever(storage_path=str(storage_path), collection_name="strategy_b")
        try:
            return HybridRetriever(dense, BM25Retriever(chunks)), dense
        except Exception:
            dense.close()
            raise

    if corpus == "gsdr":
        from src.advanced.retriever import (
            GSDRBM25Retriever,
            GSDRDenseRetriever,
            GSDRHybridRetriever,
        )

        chunks_path = ROOT_DIR / "data" / "gsdr_index" / "gsdr_chunks.json"
        storage_path = ROOT_DIR / "data" / "gsdr_qdrant_storage"
        if not chunks_path.is_file():
            raise FileNotFoundError(f"GSDR chunks were not found: {chunks_path}")
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        dense = GSDRDenseRetriever(storage_path=str(storage_path), collection_name="gsdr_advanced")
        try:
            return GSDRHybridRetriever(dense, GSDRBM25Retriever(chunks)), dense
        except Exception:
            dense.close()
            raise

    raise ValueError("corpus must be 'nist' or 'gsdr'")


def print_banner() -> None:
    print("=" * 65)
    print("         🤖 DAY 7 — AGENTIC RAG INTERACTIVE CHAT 🤖         ")
    print("  NIST AI RMF 1.0 and UN Global Sustainable Development Report")
    print("  Gemini API with an extractive offline fallback")
    print("=" * 65)


def demo_query(question: str, controller: Any, retriever: Any, top_k: int = 5,
               budget: int = 1200) -> None:
    print(f"\n[USER QUESTION]: {question}")
    print("-" * 65)
    result = run_agent(question, controller, retriever, top_k=top_k, budget=budget)

    print("\n[AGENT EXECUTION TRACE]:")
    for index, event in enumerate(result['events'], 1):
        event_type = event.get('event')
        if event_type == 'search':
            print(f"  Step {index} [SEARCH]: query='{event.get('query')}'")
            print(f"         Retrieved: {len(event.get('results', []))} candidates")
            print(f"         Context: {event.get('context_ids', [])}")
        elif event_type == 'answer':
            print(f"  Step {index} [ANSWER]: citations validated")
        elif event_type == 'invalid_or_missing_citations':
            print(f"  Step {index} [ABSTENTION]: insufficient evidence or invalid citations")
        elif event_type == 'search_limit':
            print(f"  Step {index} [ABSTENTION]: search limit reached")

    print("\n[AGENT RESPONSE]:")
    print(f"  • Status: {result['status'].upper()}")
    print(f"  • Answer: {result['answer']}")
    print(f"  • Citations: {result['citations']}")
    if controller.last_error:
        print(f"  • Controller warning: {controller.last_error}")
    print("-" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the local Day 7 RAG corpora")
    parser.add_argument("-i", "--interactive", action="store_true", help="read questions interactively")
    parser.add_argument("--corpus", choices=("nist", "gsdr"), default="nist")
    parser.add_argument("--top-k", type=lambda value: bounded_int(value, 1, 20, "top-k"), default=5)
    parser.add_argument("--budget", type=lambda value: bounded_int(value, 1, 5000, "budget"), default=1200)
    parser.add_argument("question", nargs="*", help="optional one-shot question")
    args = parser.parse_args()
    question = " ".join(args.question).strip()

    print_banner()
    try:
        retriever, dense = create_retriever(args.corpus)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"Could not load {args.corpus} corpus: {exc}\n")

    try:
        controller = LLMController()
        if question:
            demo_query(question, controller, retriever, args.top_k, args.budget)
        elif args.interactive:
            print("Enter a question; type 'exit' or 'q' to quit.")
            while True:
                try:
                    current_question = input("\nQuestion: ").strip()
                    if current_question.casefold() in ("exit", "quit", "q"):
                        break
                    if current_question:
                        demo_query(current_question, controller, retriever, args.top_k, args.budget)
                except (KeyboardInterrupt, EOFError):
                    print("\nChat ended.")
                    break
        else:
            default_question = "What are the four Core functions described in Part 2 of the AI RMF?" if args.corpus == "nist" else "What are the three phases of transformation?"
            demo_query(default_question, controller, retriever, args.top_k, args.budget)
    finally:
        dense.close()


if __name__ == '__main__':
    main()
