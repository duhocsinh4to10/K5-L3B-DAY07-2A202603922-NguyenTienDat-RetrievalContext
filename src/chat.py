"""Interactive Chat CLI for Day 7 Agentic RAG.
Allows users to ask questions interactively against:
1. NIST AI RMF 1.0 (Core Day 7 / Day 6 corpus)
2. GSDR 2023 Report (Advanced Challenge 224-page corpus)
"""
import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.retrievers import DenseRetriever, BM25Retriever, HybridRetriever
from src.agent import LLMController
from starter.agentic import run_agent


def print_banner():
    print("=" * 65)
    print("         🤖 DAY 7 — AGENTIC RAG INTERACTIVE CHAT 🤖         ")
    print("=" * 65)
    print("Hệ thống Agentic RAG tích hợp:")
    print("  • Retriever: Hybrid RRF (Dense bge-small-en-v1.5 + BM25 Okapi)")
    print("  • Controller: Gemini API / Local semantic reasoning")
    print("  • Context Engineering: Budget packing, citation enforcement")
    print("=" * 65)


def demo_query(question: str, controller, retriever, top_k=5, budget=1200):
    print(f"\n[USER QUESTION]: {question}")
    print("-" * 65)
    res = run_agent(question, controller, retriever, top_k=top_k, budget=budget)
    
    print("\n[AGENT EXECUTION TRACE]:")
    for idx, event in enumerate(res['events'], 1):
        ev_type = event.get('event')
        if ev_type == 'search':
            print(f"  Step {idx} [TOOL CALL - SEARCH]: query='{event.get('query')}'")
            print(f"         Retrieved: {len(event.get('results', []))} candidates")
            print(f"         Context packed: {len(event.get('context_ids', []))} chunks: {event.get('context_ids')}")
        elif ev_type == 'answer':
            print(f"  Step {idx} [FINAL ANSWER]: Generated with valid citations")
        elif ev_type == 'invalid_or_missing_citations':
            print(f"  Step {idx} [ABSTENTION]: Thiếu bằng chứng hoặc trích dẫn không hợp lệ")

    print("\n[AGENT RESPONSE]:")
    print(f"  • Trạng thái: {res['status'].upper()}")
    print(f"  • Câu trả lời: {res['answer']}")
    print(f"  • Citations : {res['citations']}")
    print("-" * 65)


def main():
    print_banner()

    # Load NIST chunks and initialize retrievers
    chunks_path = Path("data/converted/chunks.json")
    if not chunks_path.exists():
        print("Lỗi: Không tìm thấy data/converted/chunks.json. Vui lòng chạy benchmark_runner trước.")
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"\nĐang tải chỉ mục NIST AI RMF ({len(chunks)} chunks)...")
    dense = DenseRetriever(storage_path="data/day6_qdrant_storage", collection_name="strategy_b")
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(dense, bm25)
    controller = LLMController()

    sample_questions = [
        "1. What are the four Core functions described in Part 2 of the AI RMF?",
        "2. What must organizational leadership foster in the GOVERN function?",
        "3. What is the specific financial penalty in USD imposed by NIST for non-compliance? (Câu hỏi test Abstention)",
        "4. How should treatment of documented AI risks be prioritized under subcategory MANAGE 1.2?",
        "5. Nhập câu hỏi tùy ý của bạn..."
    ]

    print("\nCác câu hỏi mẫu sẵn sàng để kiểm thử:")
    for s in sample_questions:
        print(f"  {s}")

    # Run interactive loop if -i or interactive terminal, else run demo
    if len(sys.argv) > 1 and sys.argv[1] not in ("-i", "--interactive"):
        user_q = " ".join(sys.argv[1:])
        demo_query(user_q, controller, hybrid)
    elif len(sys.argv) > 1 and sys.argv[1] in ("-i", "--interactive"):
        print("\n[CHẾ ĐỘ TƯƠNG TÁC] Gõ câu hỏi của bạn (gõ 'exit' hoặc 'q' để thoát):")
        while True:
            try:
                user_q = input("\n👉 Nhập câu hỏi: ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ("exit", "quit", "q"):
                    print("Tạm biệt!")
                    break
                demo_query(user_q, controller, hybrid)
            except (KeyboardInterrupt, EOFError):
                print("\nKết thúc phiên tương tác.")
                break
    else:
        # Default demo query 1
        demo_query("What are the four Core functions described in Part 2 of the AI RMF?", controller, hybrid)
        # Demo query 2 (unanswerable test)
        demo_query("What is the specific financial penalty in USD imposed by NIST for non-compliance?", controller, hybrid)

    dense.close()


if __name__ == '__main__':
    main()
