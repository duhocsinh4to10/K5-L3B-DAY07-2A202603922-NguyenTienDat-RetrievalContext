"""Web Server for Day 7 Agentic RAG Web Explorer.
Serves a sleek interactive web chat UI and opens Google Chrome automatically.
"""
import sys
import os
import json
import time
import socket
import threading
import subprocess
import webbrowser
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.retrievers import DenseRetriever, BM25Retriever, HybridRetriever
from src.advanced.retriever import GSDRDenseRetriever, GSDRBM25Retriever, GSDRHybridRetriever
from src.agent import LLMController
from starter.agentic import run_agent

HTML_PATH = ROOT_DIR / "src" / "web" / "index.html"


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads for fast concurrent response."""
    daemon_threads = True


class RAGAppServer:
    """Manages retrievers and agent controllers."""
    def __init__(self):
        self.controller = LLMController()
        self.nist_hybrid = None
        self.gsdr_hybrid = None
        self.dense_nist = None
        self.dense_gsdr = None
        self._init_corpora()

    def _init_corpora(self):
        print("Đang khởi tạo các kho tri thức và chỉ mục RAG...")
        # 1. NIST AI RMF
        nist_chunks_file = ROOT_DIR / "data" / "converted" / "chunks.json"
        if nist_chunks_file.exists():
            with open(nist_chunks_file, "r", encoding="utf-8") as f:
                nist_chunks = json.load(f)
            self.dense_nist = DenseRetriever(
                storage_path=str(ROOT_DIR / "data" / "day6_qdrant_storage"),
                collection_name="strategy_b"
            )
            bm25_nist = BM25Retriever(nist_chunks)
            self.nist_hybrid = HybridRetriever(self.dense_nist, bm25_nist)
            print(f"  ✓ NIST AI RMF 1.0 sẵn sàng ({len(nist_chunks)} chunks).")
        else:
            print("  ⚠️ Không tìm thấy data/converted/chunks.json")

        # 2. GSDR 2023
        gsdr_chunks_file = ROOT_DIR / "data" / "gsdr_index" / "gsdr_chunks.json"
        if gsdr_chunks_file.exists():
            with open(gsdr_chunks_file, "r", encoding="utf-8") as f:
                gsdr_chunks = json.load(f)
            self.dense_gsdr = GSDRDenseRetriever(
                storage_path=str(ROOT_DIR / "data" / "gsdr_qdrant_storage"),
                collection_name="gsdr_advanced"
            )
            bm25_gsdr = GSDRBM25Retriever(gsdr_chunks)
            self.gsdr_hybrid = GSDRHybridRetriever(self.dense_gsdr, bm25_gsdr)
            print(f"  ✓ GSDR 2023 Advanced sẵn sàng ({len(gsdr_chunks)} chunks).")
        else:
            print("  ⚠️ Không tìm thấy data/gsdr_index/gsdr_chunks.json")

    def query(self, question: str, corpus: str = "nist", top_k: int = 5, budget: int = 1200) -> dict:
        retriever = self.gsdr_hybrid if corpus == "gsdr" else self.nist_hybrid
        if retriever is None:
            return {
                "status": "abstain",
                "answer": f"Lỗi: Kho tri thức '{corpus}' chưa được tải.",
                "citations": [],
                "events": [],
                "retrieved_chunks": []
            }

        start = time.perf_counter()
        result = run_agent(
            question=question,
            controller=self.controller,
            retriever=retriever,
            max_searches=4,
            top_k=top_k,
            budget=budget
        )
        latency_ms = round((time.perf_counter() - start) * 1000)

        # Extract all retrieved chunks from search events
        retrieved_chunks = []
        seen_ids = set()
        for ev in result.get("events", []):
            if ev.get("event") == "search":
                for chunk in ev.get("results", []):
                    cid = chunk.get("id")
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        retrieved_chunks.append({
                            "id": cid,
                            "score": float(chunk.get("score", 0.0)),
                            "page": chunk.get("page", "N/A"),
                            "source": chunk.get("source", ""),
                            "metadata": chunk.get("metadata", {}),
                            "text": chunk.get("text", "")
                        })

        result["latency_ms"] = latency_ms
        result["retrieved_chunks"] = retrieved_chunks
        return result

    def close(self):
        if self.dense_nist:
            self.dense_nist.close()
        if self.dense_gsdr:
            self.dense_gsdr.close()


APP_ENGINE = None


class RAGRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP Handler serving Single Page Web UI and REST API."""

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(HTML_PATH, "rb") as f:
                self.wfile.write(f.read())
        elif self.path == "/api/status":
            self.send_json({
                "status": "online",
                "gemini": bool(APP_ENGINE.controller.gemini_key),
                "corpora": {
                    "nist": APP_ENGINE.nist_hybrid is not None,
                    "gsdr": APP_ENGINE.gsdr_hybrid is not None
                }
            })
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/chat":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_json({"error": f"Invalid JSON: {str(e)}"}, status=400)
                return

            question = data.get("question", "").strip()
            corpus = data.get("corpus", "nist")
            top_k = int(data.get("top_k", 5))
            budget = int(data.get("budget", 1200))

            if not question:
                self.send_json({"error": "Question is required"}, status=400)
                return

            res = APP_ENGINE.query(question=question, corpus=corpus, top_k=top_k, budget=budget)
            self.send_json(res)
        else:
            self.send_error(404, "Not Found")

    def send_json(self, data: dict, status: int = 200):
        resp_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(resp_bytes)

    def log_message(self, format, *args):
        # Clean logging
        sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] {format % args}\n")


def open_in_chrome(url: str):
    """Attempts to launch Google Chrome explicitly on Windows, falls back to default browser."""
    chrome_candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    for path in chrome_candidates:
        if os.path.isfile(path):
            try:
                subprocess.Popen([path, url])
                print(f"Đã mở Google Chrome thành công: {url}")
                return
            except Exception:
                pass
    webbrowser.open(url)
    print(f"Đã mở trình duyệt mặc định: {url}")


def find_available_port(start_port: int = 7860, max_attempts: int = 10) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port


def main():
    global APP_ENGINE
    port = find_available_port(7860)
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])

    print("=" * 65)
    print("      🚀 KHỞI ĐỘNG GIAO DIỆN WEB VINAI DAY 7 TRÊN CHROME 🚀      ")
    print("=" * 65)
    APP_ENGINE = RAGAppServer()

    server_address = ('127.0.0.1', port)
    httpd = ThreadedHTTPServer(server_address, RAGRequestHandler)
    url = f"http://127.0.0.1:{port}"

    print(f"\n[SERVER ONLINE] Máy chủ web đang chạy tại:")
    print(f"👉 URL: {url}")
    print("-" * 65)
    print("Đang tự động mở Google Chrome...")

    # Open Chrome after 0.6s
    threading.Timer(0.6, open_in_chrome, args=[url]).start()

    print("Nhấn Ctrl+C để dừng máy chủ bất kỳ lúc nào.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nĐang dừng máy chủ web...")
    finally:
        httpd.server_close()
        APP_ENGINE.close()
        print("Đã đóng kết nối an toàn.")


if __name__ == '__main__':
    main()
