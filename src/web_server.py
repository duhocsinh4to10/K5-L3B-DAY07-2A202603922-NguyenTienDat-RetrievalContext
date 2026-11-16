"""Web Server for Day 7 Agentic RAG Web Explorer.
Serves a sleek interactive web chat UI and opens Google Chrome automatically.
"""
import sys
import os
import json
import time
import socket
import logging
import argparse
import threading
import subprocess
import webbrowser
from pathlib import Path
from typing import Any
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
MAX_REQUEST_BYTES = 1_048_576

# Module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Content-Security-Policy header value
CSP_VALUE = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self';"
)


def parse_chat_request(body: bytes) -> dict:
    """Decode and validate an HTTP chat request before retrieving any data."""
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid JSON request body") from exc
    return validate_chat_request(data)


def validate_chat_request(data: Any) -> dict:
    """Validate already decoded chat input, including direct server calls."""
    if not isinstance(data, dict):
        raise ValueError("JSON request body must be an object")

    question = data.get("question")
    if not isinstance(question, str) or not question.strip() or len(question) > 4000:
        raise ValueError("question must be a non-empty string of at most 4000 characters")
    corpus = data.get("corpus", "nist")
    if corpus not in ("nist", "gsdr"):
        raise ValueError("corpus must be 'nist' or 'gsdr'")

    top_k = data.get("top_k", 5)
    budget = data.get("budget", 1200)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise ValueError("top_k must be an integer from 1 to 20")
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 5000:
        raise ValueError("budget must be an integer from 1 to 5000")
    return {"question": question.strip(), "corpus": corpus, "top_k": top_k, "budget": budget}


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
        self.init_errors = {}
        self._init_corpora()

    def _init_corpora(self):
        print("Đang khởi tạo các kho tri thức và chỉ mục RAG...")
        # 1. NIST AI RMF
        nist_chunks_file = ROOT_DIR / "data" / "converted" / "chunks.json"
        try:
            if not nist_chunks_file.exists():
                raise FileNotFoundError(f"Không tìm thấy {nist_chunks_file}")
            with open(nist_chunks_file, "r", encoding="utf-8") as f:
                nist_chunks = json.load(f)
            self.dense_nist = DenseRetriever(
                storage_path=str(ROOT_DIR / "data" / "day6_qdrant_storage"),
                collection_name="strategy_b",
            )
            self.nist_hybrid = HybridRetriever(self.dense_nist, BM25Retriever(nist_chunks))
            print(f"  ✓ NIST AI RMF 1.0 sẵn sàng ({len(nist_chunks)} chunks).")
        except Exception as exc:
            if self.dense_nist is not None:
                self.dense_nist.close()
                self.dense_nist = None
            self.nist_hybrid = None
            self.init_errors["nist"] = str(exc)
            logger.warning("Không thể tải kho NIST: %s", exc)
            print(f"  ⚠️ Không thể tải kho NIST: {exc}")

        # 2. GSDR 2023
        gsdr_chunks_file = ROOT_DIR / "data" / "gsdr_index" / "gsdr_chunks.json"
        try:
            if not gsdr_chunks_file.exists():
                raise FileNotFoundError(f"Không tìm thấy {gsdr_chunks_file}")
            with open(gsdr_chunks_file, "r", encoding="utf-8") as f:
                gsdr_chunks = json.load(f)
            self.dense_gsdr = GSDRDenseRetriever(
                storage_path=str(ROOT_DIR / "data" / "gsdr_qdrant_storage"),
                collection_name="gsdr_advanced",
            )
            self.gsdr_hybrid = GSDRHybridRetriever(self.dense_gsdr, GSDRBM25Retriever(gsdr_chunks))
            print(f"  ✓ GSDR 2023 Advanced sẵn sàng ({len(gsdr_chunks)} chunks).")
        except Exception as exc:
            if self.dense_gsdr is not None:
                self.dense_gsdr.close()
                self.dense_gsdr = None
            self.gsdr_hybrid = None
            self.init_errors["gsdr"] = str(exc)
            logger.warning("Không thể tải kho GSDR: %s", exc)
            print(f"  ⚠️ Không thể tải kho GSDR: {exc}")

    def query(self, question: str, corpus: str = "nist", top_k: int = 5, budget: int = 1200) -> dict:
        try:
            request = validate_chat_request({
                "question": question,
                "corpus": corpus,
                "top_k": top_k,
                "budget": budget,
            })
        except ValueError as exc:
            logger.error("Invalid query parameters: %s", exc)
            return {
                "status": "error",
                "answer": f"Tham số không hợp lệ: {exc}",
                "citations": [],
                "events": [],
                "retrieved_chunks": [],
                "error": str(exc),
            }

        retriever = self.gsdr_hybrid if request["corpus"] == "gsdr" else self.nist_hybrid
        if retriever is None:
            return {
                "status": "abstain",
                "answer": f"Lỗi: Kho tri thức '{corpus}' chưa được tải.",
                "citations": [],
                "events": [],
                "retrieved_chunks": [],
                "error": self.init_errors.get(corpus),
            }

        start = time.perf_counter()
        try:
            result = run_agent(
                question=request["question"],
                controller=self.controller,
                retriever=retriever,
                max_searches=4,
                top_k=request["top_k"],
                budget=request["budget"],
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000)
            logger.error("Agent run_agent raised an exception for question=%r corpus=%s: %s", question, corpus, exc)
            return {
                "status": "error",
                "answer": f"Lỗi trong quá trình xử lý: {exc}",
                "citations": [],
                "events": [],
                "retrieved_chunks": [],
                "error": str(exc),
                "latency_ms": latency_ms,
            }

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
        if self.controller.last_error:
            result["controller_warning"] = self.controller.last_error
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
            if not HTML_PATH.is_file():
                self.send_error(404, "Web UI file not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Security-Policy", CSP_VALUE)
            self.end_headers()
            with open(HTML_PATH, "rb") as f:
                self.wfile.write(f.read())
        elif self.path == "/api/status":
            if APP_ENGINE is None:
                self.send_json({"status": "starting"}, status=503)
                return
            self.send_json({
                "status": "online",
                "gemini": APP_ENGINE.controller.gemini_available,
                "corpora": {
                    "nist": APP_ENGINE.nist_hybrid is not None,
                    "gsdr": APP_ENGINE.gsdr_hybrid is not None
                },
                "corpus_errors": APP_ENGINE.init_errors,
            })
        elif self.path == "/api/usage":
            if APP_ENGINE is None:
                self.send_json({"error": "Agent server is not ready"}, status=503)
                return
            if hasattr(APP_ENGINE.controller, "usage_report"):
                report = APP_ENGINE.controller.usage_report()
            else:
                report = {
                    "call_count": 0,
                    "total_prompt_tokens": 0,
                    "total_candidates_tokens": 0,
                    "estimated_cost_usd": 0.0,
                }
            self.send_json(report)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/api/chat":
            t0 = time.perf_counter()
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            corpus = "unknown"
            question_len = 0
            response_status = 500

            try:
                content_len = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self.send_json({"error": "Content-Length must be a valid integer"}, status=400)
                return
            if content_len < 1:
                self.send_json({"error": "Request body is required"}, status=400)
                return
            if content_len > MAX_REQUEST_BYTES:
                self.send_json({"error": "Request body is too large"}, status=413)
                return
            body = self.rfile.read(content_len)
            if len(body) != content_len:
                self.send_json({"error": "Request body ended unexpectedly"}, status=400)
                return
            try:
                request = parse_chat_request(body)
                corpus = request.get("corpus", "unknown")
                question_len = len(request.get("question", ""))
            except ValueError as exc:
                self.send_json({"error": str(exc)}, status=400)
                return
            if APP_ENGINE is None:
                self.send_json({"error": "Agent server is not ready"}, status=503)
                return
            try:
                result = APP_ENGINE.query(**request)
                response_status = 200
                self.send_json(result)
            except Exception as exc:
                response_status = 502
                self.send_json({"error": f"Agent request failed: {exc}"}, status=502)
            finally:
                latency_ms = round((time.perf_counter() - t0) * 1000)
                logger.info(
                    "chat | ts=%s corpus=%s q_len=%d status=%d latency_ms=%d",
                    timestamp, corpus, question_len, response_status, latency_ms,
                )
        else:
            self.send_error(404, "Not Found")

    def do_PUT(self):
        self._method_not_allowed()

    def do_DELETE(self):
        self._method_not_allowed()

    def do_PATCH(self):
        self._method_not_allowed()

    def _method_not_allowed(self):
        self.send_json({"error": "Method Not Allowed"}, status=405)

    def send_json(self, data: dict, status: int = 200):
        resp_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.send_header("Content-Security-Policy", CSP_VALUE)
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
                # Use list form so Windows handles spaces in path correctly
                subprocess.Popen([path, url])
                print(f"Đã mở Google Chrome thành công: {url}")
                return
            except Exception:
                pass
    webbrowser.open(url)
    print(f"Đã mở trình duyệt mặc định: {url}")


def find_available_port(start_port: int = 7860, max_attempts: int = 10) -> int:
    if isinstance(start_port, bool) or not isinstance(start_port, int) or not 1 <= start_port <= 65535:
        raise ValueError("start_port must be between 1 and 65535")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    for port in range(start_port, start_port + max_attempts):
        if port > 65535:
            break
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port found from {start_port} through {min(65535, start_port + max_attempts - 1)}")


def parse_args(argv=None):
    """Parse CLI arguments. Supports --port <int>."""
    parser = argparse.ArgumentParser(description="VinAI Day 7 — Agentic RAG Web Server")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        metavar="PORT",
        help="Port to listen on (default: auto-select starting from 7860)",
    )
    return parser.parse_args(argv)


def main():
    global APP_ENGINE
    args = parse_args()

    if args.port is not None:
        if not 1 <= args.port <= 65535:
            raise ValueError("--port must be between 1 and 65535")
        port = args.port
    else:
        port = find_available_port(7860)

    print("=" * 65)
    print("      🚀 KHỞI ĐỘNG GIAO DIỆN WEB VINAI DAY 7 TRÊN CHROME 🚀      ")
    print("=" * 65)
    APP_ENGINE = RAGAppServer()

    server_address = ('127.0.0.1', port)
    url = f"http://127.0.0.1:{port}"
    httpd = None
    try:
        httpd = ThreadedHTTPServer(server_address, RAGRequestHandler)
        print(f"\n[SERVER ONLINE] Máy chủ web đang chạy tại:\n👉 URL: {url}")
        print("-" * 65)
        print("Đang tự động mở Google Chrome...")

        browser_timer = threading.Timer(0.6, open_in_chrome, args=[url])
        browser_timer.daemon = True
        browser_timer.start()
        print("Nhấn Ctrl+C để dừng máy chủ bất kỳ lúc nào.\n")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nĐang dừng máy chủ web...")
    finally:
        if httpd is not None:
            httpd.server_close()
        APP_ENGINE.close()
        APP_ENGINE = None
        print("Đã đóng kết nối an toàn.")


if __name__ == '__main__':
    main()
