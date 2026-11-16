# Source Code — Day 7 Agentic RAG

Dự án triển khai hoàn chỉnh giải pháp Agentic Retrieval-Augmented Generation (RAG) và Thử thách nâng cao cho Day 7.

## Cấu trúc thư mục

- `src/retrievers.py`: Triển khai 3 phương pháp retrieval cốt lõi; BM25 dùng inverted postings để chỉ chấm các chunk có từ truy vấn:
  1. `DenseRetriever`: FastEmbed `BAAI/bge-small-en-v1.5` + Qdrant Vector Store (kế thừa Day 6).
  2. `BM25Retriever`: Lexical BM25 Okapi retriever.
  3. `HybridRetriever`: Kết hợp Dense + BM25 sử dụng Reciprocal Rank Fusion (RRF $k=60$).
- `src/agent.py`: Agent Controller tích hợp Google Gemini API (`gemini-3.5-flash-lite`), hỗ trợ tìm kiếm nhiều lượt, citation `[chunk-id]` và từ chối trả lời. Khi API offline, fallback chỉ trích câu từ context nếu có từ khóa liên quan; nếu không thì từ chối. Cảnh báo được đưa ra trong CLI/Web.
- `src/evaluation.py`: Đánh giá xếp hạng retrieval (Hit@1, Hit@3, Context Evidence Rate).
- `src/benchmark_runner.py`: Chạy retrieval benchmark offline và agent benchmark với Gemini; lưu trace cùng review template chưa gán nhãn. Cần tự chấm câu trả lời trước khi dùng `starter.agent_metrics`.
- `src/advanced/`:
  - `ingestion.py`: Trích xuất văn bản và metadata cấu trúc cho tài liệu 224 trang `data/GSDR_2023.pdf`, sau đó batch-embed và lập chỉ mục Qdrant.
  - `retriever.py`: Hybrid retriever chuyên biệt cho GSDR 2023.
  - `benchmark_data.py`: Tập câu hỏi benchmark gồm 48 câu (18 Dev, 30 Test) bao phủ Bảng, Ảnh, Biểu đồ, Multi-hop, Text, Abstention.
  - `runner.py`: Chạy agent trên benchmark và báo retrieval/abstention diagnostics. Độ đúng của câu trả lời vẫn cần người chấm xác nhận.

## Hướng dẫn chạy và tái lập kết quả

### 0. Mở giao diện Web trên Google Chrome (Khuyên dùng)
Bạn có thể mở giao diện đồ họa Web UI trực quan trên Google Chrome:
```bash
# Cách 1: Nhấp đúp hoặc chạy file batch (Tự động mở Google Chrome)
.\run_web.bat

# Cách 2: Khởi động qua Python
python src/web_server.py
```
Giao diện sẽ chạy tại `http://127.0.0.1:7860`, cho phép:
- Chuyển đổi giữa 2 kho tri thức: NIST AI RMF (Core) và UN GSDR 2023 (Nâng cao 224 trang).
- Xem câu trả lời có kèm Citation chip clickable `[chunk-id]`.
- Bảng tra cứu bên phải hiển thị chi tiết các Chunks được trích xuất (điểm RRF, trang, loại layout) và vết thực thi (Agent trace).

### 0.1. Mở giao diện AI Agent tương tác qua dòng lệnh (CLI)
```bash
# Chạy file batch tương tác
.\chat.bat

# Hoặc qua Python CLI
python src/chat.py -i
```

### 1. Chạy Unit Tests
```bash
python -m unittest discover -s tests -v
```

### 2. Chạy Benchmark phần cơ bản (85 điểm)
```bash
python -m src.benchmark_runner
```
Kết quả được xuất ra:
- `results/benchmark_summary.json`
- `results/dev_baseline_traces.json`, `results/dev_selected_traces.json`, `results/test_baseline_traces.json`, `results/test_selected_traces.json`
- Các file `*_review_template.json` có trường `null` cần được chấm thủ công.

### 3. Chạy Thử thách nâng cao (+15 điểm bonus)
```bash
# Bước 1: Trích xuất và lập chỉ mục PDF 224 trang
python -m src.advanced.ingestion --overwrite

# Bước 2: Chạy đánh giá toàn bộ 30 câu Test
python -m src.advanced.runner
```
Kết quả được xuất ra:
- `results/advanced_summary.json`
- `results/advanced_dev_traces.json` / `results/advanced_test_traces.json` (bản sao tương thích: `results/advanced_traces.json`)

Lệnh ingestion cần `--overwrite` khi collection `gsdr_advanced` đã tồn tại. Kết quả trace cũ trong `results/` không được xem là nhãn đánh giá đã xác nhận; hãy chạy lại agent thật và tự chấm review template trước khi báo cáo độ đúng.
