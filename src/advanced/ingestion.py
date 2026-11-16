"""Text extraction and structural indexing for the 224-page GSDR 2023 report."""
import re
import json
import time
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4
import argparse


_STRUCTURE_PATTERNS = (
    ("figure", re.compile(r"\bFIGURE\s+\d+[.\-]\d+[^.\n\r]*", re.IGNORECASE)),
    ("box", re.compile(r"\bBOX\s+\d+[.\-]\d+[^.\n\r]*", re.IGNORECASE)),
    ("table", re.compile(r"\bTABLE\s+\d+[.\-]\d+[^.\n\r]*", re.IGNORECASE)),
)


def clean_text(text: str) -> str:
    """Normalize whitespace, repeated dots, and broken hyphens."""
    t = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    t = re.sub(r'(?:\.\s*){3,}', '... ', t)
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


def _chunk_type_and_tag(text: str) -> tuple[str, str | None]:
    """Classify structures from this chunk only, avoiding page-wide tag bleed."""
    for chunk_type, pattern in _STRUCTURE_PATTERNS:
        match = pattern.search(text)
        if match:
            tag = match.group(0).strip()
            if chunk_type == "figure" and "chart" in tag.casefold():
                chunk_type = "chart"
            return chunk_type, tag
    return "text", None


def _split_long_paragraph(paragraph: str, max_chars: int) -> List[str]:
    """Split oversized paragraphs without dropping text or making huge chunks."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces = []
    current = ""
    for word in paragraph.split():
        if len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(word[i:i + max_chars] for i in range(0, len(word), max_chars))
        elif not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def parse_gsdr_pdf(pdf_path: str = "data/GSDR_2023.pdf", max_chars: int = 650) -> List[Dict[str, Any]]:
    """Extract text from the PDF and attach page/section/structure metadata."""
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    try:
        import pypdf
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires pypdf") from exc

    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"Ingesting GSDR 2023 PDF: {total_pages} total pages...")

    chunks = []
    chunk_counter = 0

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        raw_text = page.extract_text() or ''
        text = clean_text(raw_text)
        if not text or len(text) < 50:
            continue

        # Detect chapter or section
        section = "general"
        if page_num < 32:
            section = "frontmatter_exec_summary"
        elif page_num <= 55:
            section = "chapter_1_where_are_we"
        elif page_num <= 75:
            section = "chapter_2_where_are_we_headed"
        elif page_num <= 97:
            section = "chapter_3_how_do_we_get_there"
        elif page_num <= 125:
            section = "chapter_4_what_levers_accelerate"
        elif page_num <= 148:
            section = "chapter_5_transformations_through_science"
        else:
            section = "chapter_6_and_annexes"

        paragraphs = [
            piece
            for paragraph in text.split('\n\n')
            for piece in _split_long_paragraph(paragraph.strip(), max_chars)
            if piece
        ]
        cur_text = []
        cur_len = 0

        def flush_chunk() -> None:
            nonlocal chunk_counter, cur_text, cur_len
            if not cur_text:
                return
            chunk_counter += 1
            chunk_text = '\n\n'.join(cur_text)
            chunk_type, tag = _chunk_type_and_tag(chunk_text)
            chunks.append({
                'id': f"gsdr-p{page_num:03d}-c{chunk_counter:04d}",
                'text': chunk_text,
                'source_id': 'gsdr-2023',
                'source': 'data/GSDR_2023.pdf',
                'page': page_num,
                'metadata': {
                    'section': section,
                    'chunk_type': chunk_type,
                    'tag': tag,
                },
            })
            cur_text = []
            cur_len = 0

        for para in paragraphs:
            if cur_text and cur_len + 2 + len(para) > max_chars:
                flush_chunk()
            cur_text.append(para)
            cur_len += len(para) + (2 if len(cur_text) > 1 else 0)
        flush_chunk()

    print(f"Generated {len(chunks)} chunks across {total_pages} pages.")
    return chunks


def build_gsdr_index(
    chunks: List[Dict[str, Any]],
    output_dir: str = "data/gsdr_index",
    qdrant_path: str = "data/gsdr_qdrant_storage",
    overwrite: bool = False,
    batch_size: int = 100,
) -> None:
    """Save chunks and batch-index them into Qdrant collection 'gsdr_advanced'."""
    if not chunks:
        raise ValueError("chunks must not be empty")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    required = ('id', 'text', 'source_id', 'source', 'page', 'metadata')
    if any(not isinstance(chunk, dict) or any(key not in chunk for key in required) for chunk in chunks):
        raise ValueError("every chunk must contain id, text, source_id, source, page, and metadata")
    chunk_ids = [chunk.get('id') for chunk in chunks]
    if any(not isinstance(cid, str) or not cid for cid in chunk_ids) or len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("chunk IDs must be non-empty and unique")
    if any(not isinstance(chunk.get('text'), str) or not chunk['text'].strip() for chunk in chunks):
        raise ValueError("every chunk must have non-empty text")
    if any(not isinstance(chunk.get('metadata'), dict) for chunk in chunks):
        raise ValueError("every chunk metadata value must be an object")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    chunks_file = out_path / "gsdr_chunks.json"

    try:
        from fastembed import TextEmbedding
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise RuntimeError("Indexing requires fastembed and qdrant-client") from exc

    print("Embedding chunks using BAAI/bge-small-en-v1.5 and indexing into Qdrant...")
    client = QdrantClient(path=qdrant_path)
    col_name = "gsdr_advanced"
    try:
        collection_exists = client.collection_exists(col_name)
        if collection_exists and not overwrite:
            raise FileExistsError(
                f"Qdrant collection '{col_name}' already exists; pass overwrite=True to replace it"
            )

        embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
        first_batch = chunks[:batch_size]
        first_vectors = list(embedder.embed([chunk['text'] for chunk in first_batch]))
        if len(first_vectors) != len(first_batch) or not first_vectors:
            raise RuntimeError("embedding model returned an invalid first batch")
        vector_size = len(first_vectors[0])
        if vector_size < 1:
            raise RuntimeError("embedding model returned zero-dimensional vectors")

        if collection_exists:
            client.delete_collection(col_name)

        client.create_collection(
            collection_name=col_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = first_vectors if start == 0 else list(embedder.embed([chunk['text'] for chunk in batch]))
            if len(vectors) != len(batch):
                raise RuntimeError("embedding model returned a different number of vectors than chunks")
            if any(len(vector) != vector_size for vector in vectors):
                raise RuntimeError("embedding model returned inconsistent vector dimensions")
            points = [
                models.PointStruct(
                    id=start + offset + 1,
                    vector=vector.tolist(),
                    payload={
                        'chunk_id': chunk['id'],
                        'text': chunk['text'],
                        'document_id': chunk['source_id'],
                        'source': chunk['source'],
                        'page': chunk['page'],
                        'section': chunk['metadata'].get('section'),
                        'chunk_type': chunk['metadata'].get('chunk_type'),
                        'tag': chunk['metadata'].get('tag'),
                    },
                )
                for offset, (chunk, vector) in enumerate(zip(batch, vectors))
            ]
            client.upsert(collection_name=col_name, points=points)

        tmp_file = chunks_file.with_name(f"{chunks_file.name}.{uuid4().hex}.tmp")
        try:
            tmp_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + '\n', encoding="utf-8")
            tmp_file.replace(chunks_file)
        finally:
            tmp_file.unlink(missing_ok=True)
    finally:
        client.close()
    print(f"Saved {len(chunks)} chunks and indexed them into Qdrant collection '{col_name}'.")


def main():
    parser = argparse.ArgumentParser(description="Index the GSDR 2023 PDF")
    parser.add_argument('--pdf', type=Path, default=Path("data/GSDR_2023.pdf"))
    parser.add_argument(
        '--overwrite', action='store_true',
        help="replace the existing 'gsdr_advanced' Qdrant collection",
    )
    args = parser.parse_args()
    t0 = time.time()
    chunks = parse_gsdr_pdf(str(args.pdf))
    build_gsdr_index(chunks, overwrite=args.overwrite)
    print(f"Done in {time.time() - t0:.2f}s")


if __name__ == '__main__':
    main()
