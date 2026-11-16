"""Retrieval implementations for Day 7 Agentic RAG:
1. Dense Retrieval: FastEmbed (BAAI/bge-small-en-v1.5) + Qdrant Vector Store
2. Lexical Retrieval: BM25 Okapi over corpus chunks
3. Hybrid Retrieval: Reciprocal Rank Fusion (RRF) combining Dense and BM25
"""
import math
import re
import unicodedata
from collections import Counter
from heapq import nsmallest
from typing import Any, Dict, List, Optional


def tokenize(text: str) -> List[str]:
    """Tokenize normalized text into lowercase words."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    norm = unicodedata.normalize('NFC', text).casefold()
    return re.findall(r'\w+', norm)


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")


def _validate_query(query: str) -> None:
    if not isinstance(query, str):
        raise TypeError("query must be a string")


def pack_context(results: List[Dict[str, Any]], budget: int = 800) -> tuple[str, List[str]]:
    """Pack candidate chunks into context within a character budget."""
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a non-negative integer")

    blocks: List[str] = []
    ids: List[str] = []
    seen = set()
    context_chars = 0
    for c in results:
        cid = c['id']
        if not isinstance(cid, str) or not cid:
            raise ValueError("each chunk must have a non-empty string id")
        if not isinstance(c.get('text'), str):
            raise ValueError(f"chunk {cid} must have string text")
        if cid in seen:
            continue
        seen.add(cid)
        source = c.get('source', c.get('source_id', 'unknown'))
        page = c.get('page', 0)
        block = f"[{cid} | {source} | page {page}]\n{c['text']}"
        separator_chars = 2 if blocks else 0
        if context_chars + separator_chars + len(block) <= budget:
            blocks.append(block)
            ids.append(cid)
            context_chars += separator_chars + len(block)
    return '\n\n'.join(blocks), ids


class DenseRetriever:
    """Dense retriever using FastEmbed BAAI/bge-small-en-v1.5 and Qdrant."""
    def __init__(self, storage_path: str = "data/day6_qdrant_storage",
                 collection_name: str = "strategy_b",
                 model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise RuntimeError("Dense retrieval requires fastembed and qdrant-client") from exc

        self.storage_path = storage_path
        self.collection_name = collection_name
        self.embedder = TextEmbedding(model_name)
        self.client = QdrantClient(path=storage_path)
        self.models = models

    def search(self, query: str, top_k: int = 3,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        _validate_query(query)
        _validate_top_k(top_k)
        if filters is not None and not isinstance(filters, dict):
            raise TypeError("filters must be a dictionary or None")
        if top_k == 0 or not query.strip():
            return []

        query_vector = list(self.embedder.embed([query]))[0].tolist()
        qdrant_filter = None
        if filters:
            conditions = []
            for k, v in filters.items():
                conditions.append(
                    self.models.FieldCondition(key=k, match=self.models.MatchValue(value=v))
                )
            if conditions:
                qdrant_filter = self.models.Filter(must=conditions)

        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True
        )

        results = []
        for r in res.points:
            p = r.payload or {}
            results.append({
                'id': p.get('chunk_id', str(r.id)),
                'text': p.get('text', ''),
                'source_id': p.get('document_id', 'nist-ai-rmf-1.0'),
                'source': p.get('source', 'data/NIST.AI.100-1.pdf'),
                'page': p.get('page', 0),
                'metadata': {
                    'section': p.get('section'),
                    'section_title': p.get('section_title'),
                    'chunk_type': p.get('chunk_type')
                },
                'score': float(r.score),
                'strategy': 'dense'
            })
        return results

    def close(self):
        self.client.close()


class BM25Retriever:
    """BM25 Okapi lexical retriever over a collection of chunks."""
    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        if isinstance(k1, bool) or not isinstance(k1, (int, float)) or not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be a finite positive number")
        if isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.chunks = list(chunks)
        if any(not isinstance(c, dict) or not isinstance(c.get('text'), str) for c in self.chunks):
            raise ValueError("each chunk must be an object with string text")
        if any(c.get('metadata') is not None and not isinstance(c['metadata'], dict) for c in self.chunks):
            raise ValueError("chunk metadata must be an object")
        chunk_ids = [c.get('id') for c in self.chunks]
        if any(not isinstance(cid, str) or not cid for cid in chunk_ids) or len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunk IDs must be non-empty strings and unique")
        self.k1 = k1
        self.b = b
        self.doc_lens: List[int] = []
        self.doc_freqs: Counter = Counter()
        self.postings: Dict[str, List[tuple[int, int]]] = {}
        for idx, chunk in enumerate(self.chunks):
            tf = Counter(tokenize(chunk['text']))
            self.doc_lens.append(sum(tf.values()))
            for term, frequency in tf.items():
                self.doc_freqs[term] += 1
                self.postings.setdefault(term, []).append((idx, frequency))

        self.avg_doc_len = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 1.0
        self.avg_doc_len = self.avg_doc_len or 1.0
        self.doc_norms = [
            self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            for doc_len in self.doc_lens
        ]

        # Postings restrict query-time work to documents containing query terms.
        # Precompute IDF
        self.idf = {}
        total_docs = len(self.chunks)
        for term, df in self.doc_freqs.items():
            # Standard BM25 IDF with smoothing
            self.idf[term] = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 3,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        _validate_query(query)
        _validate_top_k(top_k)
        if filters is not None and not isinstance(filters, dict):
            raise TypeError("filters must be a dictionary or None")
        if top_k == 0:
            return []

        query_tf = Counter(tokenize(query))
        if not query_tf:
            return []

        candidate_ids = set()
        for term in query_tf:
            candidate_ids.update(idx for idx, _ in self.postings.get(term, ()))

        if filters:
            candidate_ids = {
                idx for idx in candidate_ids
                if all(
                    (self.chunks[idx].get('metadata') or {}).get(key) == value
                    or self.chunks[idx].get(key) == value
                    for key, value in filters.items()
                )
            }

        scores_by_id: Dict[int, float] = {}
        for term, query_frequency in query_tf.items():
            for idx, term_tf in self.postings.get(term, ()):
                if idx not in candidate_ids:
                    continue
                norm = self.doc_norms[idx]
                scores_by_id[idx] = scores_by_id.get(idx, 0.0) + (
                    query_frequency * self.idf[term] * (term_tf * (self.k1 + 1)) / (term_tf + norm)
                )

        scores = [(score, self.chunks[idx]) for idx, score in scores_by_id.items() if score > 0]

        # Keep deterministic score/ID ordering without sorting the entire match set.
        best = nsmallest(top_k, scores, key=lambda item: (-item[0], item[1]['id']))
        results = []
        for sc, c in best:
            res = dict(c)
            res['score'] = float(sc)
            res['strategy'] = 'bm25'
            results.append(res)
        return results


class HybridRetriever:
    """Hybrid Retriever fusing Dense and BM25 using Reciprocal Rank Fusion (RRF)."""
    def __init__(self, dense_retriever: DenseRetriever,
                 bm25_retriever: BM25Retriever,
                 rrf_k: int = 60):
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
            raise ValueError("rrf_k must be a positive integer")
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k
        self.chunks_by_id = {c['id']: c for c in bm25_retriever.chunks}

    def search(self, query: str, top_k: int = 3,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        _validate_query(query)
        _validate_top_k(top_k)
        if filters is not None and not isinstance(filters, dict):
            raise TypeError("filters must be a dictionary or None")
        if top_k == 0:
            return []

        # Fetch up to candidate pool
        pool_size = max(top_k * 4, 15)
        dense_hits = self.dense.search(query, top_k=pool_size, filters=filters)
        bm25_hits = self.bm25.search(query, top_k=pool_size, filters=filters)

        rrf_scores: Dict[str, float] = {}
        dense_by_id = {hit['id']: hit for hit in dense_hits}
        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit['id']
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, hit in enumerate(bm25_hits, start=1):
            cid = hit['id']
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: (-rrf_scores[cid], cid))[:top_k]
        results = []
        for cid in sorted_ids:
            chunk = self.chunks_by_id.get(cid)
            if chunk is None:
                chunk = dense_by_id[cid]
            res = dict(chunk)
            res['score'] = rrf_scores[cid]
            res['strategy'] = 'hybrid_rrf'
            results.append(res)
        return results
