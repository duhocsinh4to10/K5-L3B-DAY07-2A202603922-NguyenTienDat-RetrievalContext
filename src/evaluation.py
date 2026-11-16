"""Evaluation metrics and benchmarking logic for Day 7."""
import statistics
import time
from typing import Any, Dict, List, Optional
from src.retrievers import pack_context


def hit(chunk: Dict[str, Any], gold: List[Dict[str, Any]]) -> bool:
    """Check if chunk text contains evidence on the same page and source."""
    for g in gold:
        evidence = g.get('evidence')
        if not isinstance(evidence, str) or not evidence:
            continue
        if (chunk.get('source_id') == g.get('source_id') and
                chunk.get('page') == g.get('page') and
                evidence in chunk.get('text', '')):
            return True
    return False


def _empty_result() -> Dict[str, Any]:
    """Keep the result schema stable when a split has no answerable cases."""
    return {
        'n': 0,
        'hit1': 0,
        'hit1_rate': 0.0,
        'hit3': 0,
        'hit3_rate': 0.0,
        'context_hit': 0,
        'context_evidence_rate': 0.0,
        'avg_context_chars': 0.0,
        'median_latency_ms': 0.0,
        'mean_latency_ms': 0.0,
        'traces': [],
    }


def evaluate_retriever_on_cases(
    retriever: Any,
    cases: List[Dict[str, Any]],
    top_k: int = 3,
    budget: int = 800,
    use_case_filters: bool = False
) -> Dict[str, Any]:
    """Evaluate retrieval and context evidence metrics on answerable cases."""
    answerable_cases = [c for c in cases if c.get('answerable', True)]
    n = len(answerable_cases)
    if n == 0:
        return _empty_result()

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a non-negative integer")

    hit1_count = 0
    hit3_count = 0
    context_hit_count = 0
    context_lengths = []
    latencies_ms = []
    traces = []

    for c in answerable_cases:
        qid = c['id']
        question = c['question']
        gold = c.get('gold', [])
        filters = c.get('filter') if use_case_filters else None

        t0 = time.perf_counter()
        results = retriever.search(question, top_k=top_k, filters=filters)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        # Mark hits on results
        marked_results = []
        for r in results:
            mr = dict(r)
            mr['hit'] = hit(r, gold)
            marked_results.append(mr)

        # Ranking hits
        is_hit1 = bool(marked_results and marked_results[0]['hit'])
        is_hit3 = any(mr['hit'] for mr in marked_results[:3])
        if is_hit1:
            hit1_count += 1
        if is_hit3:
            hit3_count += 1

        # Pack context
        context_str, context_ids = pack_context(results, budget=budget)
        context_lengths.append(len(context_str))

        # Check if context contains evidence
        # Evidence is present in context if any packed chunk has hit == True
        context_id_set = set(context_ids)
        is_context_hit = any(r['hit'] and r['id'] in context_id_set for r in marked_results)
        if is_context_hit:
            context_hit_count += 1

        traces.append({
            'query_id': qid,
            'question': question,
            'gold': gold,
            'filters': filters,
            'top_k': top_k,
            'budget': budget,
            'results': marked_results,
            'hit1': is_hit1,
            'hit3': is_hit3,
            'context_hit': is_context_hit,
            'context_length': len(context_str),
            'context_ids': context_ids,
            'latency_ms': elapsed_ms
        })

    return {
        'n': n,
        'hit1': hit1_count,
        'hit1_rate': hit1_count / n,
        'hit3': hit3_count,
        'hit3_rate': hit3_count / n,
        'context_hit': context_hit_count,
        'context_evidence_rate': context_hit_count / n,
        'avg_context_chars': sum(context_lengths) / n if n else 0,
        'median_latency_ms': statistics.median(latencies_ms) if latencies_ms else 0,
        'mean_latency_ms': statistics.mean(latencies_ms) if latencies_ms else 0,
        'traces': traces
    }
