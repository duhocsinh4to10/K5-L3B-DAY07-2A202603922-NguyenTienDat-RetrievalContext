"""Evaluation metrics and benchmarking logic for Day 7.

Metric definitions (see BENCHMARK.md):

* **Hit@1** – The top-ranked retrieved chunk contains the gold evidence string
  on the correct source/page.
* **Evidence Hit@3** (hit3) – At least one of the top-3 retrieved chunks
  contains the gold evidence string on the correct source/page.
* **Context Evidence Rate** – The gold evidence is present in at least one
  chunk that survived the context-packing budget (see
  :func:`~src.retrievers.pack_context`).
* **Avg latency ms** – Mean per-query retrieval latency in **milliseconds**.
"""
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
    use_case_filters: bool = False,
    score_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate retrieval and context-evidence metrics on answerable cases.

    Metrics returned (see BENCHMARK.md for definitions):

    * ``hit1`` / ``hit1_rate`` – **Hit@1**: top-1 chunk contains gold evidence.
    * ``hit3`` / ``hit3_rate`` – **Evidence Hit@3**: any top-3 chunk contains
      gold evidence.
    * ``context_hit`` / ``context_evidence_rate`` – **Context Evidence Rate**:
      gold evidence survives the context budget (budget is in **chars**).
    * ``mean_latency_ms`` / ``median_latency_ms`` – per-query retrieval latency
      in **milliseconds**.

    Args:
        retriever:        Any retriever with a ``search(query, top_k, filters)``
                          method.  If it also exposes
                          ``search_with_score_threshold``, that method is used
                          when *score_threshold* > 0.
        cases:            List of evaluation case dicts.
        top_k:            Number of chunks to retrieve per query (default 3).
        budget:           Context packing budget in **characters** (default 800).
        use_case_filters: Whether to pass per-case ``filter`` dicts to the
                          retriever (default False).
        score_threshold:  Minimum retriever score; chunks below this threshold
                          are removed before evaluation.  Forwarded to
                          ``search_with_score_threshold`` when the retriever
                          supports it; otherwise ignored (default 0.0).

    Returns:
        Dict with aggregated metrics and per-query ``traces``.
    """
    answerable_cases = [c for c in cases if c.get('answerable', True)]
    n = len(answerable_cases)
    if n == 0:
        return _empty_result()

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a non-negative integer")

    # Decide which search method to call
    use_threshold_method = (
        score_threshold > 0.0
        and hasattr(retriever, 'search_with_score_threshold')
        and callable(getattr(retriever, 'search_with_score_threshold'))
    )

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
        if use_threshold_method:
            results = retriever.search_with_score_threshold(
                question, top_k=top_k, min_score=score_threshold, filters=filters
            )
        else:
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


def format_benchmark_table(results_dict: Dict[str, Dict[str, Any]]) -> str:
    """Format a markdown benchmark table from a dict of evaluation results.

    Produces one row per configuration with the following columns:

    * **Config** – the configuration name (key in *results_dict*).
    * **Hit@1** – ``n/N (pct%)`` as defined in BENCHMARK.md.
    * **Hit@3** – ``n/N (pct%)`` as defined in BENCHMARK.md (Evidence Hit@3).
    * **Context** – ``n/N (pct%)`` Context Evidence Rate.
    * **Avg latency ms** – mean per-query retrieval latency in **milliseconds**.

    Args:
        results_dict: Mapping of config name → result dict as returned by
                      :func:`evaluate_retriever_on_cases`.

    Returns:
        A multi-line markdown string ready for printing or embedding in a report.

    Example::

        print(format_benchmark_table({"dense": dense_results, "bm25": bm25_results}))
    """
    header = (
        "| Config | Hit@1 n/N (%) | Hit@3 n/N (%) | Context n/N (%) | Avg latency ms |\n"
        "|--------|--------------|--------------|-----------------|----------------|\n"
    )
    rows = []
    for name, r in results_dict.items():
        n = r.get('n', 0)
        h1 = r.get('hit1', 0)
        h3 = r.get('hit3', 0)
        ctx = r.get('context_hit', 0)
        lat = r.get('mean_latency_ms', 0.0)

        def _fmt(count: int, total: int) -> str:
            pct = (count / total * 100) if total else 0.0
            return f"{count}/{total} ({pct:.1f}%)"

        rows.append(
            f"| {name} | {_fmt(h1, n)} | {_fmt(h3, n)} | {_fmt(ctx, n)} | {lat:.1f} |"
        )
    return header + "\n".join(rows)
