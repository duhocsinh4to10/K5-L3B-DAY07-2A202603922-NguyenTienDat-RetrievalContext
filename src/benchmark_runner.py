"""Run retrieval benchmarks and collect agent traces for manual review."""
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from src.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from src.evaluation import evaluate_retriever_on_cases
from src.agent import LLMController
from starter.agentic import run_agent


def run_agent_benchmark(
    controller: LLMController,
    retriever: Any,
    cases: List[Dict[str, Any]],
    top_k: int = 3,
    budget: int = 800,
    use_filters: bool = False,
) -> List[Dict[str, Any]]:
    """Run the agent without passing benchmark answers or gold evidence to it."""
    traces = []
    for case in cases:
        filters = case.get('filter') if use_filters else None
        started = time.perf_counter()
        result = run_agent(
            question=case['question'],
            controller=controller,
            retriever=retriever,
            filters=filters,
            max_searches=4,
            top_k=top_k,
            budget=budget,
        )
        traces.append({
            'query_id': case['id'],
            'question': case['question'],
            'status': result['status'],
            'answer': result['answer'],
            'citations': result['citations'],
            'events': result['events'],
            'latency_ms': (time.perf_counter() - started) * 1000.0,
        })
    return traces


def summarize_agent_traces(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Report observable behavior only; answer quality needs human review."""
    return {
        'n': len(traces),
        'answered': sum(trace['status'] == 'answered' for trace in traces),
        'abstained': sum(trace['status'] == 'abstain' for trace in traces),
        'search_calls': sum(
            event.get('event') == 'search'
            for trace in traces
            for event in trace.get('events', [])
        ),
        'manual_answer_review_required': True,
    }


def make_review_template(
    cases: List[Dict[str, Any]], traces: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Create unlabelled review rows; never infer human judgments from retrieval."""
    trace_by_id = {trace['query_id']: trace for trace in traces}
    reviews = []
    for case in cases:
        trace = trace_by_id[case['id']]
        row: Dict[str, Any] = {
            'query_id': case['id'],
            'citation_support': [None] * len(trace['citations']),
        }
        if case['answerable']:
            row['answer_grade'] = None
        else:
            row['abstention_correct'] = None
        reviews.append(row)
    return reviews


def _retrieval_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'n': result['n'],
        'hit1': result['hit1'],
        'hit1_rate': result['hit1_rate'],
        'hit3': result['hit3'],
        'hit3_rate': result['hit3_rate'],
        'context_hit': result['context_hit'],
        'context_evidence_rate': result['context_evidence_rate'],
        'avg_context_chars': result['avg_context_chars'],
        'median_latency_ms': result['median_latency_ms'],
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main() -> None:
    print("=" * 60)
    print("DAY 7 — RETRIEVAL & AGENTIC BENCHMARK EXECUTION")
    print("=" * 60)

    chunks = json.loads(Path("data/converted/chunks.json").read_text(encoding="utf-8"))
    cases = json.loads(Path("data/benchmark_cases.json").read_text(encoding="utf-8"))
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    dev_cases = [case for case in cases if case.get('split') == 'dev']
    test_cases = [case for case in cases if case.get('split') == 'test']
    print(f"Loaded {len(chunks)} chunks, {len(dev_cases)} dev cases, {len(test_cases)} test cases.")

    dense = DenseRetriever(storage_path="data/day6_qdrant_storage", collection_name="strategy_b")
    controller = LLMController()
    try:
        bm25 = BM25Retriever(chunks=chunks)
        hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)
        retrieval_benchmarks: Dict[str, Any] = {}

        configs = [
            ("dev_dense_k3", dense, dev_cases, 3, 800, False),
            ("dev_bm25_k3", bm25, dev_cases, 3, 800, False),
            ("dev_hybrid_k3", hybrid, dev_cases, 3, 800, False),
            ("dev_hybrid_k5", hybrid, dev_cases, 5, 1200, False),
            ("dev_hybrid_filtered_k5", hybrid, dev_cases, 5, 1200, True),
            ("test_dense_k3", dense, test_cases, 3, 800, False),
            ("test_hybrid_filtered_k5", hybrid, test_cases, 5, 1200, True),
        ]
        for name, retriever, split_cases, top_k, budget, use_filters in configs:
            result = evaluate_retriever_on_cases(
                retriever, split_cases, top_k=top_k, budget=budget,
                use_case_filters=use_filters,
            )
            retrieval_benchmarks[name] = _retrieval_summary(result)
            print(
                f"[{name}] Hit@1 {result['hit1']}/{result['n']}, "
                f"Hit@3 {result['hit3']}/{result['n']}, "
                f"context evidence {result['context_hit']}/{result['n']}"
            )

        agent_runs = {}
        for name, retriever, split_cases, top_k, budget, use_filters in (
            ("dev_baseline", dense, dev_cases, 3, 800, False),
            ("dev_selected", hybrid, dev_cases, 5, 1200, True),
            ("test_baseline", dense, test_cases, 3, 800, False),
            ("test_selected", hybrid, test_cases, 5, 1200, True),
        ):
            traces = run_agent_benchmark(
                controller, retriever, split_cases,
                top_k=top_k, budget=budget, use_filters=use_filters,
            )
            agent_runs[name] = summarize_agent_traces(traces)
            _write_json(results_dir / f"{name}_traces.json", traces)
            _write_json(
                results_dir / f"{name}_review_template.json",
                make_review_template(split_cases, traces),
            )

        outputs = {
            'retrieval_benchmarks': retrieval_benchmarks,
            'agent_runs': agent_runs,
            'review_note': (
                "Review template entries are null until manually labelled. "
                "Use starter.agent_metrics after completing them."
            ),
        }
        _write_json(results_dir / "benchmark_summary.json", outputs)
        print("Benchmarks and unlabelled review templates written to results/.")
    finally:
        dense.close()


if __name__ == '__main__':
    main()
