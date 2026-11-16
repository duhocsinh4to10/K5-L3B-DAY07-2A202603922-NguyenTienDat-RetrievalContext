"""Run the GSDR benchmark with a real controller and post-hoc diagnostics."""
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from src.advanced.retriever import (
    GSDRBM25Retriever,
    GSDRDenseRetriever,
    GSDRHybridRetriever,
)
from src.agent import LLMController
from src.evaluation import hit
from starter.agentic import run_agent


def run_advanced_agent(
    question: str,
    controller: LLMController,
    retriever: Any,
    top_k: int = 5,
    budget: int = 1500,
) -> Dict[str, Any]:
    """Answer from retrieved context; benchmark labels never enter this call."""
    started = time.perf_counter()
    result = run_agent(
        question=question,
        controller=controller,
        retriever=retriever,
        max_searches=4,
        top_k=top_k,
        budget=budget,
    )
    result['latency_ms'] = (time.perf_counter() - started) * 1000.0
    return result


def add_posthoc_diagnostics(
    case: Dict[str, Any], result: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare retrieved chunks with gold after generation, without grading answers."""
    gold = case.get('gold', [])
    search_events = [event for event in result['events'] if event.get('event') == 'search']
    retrieved = {
        chunk['id']: chunk
        for event in search_events
        for chunk in event.get('results', [])
    }
    gold_chunk_ids = {
        chunk_id for chunk_id, chunk in retrieved.items() if hit(chunk, gold)
    }
    context_ids = {
        chunk_id
        for event in search_events
        for chunk_id in event.get('context_ids', [])
    }
    citations = result.get('citations', [])

    result['retrieved_gold_chunk_ids'] = sorted(gold_chunk_ids)
    result['gold_evidence_retrieved'] = bool(gold_chunk_ids) if case['answerable'] else None
    result['gold_evidence_in_context'] = bool(gold_chunk_ids & context_ids) if case['answerable'] else None
    result['answerability_status_match'] = (
        (result['status'] == 'answered') == case['answerable']
    )
    result['citation_support_candidates'] = [
        chunk_id in gold_chunk_ids if case['answerable'] else None
        for chunk_id in citations
    ]
    result['answer_grade'] = None
    return result


def summarize(cases: List[Dict[str, Any]], traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize retrieval and abstention signals; answer correctness is manual."""
    trace_by_id = {trace['query_id']: trace for trace in traces}
    answerable = [case for case in cases if case['answerable']]
    unanswerable = [case for case in cases if not case['answerable']]
    answerable_traces = [trace_by_id[case['id']] for case in answerable]
    unanswerable_traces = [trace_by_id[case['id']] for case in unanswerable]

    evidence_retrieved = sum(trace['gold_evidence_retrieved'] for trace in answerable_traces)
    evidence_in_context = sum(trace['gold_evidence_in_context'] for trace in answerable_traces)
    correct_answerable_status = sum(trace['status'] == 'answered' for trace in answerable_traces)
    correct_unanswerable_status = sum(trace['status'] == 'abstain' for trace in unanswerable_traces)

    def rate(count: int, total: int) -> float:
        return count / total if total else 0.0

    return {
        'total': len(cases),
        'answered': sum(trace['status'] == 'answered' for trace in traces),
        'abstained': sum(trace['status'] == 'abstain' for trace in traces),
        'answerable_cases': {
            'n': len(answerable),
            'gold_evidence_retrieved': evidence_retrieved,
            'gold_evidence_retrieval_rate': rate(evidence_retrieved, len(answerable)),
            'gold_evidence_in_context': evidence_in_context,
            'gold_evidence_context_rate': rate(evidence_in_context, len(answerable)),
            'answered': correct_answerable_status,
            'answer_status_rate': rate(correct_answerable_status, len(answerable)),
        },
        'unanswerable_cases': {
            'n': len(unanswerable),
            'correct_abstentions': correct_unanswerable_status,
            'abstention_rate': rate(correct_unanswerable_status, len(unanswerable)),
        },
        'answer_quality': 'requires manual review',
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main() -> None:
    benchmark_path = Path("data/advanced_benchmark.json")
    chunks_path = Path("data/gsdr_index/gsdr_chunks.json")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    cases = json.loads(benchmark_path.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    dev_cases = [case for case in cases if case.get('split') == 'dev']
    test_cases = [case for case in cases if case.get('split') == 'test']

    dense = GSDRDenseRetriever()
    controller = LLMController()
    try:
        bm25 = GSDRBM25Retriever(chunks)
        hybrid = GSDRHybridRetriever(dense, bm25)
        evaluations = {}
        traces_by_split = {}

        for split, split_cases in (("dev", dev_cases), ("test", test_cases)):
            traces = []
            for case in split_cases:
                result = run_advanced_agent(case['question'], controller, hybrid)
                result = add_posthoc_diagnostics(case, result)
                traces.append({
                    'query_id': case['id'],
                    'split': split,
                    'group': case.get('group'),
                    'question': case['question'],
                    **result,
                })
            evaluations[split] = summarize(split_cases, traces)
            traces_by_split[split] = traces
            _write_json(results_dir / f"advanced_{split}_traces.json", traces)
            print(f"{split}: {evaluations[split]}")

        summary = {
            'document': 'Global Sustainable Development Report (GSDR) 2023',
            'dev_evaluation': evaluations['dev'],
            'test_evaluation': evaluations['test'],
            'manual_review_required': True,
        }
        _write_json(results_dir / "advanced_summary.json", summary)
        # Preserve the expected test trace filename used by the report template.
        _write_json(results_dir / "advanced_traces.json", traces_by_split["test"])
    finally:
        dense.close()


if __name__ == '__main__':
    main()
