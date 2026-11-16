"""Example agent loop; connect a real model and the Day 6 retriever via adapters.

Adapters are intentionally not provided: Day 6 has no fixed database schema and
model providers differ. Never use fake adapters to report benchmark quality.
"""
import re
from typing import Any, Protocol
try:
    from baseline import pack_context
except ImportError:
    from starter.baseline import pack_context


class Controller(Protocol):
    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Call a real model with a search tool; return normalized action.

        Search: {'action': 'search', 'query': '...'}.
        Final: {'action': 'answer', 'answer': '...', 'citations': ['chunk-id']}.
        Citations must also appear as [chunk-id] in the answer text.
        """
        ...


class Retriever(Protocol):
    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        """Query the real Day 6 vector store with a compatible query embedding.

        Return each chunk's id, text, source_id, source, page, metadata and score.
        The query embedding model and dimension must match the document index.
        """
        ...


def run_agent(question: str, controller: Controller, retriever: Retriever, *,
              filters: dict[str, Any] | None = None, max_searches: int = 4,
              top_k: int = 3, budget: int = 800) -> dict[str, Any]:
    """Return answer, citation IDs and trace; no benchmark gold enters this loop.

    A final answer without valid citations, an empty context, and a search limit
    lead to abstention. This only checks citation IDs, not semantic grounding.
    Provider/retriever errors propagate; no silent mock fallback.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError('question must be a non-empty string')
    limits = (max_searches, top_k, budget)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in limits):
        raise ValueError('question, max_searches, top_k hoặc budget không hợp lệ')
    if filters is not None and not isinstance(filters, dict):
        raise TypeError('filters must be a dictionary or None')

    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    events: list[dict[str, Any]] = []
    searches_done = 0
    for _ in range(max_searches + 1):
        action = controller.decide(question, history)
        if not isinstance(action, dict):
            raise TypeError('controller action must be a dictionary')
        if action.get('action') == 'search':
            if searches_done >= max_searches:
                return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events + [{'event': 'search_limit'}]}
            query = action.get('query')
            if not isinstance(query, str) or not query.strip():
                raise ValueError('search query phải là chuỗi không rỗng')
            results = retriever.search(query, filters=filters, top_k=top_k)
            if not isinstance(results, list):
                raise TypeError('retriever results must be a list')
            if len(results) > top_k:
                raise ValueError('retriever phải trả tối đa top_k chunk ID duy nhất')
            result_ids = []
            for result in results:
                if not isinstance(result, dict):
                    raise TypeError('retriever results must contain dictionaries')
                cid = result.get('id')
                if not isinstance(cid, str) or not cid:
                    raise ValueError('retrieved chunk IDs must be non-empty strings')
                if not isinstance(result.get('text'), str):
                    raise ValueError(f'retrieved chunk {cid} must have string text')
                result_ids.append(cid)
            if len(set(result_ids)) != len(result_ids):
                raise ValueError('retriever must return unique chunk IDs per search')

            searches_done += 1
            for result in results:
                if result['id'] not in candidate_ids:
                    candidates.append(result)
                    candidate_ids.add(result['id'])
            context, ids = pack_context(candidates, budget=budget)
            chunks = {c['id']: c for c in candidates if c['id'] in ids}
            events.append({'event': 'search', 'query': query, 'filter': filters, 'results': results,
                           'context': context, 'context_ids': ids})
            # Send only the latest cumulative context; avoid multiplying the budget.
            history = [{'tool': 'search', 'queries': [e['query'] for e in events if e['event'] == 'search'],
                        'context': context, 'context_ids': ids}]
        elif action.get('action') == 'answer':
            answer, citations = action.get('answer'), action.get('citations')
            if not isinstance(answer, str) or not isinstance(citations, list) or any(not isinstance(i, str) for i in citations):
                raise ValueError('answer/citations không hợp lệ')
            answer_citations = set(re.findall(r'\[([^\[\]]+)\]', answer))
            valid = bool(
                answer.strip()
                and citations
                and len(citations) == len(set(citations))
                and answer_citations == set(citations)
                and all(i in chunks and f'[{i}]' in answer for i in citations)
            )
            if valid:
                return {'status': 'answered', 'answer': answer, 'citations': citations, 'events': events + [{'event': 'answer'}]}
            return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events + [{'event': 'invalid_or_missing_citations'}]}
        else:
            raise ValueError('action phải là search hoặc answer')
    return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events}
