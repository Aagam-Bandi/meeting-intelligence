"""Meeting transcript summarisation and retrieval.

Imports are lazy so the evaluation harness and its scoring functions can be
used — and tested — without the Gemini SDK installed.
"""

__all__ = [
    "MeetingOutput", "get_llm", "open_store", "process", "process_sync",
    "answer", "rewrite_query",
    "EvalReport", "evaluate", "evaluate_sync",
    "score_action_items", "score_coverage", "score_grounding",
]

_LAZY = {
    "MeetingOutput": "pipeline", "get_llm": "pipeline", "open_store": "pipeline",
    "process": "pipeline", "process_sync": "pipeline",
    "answer": "chat", "rewrite_query": "chat",
    "EvalReport": "evaluate", "evaluate": "evaluate", "evaluate_sync": "evaluate",
    "score_action_items": "evaluate", "score_coverage": "evaluate",
    "score_grounding": "evaluate",
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(f".{_LAZY[name]}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
