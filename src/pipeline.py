"""Top-level orchestration: process a transcript, then query it."""

import asyncio
import logging
import os
from dataclasses import dataclass

from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from .index import build_store, chunk_transcript, load_store
from .summarise import map_summaries, reduce_action_items, reduce_briefing

logger = logging.getLogger(__name__)

MODEL = "gemini-2.0-flash"
TEMPERATURE = 0.3
DEFAULT_STORE = "./chroma_db"


@dataclass
class MeetingOutput:
    briefing: str
    action_items: str
    store: Chroma


def get_llm() -> ChatGoogleGenerativeAI:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatGoogleGenerativeAI(model=MODEL, temperature=TEMPERATURE, google_api_key=key)


async def process(path: str, persist_dir: str = DEFAULT_STORE) -> MeetingOutput:
    """Summarise a transcript and index it for later Q&A.

    Indexing and the map stage are independent, so they run concurrently —
    embedding is CPU-bound and summarisation is network-bound, and overlapping
    them removes the embedding time from the critical path entirely.
    """
    llm = get_llm()
    docs = chunk_transcript(path)

    combined, store = await asyncio.gather(
        map_summaries(llm, docs),
        build_store(docs, persist_dir),
    )

    # Both reduce prompts consume the same combined summary, so they also
    # run concurrently rather than sequentially.
    briefing, actions = await asyncio.gather(
        reduce_briefing(llm, combined),
        reduce_action_items(llm, combined),
    )

    return MeetingOutput(briefing=briefing, action_items=actions, store=store)


def process_sync(path: str, persist_dir: str = DEFAULT_STORE) -> MeetingOutput:
    return asyncio.run(process(path, persist_dir))


def open_store(persist_dir: str = DEFAULT_STORE) -> Chroma:
    return load_store(persist_dir)
