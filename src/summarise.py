"""Map-reduce summarisation of long meeting transcripts."""

import asyncio
import logging
import random
import time

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from . import prompts

logger = logging.getLogger(__name__)

CONCURRENCY_LIMIT = 20
MAX_RETRIES = 4
BASE_BACKOFF = 1.0


async def _summarise_chunk(
    llm: ChatGoogleGenerativeAI,
    chunk: str,
    task_id: int,
    sem: asyncio.Semaphore,
) -> str:
    """Summarise a single transcript chunk under the concurrency semaphore.

    Errors return a visible marker rather than raising: one failed chunk
    should not discard an otherwise complete transcript, but it must not
    silently vanish either, or the reduce stage produces a summary that
    looks complete and isn't.
    """
    async with sem:
        chain = ChatPromptTemplate.from_template(prompts.CHUNK_SUMMARY) | llm | StrOutputParser()
        for attempt in range(MAX_RETRIES):
            try:
                result = await chain.ainvoke({"chunk": chunk})
                logger.debug("chunk %d summarised", task_id)
                return result
            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    logger.error("chunk %d failed after %d attempts: %s",
                                 task_id, MAX_RETRIES, exc)
                    return f"[segment {task_id} unavailable: {type(exc).__name__}]"
                # Jittered backoff. Without jitter, every rate-limited chunk
                # retries in lockstep and the batch re-collides each round.
                delay = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning("chunk %d attempt %d failed (%s), retrying in %.1fs",
                               task_id, attempt + 1, type(exc).__name__, delay)
                await asyncio.sleep(delay)


async def map_summaries(llm: ChatGoogleGenerativeAI, docs: list[Document]) -> str:
    """Map stage: summarise every chunk concurrently, return them concatenated."""
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [
        _summarise_chunk(llm, doc.page_content, i, sem)
        for i, doc in enumerate(docs)
        if doc.page_content and not doc.page_content.isspace()
    ]

    logger.info("summarising %d chunks (concurrency %d)", len(tasks), CONCURRENCY_LIMIT)
    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    logger.info("map stage finished in %.1fs", time.perf_counter() - start)

    return "\n\n".join(results)


async def reduce_briefing(llm: ChatGoogleGenerativeAI, combined: str) -> str:
    """Reduce stage: fold partial summaries into an executive briefing."""
    chain = ChatPromptTemplate.from_template(prompts.BRIEFING) | llm | StrOutputParser()
    return await chain.ainvoke({"combined_summary": combined})


async def reduce_action_items(llm: ChatGoogleGenerativeAI, combined: str) -> str:
    """Reduce stage: extract only assigned actions and their owners."""
    chain = ChatPromptTemplate.from_template(prompts.ACTION_ITEMS) | llm | StrOutputParser()
    return await chain.ainvoke({"combined_summary": combined})
