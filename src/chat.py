"""Retrieval-augmented Q&A over indexed transcripts."""

import logging

from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from . import prompts

logger = logging.getLogger(__name__)

RETRIEVAL_K = 5


async def rewrite_query(llm: ChatGoogleGenerativeAI, query: str, history: str) -> str:
    """Resolve conversational references before retrieval.

    Embedding a follow-up like "what did he say about it?" retrieves noise —
    the pronouns carry no signal. Rewriting against the history first is what
    makes multi-turn retrieval work at all.
    """
    if not history.strip():
        return query
    chain = ChatPromptTemplate.from_template(prompts.QUERY_REWRITE) | llm | StrOutputParser()
    rewritten = await chain.ainvoke({"query": query, "chat_history": history})
    logger.info("rewrote %r -> %r", query, rewritten.strip())
    return rewritten.strip()


async def answer(
    llm: ChatGoogleGenerativeAI,
    store: Chroma,
    query: str,
    history: str = "",
) -> str:
    """Answer a question strictly from retrieved transcript context."""
    resolved = await rewrite_query(llm, query, history)

    retriever = store.as_retriever(
        search_type="similarity", search_kwargs={"k": RETRIEVAL_K}
    )
    docs = retriever.invoke(resolved)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    chain = ChatPromptTemplate.from_template(prompts.GROUNDED_ANSWER) | llm | StrOutputParser()
    return await chain.ainvoke({"context": context, "query": resolved})
