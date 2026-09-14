"""Chunking and vector store construction."""

import asyncio
import logging
import os
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1250
CHUNK_OVERLAP = 250
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def chunk_transcript(path: str) -> list[Document]:
    """Split a transcript into overlapping chunks.

    Overlap matters more here than in prose: a decision is often stated across
    a speaker turn boundary, and a hard cut loses the attribution or the
    outcome depending on which side it falls.
    """
    documents = TextLoader(path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = splitter.split_documents(documents)
    logger.info("split transcript into %d chunks", len(docs))
    return docs


async def build_store(docs: list[Document], persist_dir: str) -> Chroma:
    """Build a fresh Chroma store, replacing any existing one at this path.

    Chroma.from_documents is blocking, so it runs in a thread to stay awaitable
    alongside the summarisation stage.
    """
    if await asyncio.to_thread(os.path.exists, persist_dir):
        logger.info("removing existing store at %s", persist_dir)
        await asyncio.to_thread(shutil.rmtree, persist_dir)

    logger.info("building vector store")
    return await asyncio.to_thread(
        Chroma.from_documents, docs, get_embeddings(), persist_directory=persist_dir
    )


def load_store(persist_dir: str) -> Chroma:
    """Reopen a store built by a previous run."""
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"no vector store at {persist_dir!r} — process a transcript first"
        )
    return Chroma(persist_directory=persist_dir, embedding_function=get_embeddings())
