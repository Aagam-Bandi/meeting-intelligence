"""Summarise a transcript, then optionally open an interactive Q&A session.

    python cli.py transcript.txt
    python cli.py transcript.txt --chat
    python cli.py --chat-only          # query a previously indexed transcript
"""

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

from src import answer, get_llm, open_store, process


async def _chat_loop(store) -> None:
    llm = get_llm()
    history: list[str] = []
    print("\nAsk questions about the transcript. Ctrl-C or 'exit' to quit.\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if query.lower() in {"exit", "quit"}:
            return
        if not query:
            continue

        reply = await answer(llm, store, query, "\n".join(history[-6:]))
        print(f"\n{reply}\n")
        history += [f"User: {query}", f"Assistant: {reply}"]


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Summarise and query meeting transcripts.")
    parser.add_argument("path", nargs="?", help="path to a transcript .txt file")
    parser.add_argument("--chat", action="store_true", help="open Q&A after summarising")
    parser.add_argument("--chat-only", action="store_true", help="query an existing index")
    parser.add_argument("--store", default="./chroma_db", help="vector store directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-7s %(message)s",
    )

    if args.chat_only:
        await _chat_loop(open_store(args.store))
        return 0

    if not args.path:
        parser.error("a transcript path is required unless --chat-only is given")

    result = await process(args.path, args.store)

    print("\n" + "=" * 62)
    print("EXECUTIVE BRIEFING")
    print("=" * 62)
    print(result.briefing)
    print("\n" + "=" * 62)
    print("ACTION ITEMS")
    print("=" * 62)
    print(result.action_items)

    if args.chat:
        await _chat_loop(result.store)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(_main()))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
