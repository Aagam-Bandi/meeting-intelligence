# Meeting Intelligence

Turns long meeting transcripts into executive briefings and attributed action items, and indexes them so you can ask questions across every meeting you've processed.

```bash
pip install -r requirements.txt
python cli.py transcript.txt --chat    # summarise, then ask questions
pytest                                 # 30 tests
```

```
==============================================================
EXECUTIVE BRIEFING
==============================================================
• Doug A. presented Q3 pipeline figures, flagging a 12% shortfall
  against forecast driven mainly by two delayed enterprise renewals.
• Carin M. questioned whether the renewal delays were structural or
  timing-related; no conclusion was reached.
• The team agreed to defer the pricing change to Q4 pending the
  competitive analysis Ry. W. is preparing.
...

==============================================================
ACTION ITEMS
==============================================================
• Prepare competitive pricing analysis — Ry. W., due before Q4 planning
• Schedule follow-up with the two delayed renewal accounts — Doug A.
• Circulate revised forecast model — owner unassigned
...

> What did we decide about the pricing change?

The pricing change was deferred to Q4. The decision was contingent on
a competitive analysis being prepared by Ry. W. — the transcript does
not record a specific date for the deferred review.
```

---

## Why not just paste the transcript into a chat window

Two failure modes make naive summarisation of long transcripts unreliable in ways that are hard to notice:

**Attention degrades across long inputs.** A model given a two-hour transcript attends unevenly, favouring the beginning and end. Decisions made forty minutes in get dropped. The resulting summary is fluent, plausible, and quietly missing the thing you needed.

**A summary answers the wrong question.** It tells you what happened in *one* meeting. What you actually want to know — "when did we decide on the pricing change, and what was the reasoning?" — spans meetings and requires retrieval, not summarisation.

This handles the first with map-reduce chunking, and the second by indexing the same chunks it summarises.

---

## How it works

```
Transcript
    │
    ▼
Chunk (1250 chars, 250 overlap)
    │
    ├─────────────────┬──────────────────────────┐
    │                 │                          │
    ▼                 ▼                          │
 MAP STAGE       embed → Chroma                  │
 summarise every  (runs concurrently             │
 chunk, semaphore- with the map stage)           │
 bounded                                         │
    │                 │                          │
    ▼                 │                          │
 Combined             │                          │
 summary              │                          │
    │                 │                          │
    ├──────┬──────────┘                          │
    ▼      ▼                                     ▼
 REDUCE  REDUCE                          RAG Q&A interface
 briefing actions                        query → rewrite →
 (concurrent)                            retrieve top-5 →
                                         grounded answer
```

### Map-reduce over chunks

Each chunk is summarised independently, then the partial summaries are folded into a briefing. The trade is cost for coverage: a decision buried mid-transcript sits near the *start of its own chunk*, where the model actually attends to it.

The reduce stage runs twice against the same combined summary — once for a narrative briefing, once for action items only. Splitting these is deliberate. A single prompt asked to produce both reliably blurs the boundary, listing discussion points as if they were commitments.

### Overlapping the index build with summarisation

Embedding is CPU-bound; summarisation is network-bound. Running them sequentially means the embedding time sits on the critical path for no reason. `asyncio.gather` overlaps them, and since `Chroma.from_documents` is blocking, it's dispatched to a thread to stay awaitable.

The same applies to the two reduce prompts, which are independent of each other.

### Bounded concurrency and retries

A long transcript means a large number of simultaneous API calls. Firing them all at once gets the batch rate-limited, and the failure is quiet — some chunks return errors and the combined summary is silently incomplete, which is worse than being slow because the output still *looks* finished.

Three mechanisms:

- A semaphore caps in-flight requests at 20.
- Failures retry up to four times with exponential backoff **and jitter**. Without jitter every rate-limited chunk retries in lockstep and the batch re-collides each round.
- A chunk that exhausts its retries returns an explicit `[segment N unavailable]` marker, and the reduce prompt is instructed to note the gap rather than write around it.

### Query rewriting before retrieval

Multi-turn retrieval breaks without this. A follow-up like *"what did he say about it?"* embeds to nothing useful — the pronouns carry no semantic signal, and cosine similarity returns arbitrary chunks.

Each query is first rewritten against the recent history into a standalone question, then embedded. The rewrite prompt is explicitly constrained not to broaden, narrow, or answer the question, since a rewriter that "helpfully" expands scope retrieves confidently irrelevant material.

Answers are generated strictly from retrieved context, with instructions to say so when the transcript doesn't contain the answer.

---

## Evaluation

Summarisation has no single correct output, so the harness in `src/evaluate.py` measures the three things that actually go wrong rather than trying to score prose quality.

| Metric | Question | Failure it catches |
|---|---|---|
| **Decision coverage** | Did the briefing mention every decision in the transcript? | Silent omission — the summary reads complete and is missing a decision |
| **Action item recall + owner accuracy** | Were tasks captured, with the right person attached? | Attribution drift, unassigned work |
| **Grounding** | Did RAG answer from the transcript, or from the model's own knowledge? | Confident hallucination on questions the meeting never covered |

Grounding is the one worth having. A retrieval system that fluently answers a question the transcript is silent on is worse than one that says "not discussed", and nothing else in the pipeline catches it. The harness poses questions the transcript *does not* cover and scores the system on whether it correctly abstains.

```
Overall                    0.871

Decision coverage          0.900  ██████████████████
Action item recall         0.833  ████████████████
Owner attribution          0.800  ████████████████
Answers when present       1.000  ████████████████████
Abstains when absent       0.823  ████████████████
```

Owner accuracy is scored only over action items that were found at all — an owner can't be wrong on a task the summary never mentioned, and counting it twice would penalise the same miss under two metrics.

Ground truth format:

```json
{"decisions": ["pricing change deferred to Q4"],
 "action_items": [{"task": "prepare competitive analysis", "owner": "Ry. W."}],
 "questions": [{"query": "what was the pricing decision", "in_transcript": true},
               {"query": "what was the Q3 budget", "in_transcript": false}]}
```

Both `process_fn` and `answer_fn` are injectable, so the harness is exercised against stubs in the test suite without spending API calls.

---

## Install

```bash
git clone https://github.com/Aagam-Bandi/meeting-intelligence
cd meeting-intelligence
pip install -r requirements.txt

cp .env.example .env
# add your Google AI Studio key to .env
```

## Usage

```bash
python cli.py transcript.txt              # briefing + action items
python cli.py transcript.txt --chat       # then open Q&A
python cli.py --chat-only                 # query a previously indexed transcript
python cli.py transcript.txt -v           # show pipeline timing
```

**As a library**

```python
from src import process, answer, get_llm

result = await process("transcript.txt")
print(result.briefing)

reply = await answer(get_llm(), result.store, "What did we decide about pricing?")
```

---

## Configuration

| Constant | Location | Default | Effect |
|---|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `index.py` | `1250` / `250` | Splitter geometry |
| `CONCURRENCY_LIMIT` | `summarise.py` | `20` | Max simultaneous API calls |
| `MAX_RETRIES` / `BASE_BACKOFF` | `summarise.py` | `4` / `1.0` | Retry attempts and backoff base |
| `RETRIEVAL_K` | `chat.py` | `5` | Chunks retrieved per query |
| `MODEL` / `TEMPERATURE` | `pipeline.py` | `gemini-2.0-flash` / `0.3` | Model settings |

Temperature is low by design. Summarisation of factual proceedings has no use for creative variation, and higher settings produce attributions that drift from what was actually said.

---

## Layout

```
src/
  index.py       chunking, embeddings, vector store lifecycle
  summarise.py   map and reduce stages, retries
  chat.py        query rewriting and grounded retrieval Q&A
  evaluate.py    coverage, action item and grounding metrics
  prompts.py     prompt templates, isolated for independent tuning
  pipeline.py    orchestration and concurrency
tests/           30 tests — run without an API key
cli.py           command-line entry point
```

Package imports are lazy, so the evaluation harness can be imported and tested without the Gemini SDK installed. The suite runs offline in under a second.

---

## Limitations

- **No diarisation.** Speaker attribution is only as good as the input transcript. If it doesn't label speakers, the briefing can't attribute anything — and the owner-accuracy metric will correctly show this as a failure.
- **Cross-meeting retrieval is single-store.** Each run rebuilds the index. Querying across many meetings needs a persistent multi-document store with source metadata.
- **The reduce stage can flatten disagreement.** When two chunk summaries conflict, the briefing tends to pick one rather than surfacing the tension. The coverage metric catches outright omission but not this softer failure.

---

## Stack

Python · Google Gemini · LangChain · ChromaDB · sentence-transformers (all-MiniLM-L6-v2) · asyncio

## License

MIT
