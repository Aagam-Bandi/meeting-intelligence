"""Prompt templates, isolated so they can be tuned and diffed independently."""

CHUNK_SUMMARY = """You are summarising one segment of a longer meeting transcript.
Other segments are being summarised separately and all summaries will be
combined, so write something that stands alone but composes cleanly.

Produce 5-6 bullet points capturing discussions, decisions and action items.
Attribute statements to the speaker by name wherever the transcript identifies
one. Prioritise substance over pleasantries. Do not infer or embellish; if the
segment is inconclusive, say so.

Segment:
{chunk}"""


BRIEFING = """Below are summaries of every segment of one meeting, in order.

Produce an executive briefing of 8-10 bullet points covering the most
consequential discussions, decisions and outcomes.

- Attribute key points to the person who made them.
- State decisions as decisions, and open questions as open questions.
- Report only what happened. Do not add recommendations or next steps that
  were not raised in the meeting.
- If a segment is marked unavailable, note the gap rather than writing around it.

Segment summaries:
{combined_summary}"""


ACTION_ITEMS = """Below are summaries of every segment of one meeting.

Extract only the action items — specific tasks, commitments and assignments.
Exclude discussion, opinion, background and open questions.

For each item:
- Begin with an action verb (Send, Prepare, Investigate, Schedule).
- Name the owner. If a task was assigned without a named owner, write
  "owner unassigned" rather than guessing.
- Include any stated deadline.

Return a bulleted list and nothing else. If there are no action items, say so
plainly.

Segment summaries:
{combined_summary}"""


QUERY_REWRITE = """Rewrite the user's question into a standalone question that can be
understood without the conversation history.

Resolve pronouns and implicit references against the history. Preserve the
original intent exactly — do not broaden, narrow or answer it. If the question
is already self-contained, return it unchanged.

Return only the rewritten question.

Conversation history:
{chat_history}

Question:
{query}"""


GROUNDED_ANSWER = """Answer the question using ONLY the transcript excerpts below.

If the excerpts do not contain the answer, say so directly. Do not supplement
from general knowledge and do not speculate about what was probably meant.
Attribute statements to speakers where the excerpts identify them.

Excerpts:
{context}

Question:
{query}"""
