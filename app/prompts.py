"""Prompt engineering lives here, isolated from transport code.

The assignment is graded partly on "LLM prompt effectiveness", so the prompt is
treated as a first-class artifact, not an inline string. Design goals:

  1. STRUCTURE  -> force strict JSON so the output is renderable and parseable.
  2. LAYERING   -> TL;DR, decisions, action items, open questions, topic timeline,
                   so the reader gets depth, not one flat paragraph (a common
                   complaint about meeting tools).
  3. GROUNDING  -> only use facts present in the transcript; never invent owners,
                   dates, or decisions. Say "Unassigned"/"Not specified" instead.
                   This is the anti-hallucination guardrail.
  4. CITATIONS  -> every decision/action/topic carries the [mm:ss] timestamp it came
                   from, so a reader can jump back to the source moment.
"""

from __future__ import annotations

# The JSON contract the model must follow. Kept in one place so the prompt and the
# frontend renderer never drift apart.
SUMMARY_JSON_SHAPE = """{
  "tldr": "3-4 sentence executive summary a busy person can read in 15 seconds",
  "key_decisions": [
    {"decision": "what was decided", "timestamp": "mm:ss"}
  ],
  "action_items": [
    {"task": "concrete task", "owner": "person named, or 'Unassigned'", "due": "date/timeframe stated, or 'Not specified'", "timestamp": "mm:ss"}
  ],
  "open_questions": ["questions raised but left unresolved"],
  "topics": [
    {"title": "topic discussed", "summary": "1-2 sentence recap", "timestamp": "mm:ss"}
  ]
}"""

SYSTEM_PROMPT = f"""You are a meticulous meeting analyst. You convert a raw meeting \
transcript into a structured, action-oriented summary.

Return ONLY a single valid JSON object matching exactly this shape (no markdown, no \
prose outside the JSON):

{SUMMARY_JSON_SHAPE}

Rules you must follow:
- Ground every statement in the transcript. Do NOT invent decisions, owners, dates, or \
facts that are not present.
- If a task has no clear owner, set owner to "Unassigned". If no due date was stated, \
set due to "Not specified". Never guess a name or date.
- Use the [mm:ss] timestamps that appear in the transcript for the "timestamp" fields. \
If a point spans the whole meeting, use the timestamp where it is clearest.
- Prefer specific, verb-first action items ("Send the revised quote to the client") over \
vague ones ("Follow up").
- If the transcript contains no decisions or no action items, return an empty list for \
that field rather than fabricating entries.
- Keep the TL;DR neutral and factual."""

USER_PROMPT_TEMPLATE = """Summarize the following meeting transcript into key decisions \
and action items, following the JSON contract exactly.

Meeting title: {title}

Transcript (timestamps in [mm:ss]):
---
{transcript}
---"""


def build_messages(title: str, transcript: str) -> list[dict]:
    """Assemble the chat messages for the summarization call."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                title=title or "Untitled meeting", transcript=transcript
            ),
        },
    ]


# --------------------------------------------------------------------- RAG chat
# The chat answer must stay grounded in the retrieved excerpts. The prompt forbids
# outside knowledge and instructs an explicit refusal when the context is silent.
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about the \
user's past meetings, using ONLY the excerpts provided below.

Each excerpt is labelled with its meeting title and a [mm:ss] timestamp.

Rules:
- Answer strictly from the excerpts. Do NOT use outside knowledge or make assumptions.
- If the excerpts do not contain the answer, say clearly: "I couldn't find anything \
about that in your meetings." Do not guess.
- When you state a fact, mention which meeting it came from (by title) and the [mm:ss].
- Be concise and direct. Prefer specifics from the excerpts over generalities."""

RAG_USER_TEMPLATE = """Question: {question}

Meeting excerpts:
---
{context}
---

Answer the question using only the excerpts above."""


def build_rag_messages(question: str, context: str) -> list[dict]:
    """Assemble the chat messages for a grounded RAG answer."""
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": RAG_USER_TEMPLATE.format(question=question, context=context)},
    ]


# --------------------------------------------------------------------- query rewriting
# Retrieval matches a question against transcript text, and the two rarely share
# vocabulary: people ask "why was the basket page slow" about a meeting that said "N+1
# query in the cart serializer". This rewrites the question into something closer to how
# the answer would actually have been spoken.
#
# Deliberately narrow. The model is told to expand vocabulary, NOT to answer, reason, or
# invent specifics -- a rewriter that adds facts would smuggle a hallucination into the
# retrieval step, where the grounding prompt can no longer catch it.
QUERY_REWRITE_PROMPT = """You rewrite a question into a search query for a meeting \
transcript archive.

Rules:
- Keep every concept from the original question.
- Add likely synonyms and the words people actually say out loud in meetings
  (for example: basket -> cart, slogan -> tagline, slipped -> moved, delayed).
- Expand abbreviations you are confident about.
- Do NOT answer the question, explain anything, or invent names, numbers or dates.
- Output ONLY the rewritten query, on one line, with no preamble and no quotes.

Question: {question}"""


def build_rewrite_messages(question: str) -> list[dict]:
    """Assemble the chat messages that turn a question into a retrieval query."""
    return [{"role": "user", "content": QUERY_REWRITE_PROMPT.format(question=question)}]
