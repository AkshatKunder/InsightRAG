import re

import ollama


MODEL_NAME = "llama3.2:3b"

# Small models sometimes cite the context labels despite instructions, e.g.
# "According to SOURCE 2, DOCUMENT: report.pdf, PAGE: 95, and SOURCE 3, the ...".
# Citations are produced separately, so such references are stripped.
_SOURCE_REF = r"(?:source|excerpt|passage|document|page)s?\s*:?\s*[\w.\-]+"
_LEADING_ATTRIBUTION_RE = re.compile(
    rf"^\s*(?:according to|based on|as (?:stated|shown|reported) in|per)\s+"
    rf"(?:the\s+)?(?:{_SOURCE_REF}(?:\s*(?:,|and|&)\s*)*)+,?\s*",
    re.I,
)
_INLINE_REFERENCE_RE = re.compile(
    rf"\s*[(\[]\s*(?:{_SOURCE_REF}(?:\s*(?:,|;|and|&)\s*)*)+[)\]]",
    re.I,
)


def clean_answer(answer: str) -> str:
    """Remove context-label references (SOURCE n, PAGE n, ...) from an answer."""
    cleaned = _LEADING_ATTRIBUTION_RE.sub("", answer or "")
    cleaned = _INLINE_REFERENCE_RE.sub("", cleaned).strip()
    if cleaned and cleaned != (answer or "").strip():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


SCOPE_RULES = """8. Use group-level (consolidated, "Total") figures unless the question names a
   segment, region, subsidiary or joint venture. Text marked [Scope: ...] or
   placed under a segment heading reports only that segment or business unit.
9. Keep metrics distinct: production, shipments, capacity, run-of-mine (ROM)
   or "aggregate" figures, sales and income are different measures. Only
   report the measure the question asks for.
10. Check that each number you give matches the question's metric, entity,
   year and unit. For a change or comparison, use two values of the same
   metric and scope, and compute the difference from them.
11. If the excerpts only contain the figure for a different scope or measure,
   respond exactly: "I could not find this information in the provided document."
"""


def build_prompt(question, retrieved_chunks, scope_aware=True):
    # Page numbers and file names are deliberately left out of the context:
    # the model does not need them, and including them invites the model to
    # copy them into the answer. Scope markers and recovered chart labels
    # (app.rag.document_context) are included when the chunk carries them.
    if scope_aware:
        from app.rag.document_context import render_chunk

        render = render_chunk
    else:
        render = lambda chunk: chunk["text"]  # noqa: E731
    context = "\n\n".join(
        f"EXCERPT {i}:\n{render(chunk)}"
        for i, chunk in enumerate(retrieved_chunks, start=1)
    )
    scope_rules = f"\n{SCOPE_RULES}" if scope_aware else "\n"

    return f"""You are a precise document question-answering assistant.

Answer the user's question using only the document excerpts below.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts or numbers.
3. Do not mention excerpts, sources, pages, documents, or citations.
   Never write phrases like "According to excerpt 2".
4. Do not repeat the same answer or number unnecessarily.
5. If the excerpts contain the answer, state it directly in one or two
   sentences, including the number with its unit and the reporting period.
6. If the answer cannot be determined from the excerpts, respond exactly:
   "I could not find this information in the provided document."
7. Distinguish total sales from segment sales and other uses of
   the words "sale" or "sales".{scope_rules}
DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{question}

ANSWER:"""


def complete(prompt, options=None, model=MODEL_NAME):
    """Single-prompt completion with the project's Ollama model (used by NL-to-SQL).

    Returns ``(text, options_applied)``; falls back to Ollama defaults when the
    server rejects ``options`` (``options_applied`` is then False).
    """
    messages = [{"role": "user", "content": prompt}]
    if options:
        try:
            response = ollama.chat(model=model, messages=messages, options=dict(options))
            return response["message"]["content"], True
        except (TypeError, ValueError, ollama.ResponseError):
            response = ollama.chat(model=model, messages=messages)
            return response["message"]["content"], False
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"], None


class Generator:
    """Answer generation with Ollama.

    ``options`` (e.g. ``{"temperature": 0, "seed": 42}``) are passed to Ollama
    only when given; the default keeps Ollama's own sampling settings
    (production behaviour). If the server rejects the options, the call is
    retried without them and ``last_options_applied`` is set to False.
    """

    def __init__(self, options=None, scope_aware=True):
        self.model = MODEL_NAME
        self.options = dict(options) if options else None
        self.scope_aware = scope_aware
        self.last_options_applied = None

    def settings(self):
        return {
            "model": self.model,
            "options": self.options,
            "options_applied": self.last_options_applied,
            "scope_aware_prompt": self.scope_aware,
        }

    def generate(self, question, retrieved_chunks):

        prompt = build_prompt(question, retrieved_chunks, scope_aware=self.scope_aware)
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        if self.options:
            try:
                response = ollama.chat(model=self.model, messages=messages, options=self.options)
                self.last_options_applied = True
            except (TypeError, ValueError, ollama.ResponseError):
                # Older clients/servers may not accept these options: fall back
                # to defaults instead of failing, and record that they were not applied.
                response = ollama.chat(model=self.model, messages=messages)
                self.last_options_applied = False
        else:
            response = ollama.chat(model=self.model, messages=messages)
            self.last_options_applied = None

        return clean_answer(response["message"]["content"])
