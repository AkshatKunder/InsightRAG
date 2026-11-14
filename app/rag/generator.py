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


def build_prompt(question, retrieved_chunks):
    # Page numbers and file names are deliberately left out of the context:
    # the model does not need them, and including them invites the model to
    # copy them into the answer.
    context = "\n\n".join(
        f"EXCERPT {i}:\n{chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks, start=1)
    )

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
   the words "sale" or "sales".

DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{question}

ANSWER:"""


class Generator:

    def __init__(self):
        self.model = MODEL_NAME

    def generate(self, question, retrieved_chunks):

        prompt = build_prompt(question, retrieved_chunks)

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return clean_answer(response["message"]["content"])
