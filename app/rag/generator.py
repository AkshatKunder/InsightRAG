import ollama


MODEL_NAME = "llama3.2:3b"


class Generator:

    def __init__(self):
        self.model = MODEL_NAME

    def generate(self, question, retrieved_chunks):

        context_parts = []

        for i, chunk in enumerate(retrieved_chunks, start=1):

            context_parts.append(
                f"""
SOURCE {i}
DOCUMENT: {chunk['source']}
PAGE: {chunk['page']}

CONTENT:
{chunk['text']}
"""
            )

        context = "\n".join(context_parts)

        prompt = f"""You are a precise document question-answering assistant.

Answer the user's question using only the document context below.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts or numbers.
3. Do not mention source numbers, page numbers, document names,
   or citations.
4. Do not repeat the same answer or number unnecessarily.
5. If the context contains the answer, state it directly.
6. If the answer cannot be determined from the context, respond exactly:
   "I could not find this information in the provided document."
7. Keep the answer concise.
8. Distinguish total sales from segment sales and other uses of
   the words "sale" or "sales".

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]
