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

        prompt = f"""
You are a document question-answering system.

Your task is to answer the user's question using ONLY the provided
document context.

IMPORTANT RULES:

1. Do NOT use outside knowledge.

2. Do NOT guess or infer an answer that is not explicitly supported
   by the provided context.

3. Numbers are extremely important.
   Only provide a number if the context explicitly supports that number
   as the answer to the user's question.

4. Carefully distinguish between:
   - countries where the company operates
   - countries where the company has steel-making operations
   - number of employees
   - sales
   - percentages
   - dates
   - other numerical values

5. If multiple sources contain different numbers, determine which
   number directly answers the user's question.

6. Answer the question using the provided context.

7. Do not generate source numbers, page numbers, or citations.
   The application will display source information separately.

8. If the answer cannot be determined from the context, respond exactly:
   "I could not find this information in the provided document."

9. Keep the answer concise and do not include unrelated information.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

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
