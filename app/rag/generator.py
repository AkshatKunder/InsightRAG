import ollama


MODEL_NAME = "llama3.2:3b"


class Generator:

    def __init__(self):
        self.model = MODEL_NAME

    def generate(self, question, retrieved_chunks):
        """
        Generate an answer using only the retrieved document context.
        """

        context_parts = []

        for i, chunk in enumerate(retrieved_chunks, start=1):
            context_parts.append(
                f"""
Source {i}
File: {chunk['source']}
Page: {chunk['page']}

Content:
{chunk['text']}
"""
            )

        context = "\n".join(context_parts)

        prompt = f"""
You are an AI assistant answering questions about a company's annual report.

Answer the user's question using ONLY the information provided in the
retrieved context below.

Rules:
1. Do not use outside knowledge.
2. If the answer cannot be found in the context, say:
   "I could not find this information in the provided document."
3. Give a concise and factual answer.
4. Include the source page in your answer.

Retrieved context:
{context}

User question:
{question}

Answer:
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
