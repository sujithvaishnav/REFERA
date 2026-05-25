from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_answer(query, retrieved_docs):

    context = ""
    sources = []

    docs = retrieved_docs["documents"][0]
    metas = retrieved_docs["metadatas"][0]

    for doc, meta in zip(docs, metas):

        page = meta["page"]
        source = meta["source"]

        sources.append({
            "page": page,
            "source": source,
            "snippet": doc[:300]
        })

        context += f"""

        Page: {page}
        Source: {source}

        {doc}

        """

    prompt = f"""

    You are a technical research assistant.

    Answer ONLY using the provided context.

    IMPORTANT RULES:
    - Preserve exact technical terminology from the context
    - Do NOT overly paraphrase
    - Use exact phrases when possible
    - Include important concepts explicitly
    - If multiple sources discuss the topic, combine them carefully
    - Mention page references clearly
    - If answer is not found, say so

    CONTEXT:
    {context}

    QUESTION:
    {query}

    Provide a technically accurate answer.
    """

    completion = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "system",
                "content": "You are a helpful PDF assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,

        stream=True
    )

    return completion, sources

def generate_answer_eval(query, retrieved_docs):

    context = ""

    docs = retrieved_docs["documents"][0]
    metas = retrieved_docs["metadatas"][0]

    for doc, meta in zip(docs, metas):

        context += f"""

        Page: {meta['page']}
        Source: {meta['source']}

        {doc}

        """

    prompt = f"""

    Answer ONLY from provided context.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    """

    completion = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = completion.choices[0].message.content

    return {
        "answer": answer
    }