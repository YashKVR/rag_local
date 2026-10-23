import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

from implementation.llm import local_completion, logger


load_dotenv(override=True)

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"
SUMMARIES_PATH = Path(__file__).parent.parent / "summaries"

collection_name = "docs"
embedding_model = "text-embedding-3-large"
wait = wait_exponential(multiplier=1, min=1, max=8)

openai = OpenAI()

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)

RETRIEVAL_K = 8
FINAL_K = 5

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""


class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk_preview(chunk)}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = local_completion(
        messages=messages,
        response_format=RankOrder,
        reasoning_effort="low",
        max_tokens=200,
    )
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    ranked = []
    seen = set()
    for index in order:
        if isinstance(index, int) and 1 <= index <= len(chunks) and index not in seen:
            ranked.append(chunks[index - 1])
            seen.add(index)
    for chunk in chunks:
        if chunk not in ranked:
            ranked.append(chunk)
    return ranked


def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait, stop=stop_after_attempt(3))
def rewrite_query(question, history=None):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    if history is None:
        history = []

    message = f"""
You are in a conversation with a user.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Since the conversation is contextual, understand the meaning of the user question and add details based on the history.
Condense everything in a single contextually-rich VERY short and specific question, most likely to surface content.

EXAMPLE:
user: Who is the founder? -> Query: who is the founder?
assistant: The founder is FooBar
user: What role covers? -> Query: What role FooBar covers?
...

IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    response = local_completion(
        messages=[{"role": "system", "content": message}],
        reasoning_effort="low",
        max_tokens=80,
    )
    rewritten = (response.choices[0].message.content or "").strip()
    if rewritten.lower().startswith("query:"):
        rewritten = rewritten.split(":", 1)[1].strip()
    return rewritten


def merge_chunks(chunks, reranked):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def chunk_preview(chunk) -> str:
    """Headline and summary only, so reranking does not reread every source document."""
    return "\n\n".join(chunk.page_content.split("\n\n", 2)[:2])[:500]


def embed_questions(questions: list[str]) -> list[list[float]]:
    response = openai.embeddings.create(model=embedding_model, input=questions)
    return [item.embedding for item in response.data]


def fetch_by_embedding(embedding):
    results = collection.query(query_embeddings=[embedding], n_results=RETRIEVAL_K)
    return [
        Result(page_content=document, metadata=metadata)
        for document, metadata in zip(results["documents"][0], results["metadatas"][0])
    ]


def fetch_context_unranked(question):
    return fetch_by_embedding(embed_questions([question])[0])


def fetch_context(original_question, history=None):
    history = history or []
    if history:
        with ThreadPoolExecutor(max_workers=1) as pool:
            rewrite_future = pool.submit(rewrite_query, original_question, history)
            chunks = fetch_context_unranked(original_question)
            rewritten_question = (rewrite_future.result() or "").strip()
        logger.info("Rewritten query: %s", rewritten_question)
        if rewritten_question and rewritten_question.lower() != original_question.strip().lower():
            chunks = merge_chunks(chunks, fetch_context_unranked(rewritten_question))
    else:
        chunks = fetch_context_unranked(original_question)

    if len(chunks) <= 1:
        return chunks
    try:
        return rerank(original_question, chunks)[:FINAL_K]
    except Exception as exc:
        logger.warning("Rerank failed (%s). Using vector search order.", exc)
        return chunks[:FINAL_K]


@retry(wait=wait, stop=stop_after_attempt(3))
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """
    Answer a question using RAG and return the answer and the retrieved context
    """
    started = time.perf_counter()
    chunks = fetch_context(question, history)
    retrieved = time.perf_counter()
    messages = make_rag_messages(question, history, chunks)
    response = local_completion(messages=messages, reasoning_effort="low")
    logger.info(
        "Responded in %.1fs (retrieve %.1fs, generate %.1fs).",
        time.perf_counter() - started,
        retrieved - started,
        time.perf_counter() - retrieved,
    )
    return response.choices[0].message.content, chunks
