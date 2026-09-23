
import os
from typing import TypedDict, List

import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END
from fastapi import FastAPI
from pydantic import BaseModel, Field

MOCK_LLM = "1"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_path = "zepto-ai-platform/support_assistant/chroma_db"

client = chromadb.PersistentClient(path=chroma_path)

collection = client.get_collection(
    name="zepto_policy_documents"
)

class SupportState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: List[str]
    confidence: float

class SupportResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

def classify_intent(state: SupportState):

    query = state["query"].lower()

    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
    ]

    if any(keyword in query for keyword in policy_keywords):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        "intent": intent
    }

def retrieve_and_answer(state: SupportState):

    query = state["query"]

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_documents = results["documents"][0]
    retrieved_ids = results["ids"][0]

    top_chunk = retrieved_documents[0]

    response = SupportResponse(
        answer=f"Based on the retrieved context: {top_chunk}",
        sources=retrieved_ids,
        confidence=1.0
    )

    return response.model_dump()

def direct_answer(state: SupportState):

    response = SupportResponse(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0
    )

    return response.model_dump()


def route_question(state: SupportState):

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"

workflow = StateGraph(SupportState)

workflow.add_node(
    "classify_intent",
    classify_intent
)

workflow.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

workflow.add_node(
    "direct_answer",
    direct_answer
)

workflow.add_edge(
    START,
    "classify_intent"
)

workflow.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

workflow.add_edge(
    "retrieve_and_answer",
    END
)

workflow.add_edge(
    "direct_answer",
    END
)

support_graph = workflow.compile()

app = FastAPI(
    title="Zepto Support Assistant",
    description="Offline Zepto policy support assistant",
    version="1.0.0"
)


class AskRequest(BaseModel):
    query: str


@app.post("/ask", response_model=SupportResponse)
def ask(request: AskRequest):

    result = support_graph.invoke({
        "query": request.query
    })

    return SupportResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )
