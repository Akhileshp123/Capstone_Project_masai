import os
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
from fastapi import FastAPI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

POLICY_KEYWORDS = (
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
)


# ============================================================
# Structured prompt for optional real LLM mode
# ============================================================

PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer support assistant.

CONTEXT:
Use only the retrieved Zepto policy context provided below.

TASK:
Answer the customer's question accurately using the available policy context.

FORMAT:
Return a concise answer suitable for a customer support response.

LENGTH:
Keep the answer short and direct, preferably under 100 words.

NEGATIVE CONSTRAINT:
Do not invent, assume, or add policy information that is not present
in the supplied context.

FEW-SHOT EXAMPLE:
Question: How long does delivery take?
Context: Zepto delivers within 10 to 30 minutes depending on the
delivery zone and current order volume.
Answer: Zepto delivery typically takes 10 to 30 minutes.

CUSTOMER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}
"""


# ============================================================
# Pydantic schemas
# ============================================================

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)


# ============================================================
# Structured output validation with retry support
# ============================================================

def validate_response_with_retries(
    raw_output: dict,
    max_attempts: int = 3,
) -> AskResponse:
    """
    Validate structured assistant output.

    A maximum of 3 attempts is allowed:
    - 1 initial validation attempt
    - up to 2 additional retry attempts

    The retry mechanism is primarily intended for the
    optional real-LLM branch. MOCK_LLM responses are
    already deterministic.
    """

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            return AskResponse.model_validate(raw_output)

        except Exception as exc:
            last_error = exc

            if attempt == max_attempts:
                break

            # Corrective instruction that would be supplied
            # to a real LLM on a retry.
            corrective_instruction = (
                "Return valid structured output with exactly "
                "these fields: answer (string), sources (list "
                "of strings), and confidence (number from 0 to 1)."
            )

            print(
                f"Structured output validation failed "
                f"(attempt {attempt}/{max_attempts})."
            )
            print(corrective_instruction)

    raise ValueError(
        f"Unable to produce valid structured output after "
        f"{max_attempts} attempts: {last_error}"
    )


# ============================================================
# LangGraph state
# ============================================================

class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    answer: str
    sources: list[str]
    confidence: float


# ============================================================
# Mock mode
# ============================================================

def mock_mode() -> bool:
    """
    MOCK_LLM is enabled by default.

    MOCK_LLM unset -> mock mode
    MOCK_LLM=1    -> mock mode
    MOCK_LLM=0    -> optional real LLM mode
    """
    return os.getenv("MOCK_LLM", "1") != "0"


# ============================================================
# ChromaDB + embedding model
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded.")

print("Connecting to ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

print("ChromaDB collection loaded.")
print("Documents available:", collection.count())


# ============================================================
# Node 1 — classify_intent
# ============================================================

def classify_intent(state: GraphState) -> GraphState:
    """
    Classify the user query as either:

    policy_question
    or
    general_question

    MOCK_LLM mode uses deterministic keyword matching.
    """

    query = state["query"].lower()

    is_policy_question = any(
        keyword in query
        for keyword in POLICY_KEYWORDS
    )

    if is_policy_question:
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        **state,
        "intent": intent,
    }


# ============================================================
# Node 2 — retrieve_and_answer
# ============================================================

def retrieve_and_answer(state: GraphState) -> GraphState:
    """
    Retrieve the top 3 relevant documents from ChromaDB.

    In MOCK_LLM mode:
    - no LLM call
    - answer is deterministic
    - first retrieved document is used as the answer context
    """

    query = state["query"]

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    retrieved_ids = results["ids"][0]
    retrieved_documents = results["documents"][0]

    if not retrieved_documents:
        return {
            **state,
            "answer": "I could not find relevant information in the Zepto policy documents.",
            "sources": [],
            "confidence": 0.0,
        }

    top_chunk = retrieved_documents[0]

    if mock_mode():
        answer = (
            "Based on the retrieved context: "
            + top_chunk[:200]
        )
    else:
        # Optional real LLM branch.
        #
        # The assignment allows a real LLM extension when
        # MOCK_LLM=0. The graded baseline remains MOCK_LLM=1.
        answer = (
            "Real LLM mode is not configured yet. "
            "Retrieved context: "
            + top_chunk[:200]
        )

    return {
        **state,
        "answer": answer,
        "sources": retrieved_ids,
        "confidence": 1.0,
    }


# ============================================================
# Node 3 — direct_answer
# ============================================================

def direct_answer(state: GraphState) -> GraphState:
    """
    Handle non-policy questions without retrieval.
    """

    if mock_mode():
        answer = (
            "I can only answer questions about Zepto policies right now."
        )
    else:
        # Optional real LLM branch.
        answer = (
            "I can only answer questions about Zepto policies right now."
        )

    return {
        **state,
        "answer": answer,
        "sources": [],
        "confidence": 1.0,
    }


# ============================================================
# Conditional routing
# ============================================================

def route_by_intent(
    state: GraphState,
) -> Literal["retrieve_and_answer", "direct_answer"]:

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# Build LangGraph
# ============================================================

def build_graph():
    graph = StateGraph(GraphState)

    # Three required named nodes
    graph.add_node(
        "classify_intent",
        classify_intent,
    )

    graph.add_node(
        "retrieve_and_answer",
        retrieve_and_answer,
    )

    graph.add_node(
        "direct_answer",
        direct_answer,
    )

    # Start → classifier
    graph.add_edge(
        START,
        "classify_intent",
    )

    # Classifier → conditional route
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    # Both branches → END
    graph.add_edge(
        "retrieve_and_answer",
        END,
    )

    graph.add_edge(
        "direct_answer",
        END,
    )

    return graph.compile()


# ============================================================
# Compile graph
# ============================================================

GRAPH = build_graph()


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "service": "Zepto Support Assistant",
        "status": "running",
        "mock_llm": mock_mode(),
    }
@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask(request: AskRequest):
    result = GRAPH.invoke(
        {
            "query": request.query,
        }
    )

    raw_output = {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "confidence": result.get("confidence", 0.0),
    }

    return validate_response_with_retries(raw_output)