# Module 3 — Zepto Support Assistant

## Overview

The Zepto Support Assistant is a small GenAI-based support service designed to answer questions about Zepto policies.

The system uses:

- Sentence Transformers for local document embeddings
- ChromaDB for vector storage and retrieval
- LangGraph for query classification and routing
- Pydantic for structured output validation
- FastAPI for the REST API
- Docker for containerized deployment

The application runs in deterministic `MOCK_LLM` mode by default. This means it does not require an external LLM API, API key, or LLM network access.

---

## Architecture

```text
                         User Query
                             |
                             v
                     FastAPI POST /ask
                             |
                             v
                    LangGraph StateGraph
                             |
                             v
                     classify_intent
                       /          \
                      /            \
         policy_question        general_question
                |                     |
                v                     v
      retrieve_and_answer       direct_answer
                |                     |
                v                     |
             ChromaDB                 |
          Top-3 Retrieval             |
                |                     |
                +----------+----------+
                           |
                           v
                  Structured Response
                           |
                           v
                       Pydantic
                           |
                           v
                       API Response
Data Pipeline
8 Zepto Policy Documents
          |
          v
    Document Loading
          |
          v
        Chunking
   (one chunk per document)
          |
          v
   all-MiniLM-L6-v2
          |
          v
       ChromaDB
    zepto_policies
          |
          v
    Top-3 Retrieval
          |
          v
   Answer Generation
Corpus

The system uses the required eight Zepto policy documents:

ID	Document
doc_01	Delivery Policy
doc_02	Returns & Refunds
doc_03	Membership
doc_04	Tracking
doc_05	Cancellation
doc_06	Damaged/Missing
doc_07	Gift Cards
doc_08	Support Hours

The documents are stored in:

docs/

Each document is embedded using:

all-MiniLM-L6-v2

The embedding dimension is 384.

The embeddings are stored in the ChromaDB collection:

zepto_policies

Cosine similarity is used for retrieval.

LangGraph Workflow

The application uses a LangGraph StateGraph with three main nodes.

1. classify_intent

This node classifies the user query into:

policy_question
general_question

In MOCK_LLM mode, classification is deterministic and does not use an LLM.

The policy keywords are:

delivery
return
refund
membership
tracking
cancel
gift card
support hours

If the query contains one of these keywords, it is routed to the policy retrieval path.

Otherwise, it is routed to the general-answer path.

2. retrieve_and_answer

This node handles policy questions.

The query is converted into an embedding using:

all-MiniLM-L6-v2

ChromaDB is then queried for the top 3 most similar policy documents.

In MOCK_LLM mode, the answer follows the deterministic format:

Based on the retrieved context: <top retrieved document snippet>

The retrieved document IDs are returned through the sources field.

3. direct_answer

This node handles general questions that are outside the supported Zepto policy corpus.

In MOCK_LLM mode, it returns the fixed response:

I can only answer questions about Zepto policies right now.

No ChromaDB retrieval is performed for general questions.

Conditional Routing

The LangGraph workflow uses a conditional edge after classify_intent.

START
  |
  v
classify_intent
  |
  +---- policy_question ----> retrieve_and_answer ----> END
  |
  +---- general_question --> direct_answer ----------> END

The routing decision does not depend on an LLM in MOCK_LLM mode.

Structured Output

The final API response is validated using Pydantic.

Response format:

{
  "answer": "string",
  "sources": ["string"],
  "confidence": 1.0
}

The response schema contains:

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)

The confidence value must be between 0.0 and 1.0.

The application also contains response validation and retry handling for invalid structured output, allowing up to three validation attempts.

Prompt Design

The optional real-LLM prompt follows the required structured format:

ROLE
CONTEXT
TASK
FORMAT
LENGTH

The prompt also contains:

An explicit negative constraint
A few-shot example
Customer question
Retrieved context

The negative constraint instructs the model not to invent Zepto policy information that is not present in the supplied context.

MOCK_LLM Mode

The application is designed to work without an external LLM.

The behavior is:

MOCK_LLM unset  -> Mock mode
MOCK_LLM=1      -> Mock mode
MOCK_LLM=0      -> Optional real-LLM branch

In mock mode:

No external LLM API is called.
No API key is required.
Intent classification is deterministic.
Embeddings are generated locally.
ChromaDB is used locally for retrieval.
Answers use deterministic response templates.
The service can run without an external LLM service.
FastAPI

The application exposes a REST API using FastAPI.

Endpoint
POST /ask
Request
{
  "query": "How long does delivery take?"
}
Retrieval Example

Request:

{
  "query": "How long does delivery take?"
}

Response:

{
  "answer": "Based on the retrieved context: Delivery Policy: ...",
  "sources": [
    "doc_01",
    "doc_02",
    "doc_04"
  ],
  "confidence": 1.0
}

The policy query is routed to retrieve_and_answer and retrieves the relevant policy documents from ChromaDB.

General Question Example

Request:

{
  "query": "What is artificial intelligence?"
}

Response:

{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}

The general question is routed to direct_answer, so no document retrieval is performed.

Both example API calls were successfully tested through the Dockerized FastAPI service and returned HTTP 200.

Local Setup

From the support_assistant directory, install the dependencies:

pip install -r requirements.txt

Create the ChromaDB index:

python ingest.py

This loads all eight documents, generates embeddings using all-MiniLM-L6-v2, and stores them in the ChromaDB collection:

zepto_policies
Run Locally

Start the FastAPI server:

python -m uvicorn main:app --host 127.0.0.1 --port 7860

Open the service:

http://127.0.0.1:7860/

Swagger API documentation:

http://127.0.0.1:7860/docs
Docker
Build the Docker Image

From the support_assistant directory:

docker build -t zepto-support-assistant .
Run the Container
docker run --rm -p 7860:7860 zepto-support-assistant

The application listens inside the container on:

0.0.0.0:7860

Access it from the host using:

http://127.0.0.1:7860/

Swagger documentation:

http://127.0.0.1:7860/docs

The Dockerfile runs:

python ingest.py

during the image build so that the ChromaDB index is created inside the Docker image.

The container starts FastAPI using Uvicorn:

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
Project Structure
support_assistant/
│
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
│
├── chroma_db/
│
├── ingest.py
├── main.py
├── test_retrieval.py
├── requirements.txt
├── Dockerfile
└── README.md
Technologies Used
Python 3.13
FastAPI
Uvicorn
LangGraph
Pydantic
ChromaDB
Sentence Transformers
all-MiniLM-L6-v2
Docker
Summary

The completed Support Assistant provides a complete local GenAI pipeline:

Documents
   ↓
Embedding
   ↓
ChromaDB
   ↓
User Query
   ↓
LangGraph Intent Classification
   ↓
Conditional Routing
   ↓
Top-3 Retrieval / Direct Answer
   ↓
Pydantic Validation
   ↓
FastAPI Response