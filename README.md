# J.A.R — Joseph's Assitant Retrieval

> A production-grade RAG system that lets recruiters and collaborators
> query Joseph's professional background conversationally, in any language.

**[→ Try it live](https://ijar.azurewebsites.net/)**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3.x-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-E07B39?style=flat-square)
![Azure OpenAI](https://img.shields.io/badge/Azure_OpenAI-GPT--4o--mini-0078D4?style=flat-square&logo=microsoftazure&logoColor=white)
![AWS S3](https://img.shields.io/badge/AWS_S3-Free_Tier-FF9900?style=flat-square&logo=amazons3&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions_+_OIDC-2088FF?style=flat-square&logo=githubactions&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What is J.A.R?

J.A.R (Joseph's AI Representative) is a publicly accessible RAG application
built as a portfolio project. Instead of a static CV, recruiters and
collaborators can ask natural language questions about Joseph's experience,
skills, and projects — and get grounded, cited answers backed by real documents.

Built with a security-first, cloud-native architecture: no static credentials,
OIDC authentication throughout, and a clean separation between offline ingestion
and online inference.

---

## Architecture

![Architecture Diagram](docs/assets/Arquitectura_JAR.png)

> The architecture separates **data ingestion** (local, offline) from
> **runtime** (cloud). ChromaDB is pre-built locally and stored in S3 —
> the container downloads it at startup. This eliminates embedding costs
> and API calls at runtime.

---

## How it works

### The 3-step pipeline

**1. Ingest** — Documents (CV, articles, project notes) are chunked and
embedded using Azure OpenAI `text-embedding-3-small`, then persisted in a
ChromaDB vector store. The store is uploaded to AWS S3 as the source of truth.

**2. Retrieve** — At runtime, the container downloads the ChromaDB store from
S3. When a question arrives, the most semantically relevant chunks are
retrieved via cosine similarity search.

**3. Generate** — Retrieved context is injected into a prompt template and
sent to `GPT-4o-mini` via Azure OpenAI. The response is returned in the
user's language, with source citations and latency metrics.

> Ingestion is intentionally offline — zero embedding API calls at runtime,
> keeping inference costs minimal and startup predictable.

---

## Performance & Scale

| Metric | Value |
|---|---|
| Avg. response latency | ~2.8s |
| Documents indexed | 8 |
| Embedding model | `text-embedding-3-small` |
| Vector dimensions | 1,536 |
| Rate limit | 10 req / hour / IP |
| Uptime | Best effort (Azure student tier) |
| Deployment | Automated — GitHub Actions + OIDC |

---

## Key Engineering Decisions

| Decision | Choice | Why |
|---|---|---|
| Vector DB | ChromaDB | Persistent, file-based, zero infra overhead |
| LLM | GPT-4o-mini | 10x cheaper than GPT-4o, sufficient quality |
| Auth | OIDC (no static keys) | Follows least-privilege, AWS + Azure |
| ChromaDB location | S3 → downloaded at startup | Separates ingestion from runtime |
| Language detection | `langdetect` + dynamic prompt | No translation layer needed |
| Rate limiting | `slowapi` 10 req/hr/IP | Protects Azure OpenAI credits |

---

## Run Locally

**Prerequisites:** Python 3.12, Docker, Azure OpenAI access, AWS account
```bash
# 1. Clone and install
git clone https://github.com/PavoToXx/preguntale-a-joseph
cd preguntale-a-joseph
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your Azure OpenAI and AWS credentials

# 3. Ingest documents (local, one-time)
python backend/ingest.py

# 4. Run with Docker
docker build -t jar . && docker run -p 8000:8000 -p 8501:8501 jar
```

Then open `http://localhost:8501` in your browser.

---

## Project Structure
```
preguntale-a-joseph/
├── backend/
│   ├── main.py          ← FastAPI app + rate limiting
│   ├── rag_chain.py     ← LangChain RAG pipeline
│   └── ingest.py        ← Offline ingestion script
├── frontend/
│   └── app.py           ← Streamlit UI
├── docs/
│   └── assets/          ← Architecture diagram
├── .github/workflows/   ← CI/CD (Azure deploy + OIDC)
├── Dockerfile
├── requirements.txt     ← Pinned versions
└── .env.example         ← Template — never commit .env
```

---

*Built by [Joseph Dominguez](https://www.linkedin.com/in/josephdominguez-/) · 
[Live demo](https://ijar.azurewebsites.net/)*