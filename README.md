# PharmaAssist

PharmaAssist is a planned AI-powered customer support assistant for pharmaceutical products. The goal is for it to answer questions using only verified company documents.

The project is in an early stage. Right now it is a FastAPI service that serves a small product catalogue. The AI chatbot is not built yet.

> **Note:** All products in `data/products.json` are fictional and exist only for software testing. They are not real medicines.

## Current features

- FastAPI app with a welcome endpoint (`/`) and a health check (`/health`)
- Product catalogue loaded from `data/products.json`
- Product data validated with Pydantic schemas
- Product API endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/products/` | List all products |
| GET | `/api/products/search?q=<text>` | Search by product name, brand name or product ID |
| GET | `/api/products/{product_id}` | Get one product by ID (not case-sensitive) |

- Automated tests for all endpoints

## Technology stack

- Python 3.9+
- FastAPI (web framework)
- Pydantic (data validation)
- Uvicorn (server, installed with `fastapi[standard]`)
- pytest and httpx (testing)

## Project folder structure

```
Pharmaassist/
├── app/
│   ├── main.py                    # Creates the FastAPI app and registers routers
│   ├── api/products.py            # Product API endpoints
│   ├── schemas/product.py         # Pydantic models for product data
│   └── services/product_catalogue.py  # Loads and searches products
├── data/
│   └── products.json              # Fictional demo products
├── tests/
│   └── test_products.py           # API tests
├── pytest.ini                     # pytest settings
└── requirements.txt               # Python dependencies
```

Other empty folders (such as `app/rag`, `app/llm` and `training`) are placeholders for later development phases.

## Installation

```bash
cd Pharmaassist
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the FastAPI server

```bash
source .venv/bin/activate
fastapi dev app/main.py
```

Then open:

- http://127.0.0.1:8000 for the welcome message
- http://127.0.0.1:8000/docs for interactive API documentation, where you can try every endpoint

The server reloads automatically when you save a code change. Press `Ctrl+C` to stop it.

## Running tests

```bash
source .venv/bin/activate
pytest -v
```

To run a single test:

```bash
pytest tests/test_products.py::test_get_product_by_id -v
```

## Current development status

| Phase | Status |
|---|---|
| 1. Project setup and FastAPI | Done |
| 2. Product knowledge base | In progress (JSON catalogue and product API done) |
| 3–14. Document ingestion, embeddings, database, RAG, LLM, safety, evaluation, fine-tuning, frontend, deployment | Not started |

Document ingestion, RAG, LLM integration, fine-tuning and the chatbot are **not implemented yet**.
