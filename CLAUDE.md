
# PharmaAssist - Project Instructions

## 1. Project Overview

PharmaAssist is an AI-powered pharmaceutical customer support chatbot.

It should answer customer questions about pharmaceutical company products using verified company documents.

The project will use:
- Python
- FastAPI
- PostgreSQL and pgvector
- LangChain / LangGraph only where useful
- Gemini API initially
- Hugging Face for later fine-tuning
- React for the frontend
- Docker for deployment

The main objective is to build a working RAG chatbot first and introduce fine-tuning later.

## 2. Most Important Development Rules

I am learning AI Engineering while building this project.

Write code that is simple, readable and easy for a beginner to understand.

STRICT RULES:

- Do not overengineer anything.
- Prefer simple functions over unnecessary classes.
- Avoid unnecessary abstraction layers.
- Avoid design patterns unless genuinely needed.
- Do not create files without a clear purpose.
- Do not introduce unnecessary dependencies.
- Do not use complex frameworks when a simple Python solution works.
- Avoid premature optimization.
- Keep functions short and readable.
- Use meaningful variable and function names.
- Add comments only where the logic is not obvious.
- Prefer straightforward Python over clever one-liners.
- Do not generate excessive boilerplate.

## 3. Teaching Requirements

Explain every significant implementation.

For every task:
1. Explain what we are building.
2. Explain why we need it.
3. Explain how it works.
4. Show which files will be created or modified.
5. Implement the feature.
6. Explain the important code and libraries.
7. Run relevant tests.
8. Summarize what I should understand before moving forward.

Do not assume I understand code simply because you generated it.

## 4. Development Workflow

Follow this sequence:

1. Inspect the existing code before making changes.
2. Understand the existing project structure.
3. For changes involving multiple files, propose a plan first.
4. Wait for my approval before implementing the plan.
5. Make small, incremental changes.
6. Preserve existing working functionality.
7. Run relevant tests after implementation.
8. Fix errors before moving to the next task.

Never rewrite the entire project unnecessarily.

Never silently change existing dependencies or Python versions.

## 5. Code Quality

- Use type hints where they improve clarity.
- Handle errors explicitly.
- Use Pydantic for API data validation.
- Keep API routes simple.
- Keep business logic separate from API routes when necessary.
- Use environment variables for API keys.
- Never expose API keys or sensitive information.
- Maintain a useful README.
- Write tests for important functionality.

## 6. Pharmaceutical Safety

This is a pharmaceutical information chatbot.

- Use verified company documentation as the source of product information.
- Never invent medicine information.
- Never present fictional demonstration products as real medicines.
- Do not generate unsupported medical advice.
- Do not independently prescribe medicines or recommend dosage changes.
- Clearly identify unsupported or missing information.
- Implement escalation for adverse events and product complaints.
- Keep document sources, approval status and versions traceable.
- Do not use unverified information in production answers.

## 7. Implementation Roadmap

Follow this order:

Phase 1: Project setup and FastAPI
Phase 2: Product knowledge base
Phase 3: PDF document ingestion
Phase 4: Text extraction and chunking
Phase 5: Embeddings
Phase 6: PostgreSQL and pgvector
Phase 7: RAG retrieval
Phase 8: LLM integration
Phase 9: Safety and guardrails
Phase 10: RAG evaluation
Phase 11: Fine-tuning using LoRA/QLoRA
Phase 12: Integrate fine-tuned model with RAG
Phase 13: React frontend
Phase 14: Docker and deployment

Do not skip phases or implement everything at once.

## 8. My Preferences

I prefer:
- Beginner-friendly Python.
- Minimal and understandable code.
- Practical implementation over unnecessary theory.
- Step-by-step development.
- Complete updated files when code changes are requested.
- Clear explanations of technical decisions.

The final project should be professional and functional, but the codebase should remain simple enough for me to understand and maintain.
