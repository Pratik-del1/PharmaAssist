
from fastapi import FastAPI

from app.api.products import router as products_router

app = FastAPI(
    title="PharmaAssist API",
    description=(
        "AI-powered pharmaceutical "
        "product information assistant"
    ),
    version="0.2.0"
)

app.include_router(products_router)


@app.get("/")
async def root():
    return {
        "message": "Welcome to PharmaAssist",
        "status": "running",
        "version": "0.2.0"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "PharmaAssist API"
    }
