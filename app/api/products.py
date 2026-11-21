
from fastapi import APIRouter, HTTPException, Query

from app.schemas.product import Product
from app.services.product_catalogue import (
    get_all_products,
    get_product_by_id,
    search_products,
)

router = APIRouter(
    prefix="/api/products",
    tags=["Products"]
)


@router.get("/", response_model=list[Product])
async def list_products():
    """
    Development catalogue endpoint.
    Production access must enforce approval and authorization.
    """
    return get_all_products()


@router.get("/search", response_model=list[Product])
async def search_product(
    q: str = Query(..., min_length=1)
):
    return search_products(q)


@router.get("/{product_id}", response_model=Product)
async def retrieve_product(product_id: str):
    product = get_product_by_id(product_id)

    if product is None:
        raise HTTPException(
            status_code=404,
            detail="Product not found"
        )

    return product
