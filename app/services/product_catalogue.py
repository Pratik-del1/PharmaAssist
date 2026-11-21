
import json
from pathlib import Path
from typing import Optional

from app.schemas.product import Product


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCT_FILE = PROJECT_ROOT / "data" / "products.json"


def load_products() -> list[Product]:
    if not PRODUCT_FILE.exists():
        return []

    with PRODUCT_FILE.open("r", encoding="utf-8") as file:
        raw_products = json.load(file)

    return [
        Product.model_validate(product)
        for product in raw_products
    ]


def get_all_products() -> list[Product]:
    return load_products()


def get_product_by_id(product_id: str) -> Optional[Product]:
    products = load_products()

    for product in products:
        if product.product_id.lower() == product_id.lower():
            return product

    return None


def search_products(query: str) -> list[Product]:
    products = load_products()
    query = query.strip().lower()

    if not query:
        return products

    return [
        product
        for product in products
        if query in product.product_name.lower()
        or query in product.product_id.lower()
        or (
            product.brand_name
            and query in product.brand_name.lower()
        )
    ]
