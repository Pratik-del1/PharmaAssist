
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


class Ingredient(BaseModel):
    name: str
    strength: Optional[str] = None


class Product(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str
    product_name: str
    brand_name: Optional[str] = None

    category: str
    dosage_form: Optional[str] = None
    ingredients: list[Ingredient] = Field(default_factory=list)

    packaging: list[str] = Field(default_factory=list)
    manufacturer: Optional[str] = None

    product_status: Literal[
        "active", "discontinued", "under_review"
    ] = "active"

    approval_status: Literal[
        "approved", "pending", "rejected"
    ] = "pending"

    effective_date: Optional[date] = None

    description: Optional[str] = None
    source_document_ids: list[str] = Field(default_factory=list)
