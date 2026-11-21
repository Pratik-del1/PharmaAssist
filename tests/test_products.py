from fastapi.testclient import TestClient

from app.main import app

# The test client sends requests straight to our app, without starting a server.
client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_list_products():
    response = client.get("/api/products/")

    assert response.status_code == 200

    products = response.json()
    product_ids = [product["product_id"] for product in products]
    assert "DEMO-001" in product_ids
    assert "DEMO-002" in product_ids


def test_get_product_by_id():
    response = client.get("/api/products/DEMO-001")

    assert response.status_code == 200
    assert response.json()["product_name"] == "DemoCompound Alpha"


def test_get_product_by_lowercase_id():
    response = client.get("/api/products/demo-001")

    assert response.status_code == 200
    assert response.json()["product_id"] == "DEMO-001"


def test_get_unknown_product_returns_404():
    response = client.get("/api/products/UNKNOWN-999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Product not found"


def test_search_products_by_name():
    response = client.get("/api/products/search", params={"q": "beta"})

    assert response.status_code == 200

    products = response.json()
    assert len(products) == 1
    assert products[0]["product_id"] == "DEMO-002"


def test_search_without_query_returns_422():
    response = client.get("/api/products/search")

    assert response.status_code == 422


def test_search_with_empty_query_returns_422():
    response = client.get("/api/products/search", params={"q": ""})

    assert response.status_code == 422
