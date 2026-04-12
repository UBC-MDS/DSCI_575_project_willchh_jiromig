import pytest


@pytest.fixture()
def fake_corpus():
    """Small fake product corpus for testing. Mimics the structure from utils.load_corpus()."""
    return [
        {
            "parent_asin": "B001",
            "title": "Vitamin C Serum",
            "text": "Vitamin C Serum A brightening serum with 20% vitamin C for dark spots and uneven skin tone",
            "price": 24.99,
            "average_rating": 4.5,
            "images": [],
        },
        {
            "parent_asin": "B002",
            "title": "Coconut Oil Shampoo",
            "text": "Coconut Oil Shampoo Moisturizing shampoo with organic coconut oil for dry and damaged hair",
            "price": 12.99,
            "average_rating": 4.0,
            "images": [],
        },
        {
            "parent_asin": "B003",
            "title": "SPF 50 Sunscreen Lotion",
            "text": "SPF 50 Sunscreen Lotion Broad spectrum sunscreen with SPF 50 lightweight and non-greasy",
            "price": 15.99,
            "average_rating": 4.8,
            "images": [],
        },
        {
            "parent_asin": "B004",
            "title": "Retinol Night Cream",
            "text": "Retinol Night Cream Anti-aging night cream with retinol and hyaluronic acid for fine lines",
            "price": 29.99,
            "average_rating": 4.3,
            "images": [],
        },
        {
            "parent_asin": "B005",
            "title": "Tea Tree Face Wash",
            "text": "Tea Tree Face Wash Gentle foaming cleanser with tea tree oil for acne-prone skin",
            "price": 9.99,
            "average_rating": 4.1,
            "images": [],
        },
        {
            "parent_asin": "B006",
            "title": "Argan Oil Hair Mask",
            "text": "Argan Oil Hair Mask Deep conditioning treatment with argan oil for frizzy and dry hair",
            "price": 18.50,
            "average_rating": 4.6,
            "images": [],
        },
        {
            "parent_asin": "B007",
            "title": "Charcoal Peel Off Mask",
            "text": "Charcoal Peel Off Mask Activated charcoal mask for blackhead removal and pore cleansing",
            "price": 11.99,
            "average_rating": 3.9,
            "images": [],
        },
        {
            "parent_asin": "B008",
            "title": "Rose Water Toner",
            "text": "Rose Water Toner Natural rose water toner for hydrating and soothing sensitive skin",
            "price": 8.99,
            "average_rating": 4.7,
            "images": [],
        },
        {
            "parent_asin": "B009",
            "title": "Keratin Smoothing Treatment",
            "text": "Keratin Smoothing Treatment Professional keratin treatment for straightening and smoothing hair",
            "price": 34.99,
            "average_rating": 4.2,
            "images": [],
        },
        {
            "parent_asin": "B010",
            "title": "Aloe Vera Gel Moisturizer",
            "text": "Aloe Vera Gel Moisturizer Lightweight gel moisturizer with aloe vera for oily and combination skin",
            "price": 7.99,
            "average_rating": 4.4,
            "images": [],
        },
    ]
