
import sqlite3
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50

CATEGORIES = {
    "Travel": "catalogue/category/books/travel_2/index.html",
    "Mystery": "catalogue/category/books/mystery_3/index.html",
    "Historical Fiction": "catalogue/category/books/historical-fiction_20/index.html",
}

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "output" / "books.db"


def parse_rating(text):
    mapping = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    return mapping.get(text.strip())


def parse_price(text):
    match = re.search(r"[\d.]+", text)
    return float(match.group()) if match else None


def scrape_category(category, relative_url):
    response = requests.get(BASE_URL + relative_url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    rows = []

    while True:
        for article in soup.select("article.product_pod"):
            title = article.h3.a.get("title", "").strip()
            price = parse_price(article.select_one(".price_color").get_text())
            rating_text = article.select_one("p.star-rating").get("class", [])
            rating_word = next(
                (x for x in rating_text if x in
                 ["One", "Two", "Three", "Four", "Five"]),
                None
            )
            availability = article.select_one(".availability").get_text(
                " ", strip=True
            )

            rows.append({
                "title": title,
                "price_gbp": price,
                "star_rating": rating_word,
                "rating": parse_rating(rating_word) if rating_word else None,
                "availability": availability,
                "in_stock": "In stock" in availability,
                "category": category,
            })

        next_link = soup.select_one("li.next a")
        if not next_link:
            break

        next_url = next_link.get("href")
        current_url = response.url.rsplit("/", 1)[0] + "/" + next_url

        response = requests.get(current_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

    return rows


def build_dataset():
    all_rows = []

    for category, url in CATEGORIES.items():
        all_rows.extend(scrape_category(category, url))

    df = pd.DataFrame(all_rows)

    # Required fixed project baseline.
    df["price_gbp"] = pd.to_numeric(df["price_gbp"], errors="coerce")
    df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating"] = df["rating"].fillna(df["rating"].median()).round().astype(int)

    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

    return df


def create_database(df):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                category_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_name TEXT UNIQUE NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS books (
                book_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                star_rating TEXT,
                rating INTEGER,
                availability TEXT,
                in_stock INTEGER,
                category_id INTEGER NOT NULL,
                FOREIGN KEY (category_id)
                    REFERENCES categories(category_id)
            )
        """)

        conn.execute("DELETE FROM books")
        conn.execute("DELETE FROM categories")

        for category in sorted(df["category"].unique()):
            conn.execute(
                "INSERT INTO categories (category_name) VALUES (?)",
                (category,)
            )

        category_map = dict(
            conn.execute(
                "SELECT category_name, category_id FROM categories"
            ).fetchall()
        )

        for _, row in df.iterrows():
            conn.execute("""
                INSERT INTO books
                (title, price_gbp, price_inr, star_rating, rating,
                 availability, in_stock, category_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                row["star_rating"],
                row["rating"],
                row["availability"],
                int(row["in_stock"]),
                category_map[row["category"]],
            ))

        conn.commit()


if __name__ == "__main__":
    df = build_dataset()
    create_database(df)

    print(f"Books scraped: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print(f"Database: {DB_PATH}")
