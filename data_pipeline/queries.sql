
-- Query 1: SELECT + WHERE
SELECT title, rating, price_gbp
FROM books
WHERE rating = 5;


-- Query 2: ORDER BY + LIMIT
SELECT title, rating, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 10;


-- Query 3: DISTINCT
SELECT DISTINCT category_name
FROM categories;


-- Query 4: BETWEEN
SELECT title, price_gbp, rating
FROM books
WHERE price_gbp BETWEEN 20 AND 40;


-- Query 5: JOIN
SELECT
    books.title,
    categories.category_name,
    books.price_gbp,
    books.rating
FROM books
JOIN categories
    ON books.category_id = categories.category_id;
