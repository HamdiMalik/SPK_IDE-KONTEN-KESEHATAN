CREATE TABLE IF NOT EXISTS tabel_tweet_mentah (
    id             SERIAL PRIMARY KEY,
    created_at     TIMESTAMP,
    favorite_count INT,
    full_text      TEXT,
    case_folding   TEXT,
    cleaning       TEXT,
    normalisasi    TEXT,
    sentiment      VARCHAR(20)
);

COPY tabel_tweet_mentah (
    created_at,
    favorite_count,
    full_text,
    case_folding,
    cleaning,
    normalisasi,
    sentiment
)
FROM '/var/lib/postgresql/data_labeled.csv'
WITH (
    FORMAT CSV,
    DELIMITER ',',
    HEADER TRUE,
    ENCODING 'UTF8'
);
