import sqlite3
from datasets import load_dataset
from tqdm import tqdm

conn = sqlite3.connect("msmarco_dataset.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS queries (
    query_id INTEGER PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_type TEXT,
    split TEXT NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS passages (
    passage_id INTEGER PRIMARY KEY,
    passage_text TEXT NOT NULL,
    url TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS query_passage_map (
    query_id INTEGER,
    passage_id INTEGER,
    is_selected INTEGER,
    FOREIGN KEY (query_id) REFERENCES queries(query_id),
    FOREIGN KEY (passage_id) REFERENCES passages(passage_id),
    PRIMARY KEY (query_id, passage_id)
)
""")

conn.commit()

batch_size = 1000

def process_split(split_name):
    dataset = load_dataset("ms_marco", "v2.1", split=split_name, streaming=True)

    query_batch = []
    passage_batch = []
    map_batch = []

    for example in tqdm(dataset, desc=f"Processing {split_name} split"):
        query_id = example.get("query_id")
        query_text = example.get("query")
        query_type = example.get("query_type", None)

        query_batch.append((query_id, query_text, query_type, split_name))

        passages = example.get("passages", {})

        passage_texts = passages.get("passage_text", [])
        is_selected_list = passages.get("is_selected", [])
        urls = passages.get("url", [])

        for i in range(len(passage_texts)):
            passage_text = passage_texts[i]
            is_selected = is_selected_list[i] if i < len(is_selected_list) else 0
            url = urls[i] if i < len(urls) else None

            passage_id = hash(passage_text) & ((1 << 63) - 1)

            passage_batch.append((passage_id, passage_text, url))

            map_batch.append((query_id, passage_id, is_selected))

        if len(query_batch) >= batch_size:
            cursor.executemany("""
                INSERT OR IGNORE INTO queries (query_id, query_text, query_type, split) VALUES (?, ?, ?, ?)
            """, query_batch)
            
            cursor.executemany("""
                INSERT OR IGNORE INTO passages (passage_id, passage_text, url) VALUES (?, ?, ?)
            """, passage_batch)
            
            cursor.executemany("""
                INSERT OR IGNORE INTO query_passage_map (query_id, passage_id, is_selected) VALUES (?, ?, ?)
            """, map_batch)
            
            conn.commit()
            query_batch = []
            passage_batch = []
            map_batch = []

    if query_batch:
        cursor.executemany("""
            INSERT OR IGNORE INTO queries (query_id, query_text, query_type, split) VALUES (?, ?, ?, ?)
        """, query_batch)
    
    if passage_batch:
        cursor.executemany("""
            INSERT OR IGNORE INTO passages (passage_id, passage_text, url) VALUES (?, ?, ?)
        """, passage_batch)
    
    if map_batch:
        cursor.executemany("""
            INSERT OR IGNORE INTO query_passage_map (query_id, passage_id, is_selected) VALUES (?, ?, ?)
        """, map_batch)

    conn.commit()
    print(f"Completed processing for {split_name} split.")


for split in ['train', 'validation', 'test']:
    process_split(split)

conn.close()
print("Database population complete.")
