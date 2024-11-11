import psycopg2
from psycopg2 import pool
from datasets import load_dataset
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import math

# Updated batch size
batch_size = 5000  # Experiment with larger batch sizes to reduce transaction count
total_queries = 1010916
total_passages = 8841823
total_query_passage_map = 10069342

# Database connection pool
DB_PARAMS = {
    'dbname': 'msmarco',
    'user': 'postgres',
    'password': '1923',
    'host': 'localhost',
    'port': '5432'
}

connection_pool = psycopg2.pool.SimpleConnectionPool(1, 20, **DB_PARAMS)  # Adjust pool size as needed

def initialize_db():
    conn = connection_pool.getconn()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS queries (
        query_id BIGINT PRIMARY KEY,
        query_text TEXT NOT NULL,
        query_type TEXT,
        split TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS passages (
        passage_id BIGINT PRIMARY KEY,
        passage_text TEXT NOT NULL,
        url TEXT
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS query_passage_map (
        query_id BIGINT,
        passage_id BIGINT,
        is_selected INTEGER,
        PRIMARY KEY (query_id, passage_id),
        FOREIGN KEY (query_id) REFERENCES queries(query_id),
        FOREIGN KEY (passage_id) REFERENCES passages(passage_id)
    )
    """)
    conn.commit()
    cursor.close()
    connection_pool.putconn(conn)

def insert_batches(query_batch, passage_batch, map_batch):
    conn = connection_pool.getconn()
    cursor = conn.cursor()

    cursor.executemany("""
        INSERT INTO queries (query_id, query_text, query_type, split) 
        VALUES (%s, %s, %s, %s) ON CONFLICT (query_id) DO NOTHING
    """, query_batch)
    cursor.executemany("""
        INSERT INTO passages (passage_id, passage_text, url) 
        VALUES (%s, %s, %s) ON CONFLICT (passage_id) DO NOTHING
    """, passage_batch)
    cursor.executemany("""
        INSERT INTO query_passage_map (query_id, passage_id, is_selected) 
        VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
    """, map_batch)

    conn.commit()
    cursor.close()
    connection_pool.putconn(conn)

def process_split(split_name, total_examples):
    dataset = load_dataset("ms_marco", "v2.1", split=split_name, streaming=True)
    query_batch, passage_batch, map_batch = [], [], []

    with tqdm(total=math.ceil(total_examples / batch_size), desc=f"Processing {split_name} split", unit="batch") as progress_bar:
        with ThreadPoolExecutor(max_workers=4) as executor:
            for example in dataset:
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

                # Insert when batch size is reached
                if len(query_batch) >= batch_size:
                    executor.submit(insert_batches, query_batch, passage_batch, map_batch)
                    query_batch, passage_batch, map_batch = [], [], []
                    progress_bar.update(1)

            # Process remaining records
            if query_batch:
                executor.submit(insert_batches, query_batch, passage_batch, map_batch)

    print(f"Completed processing for {split_name} split.")

# Run each split in multiprocessing mode
def run_multiprocessing():
    initialize_db()
    with multiprocessing.Pool(processes=3) as pool:
        pool.starmap(process_split, [
            ('train', total_queries),
            ('validation', total_passages),
            ('test', total_query_passage_map)
        ])

if __name__ == '__main__':
    run_multiprocessing()
    print("Database population complete.")
