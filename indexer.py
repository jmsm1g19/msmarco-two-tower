import sqlite3
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import os
import logging
from tqdm import tqdm

# Configuration for paths, batch size, and logging
DB_PATH = "msmarco_dataset.db"
INDEX_PATH = "faiss_index.bin"
BATCH_SIZE = 1000
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Load model and tokenizer
logger.info("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Helper function to encode text using the pretrained model
def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.squeeze().numpy()

# Initialize FAISS index
dimension = model.config.hidden_size  # e.g., 384 for MiniLM
faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product search

def connect_db(db_path):
    """Establishes a connection to the SQLite database."""
    return sqlite3.connect(db_path)

def get_total_passages(conn):
    """Retrieves the total number of passages in the database for progress tracking."""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM passages")
    total = cursor.fetchone()[0]
    logger.info(f"Total passages in database: {total}")
    return total

def stream_passages_from_db(conn, batch_size=BATCH_SIZE):
    """
    Streams passages from the database in batches to minimize memory usage.
    Yields each batch as a list of (passage_id, passage_text).
    """
    cursor = conn.cursor()
    cursor.execute("SELECT passage_id, passage_text FROM passages")
    batch = []

    for passage_id, passage_text in cursor:
        batch.append((passage_id, passage_text))
        if len(batch) == batch_size:
            yield batch
            batch = []

    # Yield any remaining passages
    if batch:
        yield batch

def build_faiss_index(index_path=INDEX_PATH):
    """Builds the FAISS index by encoding passages from the database and adding them to the index."""
    conn = connect_db(DB_PATH)
    
    # Track passage IDs to align them with FAISS internal indices
    passage_ids = []

    # Get the total number of passages for progress tracking
    total_passages = get_total_passages(conn)

    logger.info("Starting to build the FAISS index...")
    with tqdm(total=total_passages, desc="Indexing Progress", unit="passage") as pbar:
        for batch in stream_passages_from_db(conn, batch_size=BATCH_SIZE):
            embeddings = []
            for passage_id, passage_text in batch:
                embedding = encode_text(passage_text)
                embeddings.append(embedding)
                passage_ids.append(passage_id)

            # Convert embeddings to numpy array for FAISS
            embeddings = np.array(embeddings, dtype="float32")
            faiss_index.add(embeddings)

            # Update progress bar with number of passages processed in the batch
            pbar.update(len(batch))

    conn.close()

    # Save passage_ids for later retrieval and save FAISS index to disk
    np.save("passage_ids.npy", np.array(passage_ids))
    faiss.write_index(faiss_index, index_path)
    logger.info(f"FAISS index built and saved to {index_path}")

# Function to load the FAISS index and passage IDs if already created
def load_faiss_index(index_path=INDEX_PATH):
    """Loads a saved FAISS index and associated passage IDs."""
    if not os.path.exists(index_path) or not os.path.exists("passage_ids.npy"):
        raise FileNotFoundError("FAISS index or passage ID mapping not found. Please build the index first.")
    
    # Load index and passage IDs
    faiss_index = faiss.read_index(index_path)
    passage_ids = np.load("passage_ids.npy")
    logger.info(f"FAISS index loaded from {index_path}")
    return faiss_index, passage_ids

# Run indexing if index does not exist
if not os.path.exists(INDEX_PATH):
    logger.info("No existing FAISS index found. Building a new index...")
    build_faiss_index()
else:
    logger.info("Existing FAISS index found. Loading the index...")

# Example usage: load index and passage IDs for further processing
faiss_index, passage_ids = load_faiss_index()
