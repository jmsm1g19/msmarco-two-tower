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
INDEX_PATH = "faiss_index_hnsw.bin"
PASSAGE_IDS_PATH = "passage_ids.npy"
PROGRESS_PATH = "index_progress.txt"
BATCH_SIZE = 4096
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Load model and tokenizer and move to GPU
logger.info("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to("cuda")

# Helper function to encode text using the pretrained model on GPU
# def encode_text(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cuda")
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.pooler_output.squeeze().cpu().numpy()  # Move result back to CPU for FAISS

def encode_text_batch(texts, tokenizer, model, max_length=128):
    """
    Encodes a batch of texts using the provided tokenizer and model on GPU.
    Args:
        texts (list of str): List of text passages to encode.
        tokenizer: Pretrained tokenizer.
        model: Pretrained model (on GPU).
        max_length (int): Maximum token length for truncation.
    Returns:
        np.ndarray: Array of text embeddings for the batch.
    """
    # Tokenize the batch of texts with truncation, padding, and max_length set to 128
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    # Move embeddings back to CPU and convert to numpy
    return outputs.pooler_output.cpu().numpy()



# Initialize or load FAISS HNSW index
dimension = model.config.hidden_size
if os.path.exists(INDEX_PATH):
    # Load existing index if available
    logger.info("Loading existing FAISS index...")
    faiss_index = faiss.read_index(INDEX_PATH)
else:
    # Create a new HNSW index
    logger.info("Creating new FAISS HNSW index...")
    faiss_index = faiss.IndexHNSWFlat(dimension, 32)  # 32: number of neighbors (tune as needed)

def connect_db(db_path):
    """Establishes a connection to the SQLite database."""
    logger.info("Connecting to the SQLite database...")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")  # Enable Write-Ahead Logging for performance
    logger.info("Connected to the database with WAL mode enabled.")
    return conn

def get_last_processed_id():
    """Retrieve the last processed passage ID to resume indexing."""
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            return int(f.read().strip())
    return 0

def update_progress(passage_id):
    """Update the last processed passage ID to resume indexing."""
    with open(PROGRESS_PATH, "w") as f:
        f.write(str(passage_id))

def stream_passages_from_db(conn, start_id=0, batch_size=BATCH_SIZE):
    """
    Streams passages from the database in batches to minimize memory usage.
    Yields each batch as a list of (passage_id, passage_text) starting from start_id.
    """
    logger.info("Starting to stream passages from the database...")
    cursor = conn.cursor()
    cursor.execute("SELECT passage_id, passage_text FROM passages WHERE passage_id > ? ORDER BY passage_id ASC", (start_id,))
    batch = []

    for passage_id, passage_text in cursor:
        batch.append((passage_id, passage_text))
        if len(batch) == batch_size:
            yield batch
            batch = []

    # Yield any remaining passages
    if batch:
        yield batch

# def build_faiss_index(index_path=INDEX_PATH):
#     """Builds the FAISS index by encoding passages from the database and adding them to the index incrementally."""
#     conn = connect_db(DB_PATH)
    
#     # Track passage IDs to align them with FAISS internal indices
#     passage_ids = [] if not os.path.exists(PASSAGE_IDS_PATH) else list(np.load(PASSAGE_IDS_PATH))

#     # Determine the last processed passage ID for incremental indexing
#     start_id = get_last_processed_id()
#     total_passages = 8841823

#     logger.info("Starting to build the FAISS index incrementally...")
#     with tqdm(total=total_passages - start_id, desc="Indexing Progress", unit="passage") as pbar:
#         for batch in stream_passages_from_db(conn, start_id=start_id, batch_size=BATCH_SIZE):
#             embeddings = []
#             for passage_id, passage_text in batch:
#                 embedding = encode_text(passage_text)
#                 embeddings.append(embedding)
#                 passage_ids.append(passage_id)

#             # Convert embeddings to numpy array for FAISS
#             embeddings = np.array(embeddings, dtype="float32")
#             faiss_index.add(embeddings)

#             # Update progress bar and progress file with the last processed ID
#             pbar.update(len(batch))
#             update_progress(passage_ids[-1])

#             # Periodically save the index and passage IDs
#             faiss.write_index(faiss_index, index_path)
#             np.save(PASSAGE_IDS_PATH, np.array(passage_ids))

#     conn.close()
#     logger.info(f"FAISS index built and saved to {index_path}")

def build_faiss_index(index_path=INDEX_PATH):
    """Builds the FAISS index by encoding passages from the database and adding them to the index incrementally."""
    conn = connect_db(DB_PATH)
    
    # Track passage IDs to align them with FAISS internal indices
    passage_ids = [] if not os.path.exists(PASSAGE_IDS_PATH) else list(np.load(PASSAGE_IDS_PATH))

    # Determine the last processed passage ID for incremental indexing
    start_id = get_last_processed_id()
    total_passages = 8841823

    logger.info("Starting to build the FAISS index incrementally...")
    with tqdm(total=total_passages - start_id, desc="Indexing Progress", unit="passage") as pbar:
        for batch in stream_passages_from_db(conn, start_id=start_id, batch_size=BATCH_SIZE):
            passage_texts = [passage_text for _, passage_text in batch]
            passage_ids.extend([passage_id for passage_id, _ in batch])

            # Encode the batch of texts using batch encoding with max_length=128
            embeddings = encode_text_batch(passage_texts, tokenizer, model, max_length=128)

            # Convert embeddings to numpy array for FAISS
            embeddings = np.array(embeddings, dtype="float32")
            faiss_index.add(embeddings)

            # Update progress bar and progress file with the last processed ID
            pbar.update(len(batch))
            update_progress(passage_ids[-1])

            # Periodically save the index and passage IDs
            faiss.write_index(faiss_index, index_path)
            np.save(PASSAGE_IDS_PATH, np.array(passage_ids))

    conn.close()
    logger.info(f"FAISS index built and saved to {index_path}")


# Function to load the FAISS index and passage IDs if already created
def load_faiss_index(index_path=INDEX_PATH):
    """Loads a saved FAISS index and associated passage IDs."""
    if not os.path.exists(index_path) or not os.path.exists(PASSAGE_IDS_PATH):
        raise FileNotFoundError("FAISS index or passage ID mapping not found. Please build the index first.")
    
    # Load index and passage IDs
    faiss_index = faiss.read_index(index_path)
    passage_ids = np.load(PASSAGE_IDS_PATH)
    logger.info(f"FAISS index loaded from {index_path}")
    return faiss_index, passage_ids

# Run indexing if index does not exist
if not os.path.exists(INDEX_PATH):
    logger.info("No existing FAISS index found. Building a new index...")
    build_faiss_index()
else:
    logger.info("Existing FAISS index found. Loading the index...")
    faiss_index, passage_ids = load_faiss_index()
