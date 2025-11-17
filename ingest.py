# ingest.py
import os
import sqlite3
import uuid
import json
import math
from pathlib import Path
from datetime import datetime

# install: pip install sentence-transformers faiss-cpu PyPDF2 python-docx tqdm
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

# simple text extraction helpers
def extract_text_from_pdf(path):
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

def extract_text_from_docx(path):
    import docx
    doc = docx.Document(path)
    paras = [p.text for p in doc.paragraphs]
    return "\n".join(paras)

def extract_text(path: Path):
    suff = path.suffix.lower()
    if suff == ".pdf":
        return extract_text_from_pdf(path)
    elif suff in (".docx",):
        return extract_text_from_docx(path)
    elif suff in (".txt", ".md"):
        return path.read_text(encoding="utf-8")
    else:
        # fallback: try reading bytes and decode
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except:
            return ""

# chunking
def chunk_text(text, chunk_size=500, overlap=50):
    tokens = text.split()
    chunks = []
    i = 0
    n = len(tokens)
    while i < n:
        chunk_tokens = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk_tokens))
        i += chunk_size - overlap
    return chunks

# DB helpers
DB_PATH = "rag_library.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    with open("schema.sql", "w") as f:  # create a backup schema file
        pass
    # create tables if not exists (quick inline)
    c.executescript("""
    CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        parent_id INTEGER,
        name TEXT NOT NULL,
        description TEXT
    );
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        filename TEXT,
        filetype TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        category_id INTEGER,
        manual_relevance INTEGER DEFAULT NULL,
        auto_relevance REAL DEFAULT 0.0,
        access_count INTEGER DEFAULT 0,
        num_chunks INTEGER DEFAULT 0,
        notes TEXT
    );
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        chunk_index INTEGER,
        text TEXT,
        token_count INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS access_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        event_type TEXT,
        user TEXT DEFAULT 'local',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    return conn

# FAISS index helpers (persist index to disk)
FAISS_INDEX_PATH = "faiss_index.bin"
EMBED_DIM = 384  # adjust to model

class VectorStore:
    def __init__(self, dim=EMBED_DIM, path=FAISS_INDEX_PATH):
        self.dim = dim
        self.path = path
        # use IndexFlatIP for cosine-like similarity after normalization
        self.index = faiss.IndexFlatIP(dim)
        self.ids = []  # map position -> chunk_id
        if os.path.exists(path):
            self.load()

    def add(self, vectors, ids):
        # vectors: numpy array (n, dim), ids: list of chunk ids
        import numpy as np
        # normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.ids.extend(ids)
        self.save()

    def search(self, query_vector, top_k=10):
        import numpy as np
        faiss.normalize_L2(query_vector)
        D, I = self.index.search(query_vector, top_k)
        # map I indices to stored chunk ids
        results = []
        for distances, indices in zip(D, I):
            for d, idx in zip(distances, indices):
                if idx < 0 or idx >= len(self.ids):
                    continue
                chunk_id = self.ids[idx]
                results.append((chunk_id, float(d)))
        return results

    def save(self):
        faiss.write_index(self.index, self.path)
        # persist ids list
        with open(self.path + ".ids.json", "w") as f:
            json.dump(self.ids, f)

    def load(self):
        if os.path.exists(self.path):
            self.index = faiss.read_index(self.path)
            with open(self.path + ".ids.json", "r") as f:
                self.ids = json.load(f)

# main ingestion function
def ingest_folder(folder_path, model_name="all-MiniLM-L6-v2", chunk_size=250, overlap=40):
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(model_name)
    emb_dim = model.get_sentence_embedding_dimension()

    vs = VectorStore(dim=emb_dim)
    conn = init_db()
    cur = conn.cursor()

    folder = Path(folder_path)
    files = [p for p in folder.rglob("*") if p.suffix.lower() in (".pdf", ".docx", ".txt", ".md")]
    for f in tqdm(files, desc="Files"):
        try:
            text = extract_text(f)
            if not text or len(text.strip()) < 50:
                print(f"skip {f} (no text)")
                continue
            title = f.stem
            filetype = f.suffix.lower()
            # insert document
            now = datetime.utcnow().isoformat()
            cur.execute("INSERT INTO documents (title, filename, filetype, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (title, str(f), filetype, now, now))
            doc_id = cur.lastrowid

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            embeddings = []
            chunk_ids = []
            for idx, ch in enumerate(chunks):
                cur.execute("INSERT INTO chunks (document_id, chunk_index, text, token_count) VALUES (?, ?, ?, ?)",
                            (doc_id, idx, ch, len(ch.split())))
                chunk_id = cur.lastrowid
                chunk_ids.append(chunk_id)
                embeddings.append(ch)

            # embed in batch
            vectors = model.encode(embeddings, show_progress_bar=False, convert_to_numpy=True)
            # Add to FAISS and persist
            vs.add(vectors, chunk_ids)

            # update document num_chunks
            cur.execute("UPDATE documents SET num_chunks = ? WHERE id = ?", (len(chunks), doc_id))
            conn.commit()
        except Exception as e:
            print("Error ingesting", f, e)

    conn.close()
    print("Ingestion done.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="folder with documents")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--overlap", type=int, default=40)
    args = p.parse_args()
    ingest_folder(args.folder, model_name=args.model, chunk_size=args.chunk, overlap=args.overlap)
