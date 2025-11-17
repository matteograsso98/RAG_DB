# RAG_DB

## Objective 
This is a personal DB with RAG capabilities to keep track of the research papers, slides, documents I have been reading (and the notes I have been taking) on different topics. 
The purpose is both to test/evaluate RAG techniques, to implement a few-shot LLM-based classification of the objects in categories, and to organise and maintain the things I have been learning.

Constraints by desing: 
- Lightweight and low-latency
- Topics must be divided into main categories, e.g., {Quantum Technologies, AI, Space Telecom}.
Split these main categories into sub-categories, e.g., {Quantum Technologies/Quantum Computing}, or {Quantum Technologies/QKD}, etc. 
- Assign a score of relevance (e.g., from 0 to 5 stars) either manually (me) or based on the number of times I consult a given document. 
- The final outcome is displayed through a simple GUI (we will define it later) 

The input in my case are messy documents: research papers, personal notes, etc. The format of those might be PDF, but is not strictly PDF (e.g., may be .docs).

<p align="center">
  <img width="341" height="591" alt="image" src="https://github.com/user-attachments/assets/7d0a1162-02d3-49d2-a243-3e98e93745b4">

The ingest.py (ingestion) code scans a folder of documents, extracts text (PDF/DOCX/TXT/MD), chunks text (configurable chunk size + overlap), computes embeddings (sentence-transformers), and stores metadata in SQLite and vectors in FAISS. 
Note that if one prefers an external embedding API (OpenAI), replace the model.encode call with embedding API (but keep vector normalization for FAISS). For very large libraries, consider using IndexIVFFlat with training for FAISS (but that complicates persistence).

The retrieve_rag.py (RAG) simply takes top N chunks, concatenate them with short separators and citations (title - page/ chunk idx). Add system + user instructions, then call LLM (locally or via API). 
Example prompt is:
~~~
System: You are a helpful research assistant. Use only the provided document excerpts to answer the question.
If insufficient, say "Insufficient evidence".

Context:
[Document A] (title) — excerpt...
[Document B] — excerpt...

Question: <user question>
~~~

* Notes & enhancements
Performance & scaling
- FAISS (IndexFlatIP) works great for small-to-medium corpora (thousands → low-latency). For many thousands or millions, move to an IVF index with training or use a small vector DB (Chroma/Weaviate/Milvus). But those add complexity.
- Keep embeddings dimension modest (e.g., 384) to keep memory low.
- Chunk size: 150–500 tokens is typical. Overlap helps continuity. Adjust for your file types (slides short; papers longer).
- Embedding model choices. all-MiniLM-L6-v2 (sentence-transformers) gives compact 384-d embeddings, fast, no external API. For higher-quality embeddings, use all-mpnet-base-v2 (768-d) or an API (OpenAI) if you want.
- Persistence & backups. Commit faiss_index.bin and .ids.json to disk backup. Store data/docs separately (or use disk paths in DB).
- Re-ranking. Optionally re-rank FAISS results with cross-encoder model for higher accuracy (compute cost higher; can be selective on top 20 results before final top-K).
- Security & Privacy. Keep embeddings and local files private. If using cloud APIs, be aware of data sharing.
- The ingestion script appends new chunks and adds to FAISS. If you change chunking or embedding model, rebuild index from scratch (script to drop old FAISS and re-ingest).


When you first run the code: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

~~~
