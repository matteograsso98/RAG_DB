# RAG_DB

## Abstract
This is a personal DB with RAG capabilities to keep track of the research papers, slides, documents I have been reading (and the notes I have been taking) on different topics. 
The purpose is to test/evaluate embedding techniques and vector search and vector databases, to organise and maintain the things I have been learning (and eventually to implement a few-shot, LLM-based, classification of items in categories).

Constraints by desing: 
- Lightweight and low-latency;
- Topics must be divided into main categories, e.g., {Quantum Technologies, AI, Space Telecom, ...};
These categories are then split into sub-categories, e.g., {Quantum Technologies/Quantum Computing}, or {Quantum Technologies/QKD}, etc.;
- Assign a score of relevance (e.g., from 0 to 5 stars) either manually (me) or automatically (e.g., based on the number of times I open a document);
- The final outcome is displayed through a simple GUI. 

The input in my case were messy documents: research papers, personal notes, etc. The format of those might be PDF, but is not strictly PDF (e.g., may be .docs).

The architecture's design is: 

<p align="center">
  <img width="341" height="591" alt="image" src="https://github.com/user-attachments/assets/7d0a1162-02d3-49d2-a243-3e98e93745b4">

### What do the scripts do? 
- **ingest.py** (ingestion) code scans a folder of documents, extracts text (PDF/DOCX/TXT/MD), chunks text (configurable chunk size + overlap), computes embeddings (sentence-transformers), and stores metadata in SQLite and vectors in FAISS. 
Note that if one prefers an external embedding API (OpenAI, Google, ...), replace the model.encode call with embedding API (but keep vector normalization for FAISS). For very large libraries, consider using IndexIVFFlat with training for FAISS (but that complicates persistence).

-  **retrieve_rag.py** (RAG) simply takes top N chunks, concatenate them with short separators and citations (title - page/ chunk idx). Add system + user instructions, then call LLM (locally or via API). 
Example prompt is:
~~~
System: You are a helpful research assistant. Use only the provided document excerpts to answer the question.
If insufficient, say "Insufficient evidence".

Context:
[Document A] (title) — excerpt...
[Document B] — excerpt...

Question: <user question>
~~~
### How does the GUI look? 

<p align="center">
<img width="600" height="800" alt="Screenshot 2025-11-18 at 08 40 32" src="https://github.com/user-attachments/assets/93bcbef8-4714-4ec3-a940-969e23f138ab" />

### Notes & enhancements

- FAISS (IndexFlatIP) works great (low latency) for small-to-medium corpora (thousands). For many thousands or millions, move to an IVF index with training or use a small vector DB (Chroma/Weaviate/Milvus;  but at the cost of more complexity).
- Keep embeddings dimension modest (e.g., 384) to keep memory low.
- Chunk size: 150–500 tokens is typical. Overlap helps continuity. Adjust for your file types (slides short; papers longer).
- Embedding model choices. all-MiniLM-L6-v2 (sentence-transformers) gives compact 384-d embeddings, fast, no external API. For higher-quality embeddings, use all-mpnet-base-v2 (768-d) or an API (e.g., Google embedding models) if you want.
- Re-ranking. Optionally re-rank FAISS results with cross-encoder model for higher accuracy (compute cost higher; can be selective on top 20 results before final top-K).
- Security & Privacy. Keep embeddings and local files private. If using cloud APIs, be aware of data sharing.
- The ingestion script appends new chunks and adds to FAISS. If you change chunking or embedding model, rebuild index from scratch (script to drop old FAISS and re-ingest).

.......................................................................

When you first run the code: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

### References
Google Whitepaper, "_Embeddings & Vector Stores_", Anant Nawalgaria, Xiaoqi Ren, and Charles Sugnet.
