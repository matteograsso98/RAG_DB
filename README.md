# RAG_DB
Personal DB with RAG capabilities to keep track of the research papers/ slides / documents I have been reading, and the notes I have been taking on different topics.  
The goal is to organise all that knowledge, and maintain it in a clean and easy way as the documents increase. 

Constraints by desing: 
- Lightweight and low-latency
- Topics must be divided into main categories, e.g., {Quantum Technologies, AI, Space Telecom}.
Split these main categories into sub-categories, e.g., {Quantum Technologies/Quantum Computing}, or {Quantum Technologies/QKD}, etc. 
- Assign a score of relevance (e.g., from 0 to 5 stars) either manually (me) or based on the number of times I consult a given document. 
- The final outcome is displayed through a simple GUI (we will define it later) 

The input in my case are messy documents: research papers, personal notes, etc. The format of those might be PDF, but is not strictly PDF (e.g., may be .docs).

<p align="center">
  <img width="341" height="591" alt="image" src="https://github.com/user-attachments/assets/7d0a1162-02d3-49d2-a243-3e98e93745b4">
p>

The ingest.py (ingestion) code scans a folder of documents, extracts text (PDF/DOCX/TXT/MD), chunks text (configurable chunk size + overlap), computes embeddings (sentence-transformers), and stores metadata in SQLite and vectors in FAISS. 
Note that if one prefers an external embedding API (OpenAI), replace the model.encode call with embedding API (but keep vector normalization for FAISS). For very large libraries, consider using IndexIVFFlat with training for FAISS (but that complicates persistence).
