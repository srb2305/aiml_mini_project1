# app.py
# Purpose: FastAPI backend for document ingestion and querying.
# Provides /ingest (file upload) and /query (question answering) endpoints.

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import tempfile
from ingestion.pipeline import ingest_document
from retrieval.vector_retriever import VectorStore

# Update this with your actual PostgreSQL connection string
PGVECTOR_URL = os.getenv("PGVECTOR_URL", "postgresql://postgres:postgres@localhost:5432/pgvector")
vector_store = VectorStore(PGVECTOR_URL)

app = FastAPI()

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
	"""
	Ingests an uploaded document: loads, chunks, embeds, and stores in pgvector.
	"""
	with tempfile.NamedTemporaryFile(delete=False) as tmp:
		tmp.write(await file.read())
		tmp_path = tmp.name
	chunks = ingest_document(tmp_path)
	vector_store.upsert_chunks(chunks)
	os.remove(tmp_path)
	return {"status": "success", "num_chunks": len(chunks)}

@app.post("/query")
async def query(question: str = Form(...)):
	"""
	Answers a user query by retrieving top-k relevant chunks from pgvector.
	"""
	results = vector_store.query(question)
	return JSONResponse({"results": [
		{"chunk": chunk, "score": score, "metadata": metadata}
		for chunk, score, metadata in results
	]})
