# vector_retriever.py
# Purpose: Handles embedding of chunks and vector search in pgvector (PostgreSQL extension).
# Provides functions to upsert (store) and retrieve chunks by vector similarity.

import psycopg2
from sentence_transformers import SentenceTransformer, util
import numpy as np

class VectorStore:
	def __init__(self, db_url, model_name="all-MiniLM-L6-v2"):
		"""
		Connects to pgvector database and loads embedding model.
		Args:
			db_url: PostgreSQL connection string.
			model_name: SentenceTransformer model for embeddings.
		"""
		self.conn = psycopg2.connect(db_url)
		self.model = SentenceTransformer(model_name)

	def upsert_chunks(self, chunks, metadata_list=None):
		"""
		Embeds and stores chunks in pgvector. Creates table if not exists.
		Args:
			chunks: List of text chunks.
			metadata_list: List of dicts with metadata for each chunk (optional).
		"""
		cur = self.conn.cursor()
		cur.execute('''CREATE TABLE IF NOT EXISTS documents (
			id SERIAL PRIMARY KEY,
			chunk TEXT,
			embedding VECTOR(384),
			metadata JSONB
		)''')
		for i, chunk in enumerate(chunks):
			emb = self.model.encode(chunk)
			metadata = metadata_list[i] if metadata_list else None
			cur.execute(
				"INSERT INTO documents (chunk, embedding, metadata) VALUES (%s, %s, %s)",
				(chunk, emb.tolist(), metadata)
			)
		self.conn.commit()

	def query(self, query_text, top_k=5):
		"""
		Embeds the query and retrieves top_k most similar chunks from pgvector.
		Args:
			query_text: The user query string.
			top_k: Number of results to return.
		Returns:
			List of (chunk, similarity_score, metadata) tuples.
		"""
		cur = self.conn.cursor()
		query_emb = self.model.encode(query_text)
		cur.execute(
			"""
			SELECT chunk, metadata, embedding <#> %s AS distance
			FROM documents
			ORDER BY distance ASC
			LIMIT %s
			""",
			(query_emb.tolist(), top_k)
		)
		results = cur.fetchall()
		return [(row[0], 1 - row[2], row[1]) for row in results]  # similarity = 1 - distance
