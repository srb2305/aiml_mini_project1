# pipeline.py
# Purpose: Orchestrates the ingestion pipeline: loads documents, chunks them semantically, and returns chunks.

from .document_loader import load_document
from .chunker import semantic_chunk

def ingest_document(file_path):
	"""
	Loads a document, splits it into semantic chunks, and returns the chunks.
	Args:
		file_path: Path to the document file.
	Returns:
		List of text chunks.
	"""
	docs = load_document(file_path)
	all_chunks = []
	for doc in docs:
		text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
		chunks = semantic_chunk(text)
		all_chunks.extend(chunks)
	return all_chunks
