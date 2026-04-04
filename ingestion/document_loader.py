# document_loader.py
# Purpose: Loads documents of various types (PDF, DOCX, TXT, etc.) using LangChain loaders.
# This module abstracts away the file type detection and uses the appropriate loader for each document.

from langchain.document_loaders import UnstructuredPDFLoader, Docx2txtLoader, TextLoader
import os

def load_document(file_path):
	"""
	Loads a document from the given file path using the appropriate LangChain loader.
	Supports PDF, DOCX, and TXT files.
	Returns a list of document objects (one per page or chunk, depending on loader).
	"""
	ext = os.path.splitext(file_path)[1].lower()
	if ext == ".pdf":
		loader = UnstructuredPDFLoader(file_path)
	elif ext == ".docx":
		loader = Docx2txtLoader(file_path)
	elif ext == ".txt":
		loader = TextLoader(file_path)
	else:
		raise ValueError(f"Unsupported file type: {ext}")
	return loader.load()
