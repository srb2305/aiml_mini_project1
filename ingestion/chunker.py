# chunker.py
# Purpose: Splits loaded documents into semantic chunks using embedding similarity.
# This helps ensure each chunk contains a coherent piece of information for better retrieval.

from sentence_transformers import SentenceTransformer, util
# embeding modals : all-MiniLM-L6-v2, all-MPNet-base-v2, all-distilroberta-v1, all-MiniLM-L12-v2, all-MiniLM-L6-v2, all-mpnet-base-v2, all-distilroberta-v1, all-MiniLM-L12-v2
def semantic_chunk(text, model_name="all-MiniLM-L6-v2", max_chunk_size=500, similarity_threshold=0.75):
	"""
	Splits a long text into semantic chunks based on embedding similarity drops.
	Args:
		text: The input string to chunk.
		model_name: SentenceTransformer model to use for embeddings.
		max_chunk_size: Maximum number of characters per chunk.
		similarity_threshold: Cosine similarity threshold to split chunks.
	Returns:
		List of text chunks.
	"""
	model = SentenceTransformer(model_name)
	sentences = text.split(". ")
	chunks = []
	current_chunk = []
	prev_embedding = None
	for sent in sentences:
		current_chunk.append(sent)
		chunk_text = ". ".join(current_chunk)
		if len(chunk_text) >= max_chunk_size:
			embedding = model.encode(chunk_text, convert_to_tensor=True)
			if prev_embedding is not None:
				sim = util.pytorch_cos_sim(embedding, prev_embedding).item()
				if sim < similarity_threshold:
					chunks.append(chunk_text)
					current_chunk = []
					prev_embedding = None
					continue
			prev_embedding = embedding
			chunks.append(chunk_text)
			current_chunk = []
	# Add any remaining text as a chunk
	if current_chunk:
		chunks.append(". ".join(current_chunk)) #list compression to string
	return chunks
