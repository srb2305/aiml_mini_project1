# entity_extractor.py
# Purpose: Extracts named entities from text chunks using spaCy NER.

import spacy # NLP library for natural language processing tasks, including NER, for text generation and understanding. It provides pre-trained models for various languages and tasks, making it easier to process and analyze text data.
from collections import defaultdict

class EntityExtractor:
	def __init__(self, model_name="en_core_web_sm"):
		self.nlp = spacy.load(model_name)

	def extract_entities(self, chunks):
		"""
		Runs NER over each chunk and returns a deduplicated list of entities.
		Args:
			chunks: List of text chunks (strings)
		Returns:
			entities: Dict mapping entity type to set of unique entity texts
		"""
		entities = defaultdict(set)
		for chunk in chunks:
			doc = self.nlp(chunk)
			for ent in doc.ents:
				entities[ent.label_].add(ent.text)
		# Convert sets to lists for easier downstream use
		return {label: list(ents) for label, ents in entities.items()}
