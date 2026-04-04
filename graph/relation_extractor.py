# relation_extractor.py
# Purpose: Extracts (subject, predicate, object) triples from text chunks using an LLM prompt.

import openai  # or use another LLM provider as needed

class RelationExtractor:
	def __init__(self, model_name="gpt-3.5-turbo"):
		self.model_name = model_name

	def extract_relations(self, chunk):
		"""
		Prompts the LLM to extract (subject, predicate, object) triples from a text chunk.
		Args:
			chunk: Text chunk (string)
		Returns:
			List of triples as dicts: [{"subject": ..., "predicate": ..., "object": ...}, ...]
		"""
		prompt = (
			"Extract all (subject, predicate, object) triples from this paragraph. "
			"Return the result as a JSON list of objects with keys 'subject', 'predicate', 'object'.\n"
			f"Paragraph: {chunk}"
		)
		response = openai.ChatCompletion.create(
			model=self.model_name,
			messages=[{"role": "user", "content": prompt}],
			max_tokens=512,
			temperature=0
		)
		import json
		# Try to extract JSON from the response
		try:
			triples = json.loads(response.choices[0].message['content'])
		except Exception:
			triples = []
		return triples
