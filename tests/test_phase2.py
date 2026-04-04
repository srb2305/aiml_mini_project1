# test_phase2.py
# Tests for Phase 2: Entity extraction, relation extraction, Neo4j integration, and graph retrieval

import os
from dotenv import load_dotenv
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

import pytest
from graph.entity_extractor import EntityExtractor
from graph.relation_extractor import RelationExtractor
from graph.graph_store import GraphStore
from retrieval.graph_retriever import GraphRetriever

@pytest.fixture(scope="module")
def entity_extractor():
    return EntityExtractor()

def test_entity_extraction(entity_extractor):
    chunks = ["John Doe works at Acme Corp in New York.", "Jane Smith joined Acme Corp in 2020."]
    entities = entity_extractor.extract_entities(chunks)
    assert "PERSON" in entities
    assert any("John Doe" in ents or "Jane Smith" in ents for ents in entities.values())

def test_relation_extraction(monkeypatch):
    # Mock LLM response
    class DummyResponse:
        choices = [type("obj", (), {"message": {"content": '[{"subject": "John Doe", "predicate": "works at", "object": "Acme Corp"}]'}})()]
    monkeypatch.setattr("openai.ChatCompletion.create", lambda *a, **kw: DummyResponse())
    extractor = RelationExtractor()
    triples = extractor.extract_relations("John Doe works at Acme Corp.")
    assert isinstance(triples, list)
    assert triples[0]["subject"] == "John Doe"
    assert triples[0]["object"] == "Acme Corp"

@pytest.fixture(scope="module")
def graph_store():
    if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
        pytest.skip("Neo4j credentials not set in .env")
    return GraphStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

def test_graph_store_and_retriever(graph_store):
    if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
        pytest.skip("Neo4j credentials not set in .env")
    graph_store.upsert_entity("John Doe", "PERSON", chunk_id=1)
    graph_store.upsert_entity("Acme Corp", "ORG", chunk_id=1)
    graph_store.upsert_relationship("John Doe", "works at", "Acme Corp", chunk_id=1)
    neighbors = graph_store.get_neighbors("John Doe", hops=2)
    assert any(n["name"] == "Acme Corp" for n in neighbors)

def test_graph_retriever():
    if not (NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD):
        pytest.skip("Neo4j credentials not set in .env")
    retriever = GraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    chunk_ids = retriever.retrieve("Who works at Acme Corp?", hops=2)
    assert isinstance(chunk_ids, list)
