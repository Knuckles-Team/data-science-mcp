"""Fleet-enrichment assets test for data-science-mcp.

This package is a BACKEND/COMPUTE service: it ships NO native KG ingestion
(no kg_ingest/kg_media module). This suite instead validates the enrichment
deliverables that DO ship — the OWL/RDF ontology leg, the specialist prompt,
the real skills, and the Tier-1 mcp_tool source preset — so the package's
graph federation surface stays correct.
"""

import json
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "data_science_mcp"
ONTOLOGY = PKG / "ontology" / "datascience.ttl"
PROMPT = PKG / "prompts" / "datascience_specialist.json"
SKILLS_DIR = PKG / "skills"
PRESETS = PKG / "connectors" / "mcp_source_presets.json"

EXPECTED_SKILLS = {
    "data-science-model-training",
    "data-science-llm-finetuning",
    "data-science-model-reliability",
}


def test_ontology_parses_and_has_iri():
    rdflib = pytest.importorskip("rdflib")
    g = rdflib.Graph()
    g.parse(str(ONTOLOGY), format="turtle")
    assert len(g) > 0
    # per-package owl:Ontology IRI imports the shared hub
    onto = rdflib.URIRef("http://knuckles.team/kg/datascience")
    imports = rdflib.URIRef("http://www.w3.org/2002/07/owl#imports")
    assert (onto, imports, rdflib.URIRef("http://knuckles.team/kg")) in g


def test_ontology_reuses_shared_classes_not_redefines():
    """Shared classes (:Dataset/:Model/:LanguageModel/:Person/:OutcomeEvaluation/
    :Document) must be referenced but NEVER redefined as owl:Class here."""
    text = ONTOLOGY.read_text()
    for shared in (
        ":Dataset a owl:Class",
        ":Model a owl:Class",
        ":LanguageModel a owl:Class",
        ":Person a owl:Class",
        ":Document a owl:Class",
        ":OutcomeEvaluation a owl:Class",
    ):
        assert shared not in text, f"must not redefine shared class: {shared}"
    # but DS-specific classes are defined
    assert ":TrainingRun a owl:Class" in text
    assert ":ModelArtifact a owl:Class" in text


def test_prompt_is_valid_structured_prompt():
    data = json.loads(PROMPT.read_text())
    assert data["type"] == "prompt"
    assert data["schema_version"] == "1.0"
    assert data["source"] == "data-science-mcp"
    assert data["extends"] == "agent-utilities:base"
    assert data["compose"] == "append"
    assert data["identity"]["role"]
    assert data["instructions"]["core_directive"]
    # every referenced skill exists on disk
    for skill in data["skills"]:
        assert (SKILLS_DIR / skill / "SKILL.md").is_file(), skill


def test_prompt_validate_canonical_strict():
    structured = pytest.importorskip("agent_utilities.prompting.structured")
    data = json.loads(PROMPT.read_text())
    # returns None on success (or raises); either way must not raise
    structured.validate_canonical(data, strict=True)


def test_three_real_skills_present_and_prefixed():
    found = {p.parent.name for p in SKILLS_DIR.glob("*/SKILL.md")}
    assert EXPECTED_SKILLS <= found, found
    assert "data-science-starter" not in found
    for name in EXPECTED_SKILLS:
        body = (SKILLS_DIR / name / "SKILL.md").read_text()
        assert name.startswith("data-science-")
        # house-template sections
        assert "## When to use" in body
        assert "## When NOT to use" in body
        assert "## Gotchas" in body


def test_source_preset_uses_real_tool_and_fields():
    presets = json.loads(PRESETS.read_text())
    preset = presets["datascience-model-registry"]
    assert preset["server"] == "data-science-mcp"
    # rank_models is a real registered MCP tool
    assert preset["tool"] == "rank_models"
    assert preset["records_path"] == "ranked_models"
    # id/title map to real ranked_models record fields
    assert preset["id_field"] == "model_id"
    assert preset["title_field"] == "model_str"
