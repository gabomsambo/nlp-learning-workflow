"""The bundled NLP-KG vocabulary, exact lookup, ancestry and candidate retrieval."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nlp_pillars.vocabulary import (
    BUNDLED_PATH,
    CandidateIndex,
    Vocabulary,
    VocabularyError,
    VocabularySet,
    load_vocabularies,
)


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary.from_file(BUNDLED_PATH)


def test_the_bundled_file_is_the_nlp_kg_fields_of_study(vocab):
    assert vocab.key == "nlp-kg" and vocab.licence == "MIT"
    assert len(vocab) == 421
    assert sum(len(f.synonyms) for f in vocab.fields.values()) == 583
    document = json.loads(BUNDLED_PATH.read_text())
    assert document["source"].endswith("NLP-KG-WebApp") and len(document["source_ref"]) == 40
    assert "Schopf" in document["citation"]


def test_the_hierarchy_has_one_root_and_every_parent_exists(vocab):
    roots = [f for f in vocab.fields.values() if not f.parents]
    assert [r.label for r in roots] == ["Natural Language Processing"]
    for field in vocab.fields.values():
        assert all(parent in vocab.fields for parent in field.parents)


def test_a_label_or_synonym_finds_its_field_however_it_is_written(vocab):
    assert vocab.lookup("attention").id == "Attention"
    assert vocab.lookup("Attention") is vocab.lookup("  ATTENTION ")
    assert vocab.lookup("in-context learning").label == "In-Context Learning"
    assert vocab.lookup("probing").label == "Probing"
    assert vocab.lookup("no such field of study") is None


def test_ancestors_run_nearest_first_up_to_the_root(vocab):
    labels = [f.label for f in vocab.ancestors("Attention")]
    assert labels[0] == "Language Model Mechanisms"
    assert labels[-1] == "Natural Language Processing"
    assert vocab.path("Attention") == [
        "Natural Language Processing",
        "Semantic Text Processing",
        "Language Models & Neural Networks",
        "Language Model Mechanisms",
        "Attention",
    ]
    assert vocab.ancestors("not-a-field") == []


def test_a_field_with_two_parents_lists_both_once(vocab):
    multi = next(f for f in vocab.fields.values() if len(f.parents) > 1)
    ancestors = [f.id for f in vocab.ancestors(multi.id)]
    assert set(multi.parents) <= set(ancestors) and len(ancestors) == len(set(ancestors))


def test_the_anchor_string_is_vocab_colon_id(vocab):
    assert vocab.get("Attention").anchor == "nlp-kg:Attention"
    vocabularies = VocabularySet([vocab])
    assert vocabularies.get("nlp-kg:Attention").label == "Attention"
    assert vocabularies.get("pwc:Attention") is None
    assert [f.label for f in vocabularies.ancestors("nlp-kg:Attention")][0] == "Language Model Mechanisms"


def test_an_extra_file_is_a_second_vocabulary_with_its_own_key(tmp_path):
    extra = tmp_path / "pwc.json"
    extra.write_text(
        json.dumps(
            {
                "vocabulary": "pwc",
                "name": "Papers with Code methods",
                "fields": [{"id": "lora", "label": "LoRA", "synonyms": ["low-rank adaptation"], "level": 2}],
            }
        )
    )
    both = load_vocabularies([str(extra)])
    assert both.lookup("low rank adaptation").anchor == "pwc:lora"
    assert both.lookup("attention").anchor == "nlp-kg:Attention"
    assert load_vocabularies([]).lookup("lora") is None


@pytest.mark.parametrize(
    "content",
    ["not json", json.dumps({"fields": []}), json.dumps({"vocabulary": "x", "fields": [{"id": "a"}]})],
)
def test_a_vocabulary_file_that_cannot_be_read_raises(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(VocabularyError):
        Vocabulary.from_file(path)
    with pytest.raises(VocabularyError):
        Vocabulary.from_file(tmp_path / "missing.json")


def test_candidates_are_the_nearest_fields_by_embedding(vocab):
    vocabularies = VocabularySet([vocab])
    fields = vocabularies.all_fields()
    target = next(i for i, f in enumerate(fields) if f.id == "Attention")

    def embed(texts):
        rows = []
        for text in texts:
            vector = np.zeros(len(fields))
            if text == "multi-head attention":
                vector[target] = 1.0
                vector[(target + 1) % len(fields)] = 0.2
            else:
                vector[next(i for i, f in enumerate(fields) if f.text() == text)] = 1.0
            rows.append(vector.tolist())
        return rows

    top = CandidateIndex(vocabularies, embed).top("multi-head attention", k=3)
    assert top[0][0].id == "Attention" and top[0][1] > top[1][1]
    assert len(top) == 3
