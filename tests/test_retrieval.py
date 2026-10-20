from services.rag.retrieval import rank_chunks
from shared.text import expanded_query, language_of


def test_specific_section_beats_generic_high_dense_score():
    records = [
        {"id": "generic", "text": "The Transformer relies entirely on self attention and does not use recurrent networks."},
        {"id": "position", "text": "Positional encoding. We add positional encodings to the input embeddings to inject information about position in the sequence."},
        {"id": "other", "text": "QLoRA uses four bit quantization and low rank adapters."},
    ]
    result = rank_chunks("Prečo Transformer používa pozičné kódovanie?", records,
                         [("generic", .81), ("position", .76), ("other", .4)])
    assert result[0]["id"] == "position"


def test_irrelevant_low_score_is_rejected_and_duplicates_suppressed():
    records = [{"id": "a", "text": "Double quantization quantizes the quantization constants."},
               {"id": "b", "text": "Double quantization quantizes the quantization constants."}]
    assert rank_chunks("What is my wifi password?", records, [("a", .2), ("b", .1)]) == []
    assert len(rank_chunks("Explain double quantization", records, [("a", .8), ("b", .79)])) == 1


def test_language_and_search_terms_support_slovak():
    assert language_of("Čo znamená QLoRA?") == "sk"
    assert language_of("What is QLoRA?") == "en"
    assert "positional" in expanded_query("pozičné kódovanie")
    assert "encoding" in expanded_query("pozičné kódovanie")
