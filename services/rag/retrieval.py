import math
from collections import Counter

from shared.text import expanded_query, tokens


def rank_chunks(question, records, semantic, top_k=4, min_score=.48):
    """BM25 + dense retrieval with reciprocal-rank fusion and near-duplicate suppression."""
    query = expanded_query(question)
    query_terms = set(tokens(query))
    if not records or not query_terms:
        return []
    terms = {record["id"]: tokens(record["text"]) for record in records}
    counts = {key: Counter(value) for key, value in terms.items()}
    average = sum(map(len, terms.values())) / max(1, len(terms))
    frequencies = {term: sum(term in value for value in counts.values()) for term in query_terms}
    lexical = []
    for record in records:
        key, score = record["id"], 0
        for term in query_terms:
            frequency = counts[key][term]
            if frequency:
                inverse = math.log(1 + (len(records) - frequencies[term] + .5) / (frequencies[term] + .5))
                score += inverse * frequency * 2.2 / (frequency + 1.2 * (.25 + .75 * len(terms[key]) / max(1, average)))
        if score:
            lexical.append((key, score))
    lexical.sort(key=lambda item: item[1], reverse=True)
    lex_rank = {key: index for index, (key, _) in enumerate(lexical, 1)}
    lex_score = dict(lexical)
    dense_rank = {key: index for index, (key, _) in enumerate(semantic, 1)}
    dense_score = dict(semantic)
    rows = []
    for record in records:
        key = record["id"]
        semantic_score = dense_score.get(key, 0)
        matches = len(query_terms & set(terms[key]))
        # Strong literal matches can rescue translated technical terms; otherwise require dense evidence.
        lexical_evidence = matches >= min(2, len(query_terms)) and lex_score.get(key, 0) >= 2
        if semantic_score < min_score and not lexical_evidence:
            continue
        score = (1 / (30 + dense_rank[key]) if key in dense_rank else 0) + (
            1.25 / (30 + lex_rank[key]) if key in lex_rank else 0)
        rows.append((score, {**record, "score": semantic_score}))
    rows.sort(key=lambda item: item[0], reverse=True)
    selected, seen = [], []
    for _, record in rows:
        words = set(terms[record["id"]])
        if any(len(words & previous) / max(1, len(words | previous)) > .80 for previous in seen):
            continue
        selected.append(record)
        seen.append(words)
        if len(selected) == top_k:
            break
    return selected
