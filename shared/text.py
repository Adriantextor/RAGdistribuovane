import re
import unicodedata

PROMPT_VERSION = "grounded-v2"
RETRIEVAL_VERSION = "hybrid-v2"


def normalized(text):
    return " ".join(text.casefold().split())


def language_of(question):
    words = set(re.findall(r"\w+", question.casefold()))
    if re.search(r"[áäčďéíĺľňóôŕšťúýž]", question.casefold()) or words & {
        "ako", "preco", "co", "vysvetli", "ktory", "znamena", "aky", "ake", "je", "ma", "su", "porovnaj",
    }:
        return "sk"
    return "en" if words & {"what", "how", "why", "explain", "which", "is", "does", "compare", "the"} else "sk"


def refusal(language):
    return ("V nahraných dokumentoch som nenašiel dostatočný podklad na odpoveď. "
            "Skús otázku spresniť alebo nahrať relevantný dokument.") if language == "sk" else (
            "I could not find sufficient evidence in the uploaded documents. "
            "Please refine the question or upload a relevant document.")


def tokens(text):
    text = "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))
    stop = {"the", "and", "for", "that", "this", "with", "from", "what", "how", "why", "does", "are", "was",
            "its", "can", "about", "which", "give", "short", "answer", "please", "ako", "preco", "znamena",
            "vysvetli", "ktore", "ktory", "ktora", "pouziva", "pouzivaju", "odpovedz", "strucne", "slovensky",
            "tento", "tomto", "toho", "pre", "pri", "aby", "ake", "aky", "alebo", "podla", "medzi", "nie", "pdf"}
    return [word for word in re.findall(r"[a-z0-9]+", text) if len(word) > 2 and word not in stop]


def expanded_query(question):
    """Preserve the original query and add terminology used by the English AI papers."""
    aliases = {"pozicne": "positional", "kodovanie": "encoding", "pozornost": "attention",
               "kvantizacia": "quantization", "kvantizacie": "quantization", "kvantizaciu": "quantization",
               "pamat": "memory", "pamate": "memory", "pamati": "memory", "trenovanie": "training",
               "vahy": "weights", "parametre": "parameters", "vyhladavanie": "retrieval"}
    additions = [aliases[word] for word in tokens(question) if word in aliases]
    return question + (" " + " ".join(additions) if additions else "")
