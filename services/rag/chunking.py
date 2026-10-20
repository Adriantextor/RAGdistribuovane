import re


def chunks(text: str, size: int = 1100, overlap: int = 160) -> list[str]:
    if not 0 <= overlap < size:
        raise ValueError("Overlap must be smaller than the chunk size")
    text = re.sub(r"\s+", " ", text).strip()
    result = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        result.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return result
