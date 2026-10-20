from typing import Literal

from pydantic import BaseModel, Field


class EmbedInput(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=16)
    purpose: Literal["query", "document"] = "document"


class SearchInput(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=6)
    document_ids: list[str] = Field(default_factory=list, max_length=100)


class Source(BaseModel):
    id: str
    document: str
    page: int
    text: str
    score: float = 0
    document_id: str = ""


class GenerateInput(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    sources: list[Source] = Field(max_length=6)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=8)
    language: Literal["sk", "en"] = "sk"
