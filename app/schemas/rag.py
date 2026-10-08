from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question to ask about the uploaded document"
    )


class RAGSource(BaseModel):
    page: int
    chunk: int
    similarity: float


class RAGQueryResponse(BaseModel):
    answer: str
    sources: list[RAGSource]


class RAGUploadResponse(BaseModel):
    message: str
    pages: int
    chunks: int