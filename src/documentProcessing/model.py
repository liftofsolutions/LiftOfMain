"""Request and response models for the rag job endpoints."""

from datetime import datetime

from pydantic import BaseModel, Field


class SubmitRagJobRequest(BaseModel):
    """The document a job processes, the country it is for, and metadata."""

    documentLink: str = Field(
        description=(
            "http(s) link to the document: a .pdf, .docx, .txt, .md, or a "
            ".zip, .rar or .tar of them."
        )
    )
    countryId: str | None = Field(
        default=None,
        description=(
            "The country the document is for. Leave it out for knowledge "
            "that applies to every country."
        ),
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Extra details about the document; not used yet.",
    )


class RagJobResponse(BaseModel):
    """A rag job and where it has got to."""

    jobId: str
    ragId: str = Field(description="The rag the job writes into.")
    status: str


class DeleteRagJobResponse(BaseModel):
    """A deleted job, and whether it had to be taken off the queue first."""

    jobId: str
    interrupted: bool = Field(
        description=(
            "True when the job was waiting for a slot and was taken off "
            "the queue, so it never ran."
        )
    )


class DeleteRagResponse(BaseModel):
    """A deleted rag."""

    ragId: str


class RagSummary(BaseModel):
    """One rag of a project and its state, without the text it may hold."""

    ragId: str
    status: str = Field(description="indexing, ready, failed or deleting.")
    strategy: str | None = Field(
        default=None, description="plainText or simpleRag, once indexed."
    )
    documentLink: str | None = None
    tokenCount: int | None = None
    chunkCount: int | None = Field(
        default=None, description="Chunks in Pinecone, for a simple rag."
    )
    createdAt: datetime | None = None
    updatedAt: datetime | None = None


class ListRagsResponse(BaseModel):
    """Every rag of a project."""

    projectId: str
    rags: list[RagSummary]


class QueryRagRequest(BaseModel):
    """What to look for in a rag.

    How many chunks a simple rag gives back is the server's setting,
    `RAG_QUERY_TOP_K`; a plain-text rag always gives its whole text.
    """

    query: str = Field(min_length=1, description="The question to answer.")


class RagMatch(BaseModel):
    """One part of a rag that matches the query."""

    text: str
    score: float | None = Field(
        default=None,
        description=(
            "How close the chunk is to the query, higher is closer; none "
            "for a plain-text rag's whole text."
        ),
    )
    fileName: str | None = Field(
        default=None, description="The file the chunk is from."
    )
    chunk: int | None = Field(
        default=None, description="The chunk's place in the rag."
    )


class QueryRagResponse(BaseModel):
    """The parts of a rag that best answer a query, best first."""

    ragId: str
    strategy: str = Field(description="plainText or simpleRag.")
    matches: list[RagMatch]
