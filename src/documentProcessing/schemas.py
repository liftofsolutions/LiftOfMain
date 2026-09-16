"""Request and response models for the document processing endpoints."""

from pydantic import BaseModel, Field

DEFAULT_COUNTRY = "all"


class CreateRagDatabaseRequest(BaseModel):
    """A new rag database, reserved under a project."""

    projectId: str = Field(description="Project the rag database belongs to.")


class CreateRagDatabaseResponse(BaseModel):
    """Acknowledgement that a rag database id was reserved."""

    ragDbId: str
    projectId: str
    createdAt: str


class SubmitDocumentRequest(BaseModel):
    """A document handed in for ingestion under an existing rag database."""

    ragDbId: str = Field(description="Rag database the document belongs to.")
    documentLink: str = Field(
        description="Location the document can be fetched from."
    )
    country: str = Field(
        default=DEFAULT_COUNTRY,
        description='Country this rag serves; "all" applies to every country.',
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Container for any extra metadata, open for future use.",
    )


class SubmitDocumentResponse(BaseModel):
    """Acknowledgement that a document was queued for processing."""

    jobId: str
    ragDbId: str
    status: str
    createdAt: str


class FileMetadata(BaseModel):
    """What one file in a submitted document is made of."""

    fileName: str
    fileFormat: str
    sizeBytes: int
    lineCount: int | None = None
    wordCount: int | None = None
    characterCount: int | None = None
    tokenCount: int | None = None
    tableCount: int | None = None
    notes: str = ""
    extra: dict = Field(
        default_factory=dict,
        description="Format specific metadata, open for future additions.",
    )


class JobStatusResponse(BaseModel):
    """Where a submitted document's processing job has got to."""

    jobId: str
    ragDbId: str
    status: str
    createdAt: str
    updatedAt: str
    fileCount: int | None = Field(
        default=None, description="Files found, once processing has finished."
    )
    filesMetadata: list[FileMetadata] | None = Field(
        default=None, description="Per file metadata, once collected."
    )
    error: str | None = Field(
        default=None, description="Why the job failed, when it did."
    )
