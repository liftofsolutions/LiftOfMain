"""Request and response models for the document processing endpoints."""

from pydantic import BaseModel, Field

DEFAULT_COUNTRY = "all"


class CreateJobResponse(BaseModel):
    """Acknowledgement that a job id was opened under a project."""

    jobId: str
    projectId: str
    status: str
    createdAt: str


class SubmitDocumentResponse(BaseModel):
    """Acknowledgement that a document was queued for processing."""

    jobId: str
    status: str
    message: str


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
