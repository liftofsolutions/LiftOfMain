"""Request and response models for the document processing endpoints."""

from pydantic import BaseModel, Field


class SubmitDocumentRequest(BaseModel):
    """A document handed in for ingestion, addressed by its account and project."""

    accountUuid: str = Field(description="Account the project belongs to.")
    projectId: str = Field(description="Project the document belongs to.")
    documentLink: str = Field(description="Location the document can be fetched from.")


class DeleteDocumentRequest(BaseModel):
    """A document to be removed again from a project."""

    projectId: str = Field(description="Project the document belongs to.")
    documentLink: str = Field(description="Location of the document that was ingested.")


class SubmitDocumentResponse(BaseModel):
    """Acknowledgement that a document was taken in for processing."""

    status: str
    message: str
    accountUuid: str
    projectId: str
    documentLink: str
    timestamp: str


class DeleteDocumentResponse(BaseModel):
    """Acknowledgement that a document was taken out of a project."""

    status: str
    message: str
    projectId: str
    documentLink: str
    timestamp: str


class JobStatusResponse(BaseModel):
    """Where a submitted document has got to in the pipeline."""

    status: str
    message: str
    jobId: str
    timestamp: str
