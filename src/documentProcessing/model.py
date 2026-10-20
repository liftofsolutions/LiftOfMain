"""Request and response models for the rag job endpoints."""

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
    """A deleted job, and whether it had to be stopped first."""

    jobId: str
    interrupted: bool = Field(
        description="True when the job was still going and was stopped."
    )


class DeleteRagResponse(BaseModel):
    """A deleted rag."""

    ragId: str
