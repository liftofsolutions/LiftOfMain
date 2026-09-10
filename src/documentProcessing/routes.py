"""Route declarations for the document processing endpoints."""

from fastapi import APIRouter

from documentProcessing import service
from documentProcessing.schemas import (
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    JobStatusResponse,
    SubmitDocumentRequest,
    SubmitDocumentResponse,
)

router = APIRouter(prefix="/document", tags=["document processing"])


@router.post("/submit", response_model=SubmitDocumentResponse, summary="Submit a document")
def submitDocument(payload: SubmitDocumentRequest) -> dict:
    """Hand a document link in for ingestion under an account's project."""
    return service.submitDocument(payload.accountUuid, payload.projectId, payload.documentLink)


@router.post("/delete", response_model=DeleteDocumentResponse, summary="Delete a document")
def deleteDocument(payload: DeleteDocumentRequest) -> dict:
    """Remove a previously submitted document from a project."""
    return service.deleteDocument(payload.projectId, payload.documentLink)


@router.get("/status/{jobId}", response_model=JobStatusResponse, summary="Job status")
def readJobStatus(jobId: str) -> dict:
    """Report the processing state of a single submitted document."""
    return service.getJobStatus(jobId)
