"""Route declarations for the document processing endpoints.

Every endpoint is scoped to an account and a project, so both ids sit in the
path and no handler has to trust a body for them.
"""

from fastapi import APIRouter, Body, HTTPException

from documentProcessing import service
from documentProcessing.schemas import (
    JobStatusResponse,
    SubmitDocumentResponse,
)

router = APIRouter(prefix="/document", tags=["document processing"])


@router.post(
    "/{accountUuid}/{projectId}/rag",
    response_model=SubmitDocumentResponse,
    summary="Create a rag database",
)
async def createRagDatabase(accountUuid: str, projectId: str) -> dict:
    """Reserve a rag database under a project and open a job for it."""
    created = await service.createRagDatabase(accountUuid, projectId)
    if "error" in created:
        raise HTTPException(status_code=404, detail=created["error"])
    return created


@router.post(
    "/{accountUuid}/{projectId}/submit/{jobId}",
    response_model=SubmitDocumentResponse,
    summary="Submit a document",
)
async def submitDocument(
    accountUuid: str,
    projectId: str,
    jobId: str,
    documentLink: str = Body(),
    country: str | None = Body(default=None),
    metadata: dict = Body(default={}),
) -> dict:
    """Attach a document, country and metadata to a job, then queue it.

    A null country means the rag serves every country.
    """
    try:
        return await service.submitDocument(
            accountUuid,
            projectId,
            jobId,
            documentLink,
            country,
            metadata,
        )
    except service.RagDatabaseNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.delete(
    "/{accountUuid}/{projectId}/rag/{ragDbId}",
    summary="Delete a rag database",
)
async def deleteRagDatabase(
    accountUuid: str, projectId: str, ragDbId: str
) -> dict:
    """Drop a rag database and everything stored under it."""
    try:
        return await service.deleteRagDatabase(
            accountUuid, projectId, ragDbId
        )
    except service.RagDatabaseNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get(
    "/{accountUuid}/{projectId}/status/{resourceId}",
    response_model=JobStatusResponse,
    summary="Job status",
)
async def readJobStatus(
    accountUuid: str, projectId: str, resourceId: str
) -> dict:
    """Report processing state, looked up by rag database id or job id."""
    status = await service.getJobStatus(accountUuid, projectId, resourceId)
    if status is None:
        raise HTTPException(
            status_code=404, detail=f"Nothing found for {resourceId!r}."
        )
    return status
