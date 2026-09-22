"""Route declarations for the document processing endpoints.

Every endpoint is scoped to an account and a project, so both ids sit in the
path and no handler has to trust a body for them.
"""

import logging

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from documentProcessing import service
from documentProcessing.model import (
    CreateJobResponse,
    JobStatusResponse,
    SubmitDocumentResponse,
)

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/document", tags=["document processing"])

# What a caller is told when a handler breaks: never the exception itself,
# which goes to the log instead.
INTERNAL_ERROR_DETAIL = "Internal server error."

STATUS_NOT_FOUND = 404
STATUS_INTERNAL_ERROR = 500


def errorResponse(statusCode: int, message: str) -> JSONResponse:
    """The error, handed back to the caller rather than raised at them.

    Returning a `JSONResponse` sends the body as it is: a route's
    `response_model` describes what a success looks like, and is not applied
    to a response the handler builds itself.
    """
    return JSONResponse(status_code=statusCode, content={"error": message})


@router.post(
    "/{accountUuid}/{projectId}/rag",
    response_model=CreateJobResponse,
    summary="Create a rag database",
)
async def createRagDatabase(
    accountUuid: str, projectId: str
) -> dict | JSONResponse:
    """Reserve a rag database under a project and open a job for it."""
    try:
        created = await service.createRagDatabase(accountUuid, projectId)
    except Exception:
        LOGGER.exception(
            "Creating a job for account %s, project %s failed.",
            accountUuid,
            projectId,
        )
        return errorResponse(STATUS_INTERNAL_ERROR, INTERNAL_ERROR_DETAIL)

    if "error" in created:
        return errorResponse(STATUS_NOT_FOUND, created["error"])
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
) -> dict | JSONResponse:
    """Attach a document, country and metadata to a job, then queue it.

    A null country means the rag serves every country.
    """
    try:
        submitted = await service.submitDocument(
            accountUuid,
            projectId,
            jobId,
            documentLink,
            country,
            metadata,
        )
    except Exception:
        LOGGER.exception("Submitting job %s failed.", jobId)
        return errorResponse(STATUS_INTERNAL_ERROR, INTERNAL_ERROR_DETAIL)

    if "error" in submitted:
        return errorResponse(STATUS_NOT_FOUND, submitted["error"])
    return submitted


@router.delete(
    "/{accountUuid}/{projectId}/rag/{ragDbId}",
    # No schema for a delete yet, and the annotation is a union, so the
    # response model is not inferred from it.
    response_model=None,
    summary="Delete a rag database",
)
async def deleteRagDatabase(
    accountUuid: str, projectId: str, ragDbId: str
) -> dict | JSONResponse:
    """Drop a rag database and everything stored under it."""
    try:
        deleted = await service.deleteRagDatabase(
            accountUuid, projectId, ragDbId
        )
    except Exception:
        LOGGER.exception("Deleting rag database %s failed.", ragDbId)
        return errorResponse(STATUS_INTERNAL_ERROR, INTERNAL_ERROR_DETAIL)

    if "error" in deleted:
        return errorResponse(STATUS_NOT_FOUND, deleted["error"])
    return deleted


@router.get(
    "/{accountUuid}/{projectId}/status/{resourceId}",
    response_model=JobStatusResponse,
    summary="Job status",
)
async def readJobStatus(
    accountUuid: str, projectId: str, resourceId: str
) -> dict | JSONResponse:
    """Report processing state, looked up by rag database id or job id."""
    try:
        status = await service.getJobStatus(
            accountUuid, projectId, resourceId
        )
    except Exception:
        LOGGER.exception("Reading status for %s failed.", resourceId)
        return errorResponse(STATUS_INTERNAL_ERROR, INTERNAL_ERROR_DETAIL)

    if status is None:
        message = f"Nothing found for {resourceId!r}."
        LOGGER.error(message)
        return errorResponse(STATUS_NOT_FOUND, message)
    return status
