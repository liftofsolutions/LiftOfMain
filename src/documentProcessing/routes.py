"""Route declarations for the rag job endpoints.

Every endpoint is scoped to an account and a project, so both ids sit in the
path, and a job or a rag is addressed under them:

    POST   /rag/{accountUuid}/{projectId}/jobs                  create
    POST   /rag/{accountUuid}/{projectId}/jobs/{jobId}/submit   submit
    GET    /rag/{accountUuid}/{projectId}/jobs/{jobId}/status   status
    DELETE /rag/{accountUuid}/{projectId}/jobs/{jobId}          delete job
    DELETE /rag/{accountUuid}/{projectId}/rags/{ragId}          delete rag
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from documentProcessing import service
from documentProcessing.model import (
    DeleteRagJobResponse,
    DeleteRagResponse,
    RagJobResponse,
    SubmitRagJobRequest,
)

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag jobs"])

# What a caller is told when a handler breaks: never the exception itself,
# which goes to the log instead.
INTERNAL_ERROR_DETAIL = "Internal server error."

STATUS_INTERNAL_ERROR = 500
ERROR_STATUS_CODES = {
    service.ERROR_NOT_FOUND: 404,
    service.ERROR_FORBIDDEN: 403,
    service.ERROR_NOT_IMPLEMENTED: 501,
}


def errorResponse(statusCode: int, message: str) -> JSONResponse:
    """The error, handed back to the caller rather than raised at them.

    Returning a `JSONResponse` sends the body as it is: a route's
    `response_model` describes a success, and is not applied to it.
    """
    return JSONResponse(status_code=statusCode, content={"error": message})


def toResponse(result: dict) -> dict | JSONResponse:
    """The service's result, or the error it reported.

    Told apart by `errorKind`, so a success may carry an `error` field.
    """
    if "errorKind" in result:
        return errorResponse(
            ERROR_STATUS_CODES[result["errorKind"]], result["error"]
        )
    return result


def logRequest(request: Request, parameters: dict) -> None:
    """Log which route `request` reached, and the parameters it carried."""
    LOGGER.info(
        "%s %s received: %s", request.method, request.url.path, parameters
    )


def logResponse(
    request: Request, response: dict | JSONResponse
) -> dict | JSONResponse:
    """Log what the route answers `request` with, then hand it back."""
    if isinstance(response, JSONResponse):
        LOGGER.info(
            "%s %s answered %s: %s",
            request.method,
            request.url.path,
            response.status_code,
            response.body.decode(),
        )
    else:
        LOGGER.info(
            "%s %s answered: %s", request.method, request.url.path, response
        )
    return response


@router.post(
    "/{accountUuid}/{projectId}/jobs",
    response_model=RagJobResponse,
    summary="Create a rag job",
)
async def createRagJob(
    request: Request, accountUuid: str, projectId: str
) -> dict | JSONResponse:
    """Open a job under an account's project, ready to be submitted."""
    logRequest(request, {"accountUuid": accountUuid, "projectId": projectId})
    try:
        created = await service.createRagJob(accountUuid, projectId)
    except Exception:
        LOGGER.exception(
            "Creating a job for account %s, project %s failed.",
            accountUuid,
            projectId,
        )
        return logResponse(
            request,
            errorResponse(STATUS_INTERNAL_ERROR, INTERNAL_ERROR_DETAIL),
        )
    return logResponse(request, toResponse(created))


@router.post(
    "/{accountUuid}/{projectId}/jobs/{jobId}/submit",
    response_model=RagJobResponse,
    summary="Submit a document to a rag job",
)
async def submitRagJob(
    request: Request,
    accountUuid: str,
    projectId: str,
    jobId: str,
    submission: SubmitRagJobRequest,
) -> dict | JSONResponse:
    """Give the job its document, country and metadata, and start it."""
    logRequest(
        request,
        {
            "accountUuid": accountUuid,
            "projectId": projectId,
            "jobId": jobId,
            **submission.model_dump(),
        },
    )
    try:
        submitted = await service.submitRagJob(
            accountUuid,
            projectId,
            jobId,
            submission.documentId,
            submission.countryId,
            submission.metadata,
        )
    except Exception:
        LOGGER.exception("Submitting job %s failed.", jobId)
        return logResponse(
            request,
            errorResponse(STATUS_INTERNAL_ERROR, INTERNAL_ERROR_DETAIL),
        )
    return logResponse(request, toResponse(submitted))


@router.get(
    "/{accountUuid}/{projectId}/jobs/{jobId}/status",
    response_model=RagJobResponse,
    summary="Rag job status",
)
async def getRagJobStatus(
    request: Request, accountUuid: str, projectId: str, jobId: str
) -> dict | JSONResponse:
    """Report where the job has got to."""
    logRequest(
        request,
        {"accountUuid": accountUuid, "projectId": projectId, "jobId": jobId},
    )
    try:
        status = await service.getRagJobStatus(accountUuid, projectId, jobId)
    except Exception:
        LOGGER.exception("Reading status for job %s failed.", jobId)
        return logResponse(
            request,
            errorResponse(STATUS_INTERNAL_ERROR, INTERNAL_ERROR_DETAIL),
        )
    return logResponse(request, toResponse(status))


@router.delete(
    "/{accountUuid}/{projectId}/jobs/{jobId}",
    response_model=DeleteRagJobResponse,
    summary="Interrupt and delete a rag job",
)
async def deleteRagJob(
    request: Request, accountUuid: str, projectId: str, jobId: str
) -> dict | JSONResponse:
    """Stop the job if it is still going, then delete it."""
    logRequest(
        request,
        {"accountUuid": accountUuid, "projectId": projectId, "jobId": jobId},
    )
    try:
        deleted = await service.deleteRagJob(accountUuid, projectId, jobId)
    except Exception:
        LOGGER.exception("Deleting job %s failed.", jobId)
        return logResponse(
            request,
            errorResponse(STATUS_INTERNAL_ERROR, INTERNAL_ERROR_DETAIL),
        )
    return logResponse(request, toResponse(deleted))


@router.delete(
    "/{accountUuid}/{projectId}/rags/{ragId}",
    response_model=DeleteRagResponse,
    summary="Delete a rag",
)
async def deleteRag(
    request: Request, accountUuid: str, projectId: str, ragId: str
) -> dict | JSONResponse:
    """Delete the rag itself."""
    logRequest(
        request,
        {"accountUuid": accountUuid, "projectId": projectId, "ragId": ragId},
    )
    try:
        deleted = await service.deleteRag(accountUuid, projectId, ragId)
    except Exception:
        LOGGER.exception("Deleting rag %s failed.", ragId)
        return logResponse(
            request,
            errorResponse(STATUS_INTERNAL_ERROR, INTERNAL_ERROR_DETAIL),
        )
    return logResponse(request, toResponse(deleted))
