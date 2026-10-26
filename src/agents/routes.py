"""Route declarations for the agent endpoints.

An agent is configured by a member of its project, so those endpoints
are scoped to an account and a project, as the rag endpoints are:

    POST /agents/{accountUuid}/{projectId}                 make an agent
    POST /agents/{accountUuid}/{projectId}/{agentId}/rag   give it a rag

An agent is for public use, so it is asked under its project, with no
account:

    POST /agents/{projectId}/{agentId}/query   the agent's answer

There is no start and no stop: a query builds the agent if it is not
kept. A query may name its conversation by a `convoId`, and goes on
with it - or starts it - keeping its chat history.
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from agents import service
from agents.model import (
    ConfigureAgentRequest,
    ConfigureAgentResponse,
    ConfigureRagRequest,
    ConfigureRagResponse,
    QueryAgentRequest,
    QueryAgentResponse,
)

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])

# What a caller is told when a handler breaks: never the exception itself,
# which goes to the log instead.
INTERNAL_ERROR_DETAIL = "Internal server error."

STATUS_INTERNAL_ERROR = 500
ERROR_STATUS_CODES = {
    service.ERROR_NOT_FOUND: 404,
    service.ERROR_FORBIDDEN: 403,
    service.ERROR_CONFLICT: 409,
}


def errorResponse(statusCode: int, message: str) -> JSONResponse:
    """The error, handed back to the caller rather than raised at them.

    Returning a `JSONResponse` sends the body as it is: a route's
    `response_model` describes a success, and is not applied to it.
    """
    return JSONResponse(status_code=statusCode, content={"error": message})


def toResponse(result: dict) -> dict | JSONResponse:
    """The service's result, or the error it reported."""
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
    "/{accountUuid}/{projectId}",
    response_model=ConfigureAgentResponse,
    summary="Configure an agent",
)
async def configureAgent(
    request: Request,
    accountUuid: str,
    projectId: str,
    agentConfig: ConfigureAgentRequest,
) -> dict | JSONResponse:
    """Make an agent of an account's project, running a template."""
    logRequest(
        request,
        {
            "accountUuid": accountUuid,
            "projectId": projectId,
            **agentConfig.model_dump(mode="json"),
        },
    )
    try:
        configured = await service.configureAgent(
            accountUuid,
            projectId,
            agentConfig.name,
            str(agentConfig.templateId),
        )
    except Exception:
        LOGGER.exception(
            "Configuring an agent of project %s failed.", projectId
        )
        return logResponse(
            request,
            errorResponse(STATUS_INTERNAL_ERROR, INTERNAL_ERROR_DETAIL),
        )
    return logResponse(request, toResponse(configured))


@router.post(
    "/{accountUuid}/{projectId}/{agentId}/rag",
    response_model=ConfigureRagResponse,
    summary="Configure an agent's rag",
)
async def configureRag(
    request: Request,
    accountUuid: str,
    projectId: str,
    agentId: str,
    ragConfig: ConfigureRagRequest,
) -> dict | JSONResponse:
    """Give an agent of an account's project a rag of the project, as its
    knowledge base."""
    logRequest(
        request,
        {
            "accountUuid": accountUuid,
            "projectId": projectId,
            "agentId": agentId,
            **ragConfig.model_dump(mode="json"),
        },
    )
    try:
        configured = await service.configureRag(
            accountUuid, projectId, agentId, str(ragConfig.ragId)
        )
    except Exception:
        LOGGER.exception("Configuring the rag of agent %s failed.", agentId)
        return logResponse(
            request,
            errorResponse(STATUS_INTERNAL_ERROR, INTERNAL_ERROR_DETAIL),
        )
    return logResponse(request, toResponse(configured))


@router.post(
    "/{projectId}/{agentId}/query",
    response_model=QueryAgentResponse,
    summary="Query an agent",
)
async def queryAgent(
    request: Request,
    projectId: str,
    agentId: str,
    agentQuery: QueryAgentRequest,
) -> dict | JSONResponse:
    """The agent's answer to the message, in the conversation named, if
    any; the agent is built if not kept."""
    logRequest(
        request,
        {
            "projectId": projectId,
            "agentId": agentId,
            **agentQuery.model_dump(),
        },
    )
    convoId = agentQuery.convoId and str(agentQuery.convoId)
    try:
        answered = await service.queryAgent(
            projectId, agentId, agentQuery.message, convoId
        )
    except Exception:
        LOGGER.exception("Querying agent %s failed.", agentId)
        return logResponse(
            request,
            errorResponse(STATUS_INTERNAL_ERROR, INTERNAL_ERROR_DETAIL),
        )
    return logResponse(request, toResponse(answered))
