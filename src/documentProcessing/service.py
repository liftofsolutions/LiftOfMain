"""Rag jobs: create one, submit a document to it, follow it, delete it.

Only the endpoints exist so far. How a job runs is still to be designed,
so every call reports that it is not implemented yet.

Errors are returned, never raised: `{"error": message, "errorKind": kind}`,
logged, with `errorKind` one of the `ERROR_*` constants for the route to
turn into a status code.
"""

import logging

LOGGER = logging.getLogger(__name__)

ERROR_NOT_IMPLEMENTED = "notImplemented"


def notImplemented(operation: str) -> dict:
    """The error for an operation that has no logic yet, logged first."""
    message = f"{operation} is not implemented yet."
    LOGGER.error(message)
    return {"error": message, "errorKind": ERROR_NOT_IMPLEMENTED}


async def createRagJob(accountUuid: str, projectId: str) -> dict:
    """Open a job under `accountUuid`'s `projectId`."""
    return notImplemented("Creating a rag job")


async def submitRagJob(
    accountUuid: str,
    projectId: str,
    jobId: str,
    documentId: str,
    countryId: str | None,
    metadata: dict,
) -> dict:
    """Give a created job its document, country and metadata, and start it.

    No country means knowledge that applies to every country.
    """
    return notImplemented("Submitting a rag job")


async def getRagJobStatus(
    accountUuid: str, projectId: str, jobId: str
) -> dict:
    """Where the job has got to."""
    return notImplemented("Reading a rag job's status")


async def deleteRagJob(accountUuid: str, projectId: str, jobId: str) -> dict:
    """Stop the job if it is still going, then delete it."""
    return notImplemented("Deleting a rag job")


async def deleteRag(accountUuid: str, projectId: str, ragId: str) -> dict:
    """Delete the rag itself."""
    return notImplemented("Deleting a rag")
