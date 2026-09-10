"""Ingestion and removal of project documents.

The real pipeline - fetching the link, chunking, embedding and writing to the
stores in `DbAdapters` - is not wired up yet. Both entry points below already
have their final signature and payload shape, so the routes and their tests do
not change once the pipeline lands.
"""

from datetime import datetime, timezone


def utcNow() -> datetime:
    """Current UTC time, kept behind a function so tests can freeze it."""
    return datetime.now(timezone.utc)


def submitDocument(accountUuid: str, projectId: str, documentLink: str) -> dict:
    """Take a document in for processing under the given account and project.

    TODO: fetch `documentLink`, split it, embed the chunks and store them
    against `projectId` for `accountUuid`.
    """
    return {
        "status": "accepted",
        "message": "Document accepted for processing",
        "accountUuid": accountUuid,
        "projectId": projectId,
        "documentLink": documentLink,
        "timestamp": utcNow().isoformat(),
    }


def deleteDocument(projectId: str, documentLink: str) -> dict:
    """Remove a previously submitted document from the given project.

    TODO: drop the stored chunks and their vectors for `documentLink` under
    `projectId`.
    """
    return {
        "status": "accepted",
        "message": "Document accepted for deletion",
        "projectId": projectId,
        "documentLink": documentLink,
        "timestamp": utcNow().isoformat(),
    }


def getJobStatus(jobId: str) -> dict:
    """Report how far the processing job identified by `jobId` has got.

    TODO: look the job up in its store and report its real state and progress.
    """
    return {
        "status": "unknown",
        "message": "Job status lookup is not implemented yet",
        "jobId": jobId,
        "timestamp": utcNow().isoformat(),
    }
