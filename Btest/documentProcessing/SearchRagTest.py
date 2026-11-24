"""Btest for the line `searchRag` logs of the records a query returned.

The rag's record and Pinecone are faked: `fetchProjectRecord` hands back
the rag built here, and Pinecone finds the matches it is given. Each test
reads the `Rag records returned` line back from the log.

    uv run pytest Btest/documentProcessing/SearchRagTest.py -q
"""

import asyncio
import json
import logging

import pytest

from DbAdapters import QueryMatch
from documentProcessing import service
from documentProcessing.ragStrategy import (
    RAG_STATUS_READY,
    STRATEGY_PLAIN_TEXT,
    STRATEGY_SIMPLE_RAG,
    recordIdOf,
)

PROJECT_ID = "project-1"
RAG_ID = "rag-1"
QUERY = "Which planet is the largest?"


class FakePinecone:
    """Pinecone, finding `matches` for any search."""

    def __init__(self, matches: list[QueryMatch]) -> None:
        self.matches = matches

    async def search(self, namespace, text, topK, fields=None):
        return self.matches


def buildRag(**overrides) -> dict:
    """A ready simple rag of the project."""
    rag = {
        "_id": RAG_ID,
        "project_id": PROJECT_ID,
        "status": RAG_STATUS_READY,
        "strategy": STRATEGY_SIMPLE_RAG,
    }
    rag.update(overrides)
    return rag


def buildMatch(chunkIndex: int, **overrides) -> QueryMatch:
    """One chunk of the rag, as Pinecone finds it."""
    match = {
        "id": recordIdOf(RAG_ID, chunkIndex),
        "score": 0.9,
        "metadata": {
            "chunkText": f"Chunk {chunkIndex}.",
            "file_name": "planets.md",
            "chunk": float(chunkIndex),
        },
    }
    match.update(overrides)
    return QueryMatch(**match)


def useRag(monkeypatch, rag: dict, matches: list[QueryMatch]) -> None:
    """Search `rag`, with Pinecone finding `matches`."""

    async def fetchProjectRecord(collection, projectId, recordId):
        return rag

    pinecone = FakePinecone(matches)
    monkeypatch.setattr(service, "fetchProjectRecord", fetchProjectRecord)
    monkeypatch.setattr(service, "getPineconeDb", lambda: pinecone)
    monkeypatch.setattr(service, "queryTopK", lambda: 10)


def loggedRecordIds(caplog) -> list[list[str]]:
    """The record ids of every `Rag records returned` line logged."""
    prefix = service.RETURNED_RECORDS_PREFIX
    return [
        json.loads(record.getMessage().split(prefix, 1)[1])
        for record in caplog.records
        if prefix in record.getMessage()
    ]


def search(caplog) -> dict:
    """Search the rag for QUERY, logging at INFO."""
    with caplog.at_level(logging.INFO, logger=service.LOGGER.name):
        return asyncio.run(service.searchRag(PROJECT_ID, RAG_ID, QUERY))


class TestReturnedRecordsLog:
    def testSimpleRagLogsMatchIdsClosestFirst(self, monkeypatch, caplog):
        matches = [buildMatch(3), buildMatch(0), buildMatch(7)]
        useRag(monkeypatch, buildRag(), matches)

        result = search(caplog)

        assert len(result["matches"]) == 3
        assert loggedRecordIds(caplog) == [
            [f"{RAG_ID}#3", f"{RAG_ID}#0", f"{RAG_ID}#7"]
        ]

    def testNoMatchesLogsEmptyArray(self, monkeypatch, caplog):
        useRag(monkeypatch, buildRag(), [])

        search(caplog)

        assert loggedRecordIds(caplog) == [[]]

    def testPlainTextRagLogsRagId(self, monkeypatch, caplog):
        rag = buildRag(strategy=STRATEGY_PLAIN_TEXT, text="All planets.")
        useRag(monkeypatch, rag, [])

        search(caplog)

        assert loggedRecordIds(caplog) == [[RAG_ID]]

    @pytest.mark.parametrize(
        "rag",
        [None, buildRag(status="indexing")],
        ids=["missingRag", "notReady"],
    )
    def testRefusedQueryLogsNoRecords(self, monkeypatch, caplog, rag):
        useRag(monkeypatch, rag, [buildMatch(0)])

        result = search(caplog)

        assert "error" in result
        assert loggedRecordIds(caplog) == []
