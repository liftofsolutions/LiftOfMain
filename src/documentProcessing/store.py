"""In-memory storage for this component's records, mirrored into Firestore.

Single instance, early stage: this is a one person project with at most one
client's data in play at a time, so there is no concurrent-process problem to
architect around yet - a plain in-memory dict is the store, and every read
comes from it, never from Firestore. Firestore is written alongside every
write, once `GCP_PROJECT_ID` is configured, purely so records survive a
restart; it is a mirror, not a second source of truth to read from.

Used for both rag database records and job records - they are shaped
differently, but "write to memory, mirror to Firestore under a collection
name" is identical for either, so one module covers both rather than one
per kind of record.
"""

import asyncio
import logging
from typing import Any

from config import buildFirestoreAdapter, getServerConfig
from DbAdapters import Document, FirestoreAdapter

LOGGER = logging.getLogger(__name__)

# Keyed by collection name, then by record id.
IN_MEMORY_STORE: dict[str, dict[str, dict[str, Any]]] = {}

_FIRESTORE_ADAPTER: FirestoreAdapter | None = None

# Mirror writes still in flight. Held onto because asyncio only keeps a weak
# reference to a task, and a background write nobody awaits can otherwise be
# collected before it runs.
_MIRROR_TASKS: set[asyncio.Task] = set()


def isFirestoreConfigured() -> bool:
    """Whether Firestore is configured well enough to mirror writes to it."""
    return getServerConfig().hasFirestore


async def getAdapter() -> FirestoreAdapter:
    """The shared Firestore adapter, connecting on first use."""
    global _FIRESTORE_ADAPTER
    if _FIRESTORE_ADAPTER is None:
        _FIRESTORE_ADAPTER = buildFirestoreAdapter()
    await _FIRESTORE_ADAPTER.connect()
    return _FIRESTORE_ADAPTER


async def mirrorRecord(
    collection: str, recordId: str, data: dict[str, Any]
) -> None:
    """Send one record to Firestore. A network call, so it is slow."""
    adapter = await getAdapter()
    await adapter.upsertDocument(
        collection, Document(id=recordId, data=data)
    )


async def writeRecord(
    collection: str, recordId: str, data: dict[str, Any]
) -> None:
    """Write the full record for `recordId`, mirrored to Firestore."""
    IN_MEMORY_STORE.setdefault(collection, {})[recordId] = dict(data)
    if isFirestoreConfigured():
        await mirrorRecord(collection, recordId, data)


def writeRecordSoon(
    collection: str, recordId: str, data: dict[str, Any]
) -> asyncio.Task | None:
    """Write to memory now and mirror to Firestore in the background.

    Every read comes from memory, so the record is visible the moment this
    returns; what the caller skips is a network round trip it does not need
    the answer to. Use it where a reply is owed straight away, not where the
    caller must know the write is durable.

    A failed mirror is logged rather than raised - by then the caller has
    moved on and there is nobody left to raise it at.
    """
    IN_MEMORY_STORE.setdefault(collection, {})[recordId] = dict(data)
    if not isFirestoreConfigured():
        return None

    task = asyncio.create_task(mirrorQuietly(collection, recordId, data))
    _MIRROR_TASKS.add(task)
    task.add_done_callback(_MIRROR_TASKS.discard)
    return task


async def mirrorQuietly(
    collection: str, recordId: str, data: dict[str, Any]
) -> None:
    """Mirror a record, logging a failure instead of raising it."""
    try:
        await mirrorRecord(collection, recordId, data)
    except Exception:
        LOGGER.exception("Mirroring %s/%s failed.", collection, recordId)


def fetchRecord(collection: str, recordId: str) -> dict[str, Any] | None:
    """The record stored under `recordId`, or None when there is not one."""
    return IN_MEMORY_STORE.get(collection, {}).get(recordId)


def fetchRecordIds(
    collection: str, filters: dict[str, Any]
) -> set[str]:
    """Ids of the records in `collection` whose fields match `filters`.

    One read covering a whole slice of the collection, so a caller needing
    to know what is taken asks once instead of per candidate id.
    """
    return {
        recordId
        for recordId, record in IN_MEMORY_STORE.get(collection, {}).items()
        if all(record.get(key) == value for key, value in filters.items())
    }


def recordExists(collection: str, recordId: str) -> bool:
    """Whether `recordId` is already taken within `collection`."""
    return recordId in IN_MEMORY_STORE.get(collection, {})
