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

from typing import Any

from config import buildFirestoreAdapter, getSettings
from DbAdapters import Document, FirestoreAdapter

# Keyed by collection name, then by record id.
IN_MEMORY_STORE: dict[str, dict[str, dict[str, Any]]] = {}

_FIRESTORE_ADAPTER: FirestoreAdapter | None = None


def isFirestoreConfigured() -> bool:
    """Whether Firestore is configured well enough to mirror writes to it."""
    return getSettings().hasFirestore


async def getAdapter() -> FirestoreAdapter:
    """The shared Firestore adapter, connecting on first use."""
    global _FIRESTORE_ADAPTER
    if _FIRESTORE_ADAPTER is None:
        _FIRESTORE_ADAPTER = buildFirestoreAdapter()
    await _FIRESTORE_ADAPTER.connect()
    return _FIRESTORE_ADAPTER


async def writeRecord(
    collection: str, recordId: str, data: dict[str, Any]
) -> None:
    """Write the full record for `recordId`, mirrored to Firestore."""
    IN_MEMORY_STORE.setdefault(collection, {})[recordId] = dict(data)
    if isFirestoreConfigured():
        adapter = await getAdapter()
        await adapter.upsertDocument(
            collection, Document(id=recordId, data=data)
        )


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
