"""Pinecone for the whole server: one queue, one adapter, a fixed pool.

    from DbAdapters.pineconeDb import getPineconeDb
    written = await getPineconeDb().add(namespace, records)

The one way the server reaches Pinecone - `pineconeAdapter` speaks to the
index, `pineconeDb` queues every call onto the single shared adapter. Kept
under `DbAdapters` with the other stores, in its own folder because it is
more than one adapter module.
"""

from DbAdapters.pineconeDb.pineconeAdapter import (
    PineconeAdapter,
    PineconeAdapterError,
    PineconeTransactionError,
)
from DbAdapters.pineconeDb.pineconeDb import (
    MAX_CONNECTIONS,
    PineconeDb,
    PineconeDbError,
    PineconeJob,
    buildPineconeAdapter,
    getPineconeDb,
)

__all__ = [
    "MAX_CONNECTIONS",
    "PineconeAdapter",
    "PineconeAdapterError",
    "PineconeDb",
    "PineconeDbError",
    "PineconeJob",
    "PineconeTransactionError",
    "buildPineconeAdapter",
    "getPineconeDb",
]
