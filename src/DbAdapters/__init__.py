"""Adapters that put one async CRUD surface over the databases we support.

Two contracts, because two shapes of store:

* :class:`VectorDbAdapter` - embeddings searched by similarity. Pinecone.
* :class:`DocumentDbAdapter` - documents keyed by id. MongoDB, Firestore, Redis.

Import from here rather than from the vendor modules, so swapping a store is a
one line change:

    from DbAdapters import MongoAdapter, Document

How much of ACID each store actually offers differs enormously - MongoDB has real
transactions, Firestore has atomic batches, Redis has an uninterruptible block
with no rollback, and Pinecone has none of the three. Each adapter's module
docstring says exactly where its store stands; read that one before relying on
`applyAtomically`.
"""

from DbAdapters.base import DbAdapter, WriteFailure, WriteOutcome
from DbAdapters.documentBase import (
    WRITE_DELETE,
    WRITE_MERGE,
    WRITE_SET,
    Document,
    DocumentDbAdapter,
    WriteOperation,
    deleteOperation,
    mergeOperation,
    setOperation,
)
from DbAdapters.firestoreAdapter import FirestoreAdapter, FirestoreAdapterError
from DbAdapters.mongoAdapter import (
    MongoAdapter,
    MongoAdapterError,
    MongoTransactionUnsupportedError,
)
from DbAdapters.pineconeAdapter import (
    PineconeAdapter,
    PineconeAdapterError,
    PineconeTransactionError,
)
from DbAdapters.redisAdapter import RedisAdapter, RedisAdapterError, RedisContentionError
from DbAdapters.vectorBase import QueryMatch, VectorDbAdapter, VectorRecord

__all__ = [
    "WRITE_DELETE",
    "WRITE_MERGE",
    "WRITE_SET",
    "DbAdapter",
    "Document",
    "DocumentDbAdapter",
    "FirestoreAdapter",
    "FirestoreAdapterError",
    "MongoAdapter",
    "MongoAdapterError",
    "MongoTransactionUnsupportedError",
    "PineconeAdapter",
    "PineconeAdapterError",
    "PineconeTransactionError",
    "QueryMatch",
    "RedisAdapter",
    "RedisAdapterError",
    "RedisContentionError",
    "VectorDbAdapter",
    "VectorRecord",
    "WriteFailure",
    "WriteOperation",
    "WriteOutcome",
    "deleteOperation",
    "mergeOperation",
    "setOperation",
]
