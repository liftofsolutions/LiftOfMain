"""MongoDB implementation of :class:`~DbAdapters.documentBase.DocumentDbAdapter`.

Driven through PyMongo's own async client, `AsyncMongoClient`, which is what
replaced Motor. Every call here yields to the event loop while the socket is
busy, so `async def` in this module means what it says.

What "ACID" means here
----------------------
Of the stores in this package, MongoDB is the one that offers the real thing,
and :meth:`MongoAdapter.applyAtomically` and :meth:`MongoAdapter.runTransaction`
use it rather than approximating it:

* **Atomicity** - a single document write has always been atomic. Across
  documents and collections, a multi document transaction commits or aborts as a
  unit, with a genuine server side rollback.
* **Consistency** - enforced as far as you configure it: unique indexes and JSON
  schema validators are the levers, and neither is on by default.
* **Isolation** - snapshot isolation inside a transaction; uncommitted writes are
  invisible to everyone else. A write conflict aborts one side, which is why
  `with_transaction` retries the callback rather than failing outright.
* **Durability** - governed by the write concern. The default `w: "majority"`
  waits for a majority of the replica set to acknowledge; a lower concern trades
  that away, so leave it alone unless you mean to.

The one prerequisite: transactions need a replica set or a sharded cluster. A
standalone `mongod` rejects them, and this adapter turns that rejection into a
message that says so rather than passing on a bare `OperationFailure`.
"""

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any

from pymongo import AsyncMongoClient, DeleteOne, ReplaceOne, UpdateOne
from pymongo.asynchronous.client_session import AsyncClientSession
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import BulkWriteError, OperationFailure

from DbAdapters.base import WriteFailure, WriteOutcome
from DbAdapters.documentBase import (
    WRITE_DELETE,
    WRITE_MERGE,
    WRITE_SET,
    Document,
    DocumentDbAdapter,
    WriteOperation,
)

CONNECTION_URI_ENV_VAR = "MONGODB_URI"
DATABASE_NAME_ENV_VAR = "MONGODB_DATABASE"

DEFAULT_CONNECTION_URI = "mongodb://localhost:27017"
DEFAULT_TIMEOUT_MS = 30000
DEFAULT_FIND_LIMIT = 0  # 0 means "no limit", which is what the driver expects.

# MongoDB keys every document by `_id`; our Document keys it by `id`.
MONGO_ID_FIELD = "_id"

# Server error codes that mean the deployment cannot run transactions at all,
# as opposed to a transaction that merely failed this time.
NO_TRANSACTION_SUPPORT_CODES = frozenset({20, 40415})


class MongoAdapterError(RuntimeError):
    """The adapter itself is misconfigured or was driven incorrectly.

    Driver errors (`OperationFailure`, `ConnectionFailure` and the rest of
    `PyMongoError`) propagate unwrapped so callers can react to them by type.
    The one exception is a deployment that cannot run transactions, which
    :meth:`MongoAdapter.applyAtomically` reports as
    :class:`MongoTransactionUnsupportedError` because the fix is configuration,
    not a retry.
    """


class MongoTransactionUnsupportedError(MongoAdapterError):
    """This deployment cannot run transactions, so an atomic write is impossible.

    Raised when a standalone `mongod` is asked for a multi document transaction.
    Point the adapter at a replica set (even a single node one, `rs.initiate()`)
    or a sharded cluster.
    """


def toMongoDocument(document: Document) -> dict[str, Any]:
    """Adapt one of our documents into the shape the driver stores.

    The id moves into `_id`, and any `_id` already inside `data` is dropped: the
    document's own id is the one that counts.
    """
    stored = {key: value for key, value in document.data.items() if key != MONGO_ID_FIELD}
    stored[MONGO_ID_FIELD] = document.id
    return stored


def toDocument(raw: Mapping[str, Any]) -> Document:
    """Adapt one stored document back into a store agnostic one.

    `_id` is stringified: a caller that never set an id gets Mongo's ObjectId,
    and every adapter in this package keys documents by `str`.
    """
    return Document(
        id=str(raw.get(MONGO_ID_FIELD, "")),
        data={key: value for key, value in raw.items() if key != MONGO_ID_FIELD},
    )


def toWriteOperation(operation: WriteOperation) -> ReplaceOne | UpdateOne | DeleteOne:
    """Adapt one of our writes into the driver's bulk operation for it.

    A set replaces the whole document and creates it when absent; a merge is a
    `$set` of the named fields only, which leaves the rest of the document alone
    and also creates it when absent.
    """
    documentFilter = {MONGO_ID_FIELD: operation.documentId}

    if operation.kind == WRITE_DELETE:
        return DeleteOne(documentFilter)

    data = operation.data or {}
    if operation.kind == WRITE_SET:
        replacement = {key: value for key, value in data.items() if key != MONGO_ID_FIELD}
        return ReplaceOne(documentFilter, replacement, upsert=True)

    fields = {key: value for key, value in data.items() if key != MONGO_ID_FIELD}
    return UpdateOne(documentFilter, {"$set": fields}, upsert=True)


def isTransactionUnsupported(failure: OperationFailure) -> bool:
    """Whether this failure means the deployment has no transactions at all.

    A standalone `mongod` answers a `startTransaction` with an
    IllegalOperation/unrecognised-field error rather than a dedicated code, so
    the message is part of the signal.
    """
    if failure.code in NO_TRANSACTION_SUPPORT_CODES:
        return True
    message = str(failure).lower()
    return "transaction numbers" in message or "replica set" in message


def failedIdsOfBulkError(error: BulkWriteError, requestedIds: Sequence[str]) -> list[str]:
    """Work out which ids a partial bulk write left behind.

    An unordered bulk write reports the index of each failed request, which lines
    up with the ids in the order they were submitted. A report without a usable
    index falls back to naming every id, since an unknown failure is worse than
    an over-broad one.
    """
    writeErrors = error.details.get("writeErrors", []) if error.details else []
    failedIds: list[str] = []
    for writeError in writeErrors:
        index = writeError.get("index")
        if isinstance(index, int) and 0 <= index < len(requestedIds):
            failedIds.append(requestedIds[index])
        else:
            return list(requestedIds)
    return failedIds


class MongoAdapter(DocumentDbAdapter):
    """Async reads and writes against one MongoDB database.

    The client owns a connection pool, so build one adapter and share it: a
    client per request throws the pool away every time.

    Example:
        async with MongoAdapter(databaseName="rag") as store:
            await store.upsertDocument("chunks", Document(id="c-1", data={"text": "..."}))
            found = await store.findDocuments("chunks", equals={"sourceId": "doc-9"})
    """

    def __init__(
        self,
        *,
        connectionUri: str | None = None,
        databaseName: str | None = None,
        timeoutMs: int = DEFAULT_TIMEOUT_MS,
        clientOptions: Mapping[str, Any] | None = None,
    ) -> None:
        """Record the connection settings without opening anything yet.

        Args:
            connectionUri: Standard MongoDB URI, defaulting to `$MONGODB_URI` and
                then to a local server.
            databaseName: Database to work in, defaulting to `$MONGODB_DATABASE`.
                A URI that names a default database does not remove the need for
                this, since the adapter never guesses.
            timeoutMs: Server selection and socket timeout, in milliseconds.
            clientOptions: Extra keyword arguments passed straight to
                `AsyncMongoClient`, for TLS, write concern and the like.

        Raises:
            MongoAdapterError: If no database name is available.
        """
        self._connectionUri = (
            connectionUri or os.environ.get(CONNECTION_URI_ENV_VAR) or DEFAULT_CONNECTION_URI
        )
        self._databaseName = databaseName or os.environ.get(DATABASE_NAME_ENV_VAR, "")
        self._timeoutMs = timeoutMs
        self._clientOptions = dict(clientOptions or {})

        if not self._databaseName:
            raise MongoAdapterError(
                f"No database to talk to: pass databaseName or set ${DATABASE_NAME_ENV_VAR}."
            )

        self._client: AsyncMongoClient | None = None
        self._database: AsyncDatabase | None = None
        # Guards the lazy connect so concurrent first calls open one client
        # between them rather than one each.
        self._connectLock = asyncio.Lock()

    @property
    def databaseName(self) -> str:
        """Database this adapter reads and writes."""
        return self._databaseName

    @property
    def isConnected(self) -> bool:
        """Whether the client is currently open."""
        return self._database is not None

    async def connect(self) -> None:
        """Open the client, once, however often it is called.

        The driver connects lazily in the background, so this returns before any
        server has been reached; the first real operation is what surfaces an
        unreachable deployment. Call :meth:`ping` at startup to find out sooner.
        """
        if self._database is not None:
            return

        async with self._connectLock:
            if self._database is not None:
                return

            client: AsyncMongoClient = AsyncMongoClient(
                self._connectionUri,
                serverSelectionTimeoutMS=self._timeoutMs,
                **self._clientOptions,
            )
            self._client = client
            self._database = client[self._databaseName]

    async def close(self) -> None:
        """Close the client and its pool, once, however often it is called."""
        client, self._client = self._client, None
        self._database = None
        if client is not None:
            await client.aclose()

    async def getDatabase(self) -> AsyncDatabase:
        """The open database handle, connecting first if need be.

        Exposed for the operations this adapter does not wrap: aggregations,
        change streams, GridFS and so on.
        """
        await self.connect()
        if self._database is None:  # pragma: no cover - connect() raises instead
            raise MongoAdapterError("MongoDB database handle is not available.")
        return self._database

    async def getCollection(self, collection: str) -> AsyncCollection:
        """One collection handle from the open database."""
        database = await self.getDatabase()
        return database[collection]

    async def ping(self) -> bool:
        """Round trip to the server, to prove the connection really works.

        Returns True on success and raises the driver's own error otherwise, so
        a startup check reports why it failed rather than just that it did.
        """
        database = await self.getDatabase()
        await database.command("ping")
        return True

    async def fetchDocument(self, collection: str, documentId: str) -> Document | None:
        """Read one document, or None when the collection does not hold that id."""
        handle = await self.getCollection(collection)
        raw = await handle.find_one({MONGO_ID_FIELD: documentId})
        return toDocument(raw) if raw is not None else None

    async def fetchDocuments(
        self, collection: str, documentIds: Sequence[str]
    ) -> dict[str, Document]:
        """Read documents by id, keyed by id, in one round trip.

        Missing ids are absent from the result rather than an error.
        """
        if not documentIds:
            return {}

        handle = await self.getCollection(collection)
        cursor = handle.find({MONGO_ID_FIELD: {"$in": list(documentIds)}})
        return {document.id: document async for document in _documentsOf(cursor)}

    async def findDocuments(
        self,
        collection: str,
        *,
        query: Mapping[str, Any] | None = None,
        equals: Mapping[str, Any] | None = None,
        sortBy: Sequence[tuple[str, int]] | None = None,
        limit: int = DEFAULT_FIND_LIMIT,
        skip: int = 0,
    ) -> list[Document]:
        """Read documents matching a query, in the order asked for.

        `equals` is the common case spelled simply - every field has to match its
        value exactly. `query` takes the full MongoDB query language for anything
        richer, and the two merge if both are given. `limit` of 0 means no limit,
        which the driver treats as "every match", so pass one for large
        collections.
        """
        handle = await self.getCollection(collection)
        cursor = handle.find({**(query or {}), **(equals or {})})
        if sortBy:
            cursor = cursor.sort(list(sortBy))
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        return [document async for document in _documentsOf(cursor)]

    async def countDocuments(
        self, collection: str, *, query: Mapping[str, Any] | None = None
    ) -> int:
        """How many documents match, counted by the server rather than fetched."""
        handle = await self.getCollection(collection)
        return await handle.count_documents(dict(query or {}))

    async def upsertDocument(
        self, collection: str, document: Document, *, merge: bool = False
    ) -> None:
        """Write one document, replacing what is there or merging into it.

        A single document write is atomic in MongoDB whichever mode is used, so
        this either lands whole or raises.
        """
        handle = await self.getCollection(collection)
        documentFilter = {MONGO_ID_FIELD: document.id}

        if merge:
            fields = {
                key: value for key, value in document.data.items() if key != MONGO_ID_FIELD
            }
            await handle.update_one(documentFilter, {"$set": fields}, upsert=True)
        else:
            await handle.replace_one(
                documentFilter, toMongoDocument(document), upsert=True
            )

    async def upsertDocuments(
        self, collection: str, documents: Sequence[Document], *, merge: bool = False
    ) -> WriteOutcome:
        """Write many documents in one unordered bulk request.

        Unordered on purpose: one rejected document does not stop the others, so
        the result can describe a partial write. Check `WriteOutcome.succeeded`
        before treating it as done, and use :meth:`applyAtomically` when the
        batch has to land whole.

        Writes are idempotent by id, so resending failed ids is safe.
        """
        if not documents:
            return WriteOutcome(requestedCount=0, appliedCount=0)

        kind = WRITE_MERGE if merge else WRITE_SET
        operations = [
            WriteOperation(
                collection=collection, documentId=document.id, kind=kind, data=document.data
            )
            for document in documents
        ]
        return await self._bulkWrite(collection, operations)

    async def deleteDocument(self, collection: str, documentId: str) -> None:
        """Remove one document. An id that is not there is not an error."""
        handle = await self.getCollection(collection)
        await handle.delete_one({MONGO_ID_FIELD: documentId})

    async def deleteDocuments(
        self, collection: str, documentIds: Sequence[str]
    ) -> WriteOutcome:
        """Remove many documents by id in one request.

        Ids that are not there are not an error, so `appliedCount` counts the
        documents actually removed and can legitimately be lower than the number
        asked for. The outcome still reports success, since nothing failed.
        """
        if not documentIds:
            return WriteOutcome(requestedCount=0, appliedCount=0)

        handle = await self.getCollection(collection)
        result = await handle.delete_many({MONGO_ID_FIELD: {"$in": list(documentIds)}})

        if not result.acknowledged:
            # An unacknowledged write concern means the server never confirmed
            # anything, so the delete cannot be reported as applied.
            return WriteOutcome(
                requestedCount=len(documentIds),
                appliedCount=0,
                failures=[
                    WriteFailure(
                        ids=list(documentIds),
                        message="Delete was not acknowledged by the server.",
                        retryable=True,
                        disposition="unknown",
                    )
                ],
            )

        # Ids that were not there are not failures: the requested state - absent -
        # holds for all of them. `deleted_count` says how many actually existed,
        # which is what deleteMatching reports and this method does not.
        return WriteOutcome(requestedCount=len(documentIds), appliedCount=len(documentIds))

    async def deleteMatching(self, collection: str, query: Mapping[str, Any]) -> int:
        """Remove every document matching `query`, returning how many went.

        Refuses an empty query: `deleteMatching(collection, {})` would empty the
        collection, which is what :meth:`dropCollection` is for and should not
        happen by accident.
        """
        if not query:
            raise MongoAdapterError(
                "deleteMatching needs a query; use dropCollection to empty a collection."
            )
        handle = await self.getCollection(collection)
        result = await handle.delete_many(dict(query))
        return result.deleted_count

    async def dropCollection(self, collection: str) -> None:
        """Delete a collection outright, documents and indexes together."""
        handle = await self.getCollection(collection)
        await handle.drop()

    async def applyAtomically(self, operations: Sequence[WriteOperation]) -> None:
        """Apply every operation inside one transaction, or none of them.

        A real transaction, not a compensating one: the server rolls back on
        failure, other sessions never see a half applied state, and a write
        conflict retries the whole block rather than leaving it torn. Operations
        may span collections.

        The block is retried on transient errors, so keep it short and free of
        side effects outside the database - it can run more than once.

        Raises:
            MongoTransactionUnsupportedError: If the deployment is a standalone
                `mongod`, which has no transactions to offer.
        """
        if not operations:
            return

        byCollection: dict[str, list[WriteOperation]] = {}
        for operation in operations:
            byCollection.setdefault(operation.collection, []).append(operation)

        database = await self.getDatabase()

        async def applyAll(session: AsyncClientSession) -> None:
            """Every write in the block, against the transaction's session."""
            for collection, collectionOperations in byCollection.items():
                await database[collection].bulk_write(
                    [toWriteOperation(operation) for operation in collectionOperations],
                    ordered=True,
                    session=session,
                )

        await self.runTransaction(applyAll)

    async def runTransaction(
        self, operation: Callable[[AsyncClientSession], Awaitable[Any]]
    ) -> Any:
        """Run `operation` inside a transaction and return whatever it returns.

        This is the read-modify-write primitive: unlike
        :meth:`applyAtomically`, the callback can read within the transaction and
        decide what to write from what it read, under snapshot isolation.

        Pass the session on to every call inside the block - a driver call
        without it runs outside the transaction and will not be rolled back:

            async def moveCredit(session):
                source = await database["accounts"].find_one({"_id": "a"}, session=session)
                await database["accounts"].update_one(
                    {"_id": "b"}, {"$inc": {"credit": source["credit"]}}, session=session
                )

            await adapter.runTransaction(moveCredit)

        The driver retries the callback on transient errors and on a commit that
        is not yet known to have succeeded, so it must be safe to run twice.

        Raises:
            MongoTransactionUnsupportedError: If the deployment has no
                transaction support.
        """
        await self.connect()
        if self._client is None:  # pragma: no cover - connect() raises instead
            raise MongoAdapterError("MongoDB client is not available.")

        try:
            async with self._client.start_session() as session:
                return await session.with_transaction(operation)
        except OperationFailure as failure:
            if isTransactionUnsupported(failure):
                raise MongoTransactionUnsupportedError(
                    "This MongoDB deployment does not support transactions. Point the "
                    "adapter at a replica set (a single node one, initialised with "
                    "rs.initiate(), is enough) or a sharded cluster."
                ) from failure
            raise

    async def createIndex(
        self,
        collection: str,
        keys: Sequence[tuple[str, int]],
        *,
        unique: bool = False,
        name: str | None = None,
    ) -> str:
        """Create an index, returning its name. Doing so twice is harmless.

        A unique index is the closest MongoDB comes to a consistency constraint
        you declare rather than enforce in code, so it is worth having on any
        field a document is looked up by.
        """
        handle = await self.getCollection(collection)
        return await handle.create_index(list(keys), unique=unique, name=name)

    async def _bulkWrite(
        self, collection: str, operations: Sequence[WriteOperation]
    ) -> WriteOutcome:
        """Send writes unordered, turning a partial failure into an outcome.

        Kept separate from :meth:`applyAtomically` on purpose: that one wants the
        whole block to fail together, this one wants as much of it as possible to
        land and the rest reported.
        """
        handle = await self.getCollection(collection)
        requestedIds = [operation.documentId for operation in operations]

        try:
            result = await handle.bulk_write(
                [toWriteOperation(operation) for operation in operations], ordered=False
            )
        except BulkWriteError as error:
            failedIds = failedIdsOfBulkError(error, requestedIds)
            writeErrors = error.details.get("writeErrors", []) if error.details else []
            return WriteOutcome(
                requestedCount=len(operations),
                appliedCount=len(operations) - len(failedIds),
                failures=[
                    WriteFailure(
                        ids=failedIds,
                        message="; ".join(
                            str(writeError.get("errmsg", "")) for writeError in writeErrors
                        )
                        or str(error),
                        # A rejected write is usually deterministic - a schema
                        # violation or a duplicate key - so resending it unchanged
                        # fails the same way.
                        retryable=False,
                        disposition="rejected",
                    )
                ],
            )

        return WriteOutcome(
            requestedCount=len(operations),
            # Matched-but-identical documents report no modification, so counting
            # modified alone would understate a write that did land.
            appliedCount=result.upserted_count + result.matched_count + result.deleted_count,
        )


async def _documentsOf(cursor: Any) -> AsyncIterator[Document]:
    """Walk a driver cursor, adapting each raw document as it arrives.

    Async generator rather than a list build, so a large result set is converted
    a document at a time instead of twice over in memory.
    """
    async for raw in cursor:
        yield toDocument(raw)
