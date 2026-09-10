"""Firestore implementation of :class:`~DbAdapters.documentBase.DocumentDbAdapter`.

Firestore is Firebase's document database, and `google-cloud-firestore`'s
`AsyncClient` is the async way into it. The `firebase-admin` package wraps the
same service but its Firestore surface is synchronous, so calling it from a
request handler would block the event loop - hence this client instead. Both
authenticate the same way and reach the same data, so a project already using
`firebase-admin` elsewhere loses nothing here.

Credentials come from `$GOOGLE_APPLICATION_CREDENTIALS`, from an explicit service
account file, or from the ambient credentials on Cloud Run and GCE.

What "ACID" means here
----------------------
Firestore is transactional, with limits worth knowing before you lean on it:

* **Atomicity** - a batched write and a transaction both commit whole or not at
  all, across collections. :meth:`FirestoreAdapter.applyAtomically` uses a
  batched write, which is the cheaper of the two because it never reads.
* **Consistency** - security rules and index constraints, both configured outside
  this code. Firestore itself will store any shape you hand it.
* **Isolation** - serializable. A transaction that read a document another writer
  then changed is retried from the top, which is why the callback given to
  :meth:`FirestoreAdapter.runTransaction` must be free of outside side effects.
* **Durability** - a committed write is replicated synchronously before the
  commit returns, and reads are strongly consistent: unlike Pinecone, a read
  right after a write sees it.

The hard ceiling is size, not semantics: **500 operations per batch or
transaction**, and one document may not be written more than once inside the
same one. Both are checked here before the request goes out, so an oversized
block fails with a message about the limit rather than a server error.
"""

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from google.cloud import firestore
from google.cloud.firestore_v1 import AsyncTransaction, FieldFilter, async_transactional
from google.cloud.firestore_v1.base_document import DocumentSnapshot

from DbAdapters.base import WriteFailure, WriteOutcome, chunkItems
from DbAdapters.documentBase import (
    WRITE_DELETE,
    WRITE_SET,
    Document,
    DocumentDbAdapter,
    WriteOperation,
)

PROJECT_ID_ENV_VAR = "GOOGLE_CLOUD_PROJECT"
DATABASE_ID_ENV_VAR = "FIRESTORE_DATABASE"
CREDENTIALS_PATH_ENV_VAR = "GOOGLE_APPLICATION_CREDENTIALS"

# Firestore's own default database id; naming it beats passing None around.
DEFAULT_DATABASE_ID = "(default)"

# Firestore rejects a batch or transaction holding more than this many writes.
MAX_OPERATIONS_PER_COMMIT = 500

# `get_all` takes any number of references but is billed and paged per document;
# this keeps one call from holding an unbounded result set in memory.
MAX_REFERENCES_PER_READ = 300

DEFAULT_QUERY_LIMIT = 100


class FirestoreAdapterError(RuntimeError):
    """The adapter itself is misconfigured or was driven incorrectly.

    Google API errors (`NotFound`, `PermissionDenied`, `Aborted` and the rest of
    `GoogleAPICallError`) propagate unwrapped so callers can react by type.
    """


def toDocument(snapshot: DocumentSnapshot) -> Document:
    """Adapt one Firestore snapshot into a store agnostic document."""
    return Document(id=snapshot.id, data=dict(snapshot.to_dict() or {}))


def checkCommitSize(operations: Sequence[WriteOperation]) -> None:
    """Reject a block Firestore would refuse, before spending a round trip on it.

    Two limits apply to a single batch or transaction: at most
    `MAX_OPERATIONS_PER_COMMIT` writes, and at most one write per document.
    Writing the same document twice is the easier one to hit by accident, and the
    server's error for it does not say which document was duplicated.
    """
    if len(operations) > MAX_OPERATIONS_PER_COMMIT:
        raise FirestoreAdapterError(
            f"Firestore commits at most {MAX_OPERATIONS_PER_COMMIT} writes at once, "
            f"and this block has {len(operations)}. Split it, accepting that the "
            f"pieces are then no longer atomic with one another."
        )

    seen: set[tuple[str, str]] = set()
    for operation in operations:
        target = (operation.collection, operation.documentId)
        if target in seen:
            raise FirestoreAdapterError(
                f"Firestore allows one write per document per commit, but "
                f"{operation.collection}/{operation.documentId} is written twice. "
                f"Fold the two writes into one."
            )
        seen.add(target)


class FirestoreAdapter(DocumentDbAdapter):
    """Async reads and writes against one Firestore database.

    The client holds a gRPC channel, so build one adapter and share it rather
    than paying for a channel per request.

    Example:
        async with FirestoreAdapter(projectId="my-project") as store:
            await store.upsertDocument("chunks", Document(id="c-1", data={"text": "..."}))
            recent = await store.findDocuments("chunks", equals={"sourceId": "doc-9"})
    """

    def __init__(
        self,
        *,
        projectId: str | None = None,
        databaseId: str | None = None,
        credentialsPath: str | None = None,
    ) -> None:
        """Record the connection settings without opening anything yet.

        Args:
            projectId: Google Cloud project holding the database, defaulting to
                `$GOOGLE_CLOUD_PROJECT`. When neither is given the project is
                taken from the ambient credentials, which is what you want on
                Cloud Run.
            databaseId: Which Firestore database in the project, defaulting to
                `$FIRESTORE_DATABASE` and then to Firestore's own default.
            credentialsPath: Service account JSON to authenticate with,
                defaulting to `$GOOGLE_APPLICATION_CREDENTIALS`. Leave both unset
                to use ambient credentials.
        """
        self._projectId = projectId or os.environ.get(PROJECT_ID_ENV_VAR, "")
        self._databaseId = (
            databaseId or os.environ.get(DATABASE_ID_ENV_VAR) or DEFAULT_DATABASE_ID
        )
        self._credentialsPath = credentialsPath or os.environ.get(CREDENTIALS_PATH_ENV_VAR, "")

        self._client: firestore.AsyncClient | None = None
        # Guards the lazy connect so concurrent first calls open one channel
        # between them rather than one each.
        self._connectLock = asyncio.Lock()

    @property
    def databaseId(self) -> str:
        """Firestore database this adapter reads and writes."""
        return self._databaseId

    @property
    def isConnected(self) -> bool:
        """Whether the client is currently open."""
        return self._client is not None

    async def connect(self) -> None:
        """Open the client, once, however often it is called.

        Building the client resolves credentials, which can touch the filesystem
        or the metadata server, so it runs in a worker thread rather than on the
        event loop.
        """
        if self._client is not None:
            return

        async with self._connectLock:
            if self._client is not None:
                return
            self._client = await asyncio.to_thread(self._buildClient)

    def _buildClient(self) -> firestore.AsyncClient:
        """Construct the client. Blocking, so :meth:`connect` keeps it off the loop."""
        options: dict[str, Any] = {"database": self._databaseId}
        if self._projectId:
            options["project"] = self._projectId

        if self._credentialsPath:
            return firestore.AsyncClient.from_service_account_json(
                self._credentialsPath, **options
            )
        return firestore.AsyncClient(**options)

    async def close(self) -> None:
        """Close the client's gRPC channel, once, however often it is called."""
        client, self._client = self._client, None
        if client is not None:
            # The client's close is synchronous, and shutting a channel down can
            # block briefly while in-flight calls finish.
            await asyncio.to_thread(client.close)

    async def getClient(self) -> firestore.AsyncClient:
        """The open client, connecting first if need be.

        Exposed for the operations this adapter does not wrap: collection groups,
        snapshot listeners, aggregation queries and vector search.
        """
        await self.connect()
        if self._client is None:  # pragma: no cover - connect() raises instead
            raise FirestoreAdapterError("Firestore client is not available.")
        return self._client

    async def fetchDocument(self, collection: str, documentId: str) -> Document | None:
        """Read one document, or None when the collection does not hold that id."""
        client = await self.getClient()
        snapshot = await client.collection(collection).document(documentId).get()
        return toDocument(snapshot) if snapshot.exists else None

    async def fetchDocuments(
        self, collection: str, documentIds: Sequence[str]
    ) -> dict[str, Document]:
        """Read documents by id, keyed by id.

        Uses Firestore's batched read, so a page of ids costs one round trip
        rather than one each. Missing ids are absent from the result: Firestore
        returns a snapshot for them too, and those are dropped here.
        """
        if not documentIds:
            return {}

        client = await self.getClient()
        handle = client.collection(collection)
        documents: dict[str, Document] = {}

        for chunk in chunkItems(list(documentIds), MAX_REFERENCES_PER_READ):
            references = [handle.document(documentId) for documentId in chunk]
            async for snapshot in client.get_all(references):
                if snapshot.exists:
                    documents[snapshot.id] = toDocument(snapshot)

        return documents

    async def findDocuments(
        self,
        collection: str,
        *,
        equals: Mapping[str, Any] | None = None,
        conditions: Sequence[tuple[str, str, Any]] | None = None,
        orderBy: str | None = None,
        descending: bool = False,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> list[Document]:
        """Read documents matching the given conditions.

        `equals` is the common case spelled simply. `conditions` takes Firestore's
        full comparison set as `(field, operator, value)` triples - `">="`,
        `"array_contains"`, `"in"` and so on - and the two combine.

        Every condition is ANDed. Firestore needs a composite index for most
        multi field queries and for any query that filters on one field and
        orders by another; the error it raises when one is missing carries a link
        that creates it.
        """
        client = await self.getClient()
        query: Any = client.collection(collection)

        for field, value in (equals or {}).items():
            query = query.where(filter=FieldFilter(field, "==", value))
        for field, operator, value in conditions or []:
            query = query.where(filter=FieldFilter(field, operator, value))

        if orderBy:
            direction = firestore.Query.DESCENDING if descending else firestore.Query.ASCENDING
            query = query.order_by(orderBy, direction=direction)
        if limit:
            query = query.limit(limit)

        return [toDocument(snapshot) async for snapshot in query.stream()]

    async def upsertDocument(
        self, collection: str, document: Document, *, merge: bool = False
    ) -> None:
        """Write one document, replacing what is there or merging into it.

        A single document write is atomic, so this either lands whole or raises.
        With `merge` the fields given are folded into whatever is stored and the
        rest is left alone; without it the document is replaced outright.
        """
        client = await self.getClient()
        await client.collection(collection).document(document.id).set(
            dict(document.data), merge=merge
        )

    async def upsertDocuments(
        self, collection: str, documents: Sequence[Document], *, merge: bool = False
    ) -> WriteOutcome:
        """Write many documents, in batches of `MAX_OPERATIONS_PER_COMMIT`.

        Each batch is atomic in itself, but the batches are not atomic with one
        another, so a long write can land in part and the result says so. Use
        :meth:`applyAtomically` when the whole set has to commit together - at
        the cost of the 500 write ceiling.
        """
        if not documents:
            return WriteOutcome(requestedCount=0, appliedCount=0)

        operations = [
            WriteOperation(
                collection=collection, documentId=document.id, kind=WRITE_SET, data=document.data
            )
            for document in documents
        ]
        return await self._commitInBatches(operations, merge=merge)

    async def deleteDocument(self, collection: str, documentId: str) -> None:
        """Remove one document. An id that is not there is not an error."""
        client = await self.getClient()
        await client.collection(collection).document(documentId).delete()

    async def deleteDocuments(
        self, collection: str, documentIds: Sequence[str]
    ) -> WriteOutcome:
        """Remove many documents, in batches of `MAX_OPERATIONS_PER_COMMIT`.

        Ids that are not there are not an error. Deleting a document does not
        delete the subcollections beneath it - those are separate documents and
        outlive their parent; :meth:`deleteRecursively` is for that.
        """
        if not documentIds:
            return WriteOutcome(requestedCount=0, appliedCount=0)

        operations = [
            WriteOperation(collection=collection, documentId=documentId, kind=WRITE_DELETE)
            for documentId in documentIds
        ]
        return await self._commitInBatches(operations, merge=False)

    async def deleteRecursively(self, collection: str, documentId: str) -> None:
        """Remove a document and every subcollection beneath it.

        Firestore has no server side cascade, so this walks the tree and deletes
        as it goes: not atomic, and potentially many round trips. It is the only
        way to avoid leaving orphaned subcollection documents behind, which stay
        billable and stay visible to collection group queries.
        """
        client = await self.getClient()
        reference = client.collection(collection).document(documentId)
        await client.recursive_delete(reference)

    async def applyAtomically(self, operations: Sequence[WriteOperation]) -> None:
        """Apply every operation in one batched write, or none of them.

        A genuine atomic commit: no other reader sees a half applied state, and a
        failure leaves nothing behind. Operations may span collections. A merge
        folds into the stored fields server side, so no read is needed and this
        stays cheaper than a transaction.

        Use :meth:`runTransaction` instead when what to write depends on what is
        currently stored - a batch cannot read.

        Raises:
            FirestoreAdapterError: If the block exceeds
                `MAX_OPERATIONS_PER_COMMIT` writes, or writes one document twice.
        """
        if not operations:
            return

        checkCommitSize(operations)

        client = await self.getClient()
        batch = client.batch()
        for operation in operations:
            reference = client.collection(operation.collection).document(operation.documentId)
            if operation.kind == WRITE_DELETE:
                batch.delete(reference)
            else:
                batch.set(
                    reference, dict(operation.data or {}), merge=operation.kind != WRITE_SET
                )

        await batch.commit()

    async def runTransaction(
        self, operation: Callable[[AsyncTransaction], Awaitable[Any]]
    ) -> Any:
        """Run `operation` inside a transaction and return whatever it returns.

        This is the read-modify-write primitive: the callback can read documents
        and decide what to write from what it read, and Firestore guarantees
        nothing changed underneath in between.

        Every read and write inside must go through the transaction, or it
        escapes the guarantee - reads take `transaction=transaction`, writes go
        through `transaction.set` / `.update` / `.delete`:

            async def reserveSeat(transaction):
                reference = client.collection("seats").document("12a")
                snapshot = await reference.get(transaction=transaction)
                if snapshot.get("taken"):
                    return False
                transaction.update(reference, {"taken": True})
                return True

            reserved = await adapter.runTransaction(reserveSeat)

        Firestore requires every read to happen before the first write, and
        retries the whole callback when a document it read was changed by someone
        else, so the callback must be safe to run more than once and must not
        have side effects outside the database.
        """
        client = await self.getClient()
        return await async_transactional(operation)(client.transaction())

    async def _commitInBatches(
        self, operations: Sequence[WriteOperation], *, merge: bool
    ) -> WriteOutcome:
        """Commit writes in atomic batches, reporting any batch that failed.

        Batches go out one after another rather than concurrently: Firestore
        rate limits sustained writes to a collection, and parallel batches make
        contention worse rather than better.
        """
        client = await self.getClient()
        chunks = list(chunkItems(list(operations), MAX_OPERATIONS_PER_COMMIT))

        appliedCount = 0
        failures: list[WriteFailure] = []

        for chunk in chunks:
            batch = client.batch()
            for operation in chunk:
                reference = client.collection(operation.collection).document(
                    operation.documentId
                )
                if operation.kind == WRITE_DELETE:
                    batch.delete(reference)
                else:
                    batch.set(reference, dict(operation.data or {}), merge=merge)

            try:
                await batch.commit()
            except Exception as error:
                failures.append(
                    WriteFailure(
                        ids=[operation.documentId for operation in chunk],
                        message=str(error),
                        # A batch is atomic, so a failed one wrote nothing and can
                        # be resent as it stands.
                        retryable=True,
                        disposition="rejected",
                    )
                )
            else:
                appliedCount += len(chunk)

        return WriteOutcome(
            requestedCount=len(operations), appliedCount=appliedCount, failures=failures
        )
