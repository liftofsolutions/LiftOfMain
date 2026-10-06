"""Pinecone implementation of :class:`~DbAdapters.vectorBase.VectorDbAdapter`.

Part of `DbAdapters.pineconeDb`: the server never uses this directly, but
through `getPineconeDb`, whose queue holds the one adapter it shares.

Why every I/O method is `async def`
-----------------------------------
Declaring a function `async` does not on its own stop it blocking. An
`async def` that calls the *synchronous* `Pinecone` client still owns the event
loop for the entire HTTP round trip, so every other request handled by the same
worker waits behind it. What makes these methods concurrent is the transport
underneath: this adapter drives `PineconeAsyncio`, whose calls yield to the loop
while the network is in flight. The caller's choice of `await` matters, but only
because the implementation gives it something real to await.

Pure helpers stay ordinary functions: there is nothing to await while rebuilding
a dict, and marking them `async` would add call overhead and force callers to
await something that never yields.

What "ACID" can and cannot mean here
------------------------------------
Pinecone is not a transactional database and no adapter can make it one. Stated
plainly, so callers can design around it:

* **Atomicity** - per vector only. A batched upsert is sent batch by batch, so a
  failure part way leaves the earlier batches in place. :meth:`upsertVectors`
  reports that split instead of hiding it, and :meth:`upsertVectorsAtomically`
  adds a compensating rollback for callers that need all-or-nothing.
* **Consistency** - no cross record constraints, no schema enforcement on
  metadata. Validate before writing.
* **Isolation** - none. Concurrent writers to the same ids interleave freely,
  and a rollback can undo a write another caller made in between.
* **Durability** - an acknowledged upsert is durable, but reads are *eventually*
  consistent: a fetch issued straight after a write may legitimately miss it.

Configuration comes from the constructor, falling back to the environment
variables named in the `*_ENV_VAR` constants below.
"""

import asyncio
import os
from collections.abc import Mapping, Sequence
from typing import Any

from pinecone import (
    AsyncIndex,
    Hit,
    NotFoundError,
    PineconeAsyncio,
    ScoredVector,
    ServerlessSpec,
    Vector,
)

from DbAdapters.base import WriteFailure, WriteOutcome, chunkItems
from DbAdapters.vectorBase import QueryMatch, VectorDbAdapter, VectorRecord

API_KEY_ENV_VAR = "PINECONE_API_KEY"
INDEX_NAME_ENV_VAR = "PINECONE_INDEX_NAME"
INDEX_HOST_ENV_VAR = "PINECONE_INDEX_HOST"
NAMESPACE_ENV_VAR = "PINECONE_NAMESPACE"

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_UPSERT_BATCH_SIZE = 100
# The records API embeds server side, so its batches are sized by the text
# going out rather than by vector count, and stay smaller than a raw upsert's.
DEFAULT_RECORDS_BATCH_SIZE = 96
DEFAULT_TOP_K = 10
DEFAULT_METRIC = "cosine"
DEFAULT_CLOUD = "aws"
DEFAULT_REGION = "us-east-1"
DEFAULT_CREATE_TIMEOUT_SECONDS = 300

# Pinecone caps how many ids one fetch or delete request may name. Chunking
# below the cap is safe whatever the server currently allows.
MAX_IDS_PER_FETCH = 1000
MAX_IDS_PER_DELETE = 1000

# Pinecone rejects a list page size outside 1-100, so a caller's total cap has
# to be clamped before it is handed on as a page size.
MAX_LIST_PAGE_SIZE = 100

# Reported for a failed slice when the store gave no disposition of its own.
UNKNOWN_DISPOSITION = "unknown"


class PineconeAdapterError(RuntimeError):
    """The adapter itself is misconfigured or was driven incorrectly.

    Errors raised by Pinecone (`NotFoundError`, `RateLimitError` and the rest of
    the `PineconeError` family) are deliberately left to propagate unwrapped, so
    callers can react to them by type.
    """


class PineconeTransactionError(PineconeAdapterError):
    """An all-or-nothing write failed, and the rollback is accounted for here.

    Attributes:
        outcome: What the failed write did manage to apply.
        rollbackSucceeded: Whether the compensating write restored the prior
            state. When False the index is left in a mixed state and needs
            attention, and `rollbackError` says why.
        rollbackError: What defeated the rollback, when one did.
    """

    def __init__(
        self,
        message: str,
        *,
        outcome: WriteOutcome,
        rollbackSucceeded: bool,
        rollbackError: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.outcome = outcome
        self.rollbackSucceeded = rollbackSucceeded
        self.rollbackError = rollbackError


def toPineconeVector(record: VectorRecord) -> Vector:
    """Adapt one of our records into the SDK's vector type."""
    return Vector(id=record.id, values=list(record.values), metadata=dict(record.metadata))


def toVectorRecord(vector: Vector) -> VectorRecord:
    """Adapt one SDK vector back into a store agnostic record."""
    return VectorRecord(
        id=vector.id,
        values=list(vector.values or []),
        metadata=dict(vector.metadata or {}),
    )


def toQueryMatch(match: ScoredVector) -> QueryMatch:
    """Adapt one SDK search hit into a store agnostic match."""
    return QueryMatch(
        id=match.id,
        score=match.score,
        values=list(match.values) if match.values else None,
        metadata=dict(match.metadata) if match.metadata else None,
    )


def toTextMatch(hit: Hit) -> QueryMatch:
    """Adapt one integrated search hit into the same match type a query returns.

    A hit carries stored fields rather than a metadata dict, and never the
    embedding - Pinecone made it server side and does not hand it back - so
    `values` is always None here.
    """
    return QueryMatch(id=hit.id, score=hit.score, values=None, metadata=dict(hit.fields or {}))


def toWriteFailures(batchErrors: Sequence[Any]) -> list[WriteFailure]:
    """Adapt the SDK's per batch errors into our failure records.

    Each error still carries the items its batch was holding, which is what lets
    a caller resubmit exactly the writes that did not land.
    """
    failures: list[WriteFailure] = []
    for batchError in batchErrors:
        items = getattr(batchError, "items", None) or []
        failures.append(
            WriteFailure(
                ids=[str(item.get("id", "")) for item in items],
                message=getattr(batchError, "error_message", "") or str(batchError),
                retryable=bool(getattr(batchError, "retryable", False)),
                disposition=getattr(batchError, "disposition", "") or UNKNOWN_DISPOSITION,
            )
        )
    return failures


def incompleteRollbackError(step: str, outcome: WriteOutcome) -> PineconeAdapterError:
    """Describe a compensating write that itself only partly landed.

    A rollback whose own write fails leaves the index mixed, so name the step and
    the ids left behind rather than reporting a bare False.
    """
    reasons = "; ".join(failure.message for failure in outcome.failures) or "no reason given"
    return PineconeAdapterError(
        f"Rollback {step} left {len(outcome.failedIds)} of {outcome.requestedCount} "
        f"vectors unresolved ({', '.join(outcome.failedIds[:10])}): {reasons}"
    )


class PineconeAdapter(VectorDbAdapter):
    """Async reads and writes against one Pinecone index.

    The client and the index handle open lazily on first use and are then held
    for the life of the instance, so share one adapter across requests rather
    than building one per call: each construction costs a TLS handshake and,
    when the index is named rather than addressed by host, a describe round trip
    as well.

    Example:
        async with PineconeAdapter(indexName="documents") as store:
            await store.upsertVectors([VectorRecord(id="doc-1", values=embedding)])
            hits = await store.queryVectors(embedding, topK=5)
    """

    def __init__(
        self,
        *,
        apiKey: str | None = None,
        indexName: str | None = None,
        indexHost: str | None = None,
        namespace: str | None = None,
        timeoutSeconds: float = DEFAULT_TIMEOUT_SECONDS,
        upsertBatchSize: int = DEFAULT_UPSERT_BATCH_SIZE,
        connectionPoolSize: int = 0,
    ) -> None:
        """Record the connection settings without opening anything yet.

        Args:
            apiKey: Pinecone API key, defaulting to `$PINECONE_API_KEY`.
            indexName: Index to work against, defaulting to
                `$PINECONE_INDEX_NAME`. Resolving a name costs one describe
                request on connect.
            indexHost: The index's host. Passing it skips that describe request,
                and defaults to `$PINECONE_INDEX_HOST`.
            namespace: Namespace used when a call does not name one, defaulting
                to `$PINECONE_NAMESPACE` and then to the default namespace.
            timeoutSeconds: Per request timeout for every call.
            upsertBatchSize: Vectors per upsert request.
            connectionPoolSize: Most connections held open to the index, so
                at most this many requests run at once. 0 keeps the client's
                own default.

        Raises:
            PineconeAdapterError: If no API key is available, or if neither an
                index name nor an index host is.
        """
        self._apiKey = apiKey or os.environ.get(API_KEY_ENV_VAR, "")
        self._indexName = indexName or os.environ.get(INDEX_NAME_ENV_VAR, "")
        self._indexHost = indexHost or os.environ.get(INDEX_HOST_ENV_VAR, "")
        self._namespace = (
            namespace if namespace is not None else os.environ.get(NAMESPACE_ENV_VAR, "")
        )
        self._timeoutSeconds = timeoutSeconds
        self._upsertBatchSize = upsertBatchSize
        self._connectionPoolSize = connectionPoolSize

        if not self._apiKey:
            raise PineconeAdapterError(
                f"No Pinecone API key: pass apiKey or set ${API_KEY_ENV_VAR}."
            )
        if not self._indexName and not self._indexHost:
            raise PineconeAdapterError(
                f"No index to talk to: pass indexName or indexHost, or set "
                f"${INDEX_NAME_ENV_VAR} or ${INDEX_HOST_ENV_VAR}."
            )

        self._client: PineconeAsyncio | None = None
        self._index: AsyncIndex | None = None
        # Guards the lazy connect so concurrent first calls open one client
        # between them rather than one each. A Lock built outside a running loop
        # is fine on 3.10+, where it binds to the loop on first await.
        self._connectLock = asyncio.Lock()

    @property
    def namespace(self) -> str:
        """Namespace used by calls that do not name one."""
        return self._namespace

    @property
    def isConnected(self) -> bool:
        """Whether the index handle is currently open."""
        return self._index is not None

    async def connect(self) -> None:
        """Open the client and the index handle, once, however often it is called.

        Every other method calls this for you, so an explicit call is only useful
        to pay the connection cost up front, for instance during app startup.
        """
        if self._index is not None:
            return

        async with self._connectLock:
            # A caller that queued on the lock may find the work already done.
            if self._index is not None:
                return

            client = PineconeAsyncio(
                api_key=self._apiKey,
                timeout=self._timeoutSeconds,
                connection_pool_maxsize=self._connectionPoolSize,
            )
            try:
                if self._indexHost:
                    index = client.IndexAsyncio(host=self._indexHost)
                else:
                    index = await client.index(name=self._indexName)
            except BaseException:
                # Otherwise a failed host lookup leaks the control plane client.
                await client.close()
                raise

            self._client = client
            self._index = index

    async def close(self) -> None:
        """Close the index handle and the client, once, however often it is called.

        Both need closing: shutting down the client does not shut down the index
        handles opened from it.
        """
        index, self._index = self._index, None
        client, self._client = self._client, None

        try:
            if index is not None:
                await index.close()
        finally:
            if client is not None:
                await client.close()

    async def getIndex(self) -> AsyncIndex:
        """The open index handle, connecting first if need be.

        Exposed for the calls this adapter does not wrap, such as imports or
        integrated embedding search.
        """
        await self.connect()
        if self._index is None:  # pragma: no cover - connect() raises instead
            raise PineconeAdapterError("Pinecone index handle is not available.")
        return self._index

    def resolveNamespace(self, namespace: str | None) -> str:
        """The namespace a call should use: its own, else the adapter's default."""
        return self._namespace if namespace is None else namespace

    async def ensureIndex(
        self,
        *,
        dimension: int,
        metric: str = DEFAULT_METRIC,
        cloud: str = DEFAULT_CLOUD,
        region: str = DEFAULT_REGION,
        timeoutSeconds: int = DEFAULT_CREATE_TIMEOUT_SECONDS,
    ) -> bool:
        """Create the configured index if it does not exist yet.

        Returns True when this call created it, False when it was already there.
        Creation waits for the index to report ready, so it can take minutes;
        this belongs in a deploy or startup step, not on a request path.

        Raises:
            PineconeAdapterError: If the adapter was configured with a host
                rather than a name, since creating needs a name.
        """
        if not self._indexName:
            raise PineconeAdapterError(
                "ensureIndex needs indexName; a host alone names no index to create."
            )

        client = PineconeAsyncio(api_key=self._apiKey, timeout=self._timeoutSeconds)
        try:
            if await client.indexes.exists(self._indexName):
                return False
            await client.indexes.create(
                name=self._indexName,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
                timeout=timeoutSeconds,
            )
            return True
        finally:
            await client.close()

    async def upsertVectors(
        self,
        records: Sequence[VectorRecord],
        *,
        namespace: str | None = None,
        batchSize: int | None = None,
    ) -> WriteOutcome:
        """Write records, overwriting any that already carry the same id.

        The vectors go out in batches, and a batch that fails does not stop the
        ones after it, so the result can describe a partial write. Check
        `WriteOutcome.succeeded` before treating the write as done;
        :meth:`upsertVectorsAtomically` is the all-or-nothing variant.

        Writes are idempotent by id, so resending `WriteOutcome.retryableIds` is
        safe.
        """
        if not records:
            return WriteOutcome(requestedCount=0, appliedCount=0)

        index = await self.getIndex()
        response = await index.upsert(
            vectors=[toPineconeVector(record) for record in records],
            namespace=self.resolveNamespace(namespace),
            batch_size=batchSize or self._upsertBatchSize,
            show_progress=False,
        )
        return WriteOutcome(
            requestedCount=len(records),
            appliedCount=response.upserted_count,
            failures=toWriteFailures(response.errors),
        )

    async def upsertVectorsAtomically(
        self,
        records: Sequence[VectorRecord],
        *,
        namespace: str | None = None,
        batchSize: int | None = None,
    ) -> WriteOutcome:
        """Write records all-or-nothing, undoing the write if any part fails.

        Pinecone has no transactions, so this is a compensating write rather than
        a real one: the prior state of every id is read first, the upsert is
        attempted, and on failure each id is either deleted (it held nothing
        before) or rewritten with what it held. The undo covers every requested
        id, not only the reported failures, because a batch the server rejected
        may still have landed in part.

        Know what it does not buy you:

        * **No isolation.** A concurrent writer touching the same ids between the
          read and the rollback has its write undone too.
        * **Stale reads.** Reads are eventually consistent, so the snapshot can
          be out of date and the rollback can restore a superseded value.
        * **Double the round trips**, plus a copy of the prior vectors in memory.
          Use :meth:`upsertVectors` wherever a partial write is tolerable.

        Returns:
            The outcome of the write, which by then is known to have fully landed.

        Raises:
            PineconeTransactionError: If the write failed. The rollback's own
                result is carried on the exception.
        """
        if not records:
            return WriteOutcome(requestedCount=0, appliedCount=0)

        targetNamespace = self.resolveNamespace(namespace)
        ids = [record.id for record in records]
        priorRecords = await self.fetchVectors(ids, namespace=targetNamespace)

        try:
            outcome = await self.upsertVectors(
                records, namespace=targetNamespace, batchSize=batchSize
            )
        except Exception as upsertError:
            rollbackSucceeded, rollbackError = await self._rollbackUpsert(
                ids, priorRecords, namespace=targetNamespace
            )
            raise PineconeTransactionError(
                f"Atomic upsert of {len(records)} vectors failed: {upsertError}",
                outcome=WriteOutcome(requestedCount=len(records), appliedCount=0),
                rollbackSucceeded=rollbackSucceeded,
                rollbackError=rollbackError,
            ) from upsertError

        if outcome.succeeded:
            return outcome

        rollbackSucceeded, rollbackError = await self._rollbackUpsert(
            ids, priorRecords, namespace=targetNamespace
        )
        raise PineconeTransactionError(
            f"Atomic upsert applied only {outcome.appliedCount} of "
            f"{outcome.requestedCount} vectors.",
            outcome=outcome,
            rollbackSucceeded=rollbackSucceeded,
            rollbackError=rollbackError,
        )

    async def _rollbackUpsert(
        self,
        ids: Sequence[str],
        priorRecords: dict[str, VectorRecord],
        *,
        namespace: str,
    ) -> tuple[bool, BaseException | None]:
        """Put the ids back the way they were before a failed atomic upsert.

        Ids that held nothing before are deleted, ids that held something are
        rewritten with it. Returns whether the undo fully succeeded, and the
        error that stopped it when it did not: either the exception raised, or
        one describing the ids the compensating write left behind.
        """
        idsToDelete = [vectorId for vectorId in ids if vectorId not in priorRecords]
        recordsToRestore = [priorRecords[vectorId] for vectorId in ids if vectorId in priorRecords]

        try:
            if idsToDelete:
                deleteOutcome = await self.deleteVectors(idsToDelete, namespace=namespace)
                if not deleteOutcome.succeeded:
                    return False, incompleteRollbackError("delete", deleteOutcome)
            if recordsToRestore:
                restoreOutcome = await self.upsertVectors(recordsToRestore, namespace=namespace)
                if not restoreOutcome.succeeded:
                    return False, incompleteRollbackError("restore", restoreOutcome)
        except Exception as rollbackError:
            return False, rollbackError

        return True, None

    async def fetchVectors(
        self,
        ids: Sequence[str],
        *,
        namespace: str | None = None,
    ) -> dict[str, VectorRecord]:
        """Read records by id, keyed by id.

        Ids the index does not hold are absent from the result rather than an
        error. More ids than one request may carry are split across several,
        which go out concurrently.

        Because reads are eventually consistent, an id written moments ago can be
        missing here even though the write succeeded.
        """
        if not ids:
            return {}

        index = await self.getIndex()
        targetNamespace = self.resolveNamespace(namespace)
        responses = await asyncio.gather(
            *(
                index.fetch(ids=chunk, namespace=targetNamespace)
                for chunk in chunkItems(ids, MAX_IDS_PER_FETCH)
            )
        )

        records: dict[str, VectorRecord] = {}
        for response in responses:
            for vectorId, vector in (response.vectors or {}).items():
                records[vectorId] = toVectorRecord(vector)
        return records

    async def fetchVector(
        self,
        vectorId: str,
        *,
        namespace: str | None = None,
    ) -> VectorRecord | None:
        """Read a single record, or None when the index does not hold that id."""
        records = await self.fetchVectors([vectorId], namespace=namespace)
        return records.get(vectorId)

    async def queryVectors(
        self,
        vector: Sequence[float],
        *,
        topK: int = DEFAULT_TOP_K,
        metadataFilter: dict[str, Any] | None = None,
        includeValues: bool = False,
        includeMetadata: bool = True,
        namespace: str | None = None,
    ) -> list[QueryMatch]:
        """Return the `topK` nearest neighbours of `vector`, closest first.

        `metadataFilter` takes Pinecone's filter syntax, e.g.
        `{"source": {"$eq": "handbook"}}`. Leave `includeValues` off unless the
        embeddings themselves are needed; they dominate the response size.
        """
        index = await self.getIndex()
        response = await index.query(
            vector=list(vector),
            top_k=topK,
            namespace=self.resolveNamespace(namespace),
            filter=metadataFilter,
            include_values=includeValues,
            include_metadata=includeMetadata,
        )
        return [toQueryMatch(match) for match in response.matches or []]

    async def upsertTextRecords(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        namespace: str | None = None,
        batchSize: int = DEFAULT_RECORDS_BATCH_SIZE,
    ) -> int:
        """Write records to an *integrated embedding* index, and return how many.

        On an index whose schema declares a semantic text field, Pinecone does
        the embedding: hand it the text and it produces the vector. That is the
        only write path such an index has - :meth:`upsertVectors` sends raw
        vectors and bypasses the semantic field, leaving records that no text
        search will ever match.

        Each record is a dict carrying `_id` (or `id`) plus the fields to store,
        one of which must be the field the index embeds:

            await adapter.upsertTextRecords(
                [{"_id": "doc-1::0", "chunkText": "...", "chunkIndex": 0}],
                namespace=accountNamespace,
            )

        Unlike :meth:`upsertVectors` there is no default namespace to fall back
        on, so one has to be configured on the adapter or passed here.

        Raises:
            PineconeAdapterError: If no namespace resolves, or a record has no id.
        """
        if not records:
            return 0

        targetNamespace = self.resolveNamespace(namespace)
        if not targetNamespace:
            raise PineconeAdapterError(
                "upsertTextRecords needs a namespace: the records API has no default "
                "one. Pass namespace=, or set it on the adapter."
            )

        for position, record in enumerate(records):
            if not (record.get("_id") or record.get("id")):
                raise PineconeAdapterError(
                    f"Record at position {position} has no '_id'; every record needs one."
                )

        index = await self.getIndex()
        writtenCount = 0
        for chunk in chunkItems(list(records), batchSize):
            response = await index.upsert_records(records=chunk, namespace=targetNamespace)
            # record_count reports what was submitted; the call raises rather
            # than partly succeeding, so reaching here means the chunk landed.
            writtenCount += getattr(response, "record_count", len(chunk))
        return writtenCount

    async def searchByText(
        self,
        text: str,
        *,
        topK: int = DEFAULT_TOP_K,
        namespace: str | None = None,
        metadataFilter: dict[str, Any] | None = None,
        fields: Sequence[str] | None = None,
        rerankModel: str | None = None,
        rerankField: str | None = None,
    ) -> list[QueryMatch]:
        """Search an *integrated embedding* index by text, closest first.

        The counterpart to :meth:`upsertTextRecords`: Pinecone embeds the query
        with the same model it embedded the records with, so no embedding call is
        needed on this side. :meth:`queryVectors` remains the right method when
        you already hold a vector of your own.

        `fields` names the stored fields to return, which is worth setting: the
        default returns every field, and on a chunked corpus that means the full
        text of every hit. Matched fields arrive on `QueryMatch.metadata`.

        `rerankModel` runs a second, more accurate pass over the first results
        (`rerankField` says which field it reads, defaulting to the one searched).
        It costs an extra round trip server side and is worth it when precision
        at the top of the list matters more than latency.

        Raises:
            PineconeAdapterError: If no namespace resolves.
        """
        targetNamespace = self.resolveNamespace(namespace)
        if not targetNamespace:
            raise PineconeAdapterError(
                "searchByText needs a namespace: the records API has no default one. "
                "Pass namespace=, or set it on the adapter."
            )

        searchOptions: dict[str, Any] = {
            "namespace": targetNamespace,
            "top_k": topK,
            "inputs": {"text": text},
        }
        if metadataFilter is not None:
            searchOptions["filter"] = metadataFilter
        if fields is not None:
            searchOptions["fields"] = list(fields)
        if rerankModel:
            rerank: dict[str, Any] = {"model": rerankModel, "top_n": topK}
            if rerankField:
                rerank["rank_fields"] = [rerankField]
            searchOptions["rerank"] = rerank

        index = await self.getIndex()
        response = await index.search(**searchOptions)
        return [toTextMatch(hit) for hit in response.result.hits or []]

    async def updateVector(
        self,
        vectorId: str,
        *,
        values: Sequence[float] | None = None,
        metadata: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> None:
        """Patch one record in place, leaving the fields not named alone.

        `metadata` merges into whatever metadata the record already carries; it
        does not replace it, and it cannot remove a key. Use
        :meth:`upsertVectors` to write a record wholesale.

        Raises:
            PineconeAdapterError: If neither `values` nor `metadata` is given.
        """
        if values is None and metadata is None:
            raise PineconeAdapterError("updateVector needs values, metadata, or both.")

        index = await self.getIndex()
        await index.update(
            id=vectorId,
            values=list(values) if values is not None else None,
            set_metadata=metadata,
            namespace=self.resolveNamespace(namespace),
        )

    async def deleteVectors(
        self,
        ids: Sequence[str],
        *,
        namespace: str | None = None,
    ) -> WriteOutcome:
        """Remove records by id.

        Ids the index does not hold are not an error, and neither is a
        namespace that does not exist. Long id lists are split
        across concurrent requests, and one failing request does not cancel the
        rest, so the result can describe a partial delete. Deletes are
        idempotent, so resending failed ids is safe.
        """
        if not ids:
            return WriteOutcome(requestedCount=0, appliedCount=0)

        index = await self.getIndex()
        targetNamespace = self.resolveNamespace(namespace)
        chunks = list(chunkItems(ids, MAX_IDS_PER_DELETE))
        results = await asyncio.gather(
            *(index.delete(ids=chunk, namespace=targetNamespace) for chunk in chunks),
            return_exceptions=True,
        )

        appliedCount = 0
        failures: list[WriteFailure] = []
        for chunk, result in zip(chunks, results, strict=True):
            # A namespace nothing was ever written to answers 404; it holds
            # none of the ids, which is what the delete was for.
            if isinstance(result, NotFoundError):
                appliedCount += len(chunk)
            elif isinstance(result, BaseException):
                failures.append(
                    WriteFailure(
                        ids=chunk,
                        message=str(result),
                        # The request did reach the server, so the delete may have
                        # landed anyway; resending is harmless either way.
                        retryable=True,
                        disposition="rejected",
                    )
                )
            else:
                appliedCount += len(chunk)

        return WriteOutcome(requestedCount=len(ids), appliedCount=appliedCount, failures=failures)

    async def deleteByFilter(
        self,
        metadataFilter: dict[str, Any],
        *,
        namespace: str | None = None,
    ) -> None:
        """Delete every record whose metadata matches `metadataFilter`.

        Serverless indexes do not support deleting by metadata filter and reject
        this; there, list the ids first (:meth:`listVectorIds` with a prefix) and
        delete those instead.
        """
        index = await self.getIndex()
        await index.delete(filter=metadataFilter, namespace=self.resolveNamespace(namespace))

    async def deleteNamespace(self, *, namespace: str | None = None) -> None:
        """Delete every record in a namespace.

        Irreversible, and outside the reach of
        :meth:`upsertVectorsAtomically`'s rollback.
        """
        index = await self.getIndex()
        await index.delete(delete_all=True, namespace=self.resolveNamespace(namespace))

    async def listVectorIds(
        self,
        *,
        prefix: str | None = None,
        limit: int | None = None,
        namespace: str | None = None,
    ) -> list[str]:
        """Collect record ids, optionally only those starting with `prefix`.

        `limit` caps how many ids come back in total, not how many per page: the
        pages are walked for you and stop early once the cap is reached. Without
        one, a large namespace is slow to walk and every id is held in memory, so
        prefer a limit, or drive `index.list(...)` yourself through
        :meth:`getIndex` to stream the pages instead.
        """
        index = await self.getIndex()
        pageSize = min(limit, MAX_LIST_PAGE_SIZE) if limit is not None else None

        vectorIds: list[str] = []
        async for page in index.list(
            prefix=prefix, limit=pageSize, namespace=self.resolveNamespace(namespace)
        ):
            vectorIds.extend(item.id for item in page.vectors or [])
            if limit is not None and len(vectorIds) >= limit:
                return vectorIds[:limit]
        return vectorIds

    async def describeStats(
        self, *, metadataFilter: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Index vitals: dimension, fullness and the vector count per namespace."""
        index = await self.getIndex()
        response = await index.describe_index_stats(filter=metadataFilter)
        return response.to_dict()
