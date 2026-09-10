"""Redis implementation of :class:`~DbAdapters.documentBase.DocumentDbAdapter`.

Should Redis be an adapter at all, or just direct calls?
--------------------------------------------------------
Fair question, and the answer is: an adapter, but a thinner one than the others.
Redis's core operations really are fetch, write and delete by key, so it fits the
contract honestly and lets a caller swap it for MongoDB behind the same four
methods - which is the whole point of the package. What it does not have is a
query engine, so `findDocuments` is deliberately missing here; where MongoDB and
Firestore query, Redis offers :meth:`RedisAdapter.scanDocumentIds` over key
patterns, and that difference is real rather than something to paper over.

The parts of Redis that have no equivalent elsewhere - expiry, counters,
pipelining, pub/sub, streams - are better reached directly, and
:meth:`RedisAdapter.getClient` hands over the client for exactly that. Expiry is
the one exception, wired into the write methods here, because a cache entry
without a TTL is usually a bug and making callers remember a second call would
invite it.

What "ACID" means here
----------------------
Less than the name suggests, and the gap matters:

* **Atomicity** - in execution, not in rollback. `MULTI`/`EXEC` runs the queued
  commands as one uninterrupted block, but if a command fails partway the ones
  before it stand: Redis has nothing to undo them with. A command that is
  malformed is caught at queue time and aborts the block; one that is well formed
  but wrong at runtime, such as an operation against the wrong type, is not.
* **Consistency** - none to speak of. No schema, no constraints, no foreign keys.
* **Isolation** - complete while a block runs, since Redis executes commands on a
  single thread and nothing interleaves. Read-modify-write across two round trips
  is a different matter, and needs the `WATCH` loop
  :meth:`RedisAdapter.applyAtomically` runs for merges.
* **Durability** - configuration dependent, and weak by default. With the usual
  `appendfsync everysec` a crash can lose the last second of writes; with RDB
  snapshots alone it can lose far more. Treat Redis as a cache that survives
  restarts if you are lucky, not as a record of truth.

The short version: Redis is the right home for cached retrievals, dedup markers,
rate limits and session state. It is the wrong home for anything you would be
unhappy to find missing.
"""

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import WatchError

from DbAdapters.base import WriteFailure, WriteOutcome, chunkItems
from DbAdapters.documentBase import (
    WRITE_DELETE,
    WRITE_MERGE,
    Document,
    DocumentDbAdapter,
    WriteOperation,
)

CONNECTION_URL_ENV_VAR = "REDIS_URL"
KEY_PREFIX_ENV_VAR = "REDIS_KEY_PREFIX"

DEFAULT_CONNECTION_URL = "redis://localhost:6379/0"
DEFAULT_TIMEOUT_SECONDS = 10.0

# Separates the collection from the document id inside a key, by Redis convention.
KEY_SEPARATOR = ":"

# How many keys one SCAN pass asks the server for. A hint, not a guarantee:
# Redis returns roughly this many and SCAN is O(1) per call either way.
SCAN_BATCH_SIZE = 500

# How often an atomic block is replayed when WATCH says a key changed underneath.
# Contention that survives this many attempts is a design problem, not a blip.
MAX_WATCH_ATTEMPTS = 5


class RedisAdapterError(RuntimeError):
    """The adapter itself is misconfigured or was driven incorrectly.

    Redis errors (`ConnectionError`, `ResponseError` and the rest of
    `RedisError`) propagate unwrapped so callers can react to them by type.
    """


class RedisContentionError(RedisAdapterError):
    """An atomic block kept losing its `WATCH` race and was given up on.

    Every attempt found one of the documents it had read changed by another
    client before the block could commit. Nothing was written. Retrying may work,
    but repeated contention on the same keys usually means two writers own state
    that only one of them should.
    """


def encodeDocument(data: Mapping[str, Any]) -> str:
    """Serialise document fields for storage.

    JSON rather than pickle: the stored value stays readable from `redis-cli` and
    from any other language touching the same keys, and nothing unpickles data it
    did not write.
    """
    return json.dumps(data, separators=(",", ":"), default=str)


def decodeDocument(raw: str | bytes | None) -> dict[str, Any] | None:
    """Deserialise a stored value, or None when the key held nothing.

    A value that is not the JSON object this adapter writes - a plain string set
    by other code sharing the keyspace, say - is reported rather than guessed at.
    """
    if raw is None:
        return None

    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as error:
        raise RedisAdapterError(
            f"Stored value is not JSON this adapter wrote: {error}"
        ) from error

    if not isinstance(decoded, dict):
        raise RedisAdapterError(
            f"Stored value decoded to {type(decoded).__name__}, expected a JSON object."
        )
    return decoded


class RedisAdapter(DocumentDbAdapter):
    """Async reads and writes against one Redis database.

    Keys are laid out as `{prefix}{collection}:{documentId}`, so one Redis
    instance can hold several collections, and several applications can share it
    without colliding as long as their prefixes differ.

    The client owns a connection pool, so build one adapter and share it.

    Example:
        async with RedisAdapter(keyPrefix="rag:") as cache:
            await cache.upsertDocument(
                "answers", Document(id="q-1", data={"text": "..."}), ttlSeconds=3600
            )
            answer = await cache.fetchDocument("answers", "q-1")
    """

    def __init__(
        self,
        *,
        connectionUrl: str | None = None,
        keyPrefix: str | None = None,
        defaultTtlSeconds: int | None = None,
        timeoutSeconds: float = DEFAULT_TIMEOUT_SECONDS,
        clientOptions: Mapping[str, Any] | None = None,
    ) -> None:
        """Record the connection settings without opening anything yet.

        Args:
            connectionUrl: Redis URL, defaulting to `$REDIS_URL` and then to a
                local server. `rediss://` for TLS.
            keyPrefix: Prefix on every key this adapter touches, defaulting to
                `$REDIS_KEY_PREFIX`. Worth setting on a shared instance, since
                it is what keeps :meth:`scanDocumentIds` and
                :meth:`deleteCollection` from reaching another application's keys.
            defaultTtlSeconds: Expiry applied to writes that do not name one.
                None means keys persist until deleted, which for a cache is
                usually the wrong default.
            timeoutSeconds: Socket timeout for every call.
            clientOptions: Extra keyword arguments passed to `Redis.from_url`,
                for TLS settings, pool sizing and the like.
        """
        self._connectionUrl = (
            connectionUrl or os.environ.get(CONNECTION_URL_ENV_VAR) or DEFAULT_CONNECTION_URL
        )
        self._keyPrefix = keyPrefix if keyPrefix is not None else os.environ.get(
            KEY_PREFIX_ENV_VAR, ""
        )
        self._defaultTtlSeconds = defaultTtlSeconds
        self._timeoutSeconds = timeoutSeconds
        self._clientOptions = dict(clientOptions or {})

        self._client: Redis | None = None
        # Guards the lazy connect so concurrent first calls open one pool
        # between them rather than one each.
        self._connectLock = asyncio.Lock()

    @property
    def keyPrefix(self) -> str:
        """Prefix on every key this adapter reads or writes."""
        return self._keyPrefix

    @property
    def isConnected(self) -> bool:
        """Whether the client is currently open."""
        return self._client is not None

    def buildKey(self, collection: str, documentId: str) -> str:
        """The Redis key one document lives under."""
        return f"{self._keyPrefix}{collection}{KEY_SEPARATOR}{documentId}"

    def documentIdOfKey(self, collection: str, key: str | bytes) -> str:
        """Recover a document id from a key :meth:`buildKey` produced."""
        text = key.decode("utf-8") if isinstance(key, bytes) else key
        return text[len(self.buildKey(collection, "")) :]

    async def connect(self) -> None:
        """Open the client and its pool, once, however often it is called.

        Redis connects lazily, so this returns before any socket exists; the
        first real operation is what surfaces an unreachable server. Call
        :meth:`ping` at startup to find out sooner.
        """
        if self._client is not None:
            return

        async with self._connectLock:
            if self._client is not None:
                return
            self._client = Redis.from_url(
                self._connectionUrl,
                socket_timeout=self._timeoutSeconds,
                socket_connect_timeout=self._timeoutSeconds,
                # Decode here so every value this module handles is `str`, and
                # decodeDocument never has to guess at an encoding.
                decode_responses=True,
                **self._clientOptions,
            )

    async def close(self) -> None:
        """Close the client and its pool, once, however often it is called."""
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def getClient(self) -> Redis:
        """The open client, connecting first if need be.

        Exposed for everything Redis does that a document contract cannot
        express: counters, sorted sets, streams, pub/sub, Lua scripts.
        """
        await self.connect()
        if self._client is None:  # pragma: no cover - connect() raises instead
            raise RedisAdapterError("Redis client is not available.")
        return self._client

    async def ping(self) -> bool:
        """Round trip to the server, to prove the connection really works."""
        client = await self.getClient()
        return bool(await client.ping())

    async def fetchDocument(self, collection: str, documentId: str) -> Document | None:
        """Read one document, or None when the key is absent or has expired."""
        client = await self.getClient()
        raw = await client.get(self.buildKey(collection, documentId))
        data = decodeDocument(raw)
        return Document(id=documentId, data=data) if data is not None else None

    async def fetchDocuments(
        self, collection: str, documentIds: Sequence[str]
    ) -> dict[str, Document]:
        """Read documents by id, keyed by id, in one `MGET`.

        Missing and expired ids are absent from the result rather than an error.
        """
        if not documentIds:
            return {}

        client = await self.getClient()
        keys = [self.buildKey(collection, documentId) for documentId in documentIds]
        values = await client.mget(keys)

        documents: dict[str, Document] = {}
        for documentId, raw in zip(documentIds, values, strict=True):
            data = decodeDocument(raw)
            if data is not None:
                documents[documentId] = Document(id=documentId, data=data)
        return documents

    async def upsertDocument(
        self,
        collection: str,
        document: Document,
        *,
        merge: bool = False,
        ttlSeconds: int | None = None,
    ) -> None:
        """Write one document, replacing what is there or merging into it.

        A plain write is a single `SET` and therefore atomic. A merge is not one
        operation but three - read, combine, write - so it goes through the same
        `WATCH` loop as :meth:`applyAtomically` and retries if another client
        changes the key in between.

        `ttlSeconds` overrides the adapter's default expiry; pass 0 to store the
        document with no expiry at all when a default is set.
        """
        if merge:
            await self.applyAtomically(
                [
                    WriteOperation(
                        collection=collection,
                        documentId=document.id,
                        kind=WRITE_MERGE,
                        data=document.data,
                    )
                ],
                ttlSeconds=ttlSeconds,
            )
            return

        client = await self.getClient()
        await client.set(
            self.buildKey(collection, document.id),
            encodeDocument(document.data),
            ex=self._resolveTtl(ttlSeconds),
        )

    async def upsertDocuments(
        self,
        collection: str,
        documents: Sequence[Document],
        *,
        merge: bool = False,
        ttlSeconds: int | None = None,
    ) -> WriteOutcome:
        """Write many documents in one pipeline.

        The commands travel as one batch, which saves the round trips, but they
        are not sent as an atomic block: this method reports a partial write
        rather than promising all-or-nothing. Use :meth:`applyAtomically` for
        that.
        """
        if not documents:
            return WriteOutcome(requestedCount=0, appliedCount=0)

        if merge:
            return await self._applyReportingFailures(
                [
                    WriteOperation(
                        collection=collection,
                        documentId=document.id,
                        kind=WRITE_MERGE,
                        data=document.data,
                    )
                    for document in documents
                ],
                ttlSeconds=ttlSeconds,
            )

        client = await self.getClient()
        expiry = self._resolveTtl(ttlSeconds)

        # transaction=False: this is a batch for the round trips, not a block that
        # has to land together, and MULTI would only add a promise not being made.
        pipeline = client.pipeline(transaction=False)
        for document in documents:
            pipeline.set(
                self.buildKey(collection, document.id),
                encodeDocument(document.data),
                ex=expiry,
            )
        results = await pipeline.execute(raise_on_error=False)

        appliedCount = 0
        failures: list[WriteFailure] = []
        for document, result in zip(documents, results, strict=True):
            if isinstance(result, Exception):
                failures.append(
                    WriteFailure(
                        ids=[document.id],
                        message=str(result),
                        retryable=True,
                        disposition="rejected",
                    )
                )
            else:
                appliedCount += 1

        return WriteOutcome(
            requestedCount=len(documents), appliedCount=appliedCount, failures=failures
        )

    async def deleteDocument(self, collection: str, documentId: str) -> None:
        """Remove one document. A key that is not there is not an error."""
        client = await self.getClient()
        await client.delete(self.buildKey(collection, documentId))

    async def deleteDocuments(
        self, collection: str, documentIds: Sequence[str]
    ) -> WriteOutcome:
        """Remove many documents in one `DEL`.

        Ids that are not there are not an error, so the outcome counts every
        requested id as applied: the state asked for - absent - holds for all of
        them either way.
        """
        if not documentIds:
            return WriteOutcome(requestedCount=0, appliedCount=0)

        client = await self.getClient()
        keys = [self.buildKey(collection, documentId) for documentId in documentIds]
        await client.delete(*keys)
        return WriteOutcome(requestedCount=len(documentIds), appliedCount=len(documentIds))

    async def applyAtomically(
        self, operations: Sequence[WriteOperation], *, ttlSeconds: int | None = None
    ) -> None:
        """Apply every operation in one `MULTI`/`EXEC` block, or none of them.

        Read the module docstring before relying on this. In short: the block
        executes with nothing interleaved, and a merge reads its documents under
        `WATCH` so the block is abandoned and replayed if another client changes
        one first. What Redis cannot do is roll back - if a command fails during
        execution, the ones already run stand. Malformed commands are caught at
        queue time and do abort the block cleanly.

        Every key in the block must live on the same node, which is automatic on
        a single instance but needs hash tags in the prefix on a cluster.

        Raises:
            RedisContentionError: If `MAX_WATCH_ATTEMPTS` attempts all lost the
                race. Nothing was written.
        """
        if not operations:
            return

        client = await self.getClient()
        expiry = self._resolveTtl(ttlSeconds)
        mergeOperations = [
            operation for operation in operations if operation.kind == WRITE_MERGE
        ]
        watchedKeys = [
            self.buildKey(operation.collection, operation.documentId)
            for operation in mergeOperations
        ]

        for attempt in range(MAX_WATCH_ATTEMPTS):
            async with client.pipeline(transaction=True) as pipeline:
                try:
                    existing: dict[str, dict[str, Any]] = {}
                    if watchedKeys:
                        # WATCH first, then read: guarding a key after reading it
                        # leaves exactly the window this is meant to close.
                        await pipeline.watch(*watchedKeys)
                        for key, raw in zip(
                            watchedKeys, await pipeline.mget(watchedKeys), strict=True
                        ):
                            existing[key] = decodeDocument(raw) or {}

                    pipeline.multi()
                    for operation in operations:
                        key = self.buildKey(operation.collection, operation.documentId)
                        if operation.kind == WRITE_DELETE:
                            pipeline.delete(key)
                            continue
                        data = dict(operation.data or {})
                        if operation.kind == WRITE_MERGE:
                            data = {**existing.get(key, {}), **data}
                        pipeline.set(key, encodeDocument(data), ex=expiry)

                    await pipeline.execute()
                    return
                except WatchError:
                    # Someone wrote one of the watched keys between the read and
                    # the commit, so the merge was computed from stale data.
                    # Nothing was applied; read again and redo it.
                    continue

        raise RedisContentionError(
            f"Gave up after {MAX_WATCH_ATTEMPTS} attempts: {len(watchedKeys)} watched "
            f"key(s) kept changing underneath the block. Nothing was written."
        )

    async def scanDocumentIds(
        self, collection: str, *, pattern: str = "*", limit: int | None = None
    ) -> list[str]:
        """Collect document ids in a collection, optionally matching `pattern`.

        Redis has no query engine, so this is a `SCAN` over the keyspace: it
        matches on the id, never on the stored fields, and glob syntax is all it
        understands. `SCAN` is incremental and does not block the server the way
        `KEYS` would, but it still walks everything, so pass a `limit` and treat
        an unbounded scan over a large keyspace as a red flag.

        A cursor walk can return the same key twice if the keyspace changes
        underneath, so the result is de-duplicated. Keys that expire mid scan are
        simply absent.
        """
        client = await self.getClient()
        matchPattern = f"{self.buildKey(collection, '')}{pattern}"

        seen: set[str] = set()
        documentIds: list[str] = []
        async for key in client.scan_iter(match=matchPattern, count=SCAN_BATCH_SIZE):
            documentId = self.documentIdOfKey(collection, key)
            if documentId in seen:
                continue
            seen.add(documentId)
            documentIds.append(documentId)
            if limit is not None and len(documentIds) >= limit:
                break
        return documentIds

    async def deleteCollection(self, collection: str) -> int:
        """Remove every document in a collection, returning how many went.

        Scans and deletes in batches rather than in one command, so it does not
        block the server on a large keyspace. Not atomic: writes landing during
        the walk may survive it.
        """
        client = await self.getClient()
        documentIds = await self.scanDocumentIds(collection)

        deletedCount = 0
        for chunk in chunkItems(documentIds, SCAN_BATCH_SIZE):
            keys = [self.buildKey(collection, documentId) for documentId in chunk]
            deletedCount += await client.delete(*keys)
        return deletedCount

    async def getTimeToLive(self, collection: str, documentId: str) -> int | None:
        """Seconds until a document expires, or None when it has no expiry.

        Returns None for a key that does not exist either: from a caller's side
        both mean "no expiry to wait for".
        """
        client = await self.getClient()
        remaining = await client.ttl(self.buildKey(collection, documentId))
        # -1 is "exists, no expiry"; -2 is "no such key". Neither is a countdown.
        return remaining if remaining >= 0 else None

    async def setTimeToLive(self, collection: str, documentId: str, ttlSeconds: int) -> bool:
        """Give an existing document an expiry. False when the key is not there."""
        client = await self.getClient()
        return bool(await client.expire(self.buildKey(collection, documentId), ttlSeconds))

    def _resolveTtl(self, ttlSeconds: int | None) -> int | None:
        """The expiry a write should use: its own, else the adapter's default.

        An explicit 0 means "no expiry", which is how a caller overrides a
        default TTL for one write; `None` means "whatever the adapter says".
        """
        if ttlSeconds is None:
            return self._defaultTtlSeconds
        return ttlSeconds or None

    async def _applyReportingFailures(
        self, operations: Sequence[WriteOperation], *, ttlSeconds: int | None
    ) -> WriteOutcome:
        """Run each operation atomically on its own, reporting the ones that fail.

        Used for a bulk merge, where the caller asked for a batch rather than a
        block: applying them one at a time keeps each read-modify-write correct
        without promising the batch lands whole.
        """
        appliedCount = 0
        failures: list[WriteFailure] = []

        for operation in operations:
            try:
                await self.applyAtomically([operation], ttlSeconds=ttlSeconds)
            except Exception as error:
                failures.append(
                    WriteFailure(
                        ids=[operation.documentId],
                        message=str(error),
                        retryable=isinstance(error, RedisContentionError),
                        disposition="rejected",
                    )
                )
            else:
                appliedCount += 1

        return WriteOutcome(
            requestedCount=len(operations), appliedCount=appliedCount, failures=failures
        )
