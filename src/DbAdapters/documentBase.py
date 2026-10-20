"""Contract for stores that hold keyed documents: MongoDB, Firestore, Redis.

The shared surface is deliberately narrow - fetch, upsert, delete, and one
all-or-nothing write - because that is the honest intersection of three stores
that differ enormously past it. Querying is not in here: MongoDB and Firestore
have real query engines and expose one, Redis has key patterns and exposes those
instead. Reaching for a store's own strengths through its own module is better
than pretending a lowest common denominator covers them.

How much of ACID a caller actually gets varies by store, and each adapter says
so in its own module docstring. :meth:`DocumentDbAdapter.applyAtomically` is the
one place all three make an all-or-nothing promise, and even there the promise
differs: MongoDB runs a genuine transaction, Firestore commits an atomic batch,
and Redis executes a queued block that no other client can interleave with but
that cannot roll back.
"""

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from DbAdapters.base import DbAdapter, WriteOutcome

# The kinds of write `applyAtomically` and the bulk writers understand.
WRITE_SET = "set"
WRITE_MERGE = "merge"
WRITE_DELETE = "delete"

WRITE_KINDS = frozenset({WRITE_SET, WRITE_MERGE, WRITE_DELETE})


@dataclass(slots=True)
class Document:
    """One stored document: its key within a collection, and its fields."""

    id: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WriteOperation:
    """One write inside an all-or-nothing block.

    Attributes:
        collection: Collection, table or key namespace the document lives in.
        documentId: Key of the document within that collection.
        kind: `WRITE_SET` to write the document whole, `WRITE_MERGE` to fold
            `data` into the fields already there, `WRITE_DELETE` to remove it.
        data: Fields to write. Required except for a delete, which ignores it.
    """

    collection: str
    documentId: str
    kind: str
    data: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Reject a malformed operation here rather than mid-transaction."""
        if self.kind not in WRITE_KINDS:
            raise ValueError(f"Unknown write kind {self.kind!r}; expected one of {sorted(WRITE_KINDS)}.")
        if self.kind != WRITE_DELETE and self.data is None:
            raise ValueError(f"A {self.kind} of {self.documentId!r} needs data to write.")


def setOperation(collection: str, document: Document) -> WriteOperation:
    """A write that replaces `document` wholesale, creating it if absent."""
    return WriteOperation(
        collection=collection, documentId=document.id, kind=WRITE_SET, data=document.data
    )


def mergeOperation(collection: str, documentId: str, data: dict[str, Any]) -> WriteOperation:
    """A write that folds `data` into the fields already stored under `documentId`."""
    return WriteOperation(
        collection=collection, documentId=documentId, kind=WRITE_MERGE, data=data
    )


def deleteOperation(collection: str, documentId: str) -> WriteOperation:
    """A write that removes `documentId`, whether or not it is there."""
    return WriteOperation(collection=collection, documentId=documentId, kind=WRITE_DELETE)


class DocumentDbAdapter(DbAdapter):
    """Async create/read/update/delete over keyed documents."""

    @abstractmethod
    async def fetchDocument(self, collection: str, documentId: str) -> Document | None:
        """Read one document, or None when the store does not hold that id."""

    @abstractmethod
    async def fetchDocuments(
        self, collection: str, documentIds: Sequence[str]
    ) -> dict[str, Document]:
        """Read documents by id, keyed by id. Missing ids are simply absent."""

    @abstractmethod
    async def upsertDocument(
        self, collection: str, document: Document, *, merge: bool = False
    ) -> None:
        """Write one document, replacing what is there or merging into it.

        Raises on failure rather than reporting one, since a single write either
        lands or does not.
        """

    @abstractmethod
    async def upsertDocuments(
        self, collection: str, documents: Sequence[Document], *, merge: bool = False
    ) -> WriteOutcome:
        """Write many documents, reporting a partial result rather than raising.

        Not atomic: use :meth:`applyAtomically` when the batch has to land whole.
        """

    @abstractmethod
    async def deleteDocument(self, collection: str, documentId: str) -> None:
        """Remove one document. An id that is not there is not an error."""

    @abstractmethod
    async def deleteDocuments(
        self, collection: str, documentIds: Sequence[str]
    ) -> WriteOutcome:
        """Remove many documents, reporting a partial result rather than raising."""

    @abstractmethod
    async def applyAtomically(self, operations: Sequence[WriteOperation]) -> None:
        """Apply every operation, or none of them.

        The strength of that promise is the store's, not this method's; read the
        implementing adapter's docstring before relying on it. Returns nothing
        because there is no partial result to report: it either committed or it
        raised.
        """
