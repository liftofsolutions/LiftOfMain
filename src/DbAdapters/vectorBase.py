"""Contract for stores that hold embeddings and search them by similarity.

A caller that depends only on :class:`VectorDbAdapter` can be pointed at a
different vector store without changing its own code. Anything specific to one
vendor belongs in that vendor's module.
"""

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from DbAdapters.base import DbAdapter, WriteOutcome


@dataclass(slots=True)
class VectorRecord:
    """One embedding plus the payload stored beside it."""

    id: str
    values: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryMatch:
    """One nearest neighbour returned by a similarity search."""

    id: str
    score: float
    values: list[float] | None = None
    metadata: dict[str, Any] | None = None


class VectorDbAdapter(DbAdapter):
    """Async create/read/update/delete over a single vector index."""

    @abstractmethod
    async def upsertVectors(
        self,
        records: Sequence[VectorRecord],
        *,
        namespace: str | None = None,
    ) -> WriteOutcome:
        """Insert the given records, overwriting any that already exist."""

    @abstractmethod
    async def fetchVectors(
        self,
        ids: Sequence[str],
        *,
        namespace: str | None = None,
    ) -> dict[str, VectorRecord]:
        """Read records by id. Ids that do not exist are absent from the result."""

    @abstractmethod
    async def queryVectors(
        self,
        vector: Sequence[float],
        *,
        topK: int = 10,
        metadataFilter: dict[str, Any] | None = None,
        includeValues: bool = False,
        includeMetadata: bool = True,
        namespace: str | None = None,
    ) -> list[QueryMatch]:
        """Return the nearest neighbours of `vector`, closest first."""

    @abstractmethod
    async def deleteVectors(
        self,
        ids: Sequence[str],
        *,
        namespace: str | None = None,
    ) -> WriteOutcome:
        """Remove records by id. Ids that do not exist are not an error."""
