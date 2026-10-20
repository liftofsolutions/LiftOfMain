"""What every adapter in this package has in common, whatever it talks to.

The store shaped contracts build on this: :mod:`DbAdapters.vectorBase` for
similarity search over embeddings, :mod:`DbAdapters.documentBase` for keyed
documents. The connection lifecycle and the way a partly applied write is
reported are the same either way, so they live here.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any


def chunkItems(items: Sequence[Any], chunkSize: int) -> Iterator[list[Any]]:
    """Split `items` into consecutive lists of at most `chunkSize` entries.

    Every store here caps how much one request may carry, so the adapters all
    need this; the caps differ, the splitting does not.
    """
    for start in range(0, len(items), chunkSize):
        yield list(items[start : start + chunkSize])


@dataclass(slots=True)
class WriteFailure:
    """One slice of a write that did not land, and whether resending can help."""

    ids: list[str]
    message: str
    retryable: bool
    disposition: str = ""


@dataclass(slots=True)
class WriteOutcome:
    """Result of a write the store applies piecewise rather than all at once.

    A store without transactions, or a write too large for one request, can
    accept part of the work and reject the rest. The counts and the failed ids
    are what let a caller choose between retrying, compensating and giving up.

    A write that is genuinely all-or-nothing does not need this: those methods
    either return nothing or raise.
    """

    requestedCount: int
    appliedCount: int
    failures: list[WriteFailure] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        """True only when every requested item landed."""
        return not self.failures and self.appliedCount >= self.requestedCount

    @property
    def failedIds(self) -> list[str]:
        """Ids from every failed slice, ready to be resubmitted."""
        return [itemId for failure in self.failures for itemId in failure.ids]

    @property
    def retryableIds(self) -> list[str]:
        """Failed ids the store said are worth sending again.

        A deterministic failure resent unchanged fails again, so a retry loop
        should walk these rather than :attr:`failedIds`.
        """
        return [
            itemId for failure in self.failures if failure.retryable for itemId in failure.ids
        ]


class DbAdapter(ABC):
    """Connection lifecycle shared by every adapter here.

    Implementations reach their store over a genuinely async transport. An
    `async def` wrapped around a blocking client holds the event loop for the
    whole round trip and gives the caller nothing, so the concurrency has to be
    real underneath, not just in the signature.

    An instance is meant to be built once and shared: opening a connection costs
    a handshake, and pooling only pays off if the pool outlives the request.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Open the underlying connection. Safe to call more than once."""

    @abstractmethod
    async def close(self) -> None:
        """Release the connection. Safe to call more than once."""

    async def __aenter__(self) -> "DbAdapter":
        """Connect on entry, so `async with` yields a ready adapter."""
        await self.connect()
        return self

    async def __aexit__(self, excType: object, excValue: object, traceback: object) -> None:
        """Close on exit, whether or not the block raised."""
        await self.close()
