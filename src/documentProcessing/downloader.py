"""Fetching a submitted document and unpacking it into files to process.

Nothing here is written to disk. A link is read into memory as a `LoadedFile`
- its name and its bytes - and an archive is unpacked into more `LoadedFile`s
the same way, since every parser in `documentFiles` reads from a stream and
none of them needs a path. A job therefore leaves nothing behind to clean up,
and two jobs can never collide over a temporary directory.

What that costs is memory: a document, and everything inside it once
unpacked, is held at once. `MAX_UNPACKED_BYTES` caps that, so an archive that
unpacks to far more than it weighs is refused rather than taking the process
down with it.

`.rar` needs an external tool that `rarfile` shells out to - WinRAR's
`UnRAR.exe` on Windows, `unrar` or `bsdtar` on Linux. `$RAG_UNRAR_TOOL` names
it when it is not on `PATH`; without that, rarfile searches `PATH` itself.
"""

import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

import httpx
import rarfile

from config import getServerConfig

# Formats whose metadata `documentFiles` knows how to collect.
DOCUMENT_FORMATS = frozenset({".pdf", ".doc", ".docx", ".txt", ".md"})
# Formats that hold other files rather than content of their own.
ARCHIVE_FORMATS = frozenset({".zip", ".rar"})
ACCEPTED_FORMATS = DOCUMENT_FORMATS | ARCHIVE_FORMATS

DOWNLOAD_TIMEOUT_SECONDS = 120.0
DOWNLOAD_CHUNK_BYTES = 1024 * 1024

REMOTE_SCHEMES = frozenset({"http", "https"})

# Everything unpacked from one archive has to fit in memory at once.
MAX_UNPACKED_BYTES = 512 * 1024 * 1024

# Used when a link carries no usable file name of its own.
FALLBACK_FILE_NAME = "document"


class UnsupportedDocumentError(ValueError):
    """The link points at a format this service does not accept."""


class ArchiveTooLargeError(ValueError):
    """The archive unpacks to more than this service will hold in memory."""


@dataclass(slots=True)
class LoadedFile:
    """One file held in memory: what it is called, and what is in it.

    `fileName` keeps the path a file had inside its archive, so a document
    nested in folders is still identifiable once unpacked.
    """

    fileName: str
    data: bytes

    @property
    def sizeBytes(self) -> int:
        """How many bytes the file holds."""
        return len(self.data)

    @property
    def fileFormat(self) -> str:
        """The lowercased extension, e.g. `.pdf`."""
        return PurePosixPath(self.fileName).suffix.lower()

    def openStream(self) -> BytesIO:
        """A fresh stream over the bytes, for a parser that wants a file."""
        return BytesIO(self.data)


def formatOf(name: str) -> str:
    """The lowercased extension of `name`, e.g. `.pdf`."""
    return PurePosixPath(name).suffix.lower()


def isArchive(fileName: str) -> bool:
    """Whether `fileName` is an archive holding other files."""
    return formatOf(fileName) in ARCHIVE_FORMATS


def fileNameOfLink(documentLink: str) -> str:
    """The file name a link ends in, with any percent-encoding undone."""
    parsed = urlparse(documentLink)
    if parsed.scheme in REMOTE_SCHEMES:
        return PurePosixPath(unquote(parsed.path)).name
    return Path(unquote(documentLink)).name


def checkAccepted(fileName: str) -> str:
    """The accepted format of `fileName`.

    Raises:
        UnsupportedDocumentError: If the extension is not one we accept.
    """
    fileFormat = formatOf(fileName)
    if fileFormat not in ACCEPTED_FORMATS:
        accepted = ", ".join(sorted(ACCEPTED_FORMATS))
        raise UnsupportedDocumentError(
            f"{fileName!r} is a {fileFormat or 'nameless'} file; this service "
            f"accepts {accepted}."
        )
    return fileFormat


async def downloadDocument(documentLink: str) -> LoadedFile:
    """Read the document `documentLink` names into memory.

    A local path is read rather than fetched, so a document already on this
    machine can be processed without a server to serve it.

    Raises:
        UnsupportedDocumentError: If the link is not an accepted format.
        httpx.HTTPError: If the download itself fails.
    """
    fileName = fileNameOfLink(documentLink) or FALLBACK_FILE_NAME
    checkAccepted(fileName)

    if urlparse(documentLink).scheme in REMOTE_SCHEMES:
        return LoadedFile(fileName, await fetchRemote(documentLink))

    sourcePath = Path(documentLink)
    if not sourcePath.is_file():
        raise UnsupportedDocumentError(
            f"{documentLink!r} is neither an http(s) link nor a file on disk."
        )
    return LoadedFile(fileName, sourcePath.read_bytes())


async def fetchRemote(documentLink: str) -> bytes:
    """Stream `documentLink` into memory, a chunk at a time.

    Streamed rather than read whole by the client, so the bytes arrive in
    pieces this can append instead of the client buffering its own copy
    alongside ours.
    """
    timeout = httpx.Timeout(DOWNLOAD_TIMEOUT_SECONDS)
    buffer = BytesIO()
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True
    ) as client:
        async with client.stream("GET", documentLink) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(DOWNLOAD_CHUNK_BYTES):
                buffer.write(chunk)
    return buffer.getvalue()


def expandDocument(loadedFile: LoadedFile) -> list[LoadedFile]:
    """The files to process for one downloaded document.

    A single document is the one file to process; an archive is unpacked and
    every file inside it is returned, however deeply nested. Blocking work -
    this is meant to be called through `asyncio.to_thread`.
    """
    if not isArchive(loadedFile.fileName):
        return [loadedFile]
    return expandArchive(loadedFile)


def expandArchive(loadedFile: LoadedFile) -> list[LoadedFile]:
    """Read every file out of an archive, in name order.

    Directory entries are skipped - they hold nothing to collect metadata
    from, and their contents arrive under their own names anyway.

    Raises:
        ArchiveTooLargeError: If the archive unpacks to more than
            `MAX_UNPACKED_BYTES`.
    """
    if loadedFile.fileFormat == ".zip":
        with zipfile.ZipFile(loadedFile.openStream()) as archive:
            return readMembers(archive, loadedFile.fileName)

    unrarTool = getServerConfig().unrarTool
    if unrarTool:
        rarfile.UNRAR_TOOL = unrarTool
    with rarfile.RarFile(loadedFile.openStream()) as archive:
        return readMembers(archive, loadedFile.fileName)


def readMembers(
    archive: zipfile.ZipFile | rarfile.RarFile, archiveName: str
) -> list[LoadedFile]:
    """Every file in an open archive, read into memory in name order.

    Raises:
        ArchiveTooLargeError: If the members total more than
            `MAX_UNPACKED_BYTES`. Checked against the sizes the archive
            declares, before any of it is read, so an archive that unpacks to
            far more than it weighs is refused rather than consuming memory
            until it is too late.
    """
    members = [member for member in archive.infolist() if not member.is_dir()]

    unpackedBytes = sum(member.file_size for member in members)
    if unpackedBytes > MAX_UNPACKED_BYTES:
        raise ArchiveTooLargeError(
            f"{archiveName!r} unpacks to {unpackedBytes} bytes, over the "
            f"{MAX_UNPACKED_BYTES} byte limit."
        )

    return [
        LoadedFile(member.filename, archive.read(member))
        for member in sorted(members, key=lambda member: member.filename)
    ]
