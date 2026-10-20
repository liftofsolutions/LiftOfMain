"""Downloading a submitted document to this server's disk.

A document is streamed to a file one piece at a time instead of being held
in memory: memory is shared by every request the server is handling, a
document can be large, and nothing downstream needs it whole - readers
open a file by its path. So memory use stays at one piece however big the
document is, and disk, which is cheaper and larger, holds the rest.

Only the formats in ACCEPTED_FORMATS are taken. The link's extension is
checked before a byte is fetched, and the file is checked again once it is
down: a pdf, docx, zip or rar must start with its format's signature, a tar
must carry the POSIX `ustar` mark, and a txt or md must hold no NUL byte. A
file that is not what its link says is deleted. The checks read bytes only,
so they answer the same on Windows and Linux - `mimetypes` is not used, as
it reads the Windows registry on one and `/etc/mime.types` on the other.

Each job's download goes in its own folder under DOWNLOAD_ROOT, named by
the job id, so later steps and cleanup can find it from the job id alone.
A download that fails, is refused or is cancelled part way removes its
folder, so no partial file is left behind.
"""

import shutil
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

import httpx

# Inside the system temp folder: `%TEMP%` on Windows, `/tmp` on Linux.
DOWNLOAD_ROOT = Path(tempfile.gettempdir()) / "ragDownloads"

DOWNLOAD_PIECE_BYTES = 1024 * 1024
# The most disk one document may take.
MAX_DOCUMENT_BYTES = 100 * 1024 * 1024
# Generous: a download has no deadline, only a stalled one should fail.
DOWNLOAD_TIMEOUT_SECONDS = 300.0

LINK_SCHEMES = frozenset({"http", "https"})
DOWNLOAD_FILE_STEM = "document"

ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06")
# The bytes each binary format starts with; a docx is a zip inside.
FILE_SIGNATURES = {
    ".pdf": (b"%PDF-",),
    ".docx": ZIP_SIGNATURES,
    ".zip": ZIP_SIGNATURES,
    ".rar": (b"Rar!\x1a\x07",),
}
# A tar has no signature at its start; a POSIX tar marks its header here.
TAR_MARK = b"ustar"
TAR_MARK_OFFSET = 257
# Text has no signature; a NUL byte in it means the file is binary.
TEXT_FORMATS = frozenset({".txt", ".md"})
ACCEPTED_FORMATS = frozenset(FILE_SIGNATURES) | TEXT_FORMATS | {".tar"}
# How much of a file's start the format check reads.
HEAD_BYTES = 8192


class DownloadError(RuntimeError):
    """A document that could not be downloaded; the message says why."""


def formatOf(documentLink: str) -> str:
    """The lowercased extension of the link's path, e.g. `.pdf`."""
    return PurePosixPath(urlparse(documentLink).path).suffix.lower()


def fileNameOf(documentLink: str) -> str:
    """The file name a link ends in, with percent-encoding undone."""
    return PurePosixPath(unquote(urlparse(documentLink).path)).name


def checkDocumentLink(documentLink: object) -> str | None:
    """Why `documentLink` cannot be downloaded, or None when it can."""
    if not isinstance(documentLink, str):
        return "There is no link to download."
    parsed = urlparse(documentLink)
    if parsed.scheme not in LINK_SCHEMES or not parsed.netloc:
        return f"{documentLink!r} is not an http(s) link."
    if formatOf(documentLink) not in ACCEPTED_FORMATS:
        accepted = ", ".join(sorted(ACCEPTED_FORMATS))
        return f"{documentLink!r} is not one of {accepted}."
    return None


def isFileOfFormat(path: Path, fileFormat: str) -> bool:
    """Whether the file at `path` really is a `fileFormat` file."""
    with path.open("rb") as file:
        head = file.read(HEAD_BYTES)
    if fileFormat in TEXT_FORMATS:
        return b"\x00" not in head
    if fileFormat == ".tar":
        markEnd = TAR_MARK_OFFSET + len(TAR_MARK)
        return head[TAR_MARK_OFFSET:markEnd] == TAR_MARK
    return head.startswith(FILE_SIGNATURES[fileFormat])


def jobFolderOf(jobId: str) -> Path:
    """The folder a job's download is kept in."""
    return DOWNLOAD_ROOT / jobId


def downloadPathOf(jobId: str, documentLink: str) -> Path:
    """Where a job's document is saved, keeping the link's extension.

    The file is always named `document`, never after the link, so a link
    cannot name a path outside the job's folder or one Windows rejects.
    """
    return jobFolderOf(jobId) / f"{DOWNLOAD_FILE_STEM}{formatOf(documentLink)}"


def removeDownload(jobId: str) -> None:
    """Delete a job's download folder, if it has one."""
    shutil.rmtree(jobFolderOf(jobId), ignore_errors=True)


async def downloadDocument(documentLink: str, jobId: str) -> Path:
    """Stream the document at `documentLink` to the job's folder.

    Raises:
        DownloadError: If the link is not one this server takes, the server
            answers with an error, the download breaks, the document is
            over MAX_DOCUMENT_BYTES, or the file is not what its extension
            says.
    """
    problem = checkDocumentLink(documentLink)
    if problem is not None:
        raise DownloadError(problem)

    fileFormat = formatOf(documentLink)
    path = downloadPathOf(jobId, documentLink)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        async with httpx.AsyncClient(
            timeout=DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True
        ) as client:
            async with client.stream("GET", documentLink) as response:
                if response.is_error:
                    raise DownloadError(
                        "Downloading the document answered HTTP "
                        f"{response.status_code}."
                    )
                declaredBytes = int(response.headers.get("content-length", 0))
                if declaredBytes > MAX_DOCUMENT_BYTES:
                    raise tooLarge()
                await writePieces(response, path)
        if not isFileOfFormat(path, fileFormat):
            raise DownloadError(
                f"The downloaded file is not a {fileFormat} file."
            )
    except httpx.HTTPError as error:
        removeDownload(jobId)
        raise DownloadError(
            f"The document could not be downloaded: {error!r}."
        ) from error
    except BaseException:
        removeDownload(jobId)
        raise
    return path


async def writePieces(response: httpx.Response, path: Path) -> None:
    """Write the response body to `path` one piece at a time.

    Raises:
        DownloadError: Once more than MAX_DOCUMENT_BYTES have arrived, for
            a server that sent no size, or the wrong one.
    """
    writtenBytes = 0
    with path.open("wb") as file:
        async for piece in response.aiter_bytes(DOWNLOAD_PIECE_BYTES):
            writtenBytes += len(piece)
            if writtenBytes > MAX_DOCUMENT_BYTES:
                raise tooLarge()
            file.write(piece)


def tooLarge() -> DownloadError:
    """The error for a document over the size limit."""
    return DownloadError(
        f"The document is over the {MAX_DOCUMENT_BYTES} byte limit."
    )
