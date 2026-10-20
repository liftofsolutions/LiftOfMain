"""What a downloaded document holds, file by file, for the rag to use.

`analyzeDocument` takes the file a job downloaded and returns one
`DocumentAnalysis` - the umbrella - holding a `FileAnalysis` for every
document file in it. A pdf, docx, txt or md is one file. A zip, rar or tar
is opened, and every file in it, and in archives inside it, gets its own
`FileAnalysis`, named by its path through the archives
(`bundle.zip/inner.tar/report.pdf`). Whatever cannot be analyzed - a format
not taken, a file that is not what its name says, one past a limit - is
listed in `skippedFiles` with the reason, not dropped without a word.

Each file is described by what processing it later needs to know: its size
and checksum, how many characters, words, lines, paragraphs and tokens it
holds, and - where its format has them - pages, headings, tables, images
and code blocks. A file that cannot be read keeps its entry, its counts
left empty and a note saying why, so one broken file in an archive of a
hundred does not lose the other ninety nine.

Nothing is held whole in memory: text is read a batch of lines, a pdf a
page and a docx a paragraph at a time, and counted as it goes. Archive
members are copied to disk a piece at a time, into the job's folder, named
by number so a member's own name can never point outside it. Unpacking one
document stops at MAX_UNPACKED_BYTES, MAX_ARCHIVE_FILES and
MAX_ARCHIVE_DEPTH, so an archive built to unpack to far more than it weighs
is cut off before it fills the disk. Unpacked files stay on disk:
processing reads them later through `FileAnalysis.path`; `textOf` gives a
file's text.

Parsing is blocking work, so call `analyzeDocument` through
`asyncio.to_thread`.

Rar is read by an external tool that rarfile runs: `unrar`, or libarchive's
`bsdtar`. On Linux install either (`unrar` or `libarchive-tools`); `tar`
there is GNU tar, which cannot read rar. Windows ships bsdtar as `tar.exe`,
so rarfile is pointed at that. `RAG_UNRAR_TOOL` names an unrar to try
first, on either.
"""

import functools
import hashlib
import itertools
import logging
import math
import os
import re
import tarfile
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO

import docx
import pdfplumber
import rarfile
import tiktoken
from docx.opc.constants import RELATIONSHIP_TYPE

from config import getServerConfig
from documentProcessing import downloader

LOGGER = logging.getLogger(__name__)

DOCUMENT_FORMATS = frozenset({".pdf", ".docx", ".txt", ".md"})
ARCHIVE_FORMATS = frozenset({".zip", ".rar", ".tar"})

# Limits on unpacking one document, archives inside archives included.
MAX_UNPACKED_BYTES = 500 * 1024 * 1024
MAX_ARCHIVE_FILES = 1000
# An archive inside an archive is depth 2.
MAX_ARCHIVE_DEPTH = 3
# Unpacked members go in this folder, beside the downloaded file.
UNPACKED_FOLDER_NAME = "files"
COPY_PIECE_BYTES = 1024 * 1024
LINES_PER_BATCH = 1000

# The encoding GPT-4o and its family count in.
TOKEN_ENCODING_NAME = "o200k_base"
TOKEN_METHOD_ENCODED = f"tiktoken {TOKEN_ENCODING_NAME}"
TOKEN_METHOD_ESTIMATED = "estimated"
# Text is encoded in pieces of this many characters, to bound memory.
TOKEN_PIECE_CHARACTERS = 100_000
# The rough rule used when no encoding can be loaded.
CHARACTERS_PER_TOKEN = 4

WINDOWS_BSDTAR_TOOL = "tar"
HEADING_STYLE_PREFIX = "Heading"

MARKDOWN_HEADING = re.compile(r"^#{1,6}\s")
# The `|---|---|` row under a table's header; one per table.
MARKDOWN_TABLE_SEPARATOR = re.compile(r"^\s*\|?[\s:|-]*-[-\s:|]*\|[\s:|-]*$")
MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
CODE_FENCE = re.compile(r"^\s*(```|~~~)")

# The counts summed in `DocumentAnalysis.totals`.
COUNT_NAMES = (
    "characterCount",
    "wordCount",
    "lineCount",
    "paragraphCount",
    "tokenCount",
    "pageCount",
    "headingCount",
    "tableCount",
    "imageCount",
    "codeBlockCount",
)


class AnalysisError(ValueError):
    """A document none of which could be analyzed; the message says why."""


class UnpackLimitError(ValueError):
    """Unpacking a member would pass a limit on the whole document."""


@dataclass(slots=True)
class FileAnalysis:
    """One file of a document: where it is, and what it holds.

    A count is None where it does not apply to the format - a txt has no
    pages - or when the file could not be read.
    """

    # Its path through any archives, e.g. `bundle.zip/report.pdf`.
    fileName: str
    fileFormat: str
    # Where it is on disk, for processing to read.
    path: Path
    sizeBytes: int
    checksumSha256: str
    readable: bool = True
    characterCount: int | None = None
    wordCount: int | None = None
    lineCount: int | None = None
    paragraphCount: int | None = None
    tokenCount: int | None = None
    tokenCountMethod: str | None = None
    pageCount: int | None = None
    headingCount: int | None = None
    tableCount: int | None = None
    imageCount: int | None = None
    codeBlockCount: int | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SkippedFile:
    """A file in the document that was not analyzed, and why."""

    fileName: str
    reason: str


@dataclass(slots=True)
class DocumentAnalysis:
    """Everything one downloaded document holds: the umbrella over its files."""

    fileName: str
    fileFormat: str
    path: Path
    sizeBytes: int
    checksumSha256: str
    isArchive: bool
    files: list[FileAnalysis] = field(default_factory=list)
    skippedFiles: list[SkippedFile] = field(default_factory=list)

    def totals(self) -> dict[str, int]:
        """How many files there are, and every count summed over them."""
        readable = [analysis for analysis in self.files if analysis.readable]
        totals = {
            "fileCount": len(self.files),
            "readableFileCount": len(readable),
            "skippedFileCount": len(self.skippedFiles),
        }
        for countName in COUNT_NAMES:
            totals[countName] = sum(
                getattr(analysis, countName) or 0 for analysis in readable
            )
        return totals


@dataclass(slots=True)
class ArchiveMember:
    """One file inside an archive, before it is unpacked."""

    name: str
    declaredBytes: int
    open: Callable[[], IO[bytes]]


@dataclass(slots=True)
class UnpackBudget:
    """What is left of the limits on unpacking one document.

    The limits are read when a budget is made, not when this module loads.
    """

    bytesLeft: int = field(default_factory=lambda: MAX_UNPACKED_BYTES)
    filesLeft: int = field(default_factory=lambda: MAX_ARCHIVE_FILES)
    lastNumber: int = 0


def configureRarTools() -> None:
    """Point rarfile at the tools this platform has for reading rar."""
    if os.name == "nt":
        rarfile.BSDTAR_TOOL = WINDOWS_BSDTAR_TOOL
    unrarTool = getServerConfig().unrarTool
    if unrarTool:
        rarfile.UNRAR_TOOL = unrarTool


@functools.lru_cache(maxsize=1)
def getTokenEncoding() -> tiktoken.Encoding | None:
    """The tokenizer, or None when it cannot be loaded.

    tiktoken fetches an encoding the first time it is asked for and caches
    it on disk, so a server with no way out to the internet may have none.
    Counting then falls back to an estimate rather than failing the job.
    """
    try:
        return tiktoken.get_encoding(TOKEN_ENCODING_NAME)
    except Exception:
        LOGGER.warning(
            "No %s encoding; estimating tokens.", TOKEN_ENCODING_NAME
        )
        return None


def countTokens(text: str) -> int:
    """How many tokens `text` holds, encoded a piece at a time."""
    encoding = getTokenEncoding()
    if encoding is None:
        return math.ceil(len(text) / CHARACTERS_PER_TOKEN)
    return sum(
        len(
            encoding.encode(
                text[start : start + TOKEN_PIECE_CHARACTERS],
                disallowed_special=(),
            )
        )
        for start in range(0, len(text), TOKEN_PIECE_CHARACTERS)
    )


def startCounting(analysis: FileAnalysis) -> None:
    """Set the counts every readable format has to zero."""
    analysis.characterCount = 0
    analysis.wordCount = 0
    analysis.lineCount = 0
    analysis.tokenCount = 0
    analysis.tokenCountMethod = (
        TOKEN_METHOD_ENCODED
        if getTokenEncoding() is not None
        else TOKEN_METHOD_ESTIMATED
    )


def addText(analysis: FileAnalysis, text: str) -> None:
    """Count one block of a file's text: a run of lines, a page, a cell."""
    analysis.characterCount += len(text)
    analysis.wordCount += len(text.split())
    analysis.lineCount += len(text.splitlines())
    analysis.tokenCount += countTokens(text)


def countMarkdownLine(
    analysis: FileAnalysis, line: str, inCodeBlock: bool
) -> bool:
    """Count what one markdown line opens; whether a code block is open."""
    if CODE_FENCE.match(line):
        if not inCodeBlock:
            analysis.codeBlockCount += 1
        return not inCodeBlock
    if inCodeBlock:
        return True
    if MARKDOWN_HEADING.match(line):
        analysis.headingCount += 1
    if MARKDOWN_TABLE_SEPARATOR.match(line):
        analysis.tableCount += 1
    analysis.imageCount += len(MARKDOWN_IMAGE.findall(line))
    return False


def readPlainText(analysis: FileAnalysis) -> None:
    """Count a txt or md file a batch of lines at a time.

    A paragraph is a run of lines between blank ones. Markdown also has its
    headings, tables, images and code blocks counted; nothing inside a code
    block counts as any of the others.
    """
    isMarkdown = analysis.fileFormat == ".md"
    startCounting(analysis)
    analysis.paragraphCount = 0
    if isMarkdown:
        analysis.headingCount = 0
        analysis.tableCount = 0
        analysis.imageCount = 0
        analysis.codeBlockCount = 0

    previousBlank = True
    inCodeBlock = False
    with analysis.path.open(encoding="utf-8-sig", errors="replace") as file:
        for lines in itertools.batched(file, LINES_PER_BATCH):
            addText(analysis, "".join(lines))
            for line in lines:
                isBlank = not line.strip()
                if previousBlank and not isBlank:
                    analysis.paragraphCount += 1
                previousBlank = isBlank
                if isMarkdown:
                    inCodeBlock = countMarkdownLine(analysis, line, inCodeBlock)


def readPdf(analysis: FileAnalysis) -> None:
    """Count a pdf a page at a time: its text, pages, tables and images.

    Each page's parsed objects are released once it is counted. A pdf has
    no paragraphs to count reliably, so that count stays None.
    """
    startCounting(analysis)
    analysis.tableCount = 0
    analysis.imageCount = 0
    with pdfplumber.open(analysis.path) as pdf:
        analysis.pageCount = len(pdf.pages)
        for page in pdf.pages:
            addText(analysis, page.extract_text() or "")
            analysis.tableCount += len(page.find_tables())
            analysis.imageCount += len(page.images)
            page.close()
    if analysis.wordCount == 0:
        analysis.notes.append(
            "No text could be read; scanned pages need OCR to be indexed."
        )


def readDocx(analysis: FileAnalysis) -> None:
    """Count a Word document: paragraphs, headings, tables and images.

    Table cells count towards the text. Pages are not stored in a docx -
    they come from laying it out - so that count stays None.
    """
    startCounting(analysis)
    analysis.paragraphCount = 0
    analysis.headingCount = 0
    wordDocument = docx.Document(str(analysis.path))
    for paragraph in wordDocument.paragraphs:
        if not paragraph.text.strip():
            continue
        analysis.paragraphCount += 1
        addText(analysis, paragraph.text)
        style = paragraph.style
        if style is not None and (style.name or "").startswith(
            HEADING_STYLE_PREFIX
        ):
            analysis.headingCount += 1

    analysis.tableCount = len(wordDocument.tables)
    for table in wordDocument.tables:
        for row in table.rows:
            addText(analysis, " | ".join(cell.text for cell in row.cells))
    analysis.imageCount = sum(
        1
        for relationship in wordDocument.part.rels.values()
        if relationship.reltype == RELATIONSHIP_TYPE.IMAGE
    )


READERS: dict[str, Callable[[FileAnalysis], None]] = {
    ".txt": readPlainText,
    ".md": readPlainText,
    ".pdf": readPdf,
    ".docx": readDocx,
}


def plainTextOf(path: Path) -> str:
    """A txt or md file's text."""
    return path.read_text(encoding="utf-8-sig", errors="replace")


def pdfTextOf(path: Path) -> str:
    """A pdf's text, page after page, each page its own paragraph."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
            page.close()
    return "\n\n".join(pages)


def docxTextOf(path: Path) -> str:
    """A Word document's paragraphs, then its table rows, as counted."""
    wordDocument = docx.Document(str(path))
    blocks = [
        paragraph.text
        for paragraph in wordDocument.paragraphs
        if paragraph.text.strip()
    ]
    for table in wordDocument.tables:
        for row in table.rows:
            blocks.append(" | ".join(cell.text for cell in row.cells))
    return "\n\n".join(blocks)


TEXT_READERS: dict[str, Callable[[Path], str]] = {
    ".txt": plainTextOf,
    ".md": plainTextOf,
    ".pdf": pdfTextOf,
    ".docx": docxTextOf,
}


def textOf(analysis: FileAnalysis) -> str:
    """The text of a file that was analyzed, read the way it was counted.

    Paragraphs are kept apart by blank lines. Blocking, like the analysis.
    """
    return TEXT_READERS[analysis.fileFormat](analysis.path)


def checksumOf(path: Path) -> str:
    """The file's SHA-256, read a piece at a time."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while piece := file.read(COPY_PIECE_BYTES):
            digest.update(piece)
    return digest.hexdigest()


def describeError(error: BaseException) -> str:
    """An error as `Type: message`, or just its type when it has none."""
    message = str(error)
    errorType = type(error).__name__
    return f"{errorType}: {message}" if message else errorType


def analyzeFile(path: Path, fileName: str, fileFormat: str) -> FileAnalysis:
    """Everything one document file holds.

    A file that cannot be read comes back with no counts and a note.
    """
    sizeBytes = path.stat().st_size
    checksum = checksumOf(path)
    analysis = FileAnalysis(fileName, fileFormat, path, sizeBytes, checksum)
    try:
        READERS[fileFormat](analysis)
    except Exception as error:
        LOGGER.warning("Could not read %s: %s", fileName, describeError(error))
        return FileAnalysis(
            fileName,
            fileFormat,
            path,
            sizeBytes,
            checksum,
            readable=False,
            notes=[
                f"Could not be read as a {fileFormat} file: "
                f"{describeError(error)}"
            ],
        )
    return analysis


@contextmanager
def openArchive(
    path: Path, archiveFormat: str
) -> Iterator[list[ArchiveMember]]:
    """The files in an archive, ready to be unpacked one by one.

    Folders are left out, and from a tar so is anything that is not a
    plain file - links and devices are never followed or made.
    """
    if archiveFormat == ".zip":
        with zipfile.ZipFile(path) as archive:
            yield [
                ArchiveMember(
                    info.filename,
                    info.file_size,
                    functools.partial(archive.open, info),
                )
                for info in archive.infolist()
                if not info.is_dir()
            ]
    elif archiveFormat == ".tar":
        with tarfile.open(path) as archive:
            yield [
                ArchiveMember(
                    info.name,
                    info.size,
                    functools.partial(archive.extractfile, info),
                )
                for info in archive.getmembers()
                if info.isfile()
            ]
    else:
        with rarfile.RarFile(path) as archive:
            yield [
                ArchiveMember(
                    info.filename,
                    info.file_size,
                    functools.partial(archive.open, info),
                )
                for info in archive.infolist()
                if not info.is_dir()
            ]


def describeArchiveError(error: BaseException, archiveFormat: str) -> str:
    """Why an archive could not be opened, in words a caller can act on."""
    if isinstance(error, rarfile.RarCannotExec):
        return (
            "This server has no tool to read .rar files; install unrar "
            "or bsdtar."
        )
    reason = describeError(error)
    return f"The {archiveFormat} archive could not be opened: {reason}"


def unpackMember(
    member: ArchiveMember,
    memberFormat: str,
    budget: UnpackBudget,
    unpackFolder: Path,
) -> Path:
    """Copy one member to disk a piece at a time, within the budget.

    It is named by number, never by its own name, so it lands in
    `unpackFolder` whatever its name says.

    Raises:
        UnpackLimitError: Once its bytes pass what the budget has left.
    """
    unpackFolder.mkdir(parents=True, exist_ok=True)
    budget.lastNumber += 1
    budget.filesLeft -= 1
    path = unpackFolder / f"{budget.lastNumber:04d}{memberFormat}"
    try:
        with member.open() as source, path.open("wb") as target:
            while piece := source.read(COPY_PIECE_BYTES):
                budget.bytesLeft -= len(piece)
                if budget.bytesLeft < 0:
                    raise UnpackLimitError(
                        "Unpacking it passed the document's limit of "
                        f"{MAX_UNPACKED_BYTES} bytes."
                    )
                target.write(piece)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def exploreArchive(
    path: Path,
    archiveFormat: str,
    archiveName: str,
    depth: int,
    document: DocumentAnalysis,
    budget: UnpackBudget,
) -> None:
    """Analyze every file in an archive, and in archives inside it.

    Raises:
        Whatever opening the archive raises, for the caller to describe.
    """
    unpackFolder = document.path.parent / UNPACKED_FOLDER_NAME
    with openArchive(path, archiveFormat) as members:
        for member in members:
            exploreMember(
                member,
                f"{archiveName}/{member.name}",
                depth,
                document,
                budget,
                unpackFolder,
            )


def exploreMember(
    member: ArchiveMember,
    memberName: str,
    depth: int,
    document: DocumentAnalysis,
    budget: UnpackBudget,
    unpackFolder: Path,
) -> None:
    """Unpack one member and analyze it, or record why it was skipped."""

    def skip(reason: str) -> None:
        document.skippedFiles.append(SkippedFile(memberName, reason))

    memberFormat = PurePosixPath(member.name).suffix.lower()
    isArchive = memberFormat in ARCHIVE_FORMATS
    if memberFormat not in DOCUMENT_FORMATS and not isArchive:
        skip(f"{memberFormat or 'A file with no extension'} is not taken.")
        return
    if isArchive and depth >= MAX_ARCHIVE_DEPTH:
        skip(f"Archives nest more than {MAX_ARCHIVE_DEPTH} deep.")
        return
    if budget.filesLeft <= 0:
        skip(f"The document holds more than {MAX_ARCHIVE_FILES} files.")
        return
    if member.declaredBytes > budget.bytesLeft:
        skip(
            "Unpacking it would pass the document's limit of "
            f"{MAX_UNPACKED_BYTES} bytes."
        )
        return

    try:
        unpackedPath = unpackMember(member, memberFormat, budget, unpackFolder)
    except UnpackLimitError as error:
        skip(str(error))
        return
    except Exception as error:
        skip(f"Could not be unpacked: {describeError(error)}")
        return

    if not downloader.isFileOfFormat(unpackedPath, memberFormat):
        unpackedPath.unlink()
        skip(f"It is not a {memberFormat} file.")
        return

    if not isArchive:
        document.files.append(
            analyzeFile(unpackedPath, memberName, memberFormat)
        )
        return
    try:
        exploreArchive(
            unpackedPath, memberFormat, memberName, depth + 1, document, budget
        )
    except Exception as error:
        skip(describeArchiveError(error, memberFormat))
    finally:
        # Its files are unpacked now; the archive itself is not needed.
        unpackedPath.unlink(missing_ok=True)


def analyzeDocument(path: Path, fileName: str) -> DocumentAnalysis:
    """Everything the downloaded document at `path` holds.

    `fileName` is the name the document goes by, e.g. from its link.

    Raises:
        AnalysisError: If it is an archive that cannot be opened, or nothing
            in it could be read: no file in it holds a word, as with a
            scanned pdf or an empty txt.
    """
    fileFormat = path.suffix.lower()
    document = DocumentAnalysis(
        fileName=fileName,
        fileFormat=fileFormat,
        path=path,
        sizeBytes=path.stat().st_size,
        checksumSha256=checksumOf(path),
        isArchive=fileFormat in ARCHIVE_FORMATS,
    )
    if document.isArchive:
        try:
            exploreArchive(
                path, fileFormat, fileName, 1, document, UnpackBudget()
            )
        except Exception as error:
            raise AnalysisError(
                describeArchiveError(error, fileFormat)
            ) from error
    else:
        document.files.append(analyzeFile(path, fileName, fileFormat))

    # An unreadable file's wordCount is None, a file with no text's is 0.
    if not any(analysis.wordCount for analysis in document.files):
        raise AnalysisError(describeNothingReadable(document))
    return document


def describeNothingReadable(document: DocumentAnalysis) -> str:
    """Why a document gave no text to read, from its first problem."""
    message = f"Nothing in {document.fileName!r} could be read."
    notes = [note for analysis in document.files for note in analysis.notes]
    if notes:
        return f"{message} {notes[0]}"
    if document.skippedFiles:
        first = document.skippedFiles[0]
        moreSkipped = len(document.skippedFiles) - 1
        more = f" {moreSkipped} more were skipped too." if moreSkipped else ""
        return f"{message} Skipped {first.fileName}: {first.reason}{more}"
    if document.files:
        return f"{message} It holds no text."
    return f"{message} It is empty."


configureRarTools()
