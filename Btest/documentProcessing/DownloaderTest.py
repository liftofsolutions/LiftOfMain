"""Btest for the document downloader: a real download of an open source file.

Unlike most Btests this one reaches the network. It fetches the Python 3.12.8
embeddable package from python.org - PSF licensed, about 11 MB, and pinned to
one release so its size never drifts - then unpacks it the way a job would.
Nothing is written to disk: the checks below are what prove it, since the
bytes are only ever held in memory.

    uv run pytest Btest/documentProcessing -q
"""

import asyncio
import zipfile

import pytest

from documentProcessing import downloader

DOCUMENT_LINK = (
    "https://www.python.org/ftp/python/3.12.8/python-3.12.8-embed-amd64.zip"
)
EXPECTED_FILE_NAME = "python-3.12.8-embed-amd64.zip"
EXPECTED_SIZE_BYTES = 11094114

MIN_SIZE_BYTES = 10 * 1024 * 1024
MAX_SIZE_BYTES = 20 * 1024 * 1024

# Files the embeddable package is known to carry.
EXPECTED_MEMBERS = {"python.exe", "python312.dll", "python312.zip"}

# The first bytes of any zip file, and of any Windows executable.
ZIP_MAGIC = b"PK\x03\x04"
EXECUTABLE_MAGIC = b"MZ"


@pytest.fixture(scope="module")
def downloadedArchive() -> downloader.LoadedFile:
    """The sample package, downloaded once for every test in this module."""
    return asyncio.run(downloader.downloadDocument(DOCUMENT_LINK))


class TestDownloadDocument:
    """A large remote archive is downloaded whole, into memory."""

    def testDownloadsWholeFile(
        self, downloadedArchive: downloader.LoadedFile
    ) -> None:
        assert downloadedArchive.fileName == EXPECTED_FILE_NAME
        assert downloadedArchive.fileFormat == ".zip"
        assert MIN_SIZE_BYTES <= downloadedArchive.sizeBytes <= MAX_SIZE_BYTES
        assert downloadedArchive.sizeBytes == EXPECTED_SIZE_BYTES
        assert len(downloadedArchive.data) == EXPECTED_SIZE_BYTES

    def testKeepsBytesInMemory(
        self, downloadedArchive: downloader.LoadedFile
    ) -> None:
        """What comes back is the bytes themselves, not a path to them."""
        assert isinstance(downloadedArchive.data, bytes)
        assert downloadedArchive.data[:4] == ZIP_MAGIC

    def testStreamsAreIndependent(
        self, downloadedArchive: downloader.LoadedFile
    ) -> None:
        """Each stream starts at the beginning, so parsers cannot exhaust it."""
        firstStream = downloadedArchive.openStream()
        firstStream.read()

        assert downloadedArchive.openStream().read(4) == ZIP_MAGIC

    def testDownloadedBytesAreIntactArchive(
        self, downloadedArchive: downloader.LoadedFile
    ) -> None:
        with zipfile.ZipFile(downloadedArchive.openStream()) as archive:
            assert archive.testzip() is None


class TestExpandDocument:
    """An archive is unpacked into the files inside it, also in memory."""

    def testExpandsIntoArchiveMembers(
        self, downloadedArchive: downloader.LoadedFile
    ) -> None:
        loadedFiles = downloader.expandDocument(downloadedArchive)

        fileNames = [loadedFile.fileName for loadedFile in loadedFiles]
        assert EXPECTED_MEMBERS <= set(fileNames)
        assert fileNames == sorted(fileNames)

    def testMembersCarryTheirOwnBytes(
        self, downloadedArchive: downloader.LoadedFile
    ) -> None:
        loadedFiles = downloader.expandDocument(downloadedArchive)
        byName = {loadedFile.fileName: loadedFile for loadedFile in loadedFiles}

        executable = byName["python.exe"]
        assert executable.data[:2] == EXECUTABLE_MAGIC
        assert executable.sizeBytes > 0

        # The members are read out of the archive, not shared with it.
        unpackedBytes = sum(item.sizeBytes for item in loadedFiles)
        assert unpackedBytes > downloadedArchive.sizeBytes

    def testRefusesArchiveOverTheMemoryLimit(
        self,
        downloadedArchive: downloader.LoadedFile,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An archive unpacking to more than the cap is refused, not read."""
        monkeypatch.setattr(downloader, "MAX_UNPACKED_BYTES", 1024)

        with pytest.raises(downloader.ArchiveTooLargeError):
            downloader.expandDocument(downloadedArchive)
