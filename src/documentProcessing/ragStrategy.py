"""Which kind of rag a document gets, and the work each kind does.

    strategy = selectStrategy(analysis.totals()["tokenCount"])

* Up to PLAIN_TEXT_MAX_TOKENS - plain text. The document is small enough
  to give a model whole, so `savePlainText` saves its text and link on the
  rag's record.
* Above that, however large - simple rag. `chunkDocument` cuts each file's
  text with LangChain's recursive splitter: at paragraphs, then lines, then
  words. Markdown is cut before its headings first, so a heading starts the
  chunk holding its section. Tokens are counted as `documentAnalyzer`
  counts them.

A chunk holds up to `RAG_CHUNK_MAX_TOKENS` tokens (500 when not set, at
most CHUNK_MAX_TOKENS_LIMIT), and starts with up to
`RAG_CHUNK_OVERLAP_TOKENS` of the chunk before it (100 when not set). The
overlap is made of whole pieces - paragraphs, lines or words, whichever the
text was cut into there - so a piece longer than the overlap is not
repeated.

`saveChunks` then writes a rag's chunks to Pinecone, in the rag's own
namespace, each under the id `{ragId}#{chunkIndex}`; the index embeds
their `chunkText` itself as it stores them. A batch Pinecone fails to save
is sent again on its own, straight away, up to SAVE_ATTEMPTS tries in all.
`saveIndexedRag` saves the rag's chunk count - the ids follow from it, so
none is stored.

Every write here to a rag's record in Firestore goes through `saveRag`,
which merges into the record and stamps `updated_at`; nothing is cached.
Neither strategy marks the rag ready: the job does, as it ends.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from config import getServerConfig
from config.serverConfig import (
    CHUNK_MAX_TOKENS_ENV_VAR,
    CHUNK_OVERLAP_TOKENS_ENV_VAR,
    requireCount,
)
from DbAdapters import Document
from DbAdapters.pineconeDb import getPineconeDb
from DbAdapters.pineconeDb.pineconeDb import RECORDS_PER_ADD
from documentProcessing import utilities
from documentProcessing.documentAnalyzer import (
    DocumentAnalysis,
    countTokens,
    textOf,
)

LOGGER = logging.getLogger(__name__)

STRATEGY_PLAIN_TEXT = "plainText"
STRATEGY_SIMPLE_RAG = "simpleRag"

PLAIN_TEXT_MAX_TOKENS = 10_000
# Used when RAG_CHUNK_MAX_TOKENS and RAG_CHUNK_OVERLAP_TOKENS are not set.
DEFAULT_CHUNK_MAX_TOKENS = 500
DEFAULT_CHUNK_OVERLAP_TOKENS = 100
# The largest RAG_CHUNK_MAX_TOKENS taken.
CHUNK_MAX_TOKENS_LIMIT = 1_500

RAG_STATUS_READY = "ready"
RAG_STATUS_FAILED = "failed"

# The field the Pinecone index embeds, as the index declares it.
PINECONE_TEXT_FIELD = "chunkText"
# Tries in all for a batch of chunks Pinecone fails to save, one straight
# after another: Pinecone's client already waits out rate limits and server
# errors within a try.
SAVE_ATTEMPTS = 3


@dataclass(slots=True)
class Chunk:
    """One piece of a document's text, ready to be indexed."""

    # Its place in the document, from 0, across all of its files.
    chunkIndex: int
    # The file it came from, by its path through any archives.
    fileName: str
    text: str
    tokenCount: int


def selectStrategy(tokenCount: int) -> str:
    """The strategy for a document holding `tokenCount` tokens."""
    if tokenCount <= PLAIN_TEXT_MAX_TOKENS:
        return STRATEGY_PLAIN_TEXT
    return STRATEGY_SIMPLE_RAG


def documentTextOf(document: DocumentAnalysis) -> str:
    """The text of every readable file, one after another.

    In an archive each file's text is headed by its name, so the files can
    still be told apart. Blocking: reads the files.
    """
    texts = []
    for analysis in document.files:
        if not analysis.readable:
            continue
        text = textOf(analysis).strip()
        if document.isArchive:
            text = f"{analysis.fileName}\n\n{text}"
        texts.append(text)
    return "\n\n".join(texts)


async def saveRag(ragId: str, fields: dict) -> None:
    """Merge `fields` into the rag's record in Firestore; not cached."""
    await utilities.getFirestoreAdapter().upsertDocument(
        utilities.RAGS,
        Document(ragId, {**fields, "updated_at": datetime.now(timezone.utc)}),
        merge=True,
    )


async def savePlainText(
    ragId: str, documentLink: str, document: DocumentAnalysis
) -> None:
    """Save the document's text and link on the rag."""
    text = await asyncio.to_thread(documentTextOf, document)
    await saveRag(
        ragId,
        {
            "strategy": STRATEGY_PLAIN_TEXT,
            "document_link": documentLink,
            "text": text,
            "token_count": document.totals()["tokenCount"],
        },
    )


def splitterFor(fileFormat: str) -> RecursiveCharacterTextSplitter:
    """The splitter for a file's format, sized as configured.

    Raises:
        ConfigurationError: If a chunk setting is not a usable number.
        ValueError: If the overlap is larger than the chunk size.
    """
    serverConfig = getServerConfig()
    settings = {
        "chunk_size": requireCount(
            serverConfig.chunkMaxTokens,
            CHUNK_MAX_TOKENS_ENV_VAR,
            DEFAULT_CHUNK_MAX_TOKENS,
            minimum=1,
            maximum=CHUNK_MAX_TOKENS_LIMIT,
        ),
        "chunk_overlap": requireCount(
            serverConfig.chunkOverlapTokens,
            CHUNK_OVERLAP_TOKENS_ENV_VAR,
            DEFAULT_CHUNK_OVERLAP_TOKENS,
            minimum=0,
        ),
        "length_function": countTokens,
    }
    if fileFormat == ".md":
        return RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN, **settings
        )
    return RecursiveCharacterTextSplitter(**settings)


def chunkDocument(document: DocumentAnalysis) -> list[Chunk]:
    """Chunk every readable file of the document, numbered file after file.

    Blocking: reads the files and counts tokens.
    """
    chunks: list[Chunk] = []
    for analysis in document.files:
        if not analysis.readable:
            continue
        splitter = splitterFor(analysis.fileFormat)
        for text in splitter.split_text(textOf(analysis)):
            chunks.append(
                Chunk(len(chunks), analysis.fileName, text, countTokens(text))
            )
    return chunks


def recordIdOf(ragId: str, chunkIndex: int) -> str:
    """A chunk's id in Pinecone, made from its rag and its place in it."""
    return f"{ragId}#{chunkIndex}"


async def saveBatch(ragId: str, records: list[dict]) -> int:
    """Write one batch to the rag's namespace, trying again if it fails.

    Returns how many records landed. A failed try is followed at once by
    the next; after SAVE_ATTEMPTS tries the last failure is raised.
    """
    attempt = 1
    while True:
        try:
            return await getPineconeDb().add(ragId, records)
        except Exception as error:
            if attempt == SAVE_ATTEMPTS:
                raise
            LOGGER.warning(
                "Saving %s chunks in %s failed, try %s of %s: %s. Trying "
                "again.",
                len(records),
                ragId,
                attempt,
                SAVE_ATTEMPTS,
                error,
            )
            attempt += 1


async def saveChunks(ragId: str, projectId: str, chunks: list[Chunk]) -> int:
    """Write the chunks to the rag's namespace in Pinecone; how many landed.

    Pinecone embeds each chunk's text as it stores it. The chunks go in
    batches of one request each, all at once, and a batch that fails is
    tried again on its own, so chunks already saved are not embedded
    twice. If one still fails, its error is raised once every batch is
    over; the batches that landed stay, and a rerun overwrites them, since
    an id depends only on the rag and the chunk's place.
    """
    records = [
        {
            "_id": recordIdOf(ragId, chunk.chunkIndex),
            PINECONE_TEXT_FIELD: chunk.text,
            "rag_id": ragId,
            "project_id": projectId,
            "file_name": chunk.fileName,
            "chunk": chunk.chunkIndex,
        }
        for chunk in chunks
    ]
    batches = [
        records[start : start + RECORDS_PER_ADD]
        for start in range(0, len(records), RECORDS_PER_ADD)
    ]
    results = await asyncio.gather(
        *(saveBatch(ragId, batch) for batch in batches),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return sum(results)


async def saveIndexedRag(
    ragId: str, documentLink: str, document: DocumentAnalysis, chunkCount: int
) -> None:
    """Save how the rag's document was indexed.

    Its chunks are `{ragId}#0` to `{ragId}#{chunkCount - 1}` in Pinecone.
    """
    await saveRag(
        ragId,
        {
            "strategy": STRATEGY_SIMPLE_RAG,
            "document_link": documentLink,
            "token_count": document.totals()["tokenCount"],
            "chunk_count": chunkCount,
        },
    )
