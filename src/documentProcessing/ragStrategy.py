"""Which kind of rag a document gets, and the work each kind does.

    strategy = selectStrategy(analysis.totals()["tokenCount"])

* Up to PLAIN_TEXT_MAX_TOKENS - plain text. The document is small enough
  to give a model whole, so `savePlainText` saves its text and link on the
  rag's record, and the rag is ready.
* Above that, however large - simple rag. `chunkDocument` cuts each file's
  text with LangChain's recursive splitter: at paragraphs, then lines, then
  words. Markdown is cut before its headings first, so a heading starts the
  chunk holding its section. Tokens are counted as `documentAnalyzer`
  counts them.

A chunk holds up to `RAG_CHUNK_MAX_TOKENS` tokens (500 when not set), and
starts with up to `RAG_CHUNK_OVERLAP_TOKENS` of the chunk before it (none
when not set). The overlap is made of whole pieces - paragraphs, lines or
words, whichever the text was cut into there - so a piece longer than the
overlap is not repeated.

A rag's chunks then go to Pinecone, whose index embeds them itself; that
step, and saving their Pinecone ids in Firestore, are still to be written.
"""

import asyncio
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
from documentProcessing import utilities
from documentProcessing.documentAnalyzer import (
    DocumentAnalysis,
    countTokens,
    textOf,
)

STRATEGY_PLAIN_TEXT = "plainText"
STRATEGY_SIMPLE_RAG = "simpleRag"

PLAIN_TEXT_MAX_TOKENS = 10_000
# Used when RAG_CHUNK_MAX_TOKENS and RAG_CHUNK_OVERLAP_TOKENS are not set.
DEFAULT_CHUNK_MAX_TOKENS = 500
DEFAULT_CHUNK_OVERLAP_TOKENS = 0

RAG_STATUS_READY = "ready"


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


async def savePlainText(
    ragId: str, documentLink: str, document: DocumentAnalysis
) -> None:
    """Save the document's text and link on the rag, and mark it ready.

    Merged into the rag's record in Firestore, and not cached.
    """
    text = await asyncio.to_thread(documentTextOf, document)
    await utilities.getFirestoreAdapter().upsertDocument(
        utilities.RAGS,
        Document(
            ragId,
            {
                "strategy": STRATEGY_PLAIN_TEXT,
                "document_link": documentLink,
                "text": text,
                "token_count": document.totals()["tokenCount"],
                "status": RAG_STATUS_READY,
                "updated_at": datetime.now(timezone.utc),
            },
        ),
        merge=True,
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
