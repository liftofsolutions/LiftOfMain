"""The records an agent is built from, and the chat history of its
conversations: Firestore, cached in memoryDb.

    agent = await fetchAgent(agentId)
    template = await fetchTemplate(agent["template_id"])
    nodeType = await fetchNodeType(node["node_type_id"])
    chatHistory = await fetchChatHistory(agentId, convoId)
    await saveChatHistory(agentId, projectId, convoId, chatHistory)

The `agents`, `agent_templates` and `node_types` collections of
`db/agent-schema`, read through `documentProcessing.utilities`: from the
cache if there, else from Firestore, which holds the record of truth, and
then cached. Each collection has its own cache in memoryDb, of at most
CACHE_MAX_RECORDS records, the least recently used dropped first; they
are registered when this module is imported.

A cached record is not read from Firestore again until it is dropped. A
record is handed out as it is cached, so it is read, never changed.

A conversation's chat history is an `agent_memory` record: whatever the
agent's nodes keep of it, a plain dict, read as the records above are,
and saved after every answer the agent gives in the conversation - to
Firestore, then to its cache, of at most MEMORY_CACHE_MAX_RECORDS
conversations, so the next query reads it from memoryDb rather than
Firestore.
"""

from datetime import datetime, timezone

from DbAdapters import Document
from DbAdapters.memoryDb import EVICTION_LRU, CollectionConfig, getMemoryDb
from documentProcessing import utilities

AGENTS = "agents"
AGENT_TEMPLATES = "agent_templates"
NODE_TYPES = "node_types"
CACHED_COLLECTIONS = (AGENTS, AGENT_TEMPLATES, NODE_TYPES)
AGENT_MEMORY = "agent_memory"

# Records each collection's cache keeps before it drops the least used.
CACHE_MAX_RECORDS = 100
CACHE_CONFIG = CollectionConfig(
    evictionPolicy=EVICTION_LRU, maxRecords=CACHE_MAX_RECORDS
)
# Conversations whose chat history is cached before the least recently
# used is dropped: more than agents, as an agent has many conversations.
MEMORY_CACHE_MAX_RECORDS = 2000
MEMORY_CACHE_CONFIG = CollectionConfig(
    evictionPolicy=EVICTION_LRU, maxRecords=MEMORY_CACHE_MAX_RECORDS
)


def registerCaches() -> None:
    """Give each collection its own cache in memoryDb."""
    memoryDb = getMemoryDb()
    for collection in CACHED_COLLECTIONS:
        memoryDb.createCollection(collection, CACHE_CONFIG)
    memoryDb.createCollection(AGENT_MEMORY, MEMORY_CACHE_CONFIG)


async def fetchAgent(agentId: str) -> dict | None:
    """The agent's record."""
    return await utilities.fetchRecord(AGENTS, agentId)


async def fetchTemplate(templateId: str) -> dict | None:
    """The agent template's record, its nodes and triggers in it."""
    return await utilities.fetchRecord(AGENT_TEMPLATES, templateId)


async def fetchNodeType(nodeTypeId: str) -> dict | None:
    """The node type's record."""
    return await utilities.fetchRecord(NODE_TYPES, nodeTypeId)


def chatHistoryIdOf(agentId: str, convoId: str) -> str:
    """The id of a conversation's chat history record: the agent's and the
    conversation's, so a conversation is of one agent alone."""
    return f"{agentId}_{convoId}"


async def fetchChatHistory(agentId: str, convoId: str) -> dict | None:
    """The chat history of the agent's conversation `convoId`, as it is
    cached; None before the conversation's first answer."""
    record = await utilities.fetchRecord(
        AGENT_MEMORY, chatHistoryIdOf(agentId, convoId)
    )
    return None if record is None else record["chat_history"]


async def saveChatHistory(
    agentId: str, projectId: str, convoId: str, chatHistory: dict
) -> None:
    """Save `chatHistory` as the agent's conversation `convoId`'s: to
    Firestore, then to the cache, so the next query finds it there.

    The cache is written last, so it never holds what Firestore does not.
    """
    chatHistoryId = chatHistoryIdOf(agentId, convoId)
    record = {
        "agent_id": agentId,
        "project_id": projectId,
        "convo_id": convoId,
        "chat_history": chatHistory,
        "updated_at": datetime.now(timezone.utc),
    }
    await utilities.getFirestoreAdapter().upsertDocument(
        AGENT_MEMORY, Document(chatHistoryId, record)
    )
    getMemoryDb().addRecord(AGENT_MEMORY, chatHistoryId, record)


registerCaches()
