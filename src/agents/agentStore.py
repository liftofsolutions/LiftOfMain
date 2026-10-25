"""The records an agent is built from: Firestore, cached in memoryDb.

    agent = await fetchAgent(agentId)
    template = await fetchTemplate(agent["template_id"])
    nodeType = await fetchNodeType(node["node_type_id"])

The `agents`, `agent_templates` and `node_types` collections of
`db/agent-schema`, read through `documentProcessing.utilities`: from the
cache if there, else from Firestore, which holds the record of truth, and
then cached. Each collection has its own cache in memoryDb, of at most
CACHE_MAX_RECORDS records, the least recently used dropped first; they
are registered when this module is imported.

A cached record is not read from Firestore again until it is dropped. A
record is handed out as it is cached, so it is read, never changed.
"""

from DbAdapters.memoryDb import EVICTION_LRU, CollectionConfig, getMemoryDb
from documentProcessing import utilities

AGENTS = "agents"
AGENT_TEMPLATES = "agent_templates"
NODE_TYPES = "node_types"
CACHED_COLLECTIONS = (AGENTS, AGENT_TEMPLATES, NODE_TYPES)

# Records each collection's cache keeps before it drops the least used.
CACHE_MAX_RECORDS = 100
CACHE_CONFIG = CollectionConfig(
    evictionPolicy=EVICTION_LRU, maxRecords=CACHE_MAX_RECORDS
)


def registerCaches() -> None:
    """Give each collection its own cache in memoryDb."""
    memoryDb = getMemoryDb()
    for collection in CACHED_COLLECTIONS:
        memoryDb.createCollection(collection, CACHE_CONFIG)


async def fetchAgent(agentId: str) -> dict | None:
    """The agent's record."""
    return await utilities.fetchRecord(AGENTS, agentId)


async def fetchTemplate(templateId: str) -> dict | None:
    """The agent template's record, its nodes and triggers in it."""
    return await utilities.fetchRecord(AGENT_TEMPLATES, templateId)


async def fetchNodeType(nodeTypeId: str) -> dict | None:
    """The node type's record."""
    return await utilities.fetchRecord(NODE_TYPES, nodeTypeId)


registerCaches()
