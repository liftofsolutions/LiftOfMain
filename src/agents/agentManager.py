"""The agents this process keeps, and the queries they answer.

    answer = await getAgentManager().query(projectId, agentId, "Hi!")

`query` answers a message with the agent `agentId` of `projectId`, which
`getAgent` hands it: the agent kept, used as it is, else one loaded -
read from its record and built - and kept. An agent is never started or
stopped as a step of its own: it is loaded when a query needs it and
none is kept. It is loaded once: queries at once for an agent not kept
all wait on the one load, rather than each loading it.

The manager keeps up to `maxAgents` agents - MAX_AGENTS unless it is
built with another limit - in an LRU of its own: a query makes its agent
the most recently used, and keeping one past the limit drops the least
recently used, which is released: its nodes go back to the node manager,
once its running queries are answered. It is the manager's own dict, not
memoryDb, which caches the records an agent is built from. A dropped
agent is loaded again on its next query.

A kept agent is not read from Firestore again while it is kept, nor are
its records while memoryDb caches them: pausing an agent, or editing its
template, reaches it once it is dropped and its records are gone from
memoryDb, or the process restarts.
"""

import asyncio
import functools
import logging
from collections import OrderedDict

from agents.agent import Agent, AgentNotFoundError, loadAgent
from agents.nodeManager import NodeManager, getNodeManager

LOGGER = logging.getLogger(__name__)

# Agents kept; past this, the least recently used is dropped.
MAX_AGENTS = 10


class AgentManager:
    """Loads agents, keeps the most recently used, and runs queries."""

    def __init__(
        self,
        nodeManager: NodeManager | None = None,
        maxAgents: int = MAX_AGENTS,
    ) -> None:
        """A manager with no agent kept, whose agents reserve their nodes
        from `nodeManager` - the process wide one unless given."""
        self.nodeManager = nodeManager or getNodeManager()
        self.maxAgents = maxAgents
        # Kept agents by id, the least recently used first.
        self.agents: OrderedDict[str, Agent] = OrderedDict()
        # Loads under way, by project and agent id; each is dropped once
        # done, so a load that failed is tried afresh on the next query.
        self.loading: dict[tuple[str, str], asyncio.Task] = {}

    async def getAgent(self, projectId: str, agentId: str) -> Agent:
        """The agent `agentId` of `projectId`, kept and now the most
        recently used: the one kept, used as it is; else the one loading,
        once loaded; else one loaded now.

        Raises:
            AgentNotFoundError: If the project has no such agent.
            AgentNotRunnableError: If the agent, or its template, cannot
                be run as it is.
        """
        key = (projectId, agentId)
        while True:
            agent = self.agents.get(agentId)
            if agent is not None:
                if agent.projectId != projectId:
                    raise AgentNotFoundError(
                        f"Project {projectId!r} has no agent {agentId!r}."
                    )
                self.agents.move_to_end(agentId)
                return agent

            load = self.loading.get(key)
            if load is None:
                load = asyncio.create_task(
                    self._loadAndKeep(projectId, agentId)
                )
                load.add_done_callback(functools.partial(self._forgetLoad, key))
                self.loading[key] = load
            # Shielded: a query cancelled while it waits leaves the load
            # to the others. Once loaded, the agent is looked up again: it
            # may have been dropped, and released, while this one waited.
            await asyncio.shield(load)

    def _forgetLoad(self, key: tuple[str, str], load: asyncio.Task) -> None:
        """Forget the load of `key` once it is done - loaded, failed or
        cancelled - so the next query of an agent not kept loads it
        afresh. asyncio calls it with `load`, the task that is done."""
        self.loading.pop(key)

    async def _loadAndKeep(self, projectId: str, agentId: str) -> None:
        """Load the agent and keep it as the most recently used; past the
        limit, the least recently used is dropped and released."""
        agent = await loadAgent(self.nodeManager, projectId, agentId)
        self.agents[agentId] = agent
        self.agents.move_to_end(agentId)
        LOGGER.info("Kept %s; %s agents kept.", agent, len(self.agents))
        if len(self.agents) > self.maxAgents:
            _, dropped = self.agents.popitem(last=False)
            dropped.release()
            LOGGER.info("Dropped %s, the least recently used.", dropped)

    async def query(self, projectId: str, agentId: str, message: str) -> str:
        """The agent's answer to `message`; it is loaded if not kept.

        Raises:
            AgentNotFoundError: If the project has no such agent.
            AgentNotRunnableError: If the agent, or its template, cannot
                be run as it is.
        """
        agent = await self.getAgent(projectId, agentId)
        return await agent.query(message)


_AGENT_MANAGER: AgentManager | None = None


def getAgentManager() -> AgentManager:
    """The process wide agent manager, built once and shared."""
    global _AGENT_MANAGER
    if _AGENT_MANAGER is None:
        _AGENT_MANAGER = AgentManager()
    return _AGENT_MANAGER
