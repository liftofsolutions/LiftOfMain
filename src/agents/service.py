"""Agents: configured by a member of their project, and asked by anyone.

    await configureAgent(accountUuid, projectId, "Acme support", templateId)
    await configureRag(accountUuid, projectId, agentId, ragId)
    await queryAgent(projectId, agentId, "Do you deliver to Pune?")

Configuring is for a member of the project, as `checkAccess` checks it
for the rag endpoints: the account must exist, the project must be
valid, and the account must be a member of it.

* `configureAgent` makes a new agent of the project, active, running a
  template: one of the project's own, or a premade platform template,
  which is copied into the project first, as `db/agent-schema` says. A
  template of another project is not found; an inactive one is refused.
* `configureRag` gives an agent its knowledge base: a rag of the
  project, ready, in the agent's own record - `rag_enabled`, `rag_id` -
  so other agents of its template keep theirs. The agent, if kept in
  this process, is dropped, so it is built afresh on its next query,
  its nodes with `rag_search` searching that rag.

Records are read from Firestore itself, not the cache, as they are about
to be changed, and saved to both: see `agentStore`.

Anyone may ask an agent: no account is checked, nor the project's own
record - only that the agent is of the project named, which the
`AgentManager` checks as it hands the agent over, building it first if
it is not kept. A message may name its conversation by a `convoId`, and
the agent keeps the conversation's chat history between its answers;
one that names none is of no conversation. An agent the project does not
have is answered as not found, one that cannot run - paused, say - as a
conflict.

Errors are returned, never raised: `{"error": message, "errorKind":
kind}`, logged, with `errorKind` one of the `ERROR_*` constants for the
route to turn into a status code.
"""

import copy
import logging

from agents import agentStore
from agents.agent import (
    AGENT_STATUS_ACTIVE,
    AgentNotFoundError,
    AgentNotRunnableError,
)
from agents.agentManager import getAgentManager
from agents.agentStore import AGENT_TEMPLATES, AGENTS
from documentProcessing import service as ragService
from documentProcessing import utilities
from documentProcessing.ragStrategy import RAG_STATUS_READY

LOGGER = logging.getLogger(__name__)

# The error kinds the route turns into status codes; the first two are
# also those `checkAccess` reports.
ERROR_NOT_FOUND = ragService.ERROR_NOT_FOUND
ERROR_FORBIDDEN = ragService.ERROR_FORBIDDEN
ERROR_CONFLICT = "conflict"


def errorResult(errorKind: str, message: str) -> dict:
    """An error to hand back to the route, logged first."""
    LOGGER.error(message)
    return {"error": message, "errorKind": errorKind}


async def configureAgent(
    accountUuid: str, projectId: str, name: str, templateId: str
) -> dict:
    """A new agent of `projectId`, named `name`, running the template
    `templateId`, made by `accountUuid`, a member of the project.

    A premade platform template is copied into the project, under a new
    id, and the agent runs the copy; the copy and the agent are saved
    together, so neither is saved without the other.
    """
    denied = await ragService.checkAccess(accountUuid, projectId)
    if denied is not None:
        return denied
    template = await agentStore.fetchStored(AGENT_TEMPLATES, templateId)
    if template is None or template.get("project_id") not in (
        None,
        projectId,
    ):
        return errorResult(
            ERROR_NOT_FOUND,
            f"Project {projectId!r} has no template {templateId!r}.",
        )
    if not template.get("is_active"):
        return errorResult(
            ERROR_CONFLICT, f"Template {templateId!r} is inactive."
        )

    now = ragService.utcNow()
    records = []
    if template.get("project_id") is None:
        templateId = await ragService.generateUniqueId(AGENT_TEMPLATES)
        template = {
            **copy.deepcopy(template),
            "_id": templateId,
            "project_id": projectId,
            "created_at": now,
            "updated_at": now,
        }
        records.append((AGENT_TEMPLATES, template))
    agent = {
        "_id": await ragService.generateUniqueId(AGENTS),
        "project_id": projectId,
        "template_id": templateId,
        "name": name,
        "status": AGENT_STATUS_ACTIVE,
        "created_by": accountUuid,
        "created_at": now,
        "updated_at": now,
    }
    records.append((AGENTS, agent))
    await agentStore.saveRecords(records)
    LOGGER.info("Configured agent %s of template %s.", agent, templateId)
    return {
        "agentId": agent["_id"],
        "projectId": projectId,
        "templateId": templateId,
        "name": name,
        "status": AGENT_STATUS_ACTIVE,
    }


async def configureRag(
    accountUuid: str, projectId: str, agentId: str, ragId: str
) -> dict:
    """Give the agent `agentId` the rag `ragId` as its knowledge base,
    for `accountUuid`, a member of `projectId`: a rag of the project,
    ready. The agent, if kept, is dropped, to be built afresh with it."""
    denied = await ragService.checkAccess(accountUuid, projectId)
    if denied is not None:
        return denied
    agent = await agentStore.fetchStored(AGENTS, agentId)
    if (
        agent is None
        or agent.get("project_id") != projectId
        or agent.get("deleted_at") is not None
    ):
        return errorResult(
            ERROR_NOT_FOUND, f"Project {projectId!r} has no agent {agentId!r}."
        )
    rag = await ragService.fetchProjectRecord(
        utilities.RAGS, projectId, ragId
    )
    if rag is None:
        return errorResult(
            ERROR_NOT_FOUND, f"Project {projectId!r} has no rag {ragId!r}."
        )
    if rag.get("status") != RAG_STATUS_READY:
        return errorResult(
            ERROR_CONFLICT,
            f"Rag {ragId!r} is {rag.get('status')}; only a ready rag can be "
            "an agent's.",
        )
    agent = {
        **agent,
        "rag_enabled": True,
        "rag_id": ragId,
        "updated_at": ragService.utcNow(),
    }
    await agentStore.saveRecords([(AGENTS, agent)])
    wasKept = getAgentManager().dropAgent(agentId)
    LOGGER.info(
        "Gave agent %s rag %s; it was %s.",
        agentId,
        ragId,
        "kept, and is dropped" if wasKept else "not kept",
    )
    return {"agentId": agentId, "ragId": ragId}


async def queryAgent(
    projectId: str, agentId: str, message: str, convoId: str | None = None
) -> dict:
    """The agent's answer to `message`, in the conversation `convoId` if
    one is named, and that id, for anyone who asks."""
    try:
        reply = await getAgentManager().query(
            projectId, agentId, message, convoId
        )
    except AgentNotFoundError as error:
        return errorResult(ERROR_NOT_FOUND, str(error))
    except AgentNotRunnableError as error:
        return errorResult(ERROR_CONFLICT, str(error))
    return {
        "agentId": agentId,
        "convoId": reply.convoId,
        "answer": reply.answer,
    }
