"""Agents: configured by a member of their project, and asked by anyone.

    await configureAgent(
        accountUuid, projectId, "Acme support", templateId,
        "Answers customers.", {nodeId: credentialId},
    )
    await configureRag(accountUuid, projectId, agentId, ragId)
    await queryAgent(projectId, agentId, "Do you deliver to Pune?")

Configuring is for a member of the project, as `checkAccess` checks it
for the rag endpoints: the account must exist, the project must be
valid, and the account must be a member of it.

* `configureAgent` makes a new agent of the project, active, running a
  template: one of the project's own, or a premade platform template,
  which is copied into the project first, as `db/agent-schema` says. A
  template of another project is not found; an inactive one is refused.
  It may name credentials of the project for nodes of the template, by
  node id: the agent then runs a copy of the template of its own, those
  nodes holding their `credential_id`, so the template it was made from,
  and its other agents, are left as they were. A node the template does
  not have, or a credential the project does not, is not found.
* `configureRag` gives an agent its knowledge base: a rag of the
  project, ready, in the agent's own record - `rag_enabled`, `rag_id` -
  so other agents of its template keep theirs. The agent, if kept in
  this process, is dropped, to be destroyed, so it is built afresh on
  its next query, its nodes with `rag_search` searching that rag.

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

A provider - WhatsApp, LINE - reaches an agent at its query URL too,
naming itself: `receiveProviderEvent` takes its event, answered in the
background and sent back by the agent's io node of it, as
`providerEvents` does. An agent with no enabled trigger of the provider,
or none at all, is PAGE_NOT_FOUND; a call that fails the node's check,
forbidden. WhatsApp's check of the webhook is not answered for now:
GitHub issue #15; LINE's is an event like any other.

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
from agents import providerEvents
from agents.agentManager import getAgentManager
from agents.agentStore import AGENT_TEMPLATES, AGENTS, CREDENTIALS
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


async def missingNodeCredential(
    projectId: str, template: dict, nodeCredentials: dict[str, str]
) -> str | None:
    """What `nodeCredentials` - credential ids by node id - names that is
    not there: a node `template` does not have, or a credential that is
    not `projectId`'s; None if nothing."""
    nodeIds = {node["id"] for node in template.get("nodes", [])}
    for nodeId, credentialId in nodeCredentials.items():
        if nodeId not in nodeIds:
            return f"Template {template['_id']!r} has no node {nodeId!r}."
        credential = await agentStore.fetchStored(CREDENTIALS, credentialId)
        if (
            credential is None
            or credential.get("project_id") != projectId
            or credential.get("deleted_at") is not None
        ):
            return (
                f"Project {projectId!r} has no credential {credentialId!r}."
            )
    return None


async def configureAgent(
    accountUuid: str,
    projectId: str,
    name: str,
    templateId: str,
    description: str | None = None,
    nodeCredentials: dict[str, str] | None = None,
) -> dict:
    """A new agent of `projectId`, named `name`, running the template
    `templateId`, made by `accountUuid`, a member of the project; with
    its `description`, and `nodeCredentials` - credential ids by node
    id - the credentials its nodes are formed with.

    A premade platform template is copied into the project, under a new
    id, and the agent runs the copy; so is a template whose nodes are
    given credentials, the copy's nodes holding them. The copy and the
    agent are saved together, so neither is saved without the other.
    """
    nodeCredentials = nodeCredentials or {}
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

    missing = await missingNodeCredential(
        projectId, template, nodeCredentials
    )
    if missing is not None:
        return errorResult(ERROR_NOT_FOUND, missing)

    now = ragService.utcNow()
    records = []
    if template.get("project_id") is None or nodeCredentials:
        templateId = await ragService.generateUniqueId(AGENT_TEMPLATES)
        template = {
            **copy.deepcopy(template),
            "_id": templateId,
            "project_id": projectId,
            "created_at": now,
            "updated_at": now,
        }
        for node in template.get("nodes", []):
            if node["id"] in nodeCredentials:
                node["credential_id"] = nodeCredentials[node["id"]]
        records.append((AGENT_TEMPLATES, template))
    agent = {
        "_id": await ragService.generateUniqueId(AGENTS),
        "project_id": projectId,
        "template_id": templateId,
        "name": name,
        **({} if description is None else {"description": description}),
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
        "description": description,
        "status": AGENT_STATUS_ACTIVE,
        "nodes": [
            {
                "nodeId": node["id"],
                "name": node["name"],
                "credentialId": node.get("credential_id"),
            }
            for node in template.get("nodes", [])
        ],
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


def providerErrorOf(error: Exception) -> dict:
    """The error a provider's call is answered with: the page not found,
    for anything missing, so nothing tells an agent with no trigger from
    no agent; a conflict for an agent that cannot run; forbidden for a
    call that is not the provider's."""
    if isinstance(
        error, (providerEvents.PageNotFoundError, AgentNotFoundError)
    ):
        LOGGER.info("Provider call not found: %s", error)
        return errorResult(ERROR_NOT_FOUND, providerEvents.PAGE_NOT_FOUND)
    if isinstance(error, AgentNotRunnableError):
        return errorResult(ERROR_CONFLICT, str(error))
    return errorResult(ERROR_FORBIDDEN, str(error))


async def receiveProviderEvent(
    projectId: str,
    agentId: str,
    provider: str,
    body: bytes,
    headers: dict,
) -> dict:
    """Take `provider`'s event for the agent: how many messages it holds,
    which the agent answers in the background, sent back by its io node
    of `provider`."""
    try:
        count = await providerEvents.receiveEvent(
            projectId, agentId, provider, body, headers
        )
    except (
        providerEvents.PageNotFoundError,
        providerEvents.NotFromProviderError,
        AgentNotFoundError,
        AgentNotRunnableError,
    ) as error:
        return providerErrorOf(error)
    return {"status": "accepted", "messages": count}
