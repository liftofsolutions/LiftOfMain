"""What a provider - WhatsApp - sends an agent: its check of the agent's
webhook, and its events, each answered by the agent and sent back.

An agent's webhook is its query URL, with the provider named:

    GET  /agents/{projectId}/{agentId}/query?provider=whatsapp&hub.mode=...
    POST /agents/{projectId}/{agentId}/query?provider=whatsapp

so an agent has one URL for every provider, and no new one to make for
each. What the provider sends is handled by the agent's io nodes of it:
the nodes of its template a trigger, enabled, points to, whose node
type's handler is the provider's, each built - as an io node is, out of
the graph - with its merged config, its secrets in its credential.
An agent with none, an agent not found, and a provider no io node
handles, are all PAGE_NOT_FOUND: nothing tells them apart.

* `verifyCallback` - the challenge the first of those nodes answers the
  provider's check with, given the node's token.
* `receiveEvent` - the messages the nodes read out of an event, each
  checking it came from the provider; it hands back how many at once,
  and the agent answers them in the background, each in the
  conversation of its sender with that node, `convoIdOf`, its answer
  sent back by the node. An answer that fails is logged.

A node is configured for the call, and deconfigured once its work is
done: its secrets are kept no longer. A secret is never logged.
"""

import asyncio
import logging
import uuid

from agents import agentStore
from agents.agent import AgentNotRunnableError, mergeConfig
from agents.agentManager import getAgentManager
from agents.nodes.credential import Credential
from agents.nodes.whatsappNode import WhatsAppMessage, WhatsAppNode

LOGGER = logging.getLogger(__name__)

# The io node class that handles each provider, by its handler.
IO_NODE_CLASSES = {WhatsAppNode.handler: WhatsAppNode}
# What a caller is told of a webhook with nothing behind it.
PAGE_NOT_FOUND = "Page not found."
# The namespace a conversation's id is drawn in, from its node and
# sender, so a sender's messages to a number go on with one history.
CONVO_NAMESPACE = uuid.UUID("2f5d8c4e-9a71-4b3e-8d06-1c7e5b9a4f20")

# The answers going on in the background, kept until they end, as
# asyncio keeps only a weak hold on a task.
ANSWERING: set[asyncio.Task] = set()


class PageNotFoundError(LookupError):
    """No agent, or no io node of the provider for it to handle a call."""


class NotFromProviderError(PermissionError):
    """The call failed every node's check that it came from the
    provider."""


def convoIdOf(nodeId: str, sender: str) -> str:
    """The id of the conversation of `sender` with the io node `nodeId`."""
    return str(uuid.uuid5(CONVO_NAMESPACE, f"{nodeId}:{sender}"))


async def ioNodesOf(projectId: str, agentId: str, provider: str) -> list:
    """The agent's io nodes of `provider`, each configured: its merged
    config, the secrets its class names in its credential.

    Raises:
        PageNotFoundError: If no node class handles `provider`, or the
            agent has no enabled trigger on one of its nodes.
        AgentNotFoundError: If there is no such agent.
        AgentNotRunnableError: If the agent cannot run, or a node of
            `provider` lacks what it needs; the error names no value.
    """
    nodeClass = IO_NODE_CLASSES.get(provider)
    if nodeClass is None:
        raise PageNotFoundError(f"No io node handles {provider!r}.")
    agent = await getAgentManager().getAgent(projectId, agentId)
    template = await agentStore.fetchTemplate(agent.templateId) or {}
    nodesById = {node["id"]: node for node in template.get("nodes", [])}
    nodeIds = []
    for trigger in template.get("triggers", []):
        nodeId = trigger.get("node_id")
        if agent.isTriggerAvailable(trigger.get("id")) and (
            nodeId in nodesById and nodeId not in nodeIds
        ):
            nodeIds.append(nodeId)
    ioNodes = []
    for nodeId in nodeIds:
        node = nodesById[nodeId]
        nodeType = await agentStore.fetchNodeType(node["node_type_id"])
        if not isNodeTypeOf(nodeType, provider, projectId):
            continue
        config = mergeConfig(
            nodeType.get("config", {}), node.get("config", {})
        )
        secrets = {
            key: config.pop(key)
            for key in nodeClass.secretKeys
            if key in config
        }
        ioNode = nodeClass(nodeId, node["name"])
        try:
            ioNode.configure(Credential(secrets), config)
        except ValueError as error:
            ioNode.deconfigure()
            for built in ioNodes:
                built.deconfigure()
            raise AgentNotRunnableError(str(error)) from None
        ioNodes.append(ioNode)
    if not ioNodes:
        raise PageNotFoundError(
            f"Agent {agentId!r} has no enabled {provider} trigger."
        )
    return ioNodes


def isNodeTypeOf(
    nodeType: dict | None, provider: str, projectId: str
) -> bool:
    """Whether `nodeType` is an active io node type of `provider`, the
    platform's or the project's."""
    return (
        nodeType is not None
        and nodeType.get("handler") == provider
        and nodeType.get("is_active", False)
        and nodeType.get("project_id") in (None, projectId)
    )


async def verifyCallback(
    projectId: str, agentId: str, provider: str, query: dict
) -> str:
    """The challenge to answer the provider's check of the webhook with,
    given its query parameters: the first io node's that takes them.

    Raises:
        PageNotFoundError, AgentNotFoundError, AgentNotRunnableError: As
            `ioNodesOf` raises them.
        NotFromProviderError: If no node takes them.
    """
    ioNodes = await ioNodesOf(projectId, agentId, provider)
    try:
        for ioNode in ioNodes:
            challenge = ioNode.verifyCallback(query)
            if challenge is not None:
                LOGGER.info(
                    "%s webhook of agent %s verified by node %s.",
                    provider,
                    agentId,
                    ioNode.name,
                )
                return challenge
    finally:
        for ioNode in ioNodes:
            ioNode.deconfigure()
    raise NotFromProviderError(
        f"No {provider} node of agent {agentId!r} takes the check."
    )


async def receiveEvent(
    projectId: str,
    agentId: str,
    provider: str,
    body: bytes,
    headers: dict,
) -> int:
    """How many messages the agent's io nodes of `provider` read out of
    an event; the agent answers them in the background, and the nodes
    send its answers back.

    Raises:
        PageNotFoundError, AgentNotFoundError, AgentNotRunnableError: As
            `ioNodesOf` raises them.
        NotFromProviderError: If the event fails every node's check.
    """
    ioNodes = await ioNodesOf(projectId, agentId, provider)
    received = [(ioNode, ioNode.receive(body, headers)) for ioNode in ioNodes]
    for ioNode, messages in received:
        if not messages:
            ioNode.deconfigure()
    if all(messages is None for _, messages in received):
        raise NotFromProviderError(
            f"The {provider} event fails every check of agent {agentId!r}."
        )
    count = 0
    for ioNode, messages in received:
        if messages:
            count += len(messages)
            answering = asyncio.create_task(
                answerMessages(projectId, agentId, ioNode, messages)
            )
            ANSWERING.add(answering)
            answering.add_done_callback(ANSWERING.discard)
    LOGGER.info(
        "%s event for agent %s: %s messages to answer.",
        provider,
        agentId,
        count,
    )
    return count


async def answerMessages(
    projectId: str,
    agentId: str,
    ioNode: WhatsAppNode,
    messages: list[WhatsAppMessage],
) -> None:
    """Have the agent answer each message, in its sender's conversation
    with `ioNode`, and the node send the answer back; then deconfigure
    the node. A message whose answer fails is logged, and the next
    goes on."""
    try:
        for message in messages:
            try:
                reply = await getAgentManager().query(
                    projectId,
                    agentId,
                    message.text,
                    convoIdOf(ioNode.nodeId, message.sender),
                )
                await ioNode.reply(message.sender, reply.answer)
            except Exception:
                LOGGER.exception(
                    "Answering message %s from %s to agent %s, through "
                    "node %s, failed.",
                    message.messageId,
                    message.sender,
                    agentId,
                    ioNode.name,
                )
    finally:
        ioNode.deconfigure()
