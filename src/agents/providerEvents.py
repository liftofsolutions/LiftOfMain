"""What a provider - WhatsApp, LINE - sends an agent: its events, each
answered by the agent and sent back.

An agent's webhook is its query URL, with the provider named:

    POST /agents/{projectId}/{agentId}/query?provider=whatsapp
    POST /agents/{projectId}/{agentId}/query?provider=line

so an agent has one URL for every provider, and no new one to make for
each. What the provider sends is handled by the agent's io nodes of it:
for each node of its template a trigger, enabled, points to, whose node
type's handler is the provider's, an `IoNodes` pair - its input node,
where messages come in, and its output node, where answers go out -
both built, as io nodes are, out of the graph, from the node's merged
config, each holding the secrets its class names in its credential.
An agent with none, an agent not found, and a provider no io node
handles, are all PAGE_NOT_FOUND: nothing tells them apart.

`receiveEvent` takes the messages the input nodes read out of an event,
at their entry points, each checking it came from the provider; it
hands back how many at once, and the agent answers them in the
background, each in the conversation of its sender with that node,
`convoIdOf`, its answer sent by the paired output node. An answer that
fails is logged.

LINE's check of the webhook is an event of no messages, taken as any
is. WhatsApp's - the GET with `hub.challenge` - is not answered for
now: GitHub issue #15.

Nodes are configured for the call, and deconfigured once their work is
done: their secrets are kept no longer. A secret is never logged.
"""

import asyncio
import logging
import uuid
from typing import NamedTuple

from agents import agentStore
from agents.agent import AgentNotRunnableError, mergeConfig
from agents.agentManager import getAgentManager
from agents.nodes.credential import Credential
from agents.nodes.input.lineNode import LineInputNode, LineMessage
from agents.nodes.input.whatsappNode import (
    WhatsAppInputNode,
    WhatsAppMessage,
)
from agents.nodes.nodeBase import NodeBase
from agents.nodes.output.lineNode import LineOutputNode
from agents.nodes.output.whatsappNode import WhatsAppOutputNode

LOGGER = logging.getLogger(__name__)

# The io node classes that handle each provider, by the handler of its
# node type: its input node's class, and its output node's.
IO_NODE_CLASSES = {
    WhatsAppInputNode.handler: (WhatsAppInputNode, WhatsAppOutputNode),
    LineInputNode.handler: (LineInputNode, LineOutputNode),
}
# A message an input node reads in, for its output node to answer.
ProviderMessage = WhatsAppMessage | LineMessage
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


class IoNodes(NamedTuple):
    """A template node's io nodes: its input node, where messages come
    in, and its output node, where answers go out."""

    inputNode: NodeBase
    outputNode: NodeBase

    def deconfigure(self) -> None:
        """Deconfigure both: their secrets are kept no longer."""
        self.inputNode.deconfigure()
        self.outputNode.deconfigure()


def convoIdOf(nodeId: str, sender: str) -> str:
    """The id of the conversation of `sender` with the io node `nodeId`."""
    return str(uuid.uuid5(CONVO_NAMESPACE, f"{nodeId}:{sender}"))


async def ioNodesOf(
    projectId: str, agentId: str, provider: str
) -> list[IoNodes]:
    """The agent's io nodes of `provider`, a pair for each node a trigger
    points to, each configured: the node's merged config, with every
    secret either class names taken out of it, and in its credential the
    secrets its own class names.

    Raises:
        PageNotFoundError: If no node class handles `provider`, or the
            agent has no enabled trigger on one of its nodes.
        AgentNotFoundError: If there is no such agent.
        AgentNotRunnableError: If the agent cannot run, or a node of
            `provider` lacks what it needs; the error names no value.
    """
    nodeClasses = IO_NODE_CLASSES.get(provider)
    if nodeClasses is None:
        raise PageNotFoundError(f"No io node handles {provider!r}.")
    inputClass, outputClass = nodeClasses
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
            for key in (*inputClass.secretKeys, *outputClass.secretKeys)
            if key in config
        }
        pair = IoNodes(
            inputClass(nodeId, node["name"]),
            outputClass(nodeId, node["name"]),
        )
        try:
            for ioNode in pair:
                ioNode.configure(
                    Credential(
                        {
                            key: secrets[key]
                            for key in type(ioNode).secretKeys
                            if key in secrets
                        }
                    ),
                    config,
                )
        except ValueError as error:
            pair.deconfigure()
            for built in ioNodes:
                built.deconfigure()
            raise AgentNotRunnableError(str(error)) from None
        ioNodes.append(pair)
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


async def receiveEvent(
    projectId: str,
    agentId: str,
    provider: str,
    body: bytes,
    headers: dict,
) -> int:
    """How many messages the agent's input nodes of `provider` read out
    of an event, at their entry points; the agent answers them in the
    background, and the paired output nodes send its answers back.

    Raises:
        PageNotFoundError, AgentNotFoundError, AgentNotRunnableError: As
            `ioNodesOf` raises them.
        NotFromProviderError: If the event fails every node's check.
    """
    ioNodes = await ioNodesOf(projectId, agentId, provider)
    received = [
        (pair, pair.inputNode.entryPoint(body, headers)) for pair in ioNodes
    ]
    for pair, messages in received:
        if not messages:
            pair.deconfigure()
    if all(messages is None for _, messages in received):
        raise NotFromProviderError(
            f"The {provider} event fails every check of agent {agentId!r}."
        )
    count = 0
    for pair, messages in received:
        if messages:
            count += len(messages)
            answering = asyncio.create_task(
                answerMessages(projectId, agentId, pair, messages)
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
    ioNodes: IoNodes,
    messages: list[ProviderMessage],
) -> None:
    """Have the agent answer each message, in its sender's conversation
    with the input node, and the output node send the answer back; then
    deconfigure both. A message whose answer fails is logged, and the
    next goes on."""
    try:
        for message in messages:
            try:
                reply = await getAgentManager().query(
                    projectId,
                    agentId,
                    message.text,
                    convoIdOf(ioNodes.inputNode.nodeId, message.sender),
                )
                await ioNodes.outputNode.send(message, reply.answer)
            except Exception:
                LOGGER.exception(
                    "Answering message %s from %s to agent %s, through "
                    "node %s, failed.",
                    message.messageId,
                    message.sender,
                    agentId,
                    ioNodes.inputNode.name,
                )
    finally:
        ioNodes.deconfigure()
