"""Every kind of node an agent can reserve, the base they share, and the
`Credential` they are reserved with, which is never shown.

    from agents.nodes import NODE_CLASSES, Credential, NodeBase

    nodeClass = NODE_CLASSES["openai_chat"]  # OpenAIChatNode

NODE_CLASSES is every node class by its handler: a plain dict, built once
at import from the classes below, each under the `handler` it declares.
Each node class is added there as it is written; the node manager
reserves nodes of those classes, and no others.

Only which code runs a handler lives here, and it changes only with the
code. Which node types exist, and their default configs, is data:
`node_types` in `db/agent-schema`, where `is_active` switches a node type
off without a deploy.
"""

from agents.nodes.chatModelNode import (
    ChatModelNode,
    DeepSeekChatNode,
    GeminiChatNode,
    OpenAIChatNode,
    XAIChatNode,
)
from agents.nodes.credential import Credential
from agents.nodes.nodeBase import NodeBase

# Every kind of node that can be reserved, by its handler.
NODE_CLASSES: dict[str, type[NodeBase]] = {
    nodeClass.handler: nodeClass
    for nodeClass in (
        OpenAIChatNode,
        GeminiChatNode,
        DeepSeekChatNode,
        XAIChatNode,
    )
}

__all__ = [
    "NODE_CLASSES",
    "ChatModelNode",
    "Credential",
    "DeepSeekChatNode",
    "GeminiChatNode",
    "NodeBase",
    "OpenAIChatNode",
    "XAIChatNode",
]
