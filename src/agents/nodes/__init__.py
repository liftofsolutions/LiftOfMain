"""Every kind of node an agent can reserve, the base they share, and the
`Credential` they are reserved with, which is never shown; and the nodes
an agent builds itself, never reserved: the special nodes every graph
has, `InputNode` and `OutputNode`, and the `ChoosingNode` a chat model
node with several edges out runs as.

The reservable ones: a chat model node per provider, and the
`ChatHistoryNode`, which keeps a conversation's turns, summarised once
they grow long; and, for now, `FastSupportNode`, an experiment: a guard
and support in one.

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

from agents.nodes.chatHistoryNode import ChatHistoryNode
from agents.nodes.chatModelNode import (
    ChatModelNode,
    ChoosingNode,
    DeepSeekChatNode,
    GeminiChatNode,
    OpenAIChatNode,
    XAIChatNode,
)
from agents.nodes.credential import Credential
from agents.nodes.fastSupportNode import FastSupportNode
from agents.nodes.nodeBase import NodeBase
from agents.nodes.specialNodes import (
    INPUT_NODE_ID,
    OUTPUT_ABOUT,
    OUTPUT_NODE_ID,
    RUN_REPLY,
    SPECIAL_NODE_IDS,
    AfterReplyNode,
    InputNode,
    OutputNode,
    answerOf,
)

# Every kind of node that can be reserved, by its handler.
NODE_CLASSES: dict[str, type[NodeBase]] = {
    nodeClass.handler: nodeClass
    for nodeClass in (
        OpenAIChatNode,
        GeminiChatNode,
        DeepSeekChatNode,
        XAIChatNode,
        ChatHistoryNode,
        FastSupportNode,
    )
}

__all__ = [
    "INPUT_NODE_ID",
    "NODE_CLASSES",
    "OUTPUT_ABOUT",
    "OUTPUT_NODE_ID",
    "RUN_REPLY",
    "SPECIAL_NODE_IDS",
    "AfterReplyNode",
    "ChatHistoryNode",
    "ChatModelNode",
    "ChoosingNode",
    "Credential",
    "DeepSeekChatNode",
    "FastSupportNode",
    "GeminiChatNode",
    "InputNode",
    "NodeBase",
    "OpenAIChatNode",
    "OutputNode",
    "XAIChatNode",
    "answerOf",
]
