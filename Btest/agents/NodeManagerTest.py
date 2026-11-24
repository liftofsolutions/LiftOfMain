"""Btest for the node base, and the node manager that reserves nodes.

Two kinds of node, defined here for the two sides of `requiresCredential`:
`whatsapp`, which requires a credential and "sends" the last message from
its credential's number, and `echo_chat`, which takes none, as the chat
model nodes do, and answers with its config's greeting. Every test builds
its own node, or its own `NodeManager` of the two, and reserves them the
way an agent will - a handler, the template node's id and name from
`agents.example.json`, a `Credential` when the node needs one, and a
config - then checks what is reserved, and what is left once released.

1. Credentials: a node that requires one is refused without it, a node
   that takes none is refused with one, a plain dict is refused as one,
   an unknown handler is refused, and a refusal takes no node.
2. Identity: a node is built as its template node, by id and by a name
   that must be a slug, and stays it: neither can be changed, and release
   keeps both.
3. Configure: a node takes its own copies of the credential and config,
   and deconfigure deletes them - cleared, so nothing holding them keeps
   them - leaving the borrower's own alone.
4. On demand: every reservation builds a new node, so two borrowers at
   once get two, and the next borrower never gets a released one; the
   manager keeps no node, free or reserved.
5. Releasing: the node is deconfigured - its credential and config
   cleared - and its id and name stay.
6. About: a node describes itself in LangChain's tool format, by its name
   and its config's description or its own, and a LangChain chat model
   takes it as a tool.
7. Never shown: no log line, repr or about of a node shows its
   credential, or even that it holds one.
8. In LangGraph: reserved nodes run as the nodes of a compiled graph, as
   an agent will run them, and are deconfigured once the run is over;
   the next run gets nodes of its own, sending from its own number
   alone; and two runs at once keep their own credentials.
9. Node classes: every node class written in `agents.nodes` is in
   NODE_CLASSES, under its own handler, once.

    uv run pytest Btest/agents -q
"""

import asyncio
import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from agents.nodeManager import NodeManager, NodeManagerError
from agents.nodes import NODE_CLASSES, Credential, NodeBase
from agents.providerEvents import IO_NODE_CLASSES

MANAGER_LOGGER = "agents.nodeManager"
# Where the io nodes live, out of NODE_CLASSES: no agent reserves them.
IO_NODE_PACKAGES = ("agents.nodes.input.", "agents.nodes.output.")

# The example template's reply and whatsapp nodes.
REPLY_NODE_ID = "473b3fd6-5886-40fc-bd5a-a5d414591bf2"
WHATSAPP_NODE_ID = "9e60cc8c-e30f-44f7-b025-2c613f181635"

QUESTION = "Do you deliver to Pune?"
ANSWER = f"Hi! You asked: {QUESTION}"
WHATSAPP_ABOUT = "Sends the last message to the customer on WhatsApp."


class ChannelState(MessagesState):
    """A run's state: the conversation, and what was sent on it."""

    sent: list[dict]


class WhatsAppNode(NodeBase):
    """Requires a credential, as the WhatsApp node will.

    "Sends" the last message: records it with its credential's number.
    """

    handler = "whatsapp"
    requiresCredential = True
    description = WHATSAPP_ABOUT

    async def __call__(self, state: dict) -> dict:
        """Record the last message as sent, by number and api version."""
        # Lets another run go in between, as a real send would.
        await asyncio.sleep(0)
        sent = {
            "sender": self._credential.reveal("phone_number_id"),
            "version": self.config["api_version"],
            "text": state["messages"][-1].content,
        }
        return {"sent": [sent]}


class EchoChatNode(NodeBase):
    """Takes no credential, as the chat model nodes do.

    Answers the last message with its config's greeting.
    """

    handler = "echo_chat"
    description = "Answers the last message, greeting first."

    async def __call__(self, state: dict) -> dict:
        """Answer the last message, greeting first."""
        question = state["messages"][-1].content
        answer = f"{self.config['greeting']} You asked: {question}"
        return {"messages": [AIMessage(answer)]}


def buildCredentialValues(**overrides) -> dict:
    """A WhatsApp credential's values, decrypted: its account details and
    its secret token."""
    values = {
        "phone_number_id": "111222333",
        "business_account_id": "998877",
        "access_token": "acme-secret-token",
    }
    values.update(overrides)
    return values


def buildCredential(**overrides) -> Credential:
    """A WhatsApp credential, protected, as an agent passes one."""
    return Credential(buildCredentialValues(**overrides))


def buildConfig(**overrides) -> dict:
    """A WhatsApp node's config, as `agents.example.json` gives one."""
    config = {"api_version": "v21.0", "reply_format": "text"}
    config.update(overrides)
    return config


def buildEchoConfig(**overrides) -> dict:
    """An echo chat node's config."""
    config = {"greeting": "Hi!"}
    config.update(overrides)
    return config


def buildWhatsAppReservation(**overrides) -> dict:
    """What an agent reserves the example template's whatsapp node with:
    its id and name, its credential and its config."""
    reservation = {
        "nodeId": WHATSAPP_NODE_ID,
        "name": "whatsapp",
        "credential": buildCredential(),
        "config": buildConfig(),
    }
    reservation.update(overrides)
    return reservation


def buildReplyReservation(**overrides) -> dict:
    """What an agent reserves the example template's reply node with, as
    an echo chat node: its id and name, and its config."""
    reservation = {
        "nodeId": REPLY_NODE_ID,
        "name": "reply",
        "config": buildEchoConfig(),
    }
    reservation.update(overrides)
    return reservation


def buildNodeManager() -> NodeManager:
    """A manager of the two test nodes."""
    nodeClasses = {
        nodeClass.handler: nodeClass
        for nodeClass in (WhatsAppNode, EchoChatNode)
    }
    return NodeManager(nodeClasses)


def subclassesOf(baseClass: type) -> list[type]:
    """Every class below `baseClass`, at any depth."""
    subclasses = []
    for subclass in baseClass.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(subclassesOf(subclass))
    return subclasses


def assertShowsNoCredential(text: str) -> None:
    """Fail if `text` holds any of the credential's values."""
    for value in buildCredentialValues().values():
        assert value not in text


def buildGraph(reply: NodeBase, whatsapp: NodeBase):
    """A compiled graph that answers with `reply`, then sends with
    `whatsapp`, each named as its template node is."""
    graph = StateGraph(ChannelState)
    graph.add_node(reply.name, reply)
    graph.add_node(whatsapp.name, whatsapp)
    graph.add_edge(START, reply.name)
    graph.add_edge(reply.name, whatsapp.name)
    graph.add_edge(whatsapp.name, END)
    return graph.compile()


async def runAgent(
    nodeManager: NodeManager, credential: Credential
) -> tuple[dict, list[NodeBase]]:
    """One run, as an agent will make it; the state it ends in, and the
    nodes it ran on.

    Reserves the nodes, builds the graph from them, runs it on QUESTION,
    and releases the nodes.
    """
    reply = nodeManager.reserveNode("echo_chat", **buildReplyReservation())
    whatsapp = nodeManager.reserveNode(
        "whatsapp", **buildWhatsAppReservation(credential=credential)
    )
    try:
        graph = buildGraph(reply, whatsapp)
        state = await graph.ainvoke({"messages": [HumanMessage(QUESTION)]})
        return state, [reply, whatsapp]
    finally:
        nodeManager.releaseNode(reply)
        nodeManager.releaseNode(whatsapp)


class TestCredentials:
    """What a node may be reserved with, by its `requiresCredential`."""

    def testNodeThatRequiresACredentialIsReservedWithIt(self):
        nodeManager = buildNodeManager()

        node = nodeManager.reserveNode(
            "whatsapp", **buildWhatsAppReservation()
        )

        assert isinstance(node, WhatsAppNode)
        assert node._credential.reveal("access_token") == "acme-secret-token"
        assert node.config == buildConfig()

    @pytest.mark.parametrize(
        "credential", [None, Credential({})], ids=["none", "empty"]
    )
    def testNodeThatRequiresACredentialIsRefusedWithoutOne(
        self, credential
    ):
        nodeManager = buildNodeManager()

        with pytest.raises(NodeManagerError) as raised:
            nodeManager.reserveNode(
                "whatsapp", **buildWhatsAppReservation(credential=credential)
            )

        assert "needs a credential" in str(raised.value)

    def testPlainDictIsRefusedAsACredential(self):
        nodeManager = buildNodeManager()

        with pytest.raises(NodeManagerError) as raised:
            nodeManager.reserveNode(
                "whatsapp",
                **buildWhatsAppReservation(credential=buildCredentialValues()),
            )

        assert "must be a Credential" in str(raised.value)
        assertShowsNoCredential(str(raised.value))

    def testNodeThatTakesNoCredentialIsReservedWithoutOne(self):
        nodeManager = buildNodeManager()

        node = nodeManager.reserveNode("echo_chat", **buildReplyReservation())

        assert isinstance(node, EchoChatNode)
        assert node._credential is None
        assert node.config == buildEchoConfig()

    def testNodeThatTakesNoCredentialIsRefusedWithOne(self):
        nodeManager = buildNodeManager()

        with pytest.raises(NodeManagerError) as raised:
            nodeManager.reserveNode(
                "echo_chat",
                **buildReplyReservation(credential=buildCredential()),
            )

        assert "takes no credential" in str(raised.value)

    def testUnknownHandlerIsRefused(self):
        nodeManager = buildNodeManager()

        with pytest.raises(NodeManagerError) as raised:
            nodeManager.reserveNode("line", **buildWhatsAppReservation())

        assert "'line'" in str(raised.value)


class TestIdentity:
    """Which node of the agent's template a node is, for good."""

    def testReservedNodeIsItsTemplateNode(self):
        node = buildNodeManager().reserveNode(
            "whatsapp", **buildWhatsAppReservation()
        )

        assert node.nodeId == WHATSAPP_NODE_ID
        assert node.name == "whatsapp"

    def testIdAndNameCannotBeChanged(self):
        node = WhatsAppNode(WHATSAPP_NODE_ID, "whatsapp")

        with pytest.raises(AttributeError):
            node.nodeId = REPLY_NODE_ID
        with pytest.raises(AttributeError):
            node.name = "reply"

        assert node.nodeId == WHATSAPP_NODE_ID
        assert node.name == "whatsapp"

    def testReleaseKeepsTheIdAndName(self):
        nodeManager = buildNodeManager()
        node = nodeManager.reserveNode(
            "whatsapp", **buildWhatsAppReservation()
        )

        nodeManager.releaseNode(node)

        assert node.nodeId == WHATSAPP_NODE_ID
        assert node.name == "whatsapp"

    @pytest.mark.parametrize(
        "name", ["reply", "web_search", "crm-lookup", "_draft", "x" * 64]
    )
    def testSlugIsTakenAsTheName(self, name):
        node = buildNodeManager().reserveNode(
            "whatsapp", **buildWhatsAppReservation(name=name)
        )

        assert node.name == name

    @pytest.mark.parametrize(
        "name",
        ["", None, "web search", "web:search", "1st_reply", "x" * 65],
    )
    def testNameThatIsNotASlugIsRefused(self, name):
        nodeManager = buildNodeManager()

        with pytest.raises(NodeManagerError) as raised:
            nodeManager.reserveNode(
                "whatsapp", **buildWhatsAppReservation(name=name)
            )

        assert "not a slug" in str(raised.value)

    @pytest.mark.parametrize("nodeId", ["", None])
    def testNodeWithoutATemplateIdIsRefused(self, nodeId):
        nodeManager = buildNodeManager()

        with pytest.raises(NodeManagerError) as raised:
            nodeManager.reserveNode(
                "whatsapp", **buildWhatsAppReservation(nodeId=nodeId)
            )

        assert "template node's id" in str(raised.value)


class TestConfigure:
    """What configure and deconfigure do to a node, with no manager."""

    def testConfigureGivesTheNodeItsOwnCopies(self):
        node = WhatsAppNode(WHATSAPP_NODE_ID, "whatsapp")
        credential = buildCredential()
        config = buildConfig(retry={"attempts": 3})

        node.configure(credential, config)
        credential.clear()
        config["retry"]["attempts"] = 9

        assert node._credential.reveal("access_token") == "acme-secret-token"
        assert node.config == buildConfig(retry={"attempts": 3})

    def testDeconfigureDeletesTheCredentialAndConfig(self):
        node = WhatsAppNode(WHATSAPP_NODE_ID, "whatsapp")
        node.configure(buildCredential(), buildConfig())
        heldCredential = node._credential
        heldConfig = node.config

        node.deconfigure()

        assert node._credential is None
        assert node.config == {}
        assert not heldCredential
        assert heldConfig == {}
        assert node.nodeId == WHATSAPP_NODE_ID
        assert node.name == "whatsapp"

    def testDeconfigureLeavesTheBorrowersOwnAsTheyWere(self):
        node = WhatsAppNode(WHATSAPP_NODE_ID, "whatsapp")
        credential = buildCredential()
        config = buildConfig()
        node.configure(credential, config)

        node.deconfigure()

        assert credential.reveal("access_token") == "acme-secret-token"
        assert config == buildConfig()


class TestOnDemand:
    """Every reservation is a new node; the manager keeps none."""

    def testTwoBorrowersAtOnceGetTwoNodes(self):
        nodeManager = buildNodeManager()

        first = nodeManager.reserveNode(
            "whatsapp", **buildWhatsAppReservation()
        )
        second = nodeManager.reserveNode(
            "whatsapp",
            **buildWhatsAppReservation(
                credential=buildCredential(access_token="beta-secret-token")
            ),
        )

        assert first is not second
        assert first._credential.reveal("access_token") == "acme-secret-token"
        assert second._credential.reveal("access_token") == (
            "beta-secret-token"
        )

    def testNextBorrowerGetsANewNodeWithOnlyItsCredential(self):
        nodeManager = buildNodeManager()
        node = nodeManager.reserveNode(
            "whatsapp",
            **buildWhatsAppReservation(
                credential=buildCredential(refresh_token="acme-refresh"),
                config=buildConfig(reply_format="template"),
            ),
        )
        nodeManager.releaseNode(node)

        nextNode = nodeManager.reserveNode(
            "whatsapp",
            **buildWhatsAppReservation(
                credential=buildCredential(access_token="beta-secret-token"),
                config={"api_version": "v22.0"},
            ),
        )

        assert nextNode is not node
        assert nextNode._credential.reveal("access_token") == (
            "beta-secret-token"
        )
        with pytest.raises(KeyError):
            nextNode._credential.reveal("refresh_token")
        assert nextNode.config == {"api_version": "v22.0"}

    def testManagerKeepsNoNode(self):
        nodeManager = buildNodeManager()
        nodeClasses = dict(nodeManager.nodeClasses)
        node = nodeManager.reserveNode(
            "whatsapp", **buildWhatsAppReservation()
        )

        assert vars(nodeManager) == {"nodeClasses": nodeClasses}
        nodeManager.releaseNode(node)
        assert vars(nodeManager) == {"nodeClasses": nodeClasses}


class TestReleasing:
    """What is left of a node released."""

    def testReleaseDeconfiguresTheNode(self):
        nodeManager = buildNodeManager()
        node = nodeManager.reserveNode(
            "whatsapp", **buildWhatsAppReservation()
        )
        heldCredential = node._credential

        nodeManager.releaseNode(node)

        assert node._credential is None
        assert node.config == {}
        assert not heldCredential

    def testNodeThatFailsToConfigureIsDeconfigured(self, monkeypatch):
        built = []

        def failToConfigure(node, credential, config):
            built.append(node)
            NodeBase.configure(node, credential, config)
            raise ValueError("No such api version.")

        monkeypatch.setattr(WhatsAppNode, "configure", failToConfigure)

        with pytest.raises(ValueError):
            buildNodeManager().reserveNode(
                "whatsapp", **buildWhatsAppReservation()
            )

        [node] = built
        assert node._credential is None
        assert node.config == {}


class TestAbout:
    """How a node describes itself, for a model to learn it."""

    def testAboutIsTheNodeInLangChainsToolFormat(self):
        node = buildNodeManager().reserveNode(
            "whatsapp", **buildWhatsAppReservation(name="send_reply")
        )

        about = node.about()

        assert about == {
            "type": "function",
            "function": {
                "name": "send_reply",
                "description": WHATSAPP_ABOUT,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        assert convert_to_openai_tool(about) == about

    def testConfigsDescriptionIsTheOneAModelReads(self):
        node = buildNodeManager().reserveNode(
            "echo_chat",
            **buildReplyReservation(
                config=buildEchoConfig(description="Greets Acme's customers.")
            ),
        )

        description = node.about()["function"]["description"]
        assert description == "Greets Acme's customers."

    def testChatModelTakesTheNodeAsATool(self):
        node = buildNodeManager().reserveNode(
            "echo_chat", **buildReplyReservation()
        )
        chatModel = ChatOpenAI(model="gpt-5.5", api_key="test-key")

        bound = chatModel.bind_tools([node.about()])

        assert bound.kwargs["tools"] == [node.about()]


class TestCredentialNeverShown:
    """No log line, repr or about of a node shows its credential."""

    def testLogsNeverShowACredential(self, caplog):
        caplog.set_level(logging.DEBUG, logger=MANAGER_LOGGER)
        nodeManager = buildNodeManager()

        node = nodeManager.reserveNode(
            "whatsapp", **buildWhatsAppReservation()
        )
        nodeManager.releaseNode(node)

        levels = [record.levelname for record in caplog.records]
        assert levels == ["DEBUG", "DEBUG"]
        assertShowsNoCredential(caplog.text)

    def testReprSaysNothingOfTheCredential(self):
        node = buildNodeManager().reserveNode(
            "whatsapp", **buildWhatsAppReservation()
        )

        assert "name='whatsapp'" in repr(node)
        assert "credential" not in repr(node).lower()
        assertShowsNoCredential(repr(node))

    def testAboutNeverShowsACredential(self):
        node = buildNodeManager().reserveNode(
            "whatsapp", **buildWhatsAppReservation()
        )

        assertShowsNoCredential(str(node.about()))

    def testNodesStateShowsNoCredential(self):
        node = buildNodeManager().reserveNode(
            "whatsapp", **buildWhatsAppReservation()
        )

        assertShowsNoCredential(repr(vars(node)))


class TestRunningInLangGraph:
    """Reserved nodes run as the nodes of a compiled LangGraph graph."""

    def testReservedNodesRunAsTheGraphsNodes(self):
        nodeManager = buildNodeManager()

        state, _ = asyncio.run(runAgent(nodeManager, buildCredential()))

        assert [message.content for message in state["messages"]] == [
            QUESTION,
            ANSWER,
        ]
        assert state["sent"] == [
            {"sender": "111222333", "version": "v21.0", "text": ANSWER}
        ]

    def testNodesAreDeconfiguredOnceTheRunIsOver(self):
        nodeManager = buildNodeManager()

        _, nodes = asyncio.run(runAgent(nodeManager, buildCredential()))

        for node in nodes:
            assert node._credential is None
            assert node.config == {}

    def testNextRunGetsNodesOfItsOwnWithOnlyItsCredential(self):
        nodeManager = buildNodeManager()
        _, firstNodes = asyncio.run(
            runAgent(nodeManager, buildCredential())
        )

        state, nextNodes = asyncio.run(
            runAgent(
                nodeManager, buildCredential(phone_number_id="444555666")
            )
        )

        assert not set(firstNodes) & set(nextNodes)
        assert [sent["sender"] for sent in state["sent"]] == ["444555666"]

    def testTwoRunsAtOnceKeepTheirOwnCredentials(self):
        nodeManager = buildNodeManager()

        async def runBoth() -> list[tuple[dict, list[NodeBase]]]:
            return await asyncio.gather(
                runAgent(nodeManager, buildCredential()),
                runAgent(
                    nodeManager, buildCredential(phone_number_id="444555666")
                ),
            )

        (first, firstNodes), (second, secondNodes) = asyncio.run(runBoth())

        assert [sent["sender"] for sent in first["sent"]] == ["111222333"]
        assert [sent["sender"] for sent in second["sent"]] == ["444555666"]
        assert not set(firstNodes) & set(secondNodes)


class TestNodeClasses:
    """NODE_CLASSES, the dict of node classes by handler."""

    def testEveryNodeClassWrittenIsInTheDictUnderItsHandler(self):
        written = [
            nodeClass
            for nodeClass in subclassesOf(NodeBase)
            if nodeClass.__module__.startswith("agents.nodes.")
            and not nodeClass.__module__.startswith(IO_NODE_PACKAGES)
            and nodeClass.handler
        ]

        byHandler = {nodeClass.handler: nodeClass for nodeClass in written}
        assert len(byHandler) == len(written)
        assert byHandler == NODE_CLASSES

    def testNoIoNodeIsReserved(self):
        ioClasses = [
            nodeClass
            for pair in IO_NODE_CLASSES.values()
            for nodeClass in pair
        ]

        assert ioClasses
        for nodeClass in ioClasses:
            assert nodeClass.__module__.startswith(IO_NODE_PACKAGES)
            assert nodeClass.handler not in NODE_CLASSES

    def testManagerReservesFromTheDict(self):
        assert NodeManager().nodeClasses is NODE_CLASSES
