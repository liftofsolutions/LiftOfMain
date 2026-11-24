"""Btest for the LINE io nodes - the input node of
`agents.nodes.input.lineNode`, where messages come in, and the output
node of `agents.nodes.output.lineNode`, where answers go out - and
`agents.providerEvents`, which pairs them under provider "line".

1. The input node is refused without its channel secret. Its entry point
   reads the text messages users sent the account one to one out of a
   signed event, each with its reply token, of provider "line"; a
   group's, a sticker, a follow, or one with no reply token is skipped.
   LINE's check of the webhook - a signed event of no messages - reads
   as none; an unsigned event is None.
2. The output node is refused without its token; `send` reads the
   message's provider: a LINE message's answer is sent as the reply to
   it, with its token, cut into messages LINE takes - no more than a
   reply holds - and one of any other provider is refused, nothing sent.
3. `ioNodesOf` builds the LINE pair for a triggered node, each holding
   only its own secret, the config holding none; an event's message is
   answered and replied to, and LINE's check is taken with nothing sent.

LINE is a `FakeClient` in place of `httpx.AsyncClient`; the agent, its
manager and its store are fakes too.

    uv run pytest Btest/agents -q
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging

import httpx
import pytest

from agents import agentStore, providerEvents
from agents.agent import Reply
from agents.nodes.credential import Credential
from agents.nodes.input.lineNode import PROVIDER, LineInputNode, LineMessage
from agents.nodes.output.lineNode import (
    MAX_MESSAGES,
    MAX_TEXT_LENGTH,
    LineError,
    LineOutputNode,
)

PROJECT_ID = "a7c3e2f1-9b4d-4e6a-8c2f-5d1b0e9a7f34"
AGENT_ID = "88f95300-7453-4609-b685-2c424cc181d5"
TEMPLATE_ID = "8f32ea63-10e7-4105-a05d-ea0c7c02ff52"
NODE_ID = "4c1f7a92-6b3e-4d58-9a0c-2e8f5b1d7c36"
NODE_TYPE_ID = "7e2b9d14-3a6f-4c81-b5e0-9d4a2f6c1b83"
TRIGGER_ID = "b3d8f0a6-1c4e-4a97-8f25-6e0b9c3d7a14"
BOT_ID = "U0bot0000000000000000000000000000"
SENDER = "U4af4980629be2f5a3c5d7e9b1f3a5c7e"
REPLY_TOKEN = "757913772c4646b784d4b7ce46d12671"
CHANNEL_SECRET = "channel-secret"
CHANNEL_ACCESS_TOKEN = "channel-access-token"
QUESTION = "What time is check-in?"
ANSWER = "Check-in is from 12:00, and check-out by 11:00."
REPLY_URL = "https://api.line.me/v2/bot/message/reply"


class FakeClient:
    """Stands in for `httpx.AsyncClient`, keeping every post made in
    `SENT`, and answering each with `STATUS` and an id for each message
    sent - or, with `UNREACHABLE`, failing to connect."""

    SENT: list[dict] = []
    STATUS = 200
    UNREACHABLE = False

    def __init__(self, **settings) -> None:
        """A client; `settings` are a real client's, ignored."""

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *exception) -> None:
        return None

    async def post(self, url: str, headers: dict, json: dict):
        """Keep the post; answer with STATUS."""
        if FakeClient.UNREACHABLE:
            raise httpx.ConnectError("unreachable")
        FakeClient.SENT.append({"url": url, "headers": headers, "json": json})
        sentMessages = [
            {"id": f"line.{index}"}
            for index, _ in enumerate(json["messages"], start=1)
        ]
        return httpx.Response(
            FakeClient.STATUS,
            json={"sentMessages": sentMessages},
            request=httpx.Request("POST", url),
        )


@pytest.fixture
def line(monkeypatch) -> type[FakeClient]:
    """LINE, faked: each post kept in `FakeClient.SENT`."""
    FakeClient.SENT = []
    FakeClient.STATUS = 200
    FakeClient.UNREACHABLE = False
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    return FakeClient


def buildInputNode() -> LineInputNode:
    node = LineInputNode(NODE_ID, "line")
    node.configure(Credential({"channel_secret": CHANNEL_SECRET}), {})
    return node


def buildOutputNode(**config) -> LineOutputNode:
    node = LineOutputNode(NODE_ID, "line")
    node.configure(
        Credential({"channel_access_token": CHANNEL_ACCESS_TOKEN}), config
    )
    return node


def buildLineMessage(**overrides) -> LineMessage:
    """A LINE message, as the input node reads one in."""
    values = {
        "sender": SENDER,
        "text": QUESTION,
        "messageId": "468789577898262530",
        "replyToken": REPLY_TOKEN,
    }
    values.update(overrides)
    return LineMessage(**values)


def buildEventItem(**overrides) -> dict:
    """A text message event of LINE's webhook, from a user's own chat."""
    item = {
        "type": "message",
        "mode": "active",
        "timestamp": 1764000000000,
        "webhookEventId": "01FZ74A0TDDPYRVKNK77XKC3ZR",
        "deliveryContext": {"isRedelivery": False},
        "source": {"type": "user", "userId": SENDER},
        "replyToken": REPLY_TOKEN,
        "message": {
            "type": "text",
            "id": "468789577898262530",
            "quoteToken": "q3Plxr4AgKd",
            "text": QUESTION,
        },
    }
    item.update(overrides)
    return item


def buildEvent(items=None) -> bytes:
    """LINE's webhook body, with `items` for its events."""
    events = [buildEventItem()] if items is None else items
    return json.dumps({"destination": BOT_ID, "events": events}).encode()


def signatureOf(body: bytes, secret: str = CHANNEL_SECRET) -> dict:
    """The headers LINE signs `body` with, under `secret`."""
    digest = hmac.new(secret.encode(), body, hashlib.sha256).digest()
    return {"x-line-signature": base64.b64encode(digest).decode()}


class TestLineInputNode:
    def testRefusedWithoutItsChannelSecret(self):
        node = LineInputNode(NODE_ID, "line")
        with pytest.raises(ValueError, match="channel_secret"):
            node.configure(Credential({}), {})

    def testEntryPointReadsTheMessagesOfProviderLine(self):
        body = buildEvent()

        messages = buildInputNode().entryPoint(body, signatureOf(body))

        assert messages == [buildLineMessage()]
        assert messages[0].replyToken == REPLY_TOKEN
        assert messages[0].provider == PROVIDER == "line"

    @pytest.mark.parametrize(
        "item",
        [
            buildEventItem(
                source={"type": "group", "groupId": "C1", "userId": SENDER}
            ),
            buildEventItem(message={"type": "sticker", "id": "1"}),
            buildEventItem(type="follow", message={}),
            buildEventItem(mode="standby", replyToken=None),
        ],
        ids=["group", "sticker", "follow", "noReplyToken"],
    )
    def testEntryPointSkipsWhatItDoesNotAnswer(self, item):
        body = buildEvent([item])
        assert buildInputNode().entryPoint(body, signatureOf(body)) == []

    def testLinesCheckOfTheWebhookReadsAsNoMessages(self):
        body = buildEvent([])
        assert buildInputNode().entryPoint(body, signatureOf(body)) == []

    @pytest.mark.parametrize(
        "headers",
        [{}, {"x-line-signature": "c2lnbmF0dXJl"}, signatureOf(b"", "x")],
        ids=["none", "forged", "otherSecret"],
    )
    def testEntryPointRefusesAnEventNotSignedByLine(self, headers):
        assert buildInputNode().entryPoint(buildEvent(), headers) is None


class TestLineOutputNode:
    def testRefusedWithoutItsToken(self):
        node = LineOutputNode(NODE_ID, "line")
        with pytest.raises(ValueError, match="channel_access_token"):
            node.configure(Credential({}), {})

    def testSendRepliesWithTheMessagesToken(self, line):
        sentIds = asyncio.run(
            buildOutputNode().send(buildLineMessage(), ANSWER)
        )

        assert sentIds == ["line.1"]
        [sent] = line.SENT
        assert sent["url"] == REPLY_URL
        assert sent["headers"] == {
            "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}"
        }
        assert sent["json"] == {
            "replyToken": REPLY_TOKEN,
            "messages": [{"type": "text", "text": ANSWER}],
        }

    def testSendGoesToTheConfiguredApiUrl(self, line):
        node = buildOutputNode(api_url="http://127.0.0.1:9000")

        asyncio.run(node.send(buildLineMessage(), ANSWER))

        assert line.SENT[0]["url"] == (
            "http://127.0.0.1:9000/v2/bot/message/reply"
        )

    def testSendCutsALongAnswerIntoOneReply(self, line):
        answer = "a" * (MAX_TEXT_LENGTH + 1)

        sentIds = asyncio.run(
            buildOutputNode().send(buildLineMessage(), answer)
        )

        [sent] = line.SENT
        texts = [part["text"] for part in sent["json"]["messages"]]
        assert texts == ["a" * MAX_TEXT_LENGTH, "a"]
        assert sentIds == ["line.1", "line.2"]

    def testSendCutsWhatAReplyCannotHoldAndLogsIt(self, line, caplog):
        answer = "a" * (MAX_MESSAGES * MAX_TEXT_LENGTH + 1)

        with caplog.at_level(logging.WARNING):
            asyncio.run(buildOutputNode().send(buildLineMessage(), answer))

        [sent] = line.SENT
        assert len(sent["json"]["messages"]) == MAX_MESSAGES
        assert "only its first" in caplog.text

    def testSendRefusesAnotherProviderAndSendsNothing(self, line):
        message = buildLineMessage(provider="whatsapp")

        with pytest.raises(LineError, match="whatsapp"):
            asyncio.run(buildOutputNode().send(message, ANSWER))
        assert line.SENT == []

    def testSendRaisesWhenLineRefuses(self, line):
        line.STATUS = 400

        with pytest.raises(LineError, match="400"):
            asyncio.run(buildOutputNode().send(buildLineMessage(), ANSWER))

    def testSendRaisesWhenLineCannotBeReached(self, line):
        line.UNREACHABLE = True

        with pytest.raises(LineError, match="ConnectError"):
            asyncio.run(buildOutputNode().send(buildLineMessage(), ANSWER))


class FakeAgent:
    """Stands in for a built agent: its template, and its one trigger."""

    templateId = TEMPLATE_ID

    def isTriggerAvailable(self, triggerId: str) -> bool:
        return triggerId == TRIGGER_ID


class FakeAgentManager:
    """Stands in for the agent manager: it hands over the agent, and
    answers every query with ANSWER, keeping each."""

    def __init__(self) -> None:
        self.queries: list[tuple] = []

    async def getAgent(self, projectId: str, agentId: str) -> FakeAgent:
        return FakeAgent()

    async def query(self, projectId, agentId, message, convoId):
        self.queries.append((projectId, agentId, message, convoId))
        return Reply(ANSWER, convoId)


@pytest.fixture
def agentManager(monkeypatch) -> FakeAgentManager:
    """The agent manager, and the store, faked: a template whose one
    LINE node a trigger points to, its secrets in its config."""
    manager = FakeAgentManager()
    template = {
        "nodes": [
            {
                "id": NODE_ID,
                "node_type_id": NODE_TYPE_ID,
                "name": "line",
                "config": {
                    "channel_secret": CHANNEL_SECRET,
                    "channel_access_token": CHANNEL_ACCESS_TOKEN,
                },
            }
        ],
        "triggers": [{"id": TRIGGER_ID, "node_id": NODE_ID}],
    }
    nodeType = {
        "handler": "line",
        "is_active": True,
        "config": {"api_url": "https://api.line.me"},
    }

    async def fetchTemplate(templateId):
        return template

    async def fetchNodeType(nodeTypeId):
        return nodeType

    monkeypatch.setattr(providerEvents, "getAgentManager", lambda: manager)
    monkeypatch.setattr(agentStore, "fetchTemplate", fetchTemplate)
    monkeypatch.setattr(agentStore, "fetchNodeType", fetchNodeType)
    return manager


async def receiveAndAnswer(body: bytes, headers: dict) -> int:
    """Hand LINE's event to the agent, and wait for its answers."""
    count = await providerEvents.receiveEvent(
        PROJECT_ID, AGENT_ID, "line", body, headers
    )
    await asyncio.gather(*providerEvents.ANSWERING)
    return count


class TestLineProviderEvents:
    def testEachNodeOfThePairHoldsOnlyItsOwnSecret(self, agentManager):
        [pair] = asyncio.run(
            providerEvents.ioNodesOf(PROJECT_ID, AGENT_ID, "line")
        )

        assert isinstance(pair.inputNode, LineInputNode)
        assert isinstance(pair.outputNode, LineOutputNode)
        inputCredential = pair.inputNode._credential
        outputCredential = pair.outputNode._credential
        assert inputCredential.reveal("channel_secret") == CHANNEL_SECRET
        assert outputCredential.reveal("channel_access_token") == (
            CHANNEL_ACCESS_TOKEN
        )
        with pytest.raises(KeyError):
            inputCredential.reveal("channel_access_token")
        with pytest.raises(KeyError):
            outputCredential.reveal("channel_secret")
        for node in pair:
            assert node.config == {"api_url": "https://api.line.me"}

    def testAMessageInIsAnsweredAndRepliedTo(self, agentManager, line):
        body = buildEvent()

        count = asyncio.run(receiveAndAnswer(body, signatureOf(body)))

        assert count == 1
        [(_, _, message, convoId)] = agentManager.queries
        assert message == QUESTION
        assert convoId == providerEvents.convoIdOf(NODE_ID, SENDER)
        [sent] = line.SENT
        assert sent["json"]["replyToken"] == REPLY_TOKEN
        assert sent["json"]["messages"] == [{"type": "text", "text": ANSWER}]

    def testLinesCheckOfTheWebhookIsTakenWithNothingSent(
        self, agentManager, line
    ):
        body = buildEvent([])

        count = asyncio.run(receiveAndAnswer(body, signatureOf(body)))

        assert count == 0
        assert agentManager.queries == []
        assert line.SENT == []

    def testAnUnsignedEventIsNotFromTheProvider(self, agentManager):
        with pytest.raises(providerEvents.NotFromProviderError):
            asyncio.run(
                providerEvents.receiveEvent(
                    PROJECT_ID, AGENT_ID, "line", buildEvent(), {}
                )
            )
