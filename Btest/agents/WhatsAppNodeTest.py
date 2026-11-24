"""Btest for the WhatsApp io nodes - the input node of
`agents.nodes.input.whatsappNode`, where messages come in, and the
output node of `agents.nodes.output.whatsappNode`, where answers go out -
and `agents.providerEvents`, which pairs them.

1. The input node is refused without its app secret or its number; it
   needs no verify token, and has no check of the callback URL - that
   is GitHub issue #15. Its entry point reads the text messages to its
   number out of a signed event, each of provider "whatsapp" - an
   unsigned event is None.
2. The output node is refused without its token or its number; `send`
   reads the message's provider: a WhatsApp message's answer is sent to
   its sender, cut into messages WhatsApp takes, and one of any other
   provider is refused, nothing sent.
3. `ioNodesOf` builds a pair for a triggered node, each holding only its
   own secrets - the verify token in the input node's - the config
   holding none; `answerMessages` answers each message and sends it out
   at the output node, then deconfigures both.

Meta is a `FakeClient` in place of `httpx.AsyncClient`; the agent, its
manager and its store are fakes too.

    uv run pytest Btest/agents -q
"""

import asyncio
import hashlib
import hmac
import json

import httpx
import pytest

from agents import agentStore, providerEvents
from agents.agent import Reply
from agents.nodes.credential import Credential
from agents.nodes.input.whatsappNode import (
    PROVIDER,
    WhatsAppInputNode,
    WhatsAppMessage,
)
from agents.nodes.output.whatsappNode import (
    MAX_TEXT_LENGTH,
    WhatsAppError,
    WhatsAppOutputNode,
)

PROJECT_ID = "a7c3e2f1-9b4d-4e6a-8c2f-5d1b0e9a7f34"
AGENT_ID = "88f95300-7453-4609-b685-2c424cc181d5"
TEMPLATE_ID = "8f32ea63-10e7-4105-a05d-ea0c7c02ff52"
NODE_ID = "9e60cc8c-e30f-44f7-b025-2c613f181635"
NODE_TYPE_ID = "1b06fcd1-73a2-4703-a9bc-00ae5a96a10a"
TRIGGER_ID = "d9d81456-5f87-4b46-80e1-0c4d39b36ec7"
PHONE_NUMBER_ID = "111222333"
SENDER = "919876543210"
APP_SECRET = "app-secret"
VERIFY_TOKEN = "verify-token"
ACCESS_TOKEN = "access-token"
QUESTION = "Do you deliver to Pune?"
ANSWER = "Yes, we deliver to Pune in 2-3 business days."


class FakeClient:
    """Stands in for `httpx.AsyncClient`, keeping every post made in
    `SENT`, and answering each with `STATUS` and an id of its own."""

    SENT: list[dict] = []
    STATUS = 200

    def __init__(self, **settings) -> None:
        """A client; `settings` are a real client's, ignored."""

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *exception) -> None:
        return None

    async def post(self, url: str, headers: dict, json: dict):
        """Keep the post; answer with STATUS."""
        FakeClient.SENT.append({"url": url, "headers": headers, "json": json})
        sentId = f"wamid.{len(FakeClient.SENT)}"
        return httpx.Response(
            FakeClient.STATUS,
            json={"messages": [{"id": sentId}]},
            request=httpx.Request("POST", url),
        )


@pytest.fixture
def meta(monkeypatch) -> type[FakeClient]:
    """Meta, faked: each post kept in `FakeClient.SENT`."""
    FakeClient.SENT = []
    FakeClient.STATUS = 200
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    return FakeClient


def credentialOf(values: dict) -> Credential:
    """A credential of `values`, leaving out those set to None."""
    return Credential(
        {key: value for key, value in values.items() if value is not None}
    )


def buildInputCredential(**overrides) -> Credential:
    values = {"app_secret": APP_SECRET, "verify_token": VERIFY_TOKEN}
    values.update(overrides)
    return credentialOf(values)


def buildOutputCredential(**overrides) -> Credential:
    values = {"access_token": ACCESS_TOKEN}
    values.update(overrides)
    return credentialOf(values)


def buildConfig(**overrides) -> dict:
    config = {"phone_number_id": PHONE_NUMBER_ID, "api_version": "v21.0"}
    config.update(overrides)
    return config


def buildInputNode(**overrides) -> WhatsAppInputNode:
    node = WhatsAppInputNode(NODE_ID, "whatsapp")
    node.configure(buildInputCredential(), buildConfig(**overrides))
    return node


def buildOutputNode(**overrides) -> WhatsAppOutputNode:
    node = WhatsAppOutputNode(NODE_ID, "whatsapp")
    node.configure(buildOutputCredential(), buildConfig(**overrides))
    return node


def buildMessage(**overrides) -> dict:
    """A text message of Meta's event."""
    message = {
        "from": SENDER,
        "id": "wamid.in",
        "type": "text",
        "text": {"body": QUESTION},
    }
    message.update(overrides)
    return message


def buildEvent(messages=None, phoneNumberId=PHONE_NUMBER_ID) -> bytes:
    """Meta's event, with `messages`, to `phoneNumberId`."""
    event = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "metadata": {"phone_number_id": phoneNumberId},
                            "messages": (
                                [buildMessage()]
                                if messages is None
                                else messages
                            ),
                        }
                    }
                ]
            }
        ]
    }
    return json.dumps(event).encode()


def signatureOf(body: bytes, secret: str = APP_SECRET) -> dict:
    """The headers Meta signs `body` with, under `secret`."""
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return {"x-hub-signature-256": f"sha256={digest}"}


class TestWhatsAppInputNode:
    def testRefusedWithoutItsAppSecret(self):
        node = WhatsAppInputNode(NODE_ID, "whatsapp")
        with pytest.raises(ValueError, match="app_secret"):
            node.configure(
                buildInputCredential(app_secret=None), buildConfig()
            )

    def testNeedsNoVerifyTokenAndHasNoCallbackCheck(self):
        node = WhatsAppInputNode(NODE_ID, "whatsapp")

        node.configure(buildInputCredential(verify_token=None), buildConfig())

        assert node.config == buildConfig()
        assert not hasattr(node, "verifyCallback")

    def testRefusedWithoutANumber(self):
        node = WhatsAppInputNode(NODE_ID, "whatsapp")
        with pytest.raises(ValueError, match="phone_number_id"):
            node.configure(
                buildInputCredential(), buildConfig(phone_number_id="")
            )

    def testEntryPointReadsTheMessagesOfProviderWhatsApp(self):
        body = buildEvent()
        messages = buildInputNode().entryPoint(body, signatureOf(body))
        assert messages == [
            WhatsAppMessage(SENDER, QUESTION, "wamid.in", "whatsapp")
        ]
        assert messages[0].provider == PROVIDER == "whatsapp"

    def testEntryPointSkipsAnotherNumberAndNonTextMessages(self):
        node = buildInputNode()
        toOther = buildEvent(phoneNumberId="999")
        image = buildEvent([buildMessage(type="image", text={})])
        assert node.entryPoint(toOther, signatureOf(toOther)) == []
        assert node.entryPoint(image, signatureOf(image)) == []

    @pytest.mark.parametrize(
        "headers",
        [{}, {"x-hub-signature-256": "sha256=0"}, signatureOf(b"", "other")],
    )
    def testEntryPointRefusesAnEventNotSignedByMeta(self, headers):
        assert buildInputNode().entryPoint(buildEvent(), headers) is None


class TestWhatsAppOutputNode:
    def testRefusedWithoutItsToken(self):
        node = WhatsAppOutputNode(NODE_ID, "whatsapp")
        with pytest.raises(ValueError, match="access_token"):
            node.configure(Credential({}), buildConfig())

    def testRefusedWithoutANumber(self):
        node = WhatsAppOutputNode(NODE_ID, "whatsapp")
        with pytest.raises(ValueError, match="phone_number_id"):
            node.configure(buildOutputCredential(), {})

    def testSendSendsTheAnswerToTheSender(self, meta):
        message = WhatsAppMessage(SENDER, QUESTION, "wamid.in")

        sentIds = asyncio.run(buildOutputNode().send(message, ANSWER))

        assert sentIds == ["wamid.1"]
        [sent] = meta.SENT
        assert sent["url"] == (
            f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
        )
        assert sent["headers"] == {"Authorization": f"Bearer {ACCESS_TOKEN}"}
        assert sent["json"]["to"] == SENDER
        assert sent["json"]["text"] == {"body": ANSWER}

    def testSendCutsALongAnswer(self, meta):
        message = WhatsAppMessage(SENDER, QUESTION, "wamid.in")
        answer = "a" * (MAX_TEXT_LENGTH + 1)

        asyncio.run(buildOutputNode().send(message, answer))

        bodies = [sent["json"]["text"]["body"] for sent in meta.SENT]
        assert bodies == ["a" * MAX_TEXT_LENGTH, "a"]

    def testSendRefusesAnotherProviderAndSendsNothing(self, meta):
        message = WhatsAppMessage(SENDER, QUESTION, "wamid.in", "email")

        with pytest.raises(WhatsAppError, match="email"):
            asyncio.run(buildOutputNode().send(message, ANSWER))
        assert meta.SENT == []

    def testSendRaisesWhenWhatsAppRefuses(self, meta):
        meta.STATUS = 400
        message = WhatsAppMessage(SENDER, QUESTION, "wamid.in")

        with pytest.raises(WhatsAppError, match="400"):
            asyncio.run(buildOutputNode().send(message, ANSWER))


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
    WhatsApp node a trigger points to, its secrets in its config."""
    manager = FakeAgentManager()
    template = {
        "nodes": [
            {
                "id": NODE_ID,
                "node_type_id": NODE_TYPE_ID,
                "name": "whatsapp",
                "config": {
                    "phone_number_id": PHONE_NUMBER_ID,
                    "app_secret": APP_SECRET,
                    "verify_token": VERIFY_TOKEN,
                    "access_token": ACCESS_TOKEN,
                },
            }
        ],
        "triggers": [{"id": TRIGGER_ID, "node_id": NODE_ID}],
    }
    nodeType = {
        "handler": "whatsapp",
        "is_active": True,
        "config": {"api_version": "v21.0"},
    }

    async def fetchTemplate(templateId):
        return template

    async def fetchNodeType(nodeTypeId):
        return nodeType

    monkeypatch.setattr(providerEvents, "getAgentManager", lambda: manager)
    monkeypatch.setattr(agentStore, "fetchTemplate", fetchTemplate)
    monkeypatch.setattr(agentStore, "fetchNodeType", fetchNodeType)
    return manager


class TestProviderEvents:
    def testEachNodeOfThePairHoldsOnlyItsOwnSecrets(self, agentManager):
        [pair] = asyncio.run(
            providerEvents.ioNodesOf(PROJECT_ID, AGENT_ID, "whatsapp")
        )

        assert isinstance(pair.inputNode, WhatsAppInputNode)
        assert isinstance(pair.outputNode, WhatsAppOutputNode)
        assert pair.inputNode._credential.reveal("app_secret") == APP_SECRET
        # Unread until issue #15, but out of the config all the same.
        assert pair.inputNode._credential.reveal("verify_token") == (
            VERIFY_TOKEN
        )
        assert pair.outputNode._credential.reveal("access_token") == (
            ACCESS_TOKEN
        )
        with pytest.raises(KeyError):
            pair.inputNode._credential.reveal("access_token")
        with pytest.raises(KeyError):
            pair.outputNode._credential.reveal("app_secret")
        for node in pair:
            assert node.config == buildConfig()

    def testAMessageInIsAnsweredAndSentOutThenBothAreWiped(
        self, agentManager, meta
    ):
        async def answerOneEvent():
            body = buildEvent()
            count = await providerEvents.receiveEvent(
                PROJECT_ID, AGENT_ID, "whatsapp", body, signatureOf(body)
            )
            await asyncio.gather(*providerEvents.ANSWERING)
            return count

        count = asyncio.run(answerOneEvent())

        assert count == 1
        [(_, _, message, convoId)] = agentManager.queries
        assert message == QUESTION
        assert convoId == providerEvents.convoIdOf(NODE_ID, SENDER)
        [sent] = meta.SENT
        assert sent["json"]["to"] == SENDER
        assert sent["json"]["text"] == {"body": ANSWER}

    def testAnsweringDeconfiguresBothNodes(self, agentManager, meta):
        [pair] = asyncio.run(
            providerEvents.ioNodesOf(PROJECT_ID, AGENT_ID, "whatsapp")
        )
        message = WhatsAppMessage(SENDER, QUESTION, "wamid.in")

        asyncio.run(
            providerEvents.answerMessages(
                PROJECT_ID, AGENT_ID, pair, [message]
            )
        )

        assert len(meta.SENT) == 1
        for node in pair:
            assert node.config == {}
            assert node._credential is None

    def testAnUnsignedEventIsNotFromTheProvider(self, agentManager):
        with pytest.raises(providerEvents.NotFromProviderError):
            asyncio.run(
                providerEvents.receiveEvent(
                    PROJECT_ID, AGENT_ID, "whatsapp", buildEvent(), {}
                )
            )
