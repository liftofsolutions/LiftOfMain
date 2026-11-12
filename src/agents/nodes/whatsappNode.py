"""The WhatsApp node: an `io` node, which handles a WhatsApp number's
webhook on Meta's WhatsApp Cloud API - its check of the callback URL,
the events it sends - and sends the agent's replies.

    node = WhatsAppNode(nodeId, "whatsapp")
    node.configure(credential, config)
    challenge = node.verifyCallback(queryParameters)    # GET of the hook
    messages = node.receive(body, headers)              # POST of the hook
    await node.reply(message.sender, answer)

An io node is no node of an agent's graph: a template's edges leave it
out, and no agent reserves it. It is what a trigger points to: the hooks
route, given a call for a trigger of an agent, builds the trigger's node
with its merged config and hands it the call; all it knows of WhatsApp
is the node's. The node

* `verifyCallback` - answers Meta's check of the callback URL with its
  challenge, given `hub.mode=subscribe` and the node's `verify_token`;
  else nothing;
* `receive` - checks an event came from Meta, by its
  `X-Hub-Signature-256`, the HMAC-SHA256 of the body under the app's
  secret, and reads the text messages sent to its number out of it -
  not those to another number, nor a status update or an image; an
  event not signed so is None;
* `reply` - sends a reply to a sender, cut into messages of
  MAX_TEXT_LENGTH at most.

Its credential holds SECRET_KEYS: the `access_token` it sends with, the
`app_secret` events are signed with, and the `verify_token` Meta checks
the callback URL with - for now kept in its node's config, and moved
into its credential as it is built, `secretKeys` naming them. Its
config holds the `phone_number_id` it receives and sends as, and,
optionally, the Graph API's `api_version` and `graph_api_url`. A node
lacking any is refused as it is configured.
"""

import hashlib
import hmac
import json
import logging
from collections.abc import Mapping
from typing import ClassVar, NamedTuple

import httpx

from agents.nodes.credential import Credential
from agents.nodes.nodeBase import NodeBase

LOGGER = logging.getLogger(__name__)

# The values of a WhatsApp node's credential.
SECRET_KEYS = ("access_token", "app_secret", "verify_token")
# Meta's header with an event's signature, as a header mapping that
# ignores case has it, and what starts it.
SIGNATURE_HEADER = "x-hub-signature-256"
SIGNATURE_PREFIX = "sha256="
# The query parameters of Meta's check of the callback URL.
VERIFY_MODE = "hub.mode"
VERIFY_TOKEN = "hub.verify_token"
VERIFY_CHALLENGE = "hub.challenge"
SUBSCRIBE = "subscribe"
# Where a reply is sent, unless the config says otherwise.
DEFAULT_GRAPH_API_URL = "https://graph.facebook.com"
DEFAULT_API_VERSION = "v21.0"
# The longest text a WhatsApp message may have.
MAX_TEXT_LENGTH = 4096
SEND_SECONDS = 30


class WhatsAppError(RuntimeError):
    """WhatsApp refused a reply, or could not be reached."""


class WhatsAppMessage(NamedTuple):
    """A text message sent to the node's number: who sent it - their
    WhatsApp id, a phone number - its text, and its id."""

    sender: str
    text: str
    messageId: str


class WhatsAppNode(NodeBase):
    """Handles a WhatsApp number's webhook, and sends its replies."""

    handler = "whatsapp"
    requiresCredential = True
    description = "Receives WhatsApp messages, and sends the replies."
    # The keys of its config that are secrets, kept in its credential.
    secretKeys: ClassVar[tuple[str, ...]] = SECRET_KEYS

    def configure(
        self,
        credential: Credential | None = None,
        config: dict | None = None,
    ) -> None:
        """Take the credential and config.

        Raises:
            ValueError: If the credential lacks one of SECRET_KEYS, or the
                config a `phone_number_id`; neither names a value.
        """
        super().configure(credential, config)
        for key in SECRET_KEYS:
            try:
                self._credential.reveal(key)
            except (AttributeError, KeyError):
                raise ValueError(
                    f"WhatsApp node {self.name!r} has no {key}."
                ) from None
        if not self.config.get("phone_number_id"):
            raise ValueError(
                f"WhatsApp node {self.name!r} has no phone_number_id."
            )

    def verifyCallback(self, query: Mapping[str, str]) -> str | None:
        """The challenge to answer Meta's check of the callback URL with,
        given its query parameters - if it subscribes, with the node's
        verify token; else None."""
        token = query.get(VERIFY_TOKEN)
        if query.get(VERIFY_MODE) != SUBSCRIBE or token is None:
            return None
        expected = self._credential.reveal("verify_token")
        if not hmac.compare_digest(token.encode(), expected.encode()):
            return None
        return query.get(VERIFY_CHALLENGE, "")

    def receive(
        self, body: bytes, headers: Mapping[str, str]
    ) -> list[WhatsAppMessage] | None:
        """The text messages of a WhatsApp event sent to the node's
        number, in order; None if the event is not signed by Meta, with
        the app's secret, or is not JSON."""
        if not self._isSigned(body, headers.get(SIGNATURE_HEADER)):
            return None
        try:
            event = json.loads(body)
        except ValueError:
            return None
        messages = []
        for entry in event.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                phoneNumberId = value.get("metadata", {}).get(
                    "phone_number_id"
                )
                if phoneNumberId != self.config["phone_number_id"]:
                    continue
                for message in value.get("messages", []):
                    text = message.get("text", {}).get("body")
                    if message.get("type") == "text" and text:
                        messages.append(
                            WhatsAppMessage(
                                message["from"], text, message.get("id", "")
                            )
                        )
        return messages

    def _isSigned(self, body: bytes, signature: str | None) -> bool:
        """Whether `signature` is the body's, signed with the app's
        secret."""
        if not signature or not signature.startswith(SIGNATURE_PREFIX):
            return False
        expected = hmac.new(
            self._credential.reveal("app_secret").encode(),
            body,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(
            signature.removeprefix(SIGNATURE_PREFIX), expected
        )

    async def reply(self, recipient: str, text: str) -> list[str]:
        """Send `text` to `recipient`, in messages of MAX_TEXT_LENGTH at
        most; the ids WhatsApp gave them.

        Raises:
            WhatsAppError: If WhatsApp refused a message, or could not be
                reached; the messages before it are sent.
        """
        url = "{base}/{apiVersion}/{phoneNumberId}/messages".format(
            base=self.config.get("graph_api_url", DEFAULT_GRAPH_API_URL),
            apiVersion=self.config.get("api_version", DEFAULT_API_VERSION),
            phoneNumberId=self.config["phone_number_id"],
        )
        parts = [
            text[start : start + MAX_TEXT_LENGTH]
            for start in range(0, len(text), MAX_TEXT_LENGTH)
        ] or [""]
        sentIds = []
        async with httpx.AsyncClient(timeout=SEND_SECONDS) as client:
            for part in parts:
                try:
                    response = await client.post(
                        url,
                        headers={
                            "Authorization": "Bearer "
                            + self._credential.reveal("access_token")
                        },
                        json={
                            "messaging_product": "whatsapp",
                            "recipient_type": "individual",
                            "to": recipient,
                            "type": "text",
                            "text": {"body": part},
                        },
                    )
                except httpx.HTTPError as error:
                    raise WhatsAppError(
                        "WhatsApp could not be reached: "
                        f"{type(error).__name__}."
                    ) from None
                if response.status_code != 200:
                    raise WhatsAppError(
                        f"WhatsApp refused a reply to {recipient}: "
                        f"{response.status_code} {response.text[:300]}"
                    )
                sentIds.extend(
                    message["id"]
                    for message in response.json().get("messages", [])
                )
        LOGGER.info(
            "Sent %s WhatsApp messages to %s: %s",
            len(sentIds),
            recipient,
            sentIds,
        )
        return sentIds

    async def __call__(self, state: dict) -> dict:
        """No graph runs an io node: its template's edges leave it out.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "A WhatsApp node handles its webhook through the hooks route; "
            "no graph runs it."
        )
