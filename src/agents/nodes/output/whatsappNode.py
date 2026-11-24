"""The WhatsApp output node: where an agent's answer goes out, sent over
WhatsApp to the sender of the message it answers.

    node = WhatsAppOutputNode(nodeId, "whatsapp")
    node.configure(credential, config)
    sentIds = await node.send(message, answer)

Its pair is the WhatsApp input node of `agents.nodes.input.whatsappNode`,
which reads the messages in, each with its `provider`. `send` reads it:
the answer to a message of PROVIDER is sent, in messages of
MAX_TEXT_LENGTH at most; one to a message of another provider is
refused, and nothing is sent. Both are io nodes, built by
`providerEvents` from their template node's merged config.

Its credential holds SECRET_KEYS: the `access_token` it sends with - for
now kept in its node's config, and moved into its credential as it is
built, `secretKeys` naming it. Its config holds the `phone_number_id` it
sends as, and, optionally, the Graph API's `api_version` and
`graph_api_url`. A node lacking any is refused as it is configured.
"""

import logging
from typing import ClassVar

import httpx

from agents.nodes.credential import Credential
from agents.nodes.input.whatsappNode import PROVIDER, WhatsAppMessage
from agents.nodes.nodeBase import NodeBase

LOGGER = logging.getLogger(__name__)

# The values of a WhatsApp output node's credential.
SECRET_KEYS = ("access_token",)
# Where an answer is sent, unless the config says otherwise.
DEFAULT_GRAPH_API_URL = "https://graph.facebook.com"
DEFAULT_API_VERSION = "v21.0"
# The longest text a WhatsApp message may have.
MAX_TEXT_LENGTH = 4096
SEND_SECONDS = 30


class WhatsAppError(RuntimeError):
    """WhatsApp refused an answer, or could not be reached, or the answer
    is to a message that did not come in over WhatsApp."""


class WhatsAppOutputNode(NodeBase):
    """Sends an agent's answers over WhatsApp: where they go out."""

    handler = "whatsapp_output"
    requiresCredential = True
    description = "Sends the answers to WhatsApp messages."
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

    async def send(self, message: WhatsAppMessage, answer: str) -> list[str]:
        """Send `answer` to the sender of `message`, a message of
        PROVIDER, in messages of MAX_TEXT_LENGTH at most; the ids
        WhatsApp gave them.

        Raises:
            WhatsAppError: If `message` came in over another provider,
                and nothing is sent; or WhatsApp refused a message, or
                could not be reached, the messages before it sent.
        """
        if message.provider != PROVIDER:
            raise WhatsAppError(
                f"Message {message.messageId!r} came in over "
                f"{message.provider!r}, not {PROVIDER!r}; nothing is sent."
            )
        recipient = message.sender
        url = "{base}/{apiVersion}/{phoneNumberId}/messages".format(
            base=self.config.get("graph_api_url", DEFAULT_GRAPH_API_URL),
            apiVersion=self.config.get("api_version", DEFAULT_API_VERSION),
            phoneNumberId=self.config["phone_number_id"],
        )
        parts = [
            answer[start : start + MAX_TEXT_LENGTH]
            for start in range(0, len(answer), MAX_TEXT_LENGTH)
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
                    sent["id"]
                    for sent in response.json().get("messages", [])
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
            "A WhatsApp output node sends through the hooks route; no "
            "graph runs it."
        )
