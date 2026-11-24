"""The WhatsApp input node: where a WhatsApp message comes in to an
agent, through a WhatsApp number's webhook on Meta's WhatsApp Cloud API.

    node = WhatsAppInputNode(nodeId, "whatsapp")
    node.configure(credential, config)
    messages = node.entryPoint(body, headers)    # POST of the hook

Its pair, the WhatsApp output node of `agents.nodes.output.whatsappNode`,
sends the agent's answers back. Both are io nodes: no node of an agent's
graph - a template's edges leave them out - and reserved by no agent. A
trigger points to their template node, and `providerEvents` builds both
from its merged config, each with the secrets it uses.

`entryPoint` checks an event came from Meta, by its
`X-Hub-Signature-256`, the HMAC-SHA256 of the body under the app's
secret, and reads the text messages sent to its number out of it - not
those to another number, nor a status update or an image; an event not
signed so is None. Each message is given its `provider`, PROVIDER, which
the output node reads to send its answer.

Meta's check of the callback URL - the GET with `hub.challenge` - is not
answered here, nor anywhere for now: it is to be built apart, GitHub
issue #15.

Its credential holds SECRET_KEYS: the `app_secret` events are signed
with - for now kept in its node's config, and moved into its credential
as it is built, `secretKeys` naming it, and the `verify_token` of that
check with it, so it is never left in the config. Its config holds the
`phone_number_id` it receives as. A node lacking either is refused as it
is configured.
"""

import hashlib
import hmac
import json
from collections.abc import Mapping
from typing import ClassVar, NamedTuple

from agents.nodes.credential import Credential
from agents.nodes.nodeBase import NodeBase

# The provider a message that comes in here came in over.
PROVIDER = "whatsapp"
# The values a WhatsApp input node's credential must hold.
SECRET_KEYS = ("app_secret",)
# The secrets of its config, moved into its credential: the callback
# check's `verify_token` too, unread until issue #15 brings the check
# back, so it is never left in the config.
CONFIG_SECRET_KEYS = (*SECRET_KEYS, "verify_token")
# Meta's header with an event's signature, as a header mapping that
# ignores case has it, and what starts it.
SIGNATURE_HEADER = "x-hub-signature-256"
SIGNATURE_PREFIX = "sha256="


class WhatsAppMessage(NamedTuple):
    """A text message sent to the node's number: who sent it - their
    WhatsApp id, a phone number - its text, its id, and the provider it
    came in over."""

    sender: str
    text: str
    messageId: str
    provider: str = PROVIDER


class WhatsAppInputNode(NodeBase):
    """Handles a WhatsApp number's webhook: where its messages come in."""

    handler = "whatsapp"
    requiresCredential = True
    description = "Receives WhatsApp messages."
    # The keys of its config that are secrets, kept in its credential.
    secretKeys: ClassVar[tuple[str, ...]] = CONFIG_SECRET_KEYS

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

    def entryPoint(
        self, body: bytes, headers: Mapping[str, str]
    ) -> list[WhatsAppMessage] | None:
        """The text messages of a WhatsApp event sent to the node's
        number, in order, each of PROVIDER; None if the event is not
        signed by Meta, with the app's secret, or is not JSON."""
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
                                message["from"],
                                text,
                                message.get("id", ""),
                                PROVIDER,
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

    async def __call__(self, state: dict) -> dict:
        """No graph runs an io node: its template's edges leave it out.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "A WhatsApp input node handles its webhook through the hooks "
            "route; no graph runs it."
        )
