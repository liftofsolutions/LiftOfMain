"""The LINE input node: where a LINE message comes in to an agent,
through a LINE Official Account's webhook on the LINE Messaging API.

    node = LineInputNode(nodeId, "line")
    node.configure(credential, config)
    messages = node.entryPoint(body, headers)    # POST of the hook

Its pair, the LINE output node of `agents.nodes.output.lineNode`, sends
the agent's answers back. Both are io nodes: no node of an agent's graph
- a template's edges leave them out - and reserved by no agent. A
trigger points to their template node, and `providerEvents` builds both
from its merged config, each with the secrets it uses.

`entryPoint` checks an event came from LINE, by its `X-Line-Signature`,
the base64 HMAC-SHA256 of the body under the channel's secret, and reads
the text messages a user sent the account in a one-to-one chat out of
it - not a group's or a room's, nor a follow, a sticker or an image; an
event not signed so is None. Each message keeps its `replyToken`, which
the output node answers with, and is given its `provider`, PROVIDER,
which the output node reads to send its answer.

LINE checks the webhook URL with an event of no messages, signed as any
other - the console's "Verify" button: it reads as no messages, and is
taken as any event is. No check of its own is needed.

Its credential holds SECRET_KEYS: the `channel_secret` events are signed
with - for now kept in its node's config, and moved into its credential
as it is built, `secretKeys` naming it, so it is never left in the
config. A node lacking it is refused as it is configured. One channel is
one account, so the signature names the account: the config needs
nothing more.
"""

import base64
import hashlib
import hmac
import json
from collections.abc import Mapping
from typing import ClassVar, NamedTuple

from agents.nodes.credential import Credential
from agents.nodes.nodeBase import NodeBase

# The provider a message that comes in here came in over.
PROVIDER = "line"
# The values a LINE input node's credential must hold.
SECRET_KEYS = ("channel_secret",)
# LINE's header with an event's signature, as a header mapping that
# ignores case has it.
SIGNATURE_HEADER = "x-line-signature"
# The only chat a message is answered in: a user's own with the account.
USER_SOURCE = "user"


class LineMessage(NamedTuple):
    """A text message a user sent the account: who sent it - their LINE
    user id - its text, its id, the token its answer is sent with, and
    the provider it came in over."""

    sender: str
    text: str
    messageId: str
    replyToken: str
    provider: str = PROVIDER


class LineInputNode(NodeBase):
    """Handles a LINE Official Account's webhook: where its messages come
    in."""

    handler = "line"
    requiresCredential = True
    description = "Receives LINE messages."
    # The keys of its config that are secrets, kept in its credential.
    secretKeys: ClassVar[tuple[str, ...]] = SECRET_KEYS

    def configure(
        self,
        credential: Credential | None = None,
        config: dict | None = None,
    ) -> None:
        """Take the credential and config.

        Raises:
            ValueError: If the credential lacks one of SECRET_KEYS; it
                names no value.
        """
        super().configure(credential, config)
        for key in SECRET_KEYS:
            try:
                self._credential.reveal(key)
            except (AttributeError, KeyError):
                raise ValueError(
                    f"LINE node {self.name!r} has no {key}."
                ) from None

    def entryPoint(
        self, body: bytes, headers: Mapping[str, str]
    ) -> list[LineMessage] | None:
        """The text messages users sent the account in a LINE event, in
        order, each of PROVIDER; None if the event is not signed by LINE,
        with the channel's secret, or is not JSON."""
        if not self._isSigned(body, headers.get(SIGNATURE_HEADER)):
            return None
        try:
            event = json.loads(body)
        except ValueError:
            return None
        messages = []
        for item in event.get("events", []):
            source = item.get("source", {})
            message = item.get("message", {})
            text = message.get("text")
            if (
                item.get("type") == "message"
                and message.get("type") == "text"
                and source.get("type") == USER_SOURCE
                and source.get("userId")
                and item.get("replyToken")
                and text
            ):
                messages.append(
                    LineMessage(
                        source["userId"],
                        text,
                        message.get("id", ""),
                        item["replyToken"],
                        PROVIDER,
                    )
                )
        return messages

    def _isSigned(self, body: bytes, signature: str | None) -> bool:
        """Whether `signature` is the body's, signed with the channel's
        secret."""
        if not signature:
            return False
        expected = base64.b64encode(
            hmac.new(
                self._credential.reveal("channel_secret").encode(),
                body,
                hashlib.sha256,
            ).digest()
        ).decode()
        return hmac.compare_digest(signature, expected)

    async def __call__(self, state: dict) -> dict:
        """No graph runs an io node: its template's edges leave it out.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "A LINE input node handles its webhook through the hooks "
            "route; no graph runs it."
        )
