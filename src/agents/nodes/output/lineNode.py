"""The LINE output node: where an agent's answer goes out, sent over LINE
in reply to the message it answers.

    node = LineOutputNode(nodeId, "line")
    node.configure(credential, config)
    sentIds = await node.send(message, answer)

Its pair is the LINE input node of `agents.nodes.input.lineNode`, which
reads the messages in, each with its `replyToken` and `provider`. `send`
reads both: the answer to a message of PROVIDER is sent as the reply to
it - LINE's reply API, free of the account's monthly message quota - in
one call, as a token is good for one reply only, of MAX_MESSAGES text
messages of MAX_TEXT_LENGTH at most; an answer longer still is cut, and
logged. One to a message of another provider is refused, and nothing is
sent. Both are io nodes, built by `providerEvents` from their template
node's merged config.

A reply token is good for about a minute after the message came in, so
an answer that takes longer is refused by LINE, and logged.

Its credential holds SECRET_KEYS: the `channel_access_token` it sends
with - for now kept in its node's config, and moved into its credential
as it is built, `secretKeys` naming it. Its config may hold the
Messaging API's `api_url`. A node lacking the token is refused as it is
configured.
"""

import logging
from typing import ClassVar

import httpx

from agents.nodes.credential import Credential
from agents.nodes.input.lineNode import PROVIDER, LineMessage
from agents.nodes.nodeBase import NodeBase

LOGGER = logging.getLogger(__name__)

# The values of a LINE output node's credential.
SECRET_KEYS = ("channel_access_token",)
# Where an answer is sent, unless the config says otherwise.
DEFAULT_API_URL = "https://api.line.me"
REPLY_PATH = "/v2/bot/message/reply"
# The longest text a LINE message may have, and the most messages a
# reply may hold.
MAX_TEXT_LENGTH = 5000
MAX_MESSAGES = 5
SEND_SECONDS = 30


class LineError(RuntimeError):
    """LINE refused an answer, or could not be reached, or the answer is
    to a message that did not come in over LINE."""


class LineOutputNode(NodeBase):
    """Sends an agent's answers over LINE: where they go out."""

    handler = "line_output"
    requiresCredential = True
    description = "Sends the answers to LINE messages."
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

    async def send(self, message: LineMessage, answer: str) -> list[str]:
        """Send `answer` as the reply to `message`, a message of
        PROVIDER, in MAX_MESSAGES messages of MAX_TEXT_LENGTH at most;
        the ids LINE gave them.

        Raises:
            LineError: If `message` came in over another provider, and
                nothing is sent; or LINE refused the reply - its token
                used or too old, say - or could not be reached.
        """
        if message.provider != PROVIDER:
            raise LineError(
                f"Message {message.messageId!r} came in over "
                f"{message.provider!r}, not {PROVIDER!r}; nothing is sent."
            )
        parts = [
            answer[start : start + MAX_TEXT_LENGTH]
            for start in range(0, len(answer), MAX_TEXT_LENGTH)
        ] or [""]
        if len(parts) > MAX_MESSAGES:
            LOGGER.warning(
                "The answer to LINE message %s is %s characters; only its "
                "first %s are sent.",
                message.messageId,
                len(answer),
                MAX_MESSAGES * MAX_TEXT_LENGTH,
            )
            parts = parts[:MAX_MESSAGES]
        url = self.config.get("api_url", DEFAULT_API_URL) + REPLY_PATH
        async with httpx.AsyncClient(timeout=SEND_SECONDS) as client:
            try:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": "Bearer "
                        + self._credential.reveal("channel_access_token")
                    },
                    json={
                        "replyToken": message.replyToken,
                        "messages": [
                            {"type": "text", "text": part} for part in parts
                        ],
                    },
                )
            except httpx.HTTPError as error:
                raise LineError(
                    f"LINE could not be reached: {type(error).__name__}."
                ) from None
        if response.status_code != 200:
            raise LineError(
                f"LINE refused a reply to {message.sender}: "
                f"{response.status_code} {response.text[:300]}"
            )
        sentIds = [
            sent["id"] for sent in response.json().get("sentMessages", [])
        ]
        LOGGER.info(
            "Sent %s LINE messages to %s: %s",
            len(sentIds),
            message.sender,
            sentIds,
        )
        return sentIds

    async def __call__(self, state: dict) -> dict:
        """No graph runs an io node: its template's edges leave it out.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "A LINE output node sends through the hooks route; no graph "
            "runs it."
        )
