"""Agents for public use: the answer an agent gives to a message, in a
conversation, if it names one.

Anyone may ask: no account is checked, nor the project's own record -
only that the agent is of the project named, which the `AgentManager`
checks as it hands the agent over, building it first if it is not kept.
A message may name its conversation by a `convoId`, and the agent keeps
the conversation's chat history between its answers; one that names
none is of no conversation. An agent the project does not have is
answered as not found, one that cannot run - paused, say - as a
conflict. Errors are returned, never raised: `{"error": message,
"errorKind": kind}`, logged, with `errorKind` one of the `ERROR_*`
constants for the route to turn into a status code.
"""

import logging

from agents.agent import AgentNotFoundError, AgentNotRunnableError
from agents.agentManager import getAgentManager

LOGGER = logging.getLogger(__name__)

# The error kinds the route turns into status codes.
ERROR_NOT_FOUND = "notFound"
ERROR_CONFLICT = "conflict"


def errorResult(errorKind: str, message: str) -> dict:
    """An error to hand back to the route, logged first."""
    LOGGER.error(message)
    return {"error": message, "errorKind": errorKind}


async def queryAgent(
    projectId: str, agentId: str, message: str, convoId: str | None = None
) -> dict:
    """The agent's answer to `message`, in the conversation `convoId` if
    one is named, and that id, for anyone who asks."""
    try:
        reply = await getAgentManager().query(
            projectId, agentId, message, convoId
        )
    except AgentNotFoundError as error:
        return errorResult(ERROR_NOT_FOUND, str(error))
    except AgentNotRunnableError as error:
        return errorResult(ERROR_CONFLICT, str(error))
    return {
        "agentId": agentId,
        "convoId": reply.convoId,
        "answer": reply.answer,
    }
