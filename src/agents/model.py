"""Request and response models for the agent endpoint."""

from uuid import UUID

from pydantic import BaseModel, Field


class QueryAgentRequest(BaseModel):
    """What to ask the agent, and in which conversation, if any."""

    message: str = Field(min_length=1, description="The message to answer.")
    convoId: UUID | None = Field(
        default=None,
        description=(
            "The conversation to go on with, or start; without one, the "
            "message is of no conversation."
        ),
    )


class QueryAgentResponse(BaseModel):
    """The agent's answer, and the conversation it was given in."""

    agentId: str
    convoId: str | None
    answer: str
