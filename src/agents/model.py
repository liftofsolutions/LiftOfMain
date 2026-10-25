"""Request and response models for the agent endpoint."""

from pydantic import BaseModel, Field


class QueryAgentRequest(BaseModel):
    """What to ask the agent."""

    message: str = Field(min_length=1, description="The message to answer.")


class QueryAgentResponse(BaseModel):
    """The agent's answer."""

    agentId: str
    answer: str
