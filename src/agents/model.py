"""Request and response models for the agent endpoints."""

from uuid import UUID

from pydantic import BaseModel, Field


class ConfigureAgentRequest(BaseModel):
    """The new agent's name, and the template it runs."""

    name: str = Field(min_length=1, max_length=120)
    templateId: UUID = Field(
        description=(
            "A template of the project, or a premade platform template, "
            "which is copied into the project."
        )
    )


class ConfigureAgentResponse(BaseModel):
    """The agent made: its id, and the template it runs."""

    agentId: str
    projectId: str
    templateId: str
    name: str
    status: str


class ConfigureRagRequest(BaseModel):
    """The rag of the project the agent searches."""

    ragId: UUID = Field(description="A rag of the project, ready.")


class ConfigureRagResponse(BaseModel):
    """The agent, and the rag it searches."""

    agentId: str
    ragId: str


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
