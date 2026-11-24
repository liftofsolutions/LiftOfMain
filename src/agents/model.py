"""Request and response models for the agent endpoints."""

from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class AgentNodeRequest(BaseModel):
    """A node of the template, and the credential it is formed with."""

    nodeId: str = Field(
        min_length=1, description="The id of a node of the template."
    )
    credentialId: UUID = Field(
        description="A credential of the project, for that node."
    )


class ConfigureAgentRequest(BaseModel):
    """The new agent's name and description, the template it runs, and
    the credentials its nodes are formed with."""

    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(
        default=None, max_length=500, description="What the agent is for."
    )
    templateId: UUID = Field(
        description=(
            "A template of the project, or a premade platform template, "
            "which is copied into the project."
        )
    )
    nodes: list[AgentNodeRequest] = Field(
        default_factory=list,
        description=(
            "The template's nodes that take a credential, each once. "
            "With any, the agent runs its own copy of the template."
        ),
    )

    @model_validator(mode="after")
    def checkEachNodeIsNamedOnce(self) -> "ConfigureAgentRequest":
        """Refuse a node given two credentials."""
        nodeIds = [node.nodeId for node in self.nodes]
        if len(nodeIds) != len(set(nodeIds)):
            raise ValueError("nodes names a node more than once.")
        return self


class AgentNodeResponse(BaseModel):
    """A node of the agent's template, and its credential, if any."""

    nodeId: str
    name: str
    credentialId: str | None


class ConfigureAgentResponse(BaseModel):
    """The agent made: its id, the template it runs, and that template's
    nodes."""

    agentId: str
    projectId: str
    templateId: str
    name: str
    description: str | None
    status: str
    nodes: list[AgentNodeResponse]


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


class ProviderEventResponse(BaseModel):
    """A provider's event, taken: its messages are answered in the
    background, and the answers sent back through the provider."""

    status: str
    messages: int


class QueryAgentResponse(BaseModel):
    """The agent's answer, and the conversation it was given in."""

    agentId: str
    convoId: str | None
    answer: str
