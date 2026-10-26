"""Btest for configuring agents: a member of a project makes an agent of
it, running a template, and gives the agent a rag of its own.

Firestore is the `FakeFirestore` of `AgentManagerTest`, holding its
healthy agent and template, and a member account of the project, its
project and its membership - what `checkAccess` reads - and a ready rag.
The service is called as the routes call it, on the manager the service
hands its calls to; everything between is real: `checkAccess`,
`agentStore`, the agent manager, the nodes and LangGraph.

1. Configuring an agent: a member makes an agent of the project, active,
   running one of its templates, saved to Firestore and memoryDb, which
   then answers queries. A premade platform template is copied into the
   project, and the agent runs the copy, saved with it at once. An
   account that is not a member, of a project that is not there, is
   refused, as is a template of another project, missing or inactive;
   nothing is saved.
2. Configuring a rag: the agent's record gets the rag, ready and of
   the project, as its knowledge base, saved and cached, and its
   template is left as it was; the agent, kept, is dropped and its nodes
   released, so the next query builds it afresh, its node searching the
   new rag. Another agent of the same template keeps its own rag, and
   stays kept. A refusal - no member, no such agent, a rag not there or
   not ready - saves nothing and drops nothing.

    uv run pytest Btest/agents -q
"""

import asyncio

import pytest
from AgentManagerTest import (  # noqa: F401
    ACCOUNT_ID,
    AGENT_ID,
    ANSWER,
    DELETED_AT,
    MISSING_ID,
    OTHER_PROJECT_ID,
    PROJECT_ID,
    QUESTION,
    TEMPLATE_ID,
    agentManager,
    buildAgent,
    buildReplyNode,
    buildTemplate,
    builtModels,
    firestore,
    memoryDb,
    serviceManager,
)

from agents import service
from agents.agentStore import AGENT_TEMPLATES, AGENTS
from agents.model import ConfigureAgentResponse, ConfigureRagResponse
from documentProcessing import utilities
from documentProcessing.ragStrategy import RAG_STATUS_READY

RAG_ID = "3700255f-a07c-4820-86eb-ba61fb732f77"
OLD_RAG_ID = "9c1d4e7a-2b3f-4a6e-8d5c-1f0e9b8a7c62"
PLATFORM_TEMPLATE_ID = "e7b2c9d4-5a1f-4c8e-b3d6-0a9f8e7d6c51"
OTHER_AGENT_ID = "1e6b9f3a-7c2d-4a8e-b5f1-0d3c9e8a2b47"
AGENT_NAME = "Acme support"


def buildMembership() -> list[tuple[str, dict]]:
    """The account, its project, and its membership of the project, each
    with its collection."""
    return [
        (utilities.USERS, {"_id": ACCOUNT_ID}),
        (utilities.PROJECTS, {"_id": PROJECT_ID}),
        (
            utilities.PROJECT_MEMBERS,
            {"_id": f"{PROJECT_ID}_{ACCOUNT_ID}", "user_id": ACCOUNT_ID},
        ),
    ]


def buildRag(**overrides) -> dict:
    """A rag of the project, ready."""
    rag = {
        "_id": RAG_ID,
        "project_id": PROJECT_ID,
        "status": RAG_STATUS_READY,
    }
    rag.update(overrides)
    return rag


def buildSearchingTemplate() -> dict:
    """The project's template, its reply node searching the agent's
    rag."""
    searchingNode = buildReplyNode(
        config={"system_prompt": "You are reply.", "rag_search": True}
    )
    return buildTemplate(nodes=[searchingNode])


@pytest.fixture
def members(firestore) -> None:
    """The member account of the project, and the project's rag."""
    for collection, record in buildMembership():
        firestore.put(collection, record)
    firestore.put(utilities.RAGS, buildRag())


def configureAgent(templateId: str = TEMPLATE_ID, **ids) -> dict:
    """The service configuring an agent named AGENT_NAME, running
    `templateId`, for the account and project `ids` say, else the
    member's."""
    return asyncio.run(
        service.configureAgent(
            ids.get("accountUuid", ACCOUNT_ID),
            ids.get("projectId", PROJECT_ID),
            AGENT_NAME,
            templateId,
        )
    )


def configureRag(ragId: str = RAG_ID, **ids) -> dict:
    """The service giving the agent's template `ragId`, for the account,
    project and agent `ids` say, else the member's and AGENT_ID."""
    return asyncio.run(
        service.configureRag(
            ids.get("accountUuid", ACCOUNT_ID),
            ids.get("projectId", PROJECT_ID),
            ids.get("agentId", AGENT_ID),
            ragId,
        )
    )


# An account kept out of the project, and the error kind it gets.
ACCESS_CASES = {
    "noSuchAccount": ({"accountUuid": MISSING_ID}, service.ERROR_NOT_FOUND),
    "noSuchProject": ({"projectId": MISSING_ID}, service.ERROR_NOT_FOUND),
    "notAMember": ({"projectId": OTHER_PROJECT_ID}, service.ERROR_FORBIDDEN),
}


class TestConfigureAgent:
    """A member making an agent of the project, running a template."""

    def testMemberMakesAnAgentThatAnswers(
        self, serviceManager, firestore, memoryDb, members
    ):
        configured = configureAgent()

        ConfigureAgentResponse(**configured)
        agentId = configured["agentId"]
        assert configured == {
            "agentId": agentId,
            "projectId": PROJECT_ID,
            "templateId": TEMPLATE_ID,
            "name": AGENT_NAME,
            "status": "active",
        }
        stored = firestore.records[AGENTS][agentId]
        assert {
            key: stored[key]
            for key in ("project_id", "template_id", "name", "status")
        } == {
            "project_id": PROJECT_ID,
            "template_id": TEMPLATE_ID,
            "name": AGENT_NAME,
            "status": "active",
        }
        assert stored["created_by"] == ACCOUNT_ID
        assert firestore.batches == [[(AGENTS, agentId)]]
        assert memoryDb.getRecord(AGENTS, agentId) == stored
        answered = asyncio.run(
            service.queryAgent(PROJECT_ID, agentId, QUESTION)
        )
        assert answered["answer"] == ANSWER

    def testEachAgentHasItsOwnId(self, serviceManager, members):
        agentIds = {configureAgent()["agentId"] for _ in range(3)}

        assert len(agentIds) == 3
        assert AGENT_ID not in agentIds

    def testPremadeTemplateIsCopiedIntoTheProject(
        self, serviceManager, firestore, members
    ):
        premade = buildTemplate(_id=PLATFORM_TEMPLATE_ID, project_id=None)
        firestore.put(AGENT_TEMPLATES, premade)

        configured = configureAgent(PLATFORM_TEMPLATE_ID)

        copyId = configured["templateId"]
        assert copyId not in (PLATFORM_TEMPLATE_ID, TEMPLATE_ID)
        copied = firestore.records[AGENT_TEMPLATES][copyId]
        assert copied["project_id"] == PROJECT_ID
        assert copied["nodes"] == premade["nodes"]
        assert copied["edges"] == premade["edges"]
        assert firestore.records[AGENT_TEMPLATES][PLATFORM_TEMPLATE_ID] == (
            premade
        )
        agentId = configured["agentId"]
        assert firestore.records[AGENTS][agentId]["template_id"] == copyId
        assert firestore.batches == [
            [(AGENT_TEMPLATES, copyId), (AGENTS, agentId)]
        ]
        answered = asyncio.run(
            service.queryAgent(PROJECT_ID, agentId, QUESTION)
        )
        assert answered["answer"] == ANSWER

    @pytest.mark.parametrize(
        ("ids", "errorKind"), ACCESS_CASES.values(), ids=ACCESS_CASES.keys()
    )
    def testAccountKeptOutOfTheProjectIsRefused(
        self, serviceManager, firestore, members, ids, errorKind
    ):
        firestore.put(utilities.PROJECTS, {"_id": OTHER_PROJECT_ID})

        refused = configureAgent(**ids)

        assert refused["errorKind"] == errorKind
        assert firestore.batches == []

    def testDeletedProjectIsNotFound(self, serviceManager, firestore, members):
        firestore.put(
            utilities.PROJECTS, {"_id": PROJECT_ID, "deleted_at": DELETED_AT}
        )

        refused = configureAgent()

        assert refused["errorKind"] == service.ERROR_NOT_FOUND
        assert firestore.batches == []

    @pytest.mark.parametrize(
        ("template", "errorKind", "message"),
        [
            (
                buildTemplate(project_id=OTHER_PROJECT_ID),
                service.ERROR_NOT_FOUND,
                "has no template",
            ),
            (None, service.ERROR_NOT_FOUND, "has no template"),
            (
                buildTemplate(is_active=False),
                service.ERROR_CONFLICT,
                "is inactive",
            ),
        ],
        ids=["anotherProjects", "missing", "inactive"],
    )
    def testTemplateThatCannotBeRunIsRefused(
        self, serviceManager, firestore, members, template, errorKind, message
    ):
        if template is None:
            del firestore.records[AGENT_TEMPLATES][TEMPLATE_ID]
        else:
            firestore.put(AGENT_TEMPLATES, template)

        refused = configureAgent()

        assert refused["errorKind"] == errorKind
        assert message in refused["error"]
        assert firestore.batches == []


class TestConfigureRag:
    """A member giving an agent a rag of its own."""

    def testAgentGetsTheRagAndItsTemplateIsLeftAsItWas(
        self, serviceManager, firestore, memoryDb, members
    ):
        configured = configureRag()

        ConfigureRagResponse(**configured)
        assert configured == {"agentId": AGENT_ID, "ragId": RAG_ID}
        stored = firestore.records[AGENTS][AGENT_ID]
        assert stored["rag_enabled"] is True
        assert stored["rag_id"] == RAG_ID
        assert stored["template_id"] == TEMPLATE_ID
        assert firestore.batches == [[(AGENTS, AGENT_ID)]]
        assert memoryDb.getRecord(AGENTS, AGENT_ID) == stored
        template = firestore.records[AGENT_TEMPLATES][TEMPLATE_ID]
        assert template == buildTemplate()

    def testKeptAgentIsBuiltAfreshSearchingTheRag(
        self, serviceManager, firestore, members
    ):
        firestore.put(AGENT_TEMPLATES, buildSearchingTemplate())
        firestore.put(AGENTS, buildAgent(rag_enabled=True, rag_id=OLD_RAG_ID))
        firestore.put(
            AGENTS,
            buildAgent(_id=OTHER_AGENT_ID, rag_enabled=True, rag_id=OLD_RAG_ID),
        )
        before = asyncio.run(serviceManager.getAgent(PROJECT_ID, AGENT_ID))
        [node] = before.nodes
        assert node.config["rag"]["rag_id"] == OLD_RAG_ID
        other = asyncio.run(
            serviceManager.getAgent(PROJECT_ID, OTHER_AGENT_ID)
        )

        configureRag()

        assert list(serviceManager.agents) == [OTHER_AGENT_ID]
        assert before.nodes == []
        assert node not in serviceManager.nodeManager.reservedNodes
        after = asyncio.run(serviceManager.getAgent(PROJECT_ID, AGENT_ID))
        assert after is not before
        [afterNode] = after.nodes
        assert afterNode.config["rag"] == {
            "project_id": PROJECT_ID,
            "rag_id": RAG_ID,
        }
        # The other agent of the template keeps its own rag, and is kept.
        assert serviceManager.agents[OTHER_AGENT_ID] is other
        [otherNode] = other.nodes
        assert otherNode.config["rag"]["rag_id"] == OLD_RAG_ID
        assert firestore.records[AGENTS][OTHER_AGENT_ID]["rag_id"] == (
            OLD_RAG_ID
        )

    def testAgentNotKeptIsGivenTheRag(
        self, serviceManager, firestore, members
    ):
        firestore.put(AGENT_TEMPLATES, buildSearchingTemplate())

        configured = configureRag()

        assert configured == {"agentId": AGENT_ID, "ragId": RAG_ID}
        agent = asyncio.run(serviceManager.getAgent(PROJECT_ID, AGENT_ID))
        [node] = agent.nodes
        assert node.config["rag"]["rag_id"] == RAG_ID

    @pytest.mark.parametrize(
        ("ids", "errorKind"), ACCESS_CASES.values(), ids=ACCESS_CASES.keys()
    )
    def testAccountKeptOutOfTheProjectIsRefused(
        self, serviceManager, firestore, members, ids, errorKind
    ):
        firestore.put(utilities.PROJECTS, {"_id": OTHER_PROJECT_ID})

        refused = configureRag(**ids)

        assert refused["errorKind"] == errorKind
        assert firestore.batches == []

    @pytest.mark.parametrize(
        ("agent", "rag", "errorKind", "message"),
        [
            (None, buildRag(), service.ERROR_NOT_FOUND, "has no agent"),
            (
                buildAgent(project_id=OTHER_PROJECT_ID),
                buildRag(),
                service.ERROR_NOT_FOUND,
                "has no agent",
            ),
            (
                buildAgent(deleted_at=DELETED_AT),
                buildRag(),
                service.ERROR_NOT_FOUND,
                "has no agent",
            ),
            (buildAgent(), None, service.ERROR_NOT_FOUND, "has no rag"),
            (
                buildAgent(),
                buildRag(project_id=OTHER_PROJECT_ID),
                service.ERROR_NOT_FOUND,
                "has no rag",
            ),
            (
                buildAgent(),
                buildRag(status="indexing"),
                service.ERROR_CONFLICT,
                "is indexing; only a ready rag",
            ),
        ],
        ids=[
            "noSuchAgent",
            "anotherProjectsAgent",
            "deletedAgent",
            "noSuchRag",
            "anotherProjectsRag",
            "ragNotReady",
        ],
    )
    def testAgentOrRagThatCannotBeHadIsRefused(
        self, serviceManager, firestore, members, agent, rag, errorKind, message
    ):
        kept = asyncio.run(serviceManager.getAgent(PROJECT_ID, AGENT_ID))
        for collection, record, recordId in (
            (AGENTS, agent, AGENT_ID),
            (utilities.RAGS, rag, RAG_ID),
        ):
            if record is None:
                del firestore.records[collection][recordId]
            else:
                firestore.put(collection, record)

        refused = configureRag()

        assert refused["errorKind"] == errorKind
        assert message in refused["error"]
        assert firestore.batches == []
        assert serviceManager.agents == {AGENT_ID: kept}
