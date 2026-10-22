"""Dummy records in Firestore for the Ptests, and the settings to reach it.

An account, a project, the project's owner role and the account's
membership: what the rag endpoints check before they let an account in.
They are written with the Google Firestore client itself, never through
`src/`, so a Ptest sets the database up the way any outside caller would.

Settings are read the way the server reads them: the environment first,
then the env file `RAG_ENV` picks - `server.env` unless it says `test`,
which the test suite sets - so the records land in the database the
server under test reads.

Run on its own to make or remove the records by hand:

    uv run python Ptest/documentProcessing/dummyData.py create
    uv run python Ptest/documentProcessing/dummyData.py delete \\
        <accountUuid> <projectId>
"""

import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values
from google.cloud import firestore
from google.oauth2 import service_account

# Ptest/documentProcessing/dummyData.py -> the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ENV_ENV_VAR = "RAG_ENV"
DEFAULT_RUN_ENV = "server"

USERS = "users"
PROJECTS = "projects"
PROJECT_ROLES = "project_roles"
PROJECT_MEMBERS = "project_members"

USAGE = (
    "Usage: dummyData.py create\n"
    "       dummyData.py delete <accountUuid> <projectId>"
)


@dataclass
class DummyData:
    """The ids of the records `createDummyData` wrote."""

    accountUuid: str
    projectId: str
    roleId: str


def readSetting(envVarName: str) -> str:
    """A variable from the environment, else from the env file, else empty."""
    runEnv = os.environ.get(RUN_ENV_ENV_VAR) or DEFAULT_RUN_ENV
    envFileValues = dotenv_values(PROJECT_ROOT / f"{runEnv}.env")
    return (
        os.environ.get(envVarName) or envFileValues.get(envVarName) or ""
    )


def openFirestore() -> firestore.Client:
    """A client on the Firestore database the server reads."""
    keyPath = Path(readSetting("GOOGLE_APPLICATION_CREDENTIALS"))
    if not keyPath.is_absolute():
        keyPath = PROJECT_ROOT / keyPath
    credentials = service_account.Credentials.from_service_account_file(
        str(keyPath)
    )
    return firestore.Client(
        project=readSetting("GCP_PROJECT_ID"),
        database=readSetting("FIRESTORE_DATABASE_ID") or None,
        credentials=credentials,
    )


def utcNow() -> datetime:
    """Current UTC time."""
    return datetime.now(timezone.utc)


def newId() -> str:
    """A fresh UUID v4, as the schemas' ids are."""
    return str(uuid.uuid4())


def createDummyData(client: firestore.Client) -> DummyData:
    """Write a dummy account, its project, the owner role and membership.

    The records follow `db/user-schema` and `db/project-schema`, and are
    written in one batch, so either all of them land or none.
    """
    dummy = DummyData(newId(), newId(), newId())
    now = utcNow()
    records = [
        (
            USERS,
            dummy.accountUuid,
            {
                "_id": dummy.accountUuid,
                "display_name": "Ptest dummy account",
                "email": f"ptest-{dummy.accountUuid}@example.com",
                "created_at": now,
                "updated_at": now,
            },
        ),
        (
            PROJECTS,
            dummy.projectId,
            {
                "_id": dummy.projectId,
                "name": "Ptest dummy project",
                "description": "Made by a Ptest; deleted when it ends.",
                "scope": "personal",
                "status": "active",
                "member_count": 1,
                "created_by": dummy.accountUuid,
                "created_at": now,
                "updated_at": now,
            },
        ),
        (
            PROJECT_ROLES,
            dummy.roleId,
            {
                "_id": dummy.roleId,
                "project_id": dummy.projectId,
                "name": "Owner",
                "permissions": ["view", "edit", "manage"],
                "created_at": now,
                "updated_at": now,
            },
        ),
        (
            PROJECT_MEMBERS,
            f"{dummy.projectId}_{dummy.accountUuid}",
            {
                "_id": f"{dummy.projectId}_{dummy.accountUuid}",
                "project_id": dummy.projectId,
                "user_id": dummy.accountUuid,
                "role_id": dummy.roleId,
                "added_by": dummy.accountUuid,
                "created_at": now,
                "updated_at": now,
            },
        ),
    ]
    batch = client.batch()
    for collection, recordId, fields in records:
        batch.set(client.collection(collection).document(recordId), fields)
    batch.commit()
    return dummy


def deleteDummyData(
    client: firestore.Client, accountUuid: str, projectId: str
) -> None:
    """Delete what `createDummyData` wrote for this account and project.

    The roles are found by their project, so the ids printed by `create`
    are all it takes. Rags made in the project are not touched: they are
    deleted through the API, which removes their chunks in Pinecone too.
    """
    batch = client.batch()
    batch.delete(client.collection(USERS).document(accountUuid))
    batch.delete(client.collection(PROJECTS).document(projectId))
    batch.delete(
        client.collection(PROJECT_MEMBERS).document(
            f"{projectId}_{accountUuid}"
        )
    )
    roles = client.collection(PROJECT_ROLES).where(
        filter=firestore.FieldFilter("project_id", "==", projectId)
    )
    for role in roles.stream():
        batch.delete(role.reference)
    batch.commit()


def main(arguments: list[str]) -> None:
    """Make the records and print their ids, or remove them."""
    if arguments == ["create"]:
        print(json.dumps(asdict(createDummyData(openFirestore()))))
    elif len(arguments) == 3 and arguments[0] == "delete":
        deleteDummyData(openFirestore(), arguments[1], arguments[2])
        print("Deleted.")
    else:
        raise SystemExit(USAGE)


if __name__ == "__main__":
    main(sys.argv[1:])
