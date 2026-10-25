"""Application entry point: builds the app and wires every route module in."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from agents.routes import router as agentsRouter
from DbAdapters.memoryDb import getMemoryDb
from DbAdapters.pineconeDb import getPineconeDb
from documentProcessing.routes import router as documentProcessingRouter
from serverDetails.routes import router as serverDetailsRouter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the in-memory store first; close the Pinecone queue last."""
    getMemoryDb()
    yield
    await getPineconeDb().close()


app = FastAPI(title="rag", version="0.1.0", lifespan=lifespan)

app.include_router(serverDetailsRouter)
app.include_router(documentProcessingRouter)
app.include_router(agentsRouter)


@app.get("/")
def readRoot():
    """Landing route."""
    return {"message": "Hello from rag!"}
