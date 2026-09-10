"""Application entry point: builds the app and wires every route module in."""

from fastapi import FastAPI

from documentProcessing.routes import router as documentProcessingRouter
from serverDetails.routes import router as serverDetailsRouter

app = FastAPI(title="rag", version="0.1.0")

app.include_router(serverDetailsRouter)
app.include_router(documentProcessingRouter)


@app.get("/")
def readRoot():
    """Landing route."""
    return {"message": "Hello from rag!"}
