"""Route declarations for the server details endpoints."""

from fastapi import APIRouter

from serverDetails import service
from serverDetails.model import HealthResponse, StatusResponse

router = APIRouter(tags=["server details"])


@router.get("/health", response_model=HealthResponse, summary="Full server vitals")
def readHealth() -> dict:
    """Report CPU, memory, disk, network and process vitals of the host.

    The `status` field summarises them as healthy, degraded or overloaded, and
    `warnings` names every threshold that was breached.
    """
    return service.collectServerVitals()


@router.get("/status", response_model=StatusResponse, summary="Liveness check")
def readStatus() -> dict:
    """Confirm the server is up and say how long it has been up."""
    return service.getServerStatus()
