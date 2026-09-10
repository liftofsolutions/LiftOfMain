"""Configuration for the service: `.env` loading and the adapter factories.

    from config import getSettings, buildFirestoreAdapter
"""

from config.settings import (
    ConfigurationError,
    Settings,
    buildFirestoreAdapter,
    buildMongoAdapter,
    buildPineconeAdapter,
    buildRedisAdapter,
    getSettings,
    loadEnvironment,
)

__all__ = [
    "ConfigurationError",
    "Settings",
    "buildFirestoreAdapter",
    "buildMongoAdapter",
    "buildPineconeAdapter",
    "buildRedisAdapter",
    "getSettings",
    "loadEnvironment",
]
