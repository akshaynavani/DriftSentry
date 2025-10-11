"""Environment-driven settings for the API."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DRIFTSENTRY_", env_file=".env", extra="ignore")

    artifacts_dir: str = "artifacts"
    device: str = "cpu"
    max_transactions: int = 50_000
    log_level: str = "INFO"
