"""Environment-based configuration. Every value can be overridden with an
environment variable of the same name prefixed with ``DBMS_`` (see .env.example)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DBMS_", env_file=".env", extra="ignore")

    environment: str = "development"
    secret_key: str = "dev-only-change-me-dev-only-change-me"
    access_token_minutes: int = 60 * 24 * 7
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    content_dir: Path = REPO_ROOT / "content"
    log_level: str = "INFO"

    # Application database (users, progress, notes...). Never reachable by learner SQL.
    app_database_url: str = "sqlite:///./dbms_app.db"

    # Sandbox MySQL server: learner SQL runs here, as per-lab accounts.
    sandbox_host: str = "127.0.0.1"
    sandbox_port: int = 3306
    sandbox_admin_user: str = "sandbox_admin"
    sandbox_admin_password: str = "sandbox_admin_pw"
    # Host part used for per-lab accounts ('%' in Docker, 'localhost' for a local server).
    sandbox_account_host: str = "localhost"

    # Namespace for lab ownership, so several deployments (or dev + tests) can share one sandbox server
    # without colliding. Defaults to a hash of the application database URL.
    lab_namespace: str = ""

    # Resource limits applied to learner SQL.
    statement_timeout_ms: int = 10_000
    max_statements_per_run: int = 60
    max_script_chars: int = 200_000
    max_result_rows: int = 500
    lab_quota_mb: int = 256
    lab_idle_disconnect_s: int = 15 * 60
    lab_ttl_hours: int = 24
    runs_per_minute: int = 90
    lab_max_queries_per_hour: int = 20_000
    lab_max_user_connections: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
