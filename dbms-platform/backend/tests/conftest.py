import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DBMS_APP_DATABASE_URL", "sqlite:///./test_app.db")


def sandbox_available() -> bool:
    try:
        import pymysql
        from app.config import get_settings
        s = get_settings()
        pymysql.connect(host=s.sandbox_host, port=s.sandbox_port, user=s.sandbox_admin_user,
                        password=s.sandbox_admin_password, connect_timeout=2).close()
        return True
    except Exception:
        return False


requires_sandbox = pytest.mark.skipif(not sandbox_available(), reason="sandbox MySQL not reachable")
