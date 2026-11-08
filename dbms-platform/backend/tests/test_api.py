import uuid

import pytest
from fastapi.testclient import TestClient

from conftest import requires_sandbox


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth(client):
    name = "t" + uuid.uuid4().hex[:10]
    r = client.post("/api/auth/register", json={"username": name, "password": "correct-horse"})
    assert r.status_code == 201
    return {"Authorization": "Bearer " + r.json()["token"]}


def test_auth_flow(client, auth):
    assert client.get("/api/auth/me", headers=auth).status_code == 200
    assert client.get("/api/auth/me").status_code == 401
    assert client.post("/api/auth/login", json={"username": "nobody", "password": "x"}).status_code == 401


def test_course_and_sanitized_lesson(client):
    course = client.get("/api/course").json()
    assert len(course["modules"]) == 13
    lesson = client.get("/api/lessons/m03-aggregate-functions").json()
    for seg in lesson["segments"]:
        assert "solution" not in seg and "check" not in seg
        for q in seg.get("questions", []):
            assert "answer" not in q
    assert client.get("/api/lessons/nope").status_code == 404


def test_progress_notes_bookmarks(client, auth):
    assert client.put("/api/progress/lessons/m03-aggregate-functions", json={"status": "completed"}, headers=auth).status_code == 200
    assert client.get("/api/progress", headers=auth).json()["lessons"]["m03-aggregate-functions"]["status"] == "completed"
    client.put("/api/progress/notes/m03-aggregate-functions", json={"body": "COUNT(*) counts rows"}, headers=auth)
    assert client.get("/api/progress/notes/m03-aggregate-functions", headers=auth).json()["body"] == "COUNT(*) counts rows"
    client.post("/api/progress/bookmarks", json={"target": "m03-aggregate-functions", "label": "Aggregates"}, headers=auth)
    assert len(client.get("/api/progress/bookmarks", headers=auth).json()) == 1


def test_quiz_and_weak_concepts(client, auth):
    qid = "m03-aggregate-functions/q1"
    for _ in range(3):
        assert client.post("/api/quiz/answer", json={"question_id": qid, "answer": 0}, headers=auth).json()["correct"] is False
    weak = client.get("/api/progress/weak-concepts", headers=auth).json()
    assert any(w["concept"] == "count" for w in weak)


@requires_sandbox
def test_sandbox_isolation(client, auth):
    r = client.post("/api/labs/run", json={"lab": "playground", "dataset": "shop",
                                           "sql": "SELECT COUNT(*) FROM customers; SELECT * FROM mysql.user"}, headers=auth).json()
    assert r["statements"][0]["rows"] == [[25]]
    assert r["statements"][1]["status"] == "error"
    r = client.post("/api/labs/run", json={"lab": "playground", "sql": "DROP DATABASE mysql"}, headers=auth).json()
    assert r["statements"][0]["status"] == "rejected"
    assert client.post("/api/labs/run", json={"lab": "../etc", "sql": "SELECT 1"}, headers=auth).status_code == 422


@requires_sandbox
def test_exercise_grading_and_solution_lock(client, auth):
    eid = "m03-aggregate-functions/ex1"
    url = f"/api/exercises/{eid}"
    bad = client.post(url + "/attempt", json={"submission": "SELECT COUNT(*) FROM customers WHERE phone = NULL"}, headers=auth).json()
    assert bad["passed"] is False and "solution" not in bad
    assert client.get(url + "/solution", headers=auth).status_code == 403
    good = client.post(url + "/attempt", json={"submission": "SELECT COUNT(*) FROM customers WHERE phone IS NULL"}, headers=auth).json()
    assert good["passed"] is True and good["solution"]
    assert client.get(url + "/solution", headers=auth).status_code == 200


@requires_sandbox
def test_concurrent_sessions_block_and_release(client, auth):
    lab = "concurrency"
    client.post("/api/labs/reset", json={"lab": lab, "dataset": "empty"}, headers=auth)
    run = lambda s, sql: client.post("/api/labs/run", json={"lab": lab, "session": s, "sql": sql}, headers=auth).json()  # noqa: E731
    run("A", "CREATE TABLE t (id INT PRIMARY KEY, v INT); INSERT INTO t VALUES (1, 1)")
    run("A", "START TRANSACTION; UPDATE t SET v = 2 WHERE id = 1")
    blocked = run("B", "UPDATE t SET v = 3 WHERE id = 1")
    assert blocked["statements"][-1]["status"] == "blocked"
    locks = client.get(f"/api/labs/locks?lab={lab}", headers=auth).json()
    assert any(l["status"] == "WAITING" and l["session"] == "B" for l in locks["locks"])
    run("A", "COMMIT")
    import time
    for _ in range(20):
        p = client.get(f"/api/labs/poll?lab={lab}&session=B", headers=auth).json()
        if p["finished"]:
            assert p["finished"][0]["status"] == "ok"
            break
        time.sleep(0.2)
    else:
        raise AssertionError("blocked statement never completed")
