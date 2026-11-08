import pytest

from app.sandbox.guard import classify

LAB = "lab_abc"


@pytest.mark.parametrize("sql", [
    "SELECT * FROM t", "WITH x AS (SELECT 1) SELECT * FROM x", "INSERT INTO t VALUES (1)",
    "CREATE TABLE t (id INT)", "SET SESSION sql_mode = ''", "SET @x = 1", "START TRANSACTION",
    "SET TRANSACTION ISOLATION LEVEL READ COMMITTED", "EXPLAIN SELECT 1", "CALL p()", "USE lab_abc",
    "(SELECT 1) UNION (SELECT 2)", "ANALYZE TABLE t", "SHOW INDEX FROM t",
    "SET SESSION sql_mode = @@GLOBAL.sql_mode", "SET @g = @@global.max_connections", "SELEKT 1",
])
def test_allowed(sql):
    assert classify(sql, LAB).allowed


@pytest.mark.parametrize("sql", [
    "GRANT ALL ON *.* TO x", "CREATE USER x", "DROP DATABASE lab_abc", "CREATE SCHEMA s",
    "SET GLOBAL max_connections = 1", "SET @@global.sql_mode = ''", "SET PERSIST x = 1",
    "SET SESSION sql_mode = '', GLOBAL max_connections = 1", "SET @a = 1, @@persist.x = 2",
    "LOAD DATA INFILE '/etc/passwd' INTO TABLE t", "SELECT * FROM t INTO OUTFILE '/tmp/x'",
    "SELECT LOAD_FILE('/etc/passwd')", "USE mysql", "KILL 12", "SHUTDOWN", "FLUSH PRIVILEGES",
    "SELECT SLEEP(60)", "RENAME USER a TO b", "/* c */ GRANT SELECT ON t TO u",
])
def test_rejected(sql):
    c = classify(sql, LAB)
    assert not c.allowed and c.reason
