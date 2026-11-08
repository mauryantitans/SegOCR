from app.sandbox.splitter import split_sql


def texts(s):
    return [st.sql for st in split_sql(s)]


def test_basic_and_quotes():
    assert texts("SELECT 1; SELECT ';' AS x; SELECT \"a;b\", `c;d`") == ["SELECT 1", "SELECT ';' AS x", 'SELECT "a;b", `c;d`']


def test_comments_and_escapes():
    sql = "-- note; here\nSELECT 'it''s; ok' /* ; */ ; # trailing ;\nSELECT 'a\\'b;'"
    out = texts(sql)
    assert len(out) == 2 and out[0].endswith("/* ; */") and out[1].endswith("SELECT 'a\\'b;'")


def test_comment_only_statements_dropped():
    assert texts("-- only a comment\n;;  ") == []


def test_procedure_without_delimiter():
    sql = """CREATE PROCEDURE p(IN n INT)
BEGIN
  DECLARE i INT DEFAULT 0;
  IF n > 0 THEN SET i = n; END IF;
  WHILE i > 0 DO SET i = i - 1; END WHILE;
  CASE WHEN i = 0 THEN SELECT 'done'; ELSE SELECT 'no'; END CASE;
  DROP TABLE IF EXISTS tmp;
END;
CALL p(3);"""
    out = texts(sql)
    assert len(out) == 2 and out[0].startswith("CREATE PROCEDURE") and out[0].endswith("END") and out[1] == "CALL p(3)"


def test_single_statement_function_and_trigger():
    sql = ("CREATE FUNCTION f(x INT) RETURNS INT DETERMINISTIC RETURN x * 2; "
           "CREATE TRIGGER t BEFORE INSERT ON a FOR EACH ROW BEGIN SET NEW.v = 1; END; SELECT f(2)")
    assert len(texts(sql)) == 3


def test_delimiter_directive():
    sql = "DELIMITER //\nCREATE PROCEDURE p() BEGIN SELECT 1; END //\nDELIMITER ;\nCALL p();"
    assert texts(sql) == ["CREATE PROCEDURE p() BEGIN SELECT 1; END", "CALL p()"]


def test_line_numbers():
    sts = split_sql("SELECT 1;\n\nSELECT 2;")
    assert [s.line for s in sts] == [1, 3]
