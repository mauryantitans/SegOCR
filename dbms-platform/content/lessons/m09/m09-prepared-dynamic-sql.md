---
id: m09-prepared-dynamic-sql
dataset: shop
summary: How PREPARE, EXECUTE and DEALLOCATE separate SQL code from data with ? placeholders, how to build dynamic SQL safely when identifiers must vary (whitelists and quoting), why concatenating user input enables SQL injection — demonstrated harmlessly in the lab — and how server-side and client-side prepared statements differ.
concepts: [prepared-statements, placeholders, dynamic-sql, sql-injection, identifier-quoting]
references:
  - title: "MySQL 8.4 Reference Manual — Prepared Statements"
    url: https://dev.mysql.com/doc/refman/8.4/en/sql-prepared-statements.html
  - title: "MySQL 8.4 Reference Manual — PREPARE Statement"
    url: https://dev.mysql.com/doc/refman/8.4/en/prepare.html
  - title: "MySQL 8.4 Reference Manual — EXECUTE Statement"
    url: https://dev.mysql.com/doc/refman/8.4/en/execute.html
  - title: "MySQL 8.4 Reference Manual — String Functions (QUOTE)"
    url: https://dev.mysql.com/doc/refman/8.4/en/string-functions.html#function_quote
  - title: "MySQL 8.4 Reference Manual — Schema Object Names (identifier quoting)"
    url: https://dev.mysql.com/doc/refman/8.4/en/identifiers.html
  - title: "OWASP — SQL Injection Prevention Cheat Sheet"
    url: https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html
---

## Introduction

Every SQL statement an application sends mixes two things: **code** (`SELECT … FROM customers WHERE city = …`) and **data** (the city the user typed). If both travel in one string, the database cannot tell them apart — and a user who types `x' OR '1'='1` has just written code. That is **SQL injection**, for two decades one of the most damaging classes of security vulnerability.

A **prepared statement** separates the two channels. The statement is sent once with **placeholders** (`?`) where data goes; the data is sent separately and is *never parsed as SQL*. MySQL offers this at two levels:

- the **SQL interface** — `PREPARE stmt FROM '…?…'`, `EXECUTE stmt USING @v`, `DEALLOCATE PREPARE stmt` — usable from any client and inside stored procedures;
- the **binary protocol** used by connectors (`COM_STMT_PREPARE` / `COM_STMT_EXECUTE`), which is what `cursor.execute(sql, params)` or `PreparedStatement.setInt(…)` use under the hood — sometimes, depending on driver settings.

The same SQL interface is also MySQL's only way to run **dynamic SQL** — statements whose *structure* (a table or column name, a sort order) is decided at runtime. Placeholders cannot stand for identifiers, so dynamic SQL brings back string building, and with it the injection risk unless identifiers are whitelisted and quoted.

This lesson covers both, demonstrates an injection attack harmlessly inside your own lab, and prepares the ground for the security module's [SQL Injection](/lessons/m11-sql-injection) lesson.

:::note[Dataset]
The examples use the **shop** dataset. The injection examples attack only your private lab copy; they read data that is already yours and change nothing outside the lab.
:::

## DBMS Theory

:::theory[Parsing, binding and the code/data boundary]
Processing an SQL statement has distinct phases: **parse** (text → syntax tree), **resolve** (names → tables and columns, type checking), **optimize** (choose a plan) and **execute**. A *parameterized* statement fixes the syntax tree with typed parameter markers during parse/resolve; values are **bound** to the markers afterwards. Because binding happens after parsing, a bound value can never change the tree — it is always a literal of the parameter's type. This is the formal reason parameterization defeats injection: the attacker controls data, and data cannot become syntax.

The SQL standard defines this in *dynamic SQL* (`PREPARE`, `EXECUTE … USING`, `DESCRIBE`) and in embedded/call-level interfaces (ODBC/JDBC parameter markers `?`). What may be a parameter is restricted to **values** — never identifiers or keywords — because identifiers must be known to resolve the statement.
:::

Two consequences shape all practical advice:

1. **Placeholders protect values only.** A column name, a table name, `ASC`/`DESC`, or a whole `WHERE` fragment cannot be a placeholder. When those must vary, the statement text must be built — and safety then comes from **choosing from a fixed whitelist** and **quoting identifiers**, not from escaping arbitrary input.
2. **Escaping is not parameterization.** Functions like MySQL's `QUOTE()` (or a driver's `escape_string`) turn a value into a safely quoted *literal inside the code string*. Done correctly everywhere, this prevents injection; done incorrectly once — a forgotten call, a wrong character set, a numeric context without quotes — it does not. Parameterization removes the possibility of the mistake.

## Why It Exists

Prepared statements exist for three reasons, in descending order of importance today:

- **Security.** Bound parameters make SQL injection impossible for the values they carry. OWASP's first recommendation for injection prevention is "use prepared statements with parameterized queries".
- **Correctness.** Binding handles quoting, escaping, character sets and types (dates, decimals, binary data) for you — no more broken queries for customers named `O'Brien`.
- **Efficiency.** A statement prepared once and executed many times skips repeated parsing and name resolution, and with the binary protocol, values travel in a compact binary form. MySQL still optimizes at each execution (see *Internal Working*), so the gain is modest compared with systems that cache plans.

Dynamic SQL exists because some legitimate tasks need runtime structure: a reporting procedure that sorts by a user-chosen column, maintenance that loops over a list of tables, or `LIMIT` values in older code. Without `PREPARE`, stored procedures could not do these at all.

## Internal Working

:::internals[What MySQL does for PREPARE and EXECUTE]
**`PREPARE name FROM 'text'`** parses the text (which must be exactly **one** statement), resolves names, checks privileges and records the number and positions of the `?` markers. The prepared statement is stored **in the session**: it is invisible to other connections and is freed by `DEALLOCATE PREPARE`, by re-preparing the same name, or when the session ends. A server-wide limit, `max_prepared_stmt_count`, caps how many prepared statements all sessions together may hold (default 16,382) — a guard against clients that prepare without deallocating.

**`EXECUTE name USING @a, @b`** binds the current values of the **user variables** (only user variables are accepted — not literals, not local routine variables) to the markers in order, then **optimizes** and executes the statement. MySQL does not keep a generic plan across executions: the optimizer sees the actual parameter values each time, so a selective value and a non-selective value can get different row estimates and plans (you will see this below). What is saved is the parse and resolution work.

**Binary protocol.** Connectors that use server-side prepared statements send `COM_STMT_PREPARE` (the server returns a statement id and parameter metadata), then `COM_STMT_EXECUTE` with binary-encoded values for each execution, and `COM_STMT_CLOSE`. This is the same server machinery as SQL `PREPARE`, without the user variables. Many drivers instead **emulate** prepared statements on the client: they escape the values and splice them into the SQL text, sending ordinary queries (`COM_QUERY`). Connector/J does this unless `useServerPrepStmts=true`; PHP's PDO does it by default for MySQL; PyMySQL and mysqlclient always do it. Emulation is safe *if the driver's escaping is correct for the connection character set* — which is why connection character sets must be configured through the driver, not with a manual `SET NAMES`.
:::

```mermaid
sequenceDiagram
  participant App
  participant Srv as MySQL session
  App->>Srv: PREPARE s FROM 'SELECT … WHERE city = ?'
  Srv->>Srv: parse + resolve once, store in session
  App->>Srv: SET @c = 'Austin'; EXECUTE s USING @c
  Srv->>Srv: bind value (never parsed) → optimize → execute
  App->>Srv: SET @c = 'x'' OR ''1''=''1'; EXECUTE s USING @c
  Srv->>Srv: the whole string is one value → 0 rows
  App->>Srv: DEALLOCATE PREPARE s
```

**What can be prepared.** Most DML and DDL statements, `CALL`, many `SHOW` statements and `SET` can be prepared; a few statements cannot (they fail with error 1295, "not supported in the prepared statement protocol yet"). Inside stored programs, `PREPARE`/`EXECUTE` are allowed in **procedures and events** but not in functions or triggers.

## MySQL Implementation

The three statements:

```sql
PREPARE stmt_name FROM { 'statement text' | @variable_holding_text };
EXECUTE stmt_name [USING @var1 [, @var2] ...];
{ DEALLOCATE | DROP } PREPARE stmt_name;
```

A first prepared statement with two placeholders, executed with different values:

```sql run
PREPARE orders_by_customer FROM
  'SELECT order_id, order_date, status FROM orders
   WHERE customer_id = ? AND status <> ? ORDER BY order_date, order_id';

SET @cust = 14, @excluded = 'cancelled';
EXECUTE orders_by_customer USING @cust, @excluded;
```

```sql run
SET @cust = 4;
EXECUTE orders_by_customer USING @cust, @excluded;
DEALLOCATE PREPARE orders_by_customer;
```

The statement text was parsed once; each `EXECUTE` bound new values. Placeholders work where literals work — including `LIMIT`, which does not accept user variables directly ([Variables](/lessons/m09-variables)):

```sql run
PREPARE product_page FROM
  'SELECT product_id, name, price FROM products ORDER BY product_id LIMIT ? OFFSET ?';
SET @page_size = 5, @offset = 10;
EXECUTE product_page USING @page_size, @offset;
DEALLOCATE PREPARE product_page;
```

The statement text may come from a variable — the foundation of dynamic SQL:

```sql run
SET @sql = CONCAT('SELECT COUNT(*) AS n FROM ', 'products', ' WHERE is_active = ?');
SET @active = 1;
PREPARE s FROM @sql;
EXECUTE s USING @active;
DEALLOCATE PREPARE s;
```

Your session's statement counters show the SQL-level prepared statement activity:

```sql run
SHOW SESSION STATUS WHERE Variable_name IN ('Com_prepare_sql', 'Com_execute_sql', 'Com_dealloc_sql');
```

## Syntax Breakdown

| Element | Meaning and MySQL specifics |
|---|---|
| `PREPARE s FROM '…'` | Parse and store one statement under name `s` (session scope; names are case-insensitive). Re-preparing an existing name deallocates the old one first. |
| `FROM @var` | The text may be in a user variable — the usual way to build dynamic SQL. |
| `?` | A value placeholder. Allowed where a literal value is allowed (comparisons, `VALUES`, `SET col = ?`, `LIMIT ?`). **Not** for identifiers, keywords, or lists (`IN (?)` takes exactly one value). |
| `EXECUTE s USING @a, @b` | Bind user variables to placeholders in order; the number must match (error 1210 otherwise). Literals and local routine variables are **not** accepted — copy them into user variables first. |
| `DEALLOCATE PREPARE s` / `DROP PREPARE s` | Free the statement. Executing it afterwards: error 1243 (unknown prepared statement handler). |
| `QUOTE(str)` | Returns the string as a single-quoted SQL literal with `\`, `'`, NUL and Ctrl-Z escaped; `QUOTE(NULL)` returns the word `NULL` (unquoted). For values you *must* splice into SQL text. |
| `` `identifier` `` | Backtick quoting for identifiers; a backtick inside the name is doubled (`` ` `` → ``` `` ```). |
| `max_prepared_stmt_count` | Global cap on prepared statements across all sessions. |
| Inside routines | Allowed in procedures and events; not in stored functions or triggers. |

:::standard[Standard SQL vs MySQL]
`PREPARE`, `EXECUTE … USING` and `DEALLOCATE PREPARE` follow standard dynamic SQL, but the standard binds host variables, while MySQL's SQL interface binds **user variables only**. The standard's `EXECUTE IMMEDIATE` (prepare-and-run in one step) is not supported by MySQL; PostgreSQL's PL/pgSQL `EXECUTE … USING` and its `format('%I', name)` identifier quoting, or SQL Server's `sp_executesql` with typed parameters and `QUOTENAME()`, are the analogous tools elsewhere.
:::

## Practical Examples

**1. SQL injection, demonstrated in your lab.** A search procedure that builds its query by **concatenating** user input — the pattern that causes real breaches:

```sql run
CREATE PROCEDURE find_customers_unsafe(IN p_city VARCHAR(200))
  READS SQL DATA
BEGIN
  SET @q = CONCAT('SELECT customer_id, first_name, city FROM customers WHERE city = ''', p_city,
                  ''' ORDER BY customer_id');
  PREPARE q FROM @q;
  EXECUTE q;
  DEALLOCATE PREPARE q;
END;

CALL find_customers_unsafe('Austin');
```

With honest input it works. Now an attacker types a "city" that closes the string literal and adds a condition of their own:

```sql run
CALL find_customers_unsafe('nowhere'' OR ''1''=''1');
```

The generated text was `… WHERE city = 'nowhere' OR '1'='1' ORDER BY customer_id` — a condition that is true for every row, so **all customers** come back. Worse, `UNION` lets the attacker read columns the query never selected — here the e-mail addresses and phone numbers that the search was not supposed to expose:

```sql run
CALL find_customers_unsafe('nowhere'' UNION SELECT customer_id, email, phone FROM customers WHERE customer_id <= 3 -- ');
```

The trailing `-- ` turned the procedure's own `' ORDER BY customer_id` into a comment. Nothing was harmed because this is your private lab — against a production database, the same input exfiltrates personal data.

**2. The same search, parameterized.** The value goes through a placeholder; the attacker's text is just a very strange city name:

```sql run
CREATE PROCEDURE find_customers(IN p_city VARCHAR(200))
  READS SQL DATA
BEGIN
  SET @city = p_city;
  PREPARE q FROM 'SELECT customer_id, first_name, city FROM customers WHERE city = ? ORDER BY customer_id';
  EXECUTE q USING @city;
  DEALLOCATE PREPARE q;
END;

CALL find_customers('Austin');
```

```sql run
CALL find_customers('nowhere'' OR ''1''=''1');
```

No rows: there is no city literally named `nowhere' OR '1'='1`. (In this particular procedure the query structure never changes, so plain static SQL — `SELECT … WHERE city = p_city` — would be just as safe; routine parameters in static statements are always bound values. Dynamic SQL is only needed when the *structure* varies, as in the next example.)

**3. Dynamic identifiers: whitelist, then quote.** A report procedure lets the caller choose the sort column and direction. Placeholders cannot do this, so the procedure **maps** the request onto a fixed set of allowed identifiers and rejects everything else:

```sql run
DELIMITER //
CREATE PROCEDURE product_report(IN p_sort VARCHAR(64), IN p_dir VARCHAR(4), IN p_limit INT)
  READS SQL DATA
BEGIN
  DECLARE v_col VARCHAR(64);
  DECLARE v_dir VARCHAR(4);
  SET v_col = CASE LOWER(p_sort)
                WHEN 'name'  THEN 'name'
                WHEN 'price' THEN 'price'
                WHEN 'stock' THEN 'stock_qty'
                ELSE NULL END;
  IF v_col IS NULL THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Unsupported sort column';
  END IF;
  SET v_dir = IF(UPPER(p_dir) = 'DESC', 'DESC', 'ASC');
  SET @limit = p_limit;
  SET @q = CONCAT('SELECT product_id, name, price, stock_qty FROM products ORDER BY ',
                  CONCAT('`', REPLACE(v_col, '`', '``'), '`'), ' ', v_dir, ', product_id LIMIT ?');
  PREPARE q FROM @q;
  EXECUTE q USING @limit;
  DEALLOCATE PREPARE q;
END//
DELIMITER ;

CALL product_report('price', 'desc', 3);
```

```sql run error
CALL product_report('price; DROP TABLE products', 'asc', 3);
```

The user's text never reaches the SQL string: only one of three hard-coded column names, one of two hard-coded directions, and a bound `LIMIT` value can. Backtick quoting (with embedded backticks doubled) is a second line of defence for identifiers that come from trusted metadata, such as table names read from `information_schema`.

**4. Quoting values with `QUOTE()` when you must build text.** Occasionally a value has to be embedded in SQL text — for example when generating a script that will be run elsewhere. `QUOTE()` produces a correct literal:

```sql run
SELECT QUOTE('Austin') AS plain,
       QUOTE('O''Brien') AS with_apostrophe,
       QUOTE('nowhere'' OR ''1''=''1') AS attack_attempt,
       QUOTE(NULL) AS null_value;
```

```sql run
SET @city = 'nowhere'' OR ''1''=''1';
SET @q = CONCAT('SELECT COUNT(*) AS matches FROM customers WHERE city = ', QUOTE(@city));
SELECT @q AS generated_sql;
PREPARE q FROM @q;
EXECUTE q;
DEALLOCATE PREPARE q;
```

The attack string became one harmless literal. Still prefer placeholders: `QUOTE()` must be applied every single time, and it does not help in numeric contexts written without quotes (`WHERE id = ` + input).

**5. Dynamic SQL over metadata.** A maintenance procedure that counts rows in every table of the lab — the classic legitimate use, where table names come from the data dictionary and are quoted as identifiers:

```sql run
DELIMITER //
CREATE PROCEDURE count_all_tables()
  READS SQL DATA
BEGIN
  DECLARE v_table VARCHAR(64);
  DECLARE done BOOLEAN DEFAULT FALSE;
  DECLARE cur CURSOR FOR
    SELECT TABLE_NAME FROM information_schema.TABLES
    WHERE TABLE_SCHEMA = DATABASE() AND TABLE_TYPE = 'BASE TABLE'
    ORDER BY TABLE_NAME;
  DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
  CREATE TEMPORARY TABLE IF NOT EXISTS table_counts (table_name VARCHAR(64) PRIMARY KEY, row_count BIGINT);
  DELETE FROM table_counts;
  OPEN cur;
  l: LOOP
    FETCH cur INTO v_table;
    IF done THEN
      LEAVE l;
    END IF;
    SET @q = CONCAT('INSERT INTO table_counts SELECT ', QUOTE(v_table), ', COUNT(*) FROM `',
                    REPLACE(v_table, '`', '``'), '`');
    PREPARE q FROM @q;
    EXECUTE q;
    DEALLOCATE PREPARE q;
  END LOOP;
  CLOSE cur;
END//
DELIMITER ;

CALL count_all_tables();
SELECT table_name, row_count FROM table_counts ORDER BY table_name;
```

A cursor is justified here: each row needs a *different statement* — something no single set-based query can express ([Error Handling and Cursors](/lessons/m09-error-handling-cursors)).

## Execution Walkthrough

```walkthrough
title: The life of a prepared statement — and per-execution optimization
steps:
  - title: "1. A skewed table"
    explain: >-
      100,000 support tickets, of which only 10 are 'escalated'; the rest are 'closed'. status is indexed.
      ANALYZE TABLE gives the optimizer fresh statistics for the new table.
    sql: |
      SET SESSION cte_max_recursion_depth = 100000;
      CREATE TABLE tickets (id INT PRIMARY KEY, status VARCHAR(10) NOT NULL, minutes INT NOT NULL, KEY idx_status (status));
      INSERT INTO tickets
      WITH RECURSIVE s(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM s WHERE n < 100000)
      SELECT n, IF(n % 10000 = 0, 'escalated', 'closed'), n % 60 FROM s;
      ANALYZE TABLE tickets;
  - title: "2. Prepare once"
    explain: The statement (here an EXPLAIN, so we can see the plan) is parsed and resolved once and stored in the session.
    sql: PREPARE plan_for FROM 'EXPLAIN SELECT COUNT(*), SUM(minutes) FROM tickets WHERE status = ?';
  - title: "3. Execute with a selective value"
    explain: The optimizer sees the actual value 'escalated' and estimates about 10 rows through idx_status.
    sql: |
      SET @st = 'escalated';
      EXECUTE plan_for USING @st;
  - title: "4. Execute with a non-selective value"
    explain: >-
      Same prepared statement, different value: the row estimate changes dramatically, because MySQL optimizes on
      every EXECUTE with the bound value. There is no generic plan frozen at PREPARE time.
    sql: |
      SET @st = 'closed';
      EXECUTE plan_for USING @st;
  - title: "5. Deallocate; the name is gone"
    explain: After DEALLOCATE the statement no longer exists in the session.
    sql: DEALLOCATE PREPARE plan_for;
```

```sql run error
EXECUTE plan_for USING @st;
```

## Output

Two results matter. First, from the injection demonstration: the concatenating procedure returned **every customer** for the input `nowhere' OR '1'='1` and leaked e-mail addresses and phone numbers through `UNION`, while the parameterized procedure returned **no rows** for the same input — the input was treated as one value. Second, from the walkthrough: the *same* prepared statement produced a row estimate of about 10 rows for `'escalated'` and tens of thousands for `'closed'`, demonstrating that MySQL optimizes each execution with the actual value. Read the captured plans: both use `idx_status`, but with very different `rows` estimates (10 versus about 50,000 — itself an under-estimate of the true 99,990, a reminder that estimates come from sampled statistics, see [Selectivity and Statistics](/lessons/m07-selectivity-statistics)) — on a less skewed column or a different query shape, that difference changes the chosen access path.

## Alternative Approaches

| Approach | Injection-safe? | Identifiers can vary? | Typical use |
|---|---|---|---|
| Static SQL in a stored procedure (`WHERE city = p_city`) | Yes — routine parameters are bound values | No | Most procedure logic |
| `PREPARE … ?` + `EXECUTE USING` | Yes for values | Only by building the text | Dynamic `LIMIT`, varying structure with bound values |
| Driver parameterized queries (`cursor.execute(sql, params)`) | Yes (server-side or correctly emulated) | No | All application SQL |
| String concatenation + `QUOTE()` / driver escaping | Only if never forgotten, correct charset, quoted contexts | Yes (values only) | Generating SQL scripts; legacy code |
| String concatenation of raw input | **No** | Yes | Never |
| Whitelist mapping → hard-coded identifiers | Yes | Yes | Sort columns, report dimensions, table lists |
| ORM / query builder | Usually (values bound) | Through the API; raw fragments are unsafe | Application code |

**Avoiding dynamic SQL altogether.** A few sort options can often be written statically with conditional expressions, at the cost of not using an index for the sort:

```sql run
SET @sort = 'price';
SELECT product_id, name, price, stock_qty
FROM products
ORDER BY CASE WHEN @sort = 'price' THEN price END DESC,
         CASE WHEN @sort = 'name'  THEN name  END,
         product_id
LIMIT 3;
```

No string building, no injection surface, but the `CASE` expressions force a filesort. For large tables with indexed sort columns, the whitelisted dynamic version can use the index.

## Performance Analysis

What prepared statements save — and what they do not:

| Cost component | Plain statement | MySQL prepared statement |
|---|---|---|
| Network transfer of SQL text | Every time | Once (binary protocol sends only values afterwards) |
| Parsing and name resolution | Every time | Once at `PREPARE` |
| Optimization | Every time | **Every time** (with actual values) |
| Execution | Every time | Every time |
| Server memory | None after the statement | Held until `DEALLOCATE` / session end, counted against `max_prepared_stmt_count` |

So in MySQL the benefit is real but limited to parse/resolve and protocol overhead — significant for short, very frequent statements (simple primary-key lookups), negligible for queries whose cost is dominated by execution. In return, the optimizer always plans with the actual values, so MySQL avoids the "parameter sniffing" problem of systems that reuse one cached plan for skewed data (SQL Server is the classic example).

Two operational costs deserve attention:

- **Round trips with emulation off.** A server-side prepared statement used once costs an extra round trip (prepare, execute, close) compared with one plain query. Drivers therefore cache prepared statements per connection (`cachePrepStmts` in Connector/J) or emulate by default.
- **Leaked statements.** Applications that prepare in a loop without closing eventually hit `max_prepared_stmt_count` (error 1461) for the *whole server*. Monitor the `Prepared_stmt_count` global status variable.

The `measured` comparison below runs the same 2,000 primary-key lookups through a prepared statement inside a procedure; the timing shows that the per-execution cost is small (build-machine timing — yours will differ):

```sql run
CREATE PROCEDURE lookup_loop(IN p_n INT)
  READS SQL DATA
BEGIN
  DECLARE i INT DEFAULT 0;
  PREPARE lk FROM 'SELECT price INTO @p FROM products WHERE product_id = ?';
  WHILE i < p_n DO
    SET @id = 1 + i % 30;
    EXECUTE lk USING @id;
    SET i = i + 1;
  END WHILE;
  DEALLOCATE PREPARE lk;
END;
```

```sql run measured
CALL lookup_loop(2000);
```

In the capture, 2,000 executions took about 20 ms in total — roughly 10 µs per bound primary-key lookup, including the interpreter's loop overhead.

## Best Practices

:::best-practice[Safe dynamic SQL]
1. **Parameterize every value** — in applications through the driver's parameter API, in procedures through static SQL or `EXECUTE … USING`. Never concatenate user input into SQL text.
2. **Map identifiers through a whitelist** (`CASE` → hard-coded names), and quote identifiers from metadata with backticks, doubling embedded backticks.
3. **Keep dynamic SQL in procedures small and auditable**: build the string in one place, log or `SELECT` it in debug mode, and review it like code.
4. **Always `DEALLOCATE PREPARE`** (and close statements in drivers), including on error paths.
5. **Use least privilege as a backstop.** The account that runs dynamic SQL should not be able to `DROP` tables or read other schemas; then even a missed injection has a bounded blast radius ([Privileges and Roles](/lessons/m11-privileges-roles)).
6. **Know your driver**: whether it uses server-side prepares or emulation, and set the connection character set through driver configuration so escaping is correct.
7. **Remember user-variable leakage**: `EXECUTE … USING` requires `@variables`, which live in the session after the procedure returns. Do not put secrets in them, and reset or overwrite them if pooled connections are shared.
:::

## Common Mistakes

**Mistake 1 — using a placeholder for an identifier.** A `?` in `ORDER BY` binds a *value*: the query sorts by the constant string `'price'`, i.e. not at all, and no error tells you:

```sql run
PREPARE bad_sort FROM 'SELECT product_id, name, price FROM products ORDER BY ? LIMIT 3';
SET @col = 'price';
EXECUTE bad_sort USING @col;
DEALLOCATE PREPARE bad_sort;
```

The rows come back in whatever order the scan produced, not by price. For a table name the mistake is at least loud:

```sql run error
PREPARE bad_table FROM 'SELECT COUNT(*) FROM ?';
```

**Mistake 2 — passing literals or local variables to `EXECUTE`.** Only user variables are accepted:

```sql run
PREPARE cnt FROM 'SELECT COUNT(*) AS orders FROM orders WHERE customer_id = ?';
```

```sql run error
EXECUTE cnt USING 14;
```

```sql run
SET @c = 14;
EXECUTE cnt USING @c;
DEALLOCATE PREPARE cnt;
```

**Mistake 3 — trying to prepare several statements at once.** `PREPARE` accepts exactly one statement, so "stacked" injections such as `'; DROP TABLE …` fail at the SQL level:

```sql run error
PREPARE two FROM 'SELECT 1; SELECT 2';
```

Do not rely on this: client connections opened with multi-statement support enabled (an option in most drivers) *do* execute stacked statements sent as one query string, and the `OR '1'='1'` and `UNION` attacks above need no second statement at all.

**Mistake 4 — escaping by hand.** `REPLACE(input, '''', '''''')` handles apostrophes but not backslashes (MySQL treats `\'` as an escaped quote unless `NO_BACKSLASH_ESCAPES` is set) or character-set edge cases. Use placeholders, or `QUOTE()` if you truly must build text.

**Mistake 5 — `IN (?)` with a comma-separated list.** `EXECUTE … USING @ids` with `@ids = '1,2,3'` compares against the single string `'1,2,3'` (converted to the number 1 in a numeric comparison, with a warning). Generate one placeholder per value, use `FIND_IN_SET(col, ?)` for small lists (not sargable), or pass the list through a temporary table or `JSON_TABLE`.

## Industry Application

- **Application data access.** Every mainstream driver and ORM parameterizes values — JDBC `PreparedStatement`, Python DB-API `%s` parameters, Go `database/sql` `?`, Node `mysql2` `execute()`. Code review and static analysis tools (Semgrep, CodeQL, SonarQube) flag string-built SQL as a high-severity finding.
- **Breaches.** Injection through concatenated SQL has been behind many large data breaches; it remains in the OWASP Top 10 (as part of the "Injection" category). The attacks in example 1 — tautologies and `UNION`-based extraction — are the textbook first steps; blind (boolean/time-based) variants work even when no rows are displayed ([SQL Injection](/lessons/m11-sql-injection)).
- **Reporting and admin tools.** Dynamic sort and filter options in back-office tools are a classic source of injection via identifiers; the whitelist pattern of example 3 is the standard fix.
- **Database maintenance.** DBAs use dynamic SQL in procedures and scripts to iterate over tables (`ANALYZE`, `OPTIMIZE`, partition maintenance, row counts), always driven by `information_schema` metadata with quoted identifiers — example 5.
- **Driver tuning.** High-throughput Java services enable `useServerPrepStmts` and `cachePrepStmts` to save parsing on hot statements, and monitor `Prepared_stmt_count`; others keep emulation to save round trips. Proxies such as ProxySQL track prepared statements per backend connection, which affects connection multiplexing.

## Practice Exercises

```exercise
id: ex1
title: A paged product list
difficulty: easy
concepts: [prepared-statements, placeholders]
prompt: |
  Using `PREPARE`/`EXECUTE` with placeholders for **both** the page size and the offset, return the **third page**
  of products ordered by `product_id`, with 6 products per page. Columns: `product_id`, `name`, `price`.
hints:
  - "PREPARE p FROM 'SELECT … ORDER BY product_id LIMIT ? OFFSET ?';"
  - "Page 3 with 6 rows per page starts at offset 12. Bind user variables: SET @n = 6, @o = 12;"
solution: |
  PREPARE p FROM 'SELECT product_id, name, price FROM products ORDER BY product_id LIMIT ? OFFSET ?';
  SET @n = 6, @o = 12;
  EXECUTE p USING @n, @o;
orderMatters: true
explanation: |
  `LIMIT` placeholders are bound as integers. (The reference solution ends with the `EXECUTE` so that its result
  is the one graded; in real code, follow it with `DEALLOCATE PREPARE p;`.) Plain `LIMIT 6 OFFSET 12` gives the same rows; the prepared version
  matters when the numbers come from a user or a routine parameter (user variables are not accepted directly in
  `LIMIT`). Remember that `OFFSET` paging reads and discards the skipped rows — keyset pagination
  (`WHERE product_id > last_seen`) scales better for deep pages.
```

```exercise
id: ex2
title: Close the injection hole
difficulty: medium
kind: dml
concepts: [sql-injection, placeholders, dynamic-sql]
setup: |
  CREATE PROCEDURE count_orders_by_code(IN p_code VARCHAR(100), OUT p_n INT)
  BEGIN
    SET @q = CONCAT('SELECT COUNT(*) INTO @res FROM orders WHERE discount_code = ''', p_code, '''');
    PREPARE q FROM @q;
    EXECUTE q;
    DEALLOCATE PREPARE q;
    SET p_n = @res;
  END;
prompt: |
  The setup created `count_orders_by_code(IN p_code, OUT p_n)`, which counts orders using a discount code — by
  concatenating the code into SQL. `CALL count_orders_by_code('x'' OR ''1''=''1', @n)` returns the number of **all**
  orders that have a non-NULL… in fact of all orders.

  Drop and re-create it with the **same signature** so that the code is passed as a bound value (keep using
  `PREPARE`/`EXECUTE`, or use static SQL — both are acceptable). The check calls it with a real code from the data and
  with the injection string.
hints:
  - "Replace the concatenation with a placeholder: 'SELECT COUNT(*) INTO @res FROM orders WHERE discount_code = ?'"
  - "Copy the parameter into a user variable (SET @code = p_code;) and EXECUTE q USING @code."
solution: |
  DROP PROCEDURE count_orders_by_code;
  CREATE PROCEDURE count_orders_by_code(IN p_code VARCHAR(100), OUT p_n INT)
  BEGIN
    SET @code = p_code;
    PREPARE q FROM 'SELECT COUNT(*) INTO @res FROM orders WHERE discount_code = ?';
    EXECUTE q USING @code;
    DEALLOCATE PREPARE q;
    SET p_n = @res;
  END;
check: |
  SET @real = (SELECT discount_code FROM orders WHERE discount_code IS NOT NULL ORDER BY order_id LIMIT 1);
  CALL count_orders_by_code(@real, @a);
  CALL count_orders_by_code('x'' OR ''1''=''1', @b);
  SELECT @a > 0 AS real_code_found, @b AS injection_matches;
explanation: |
  With the placeholder, the injection string is compared as one literal value and matches nothing. Static SQL —
  `SELECT COUNT(*) INTO p_n FROM orders WHERE discount_code = p_code;` — is even simpler and equally safe, because
  routine parameters in static statements are always bound values; dynamic SQL was never needed here. The original
  concatenation turned the input into code: `… = 'x' OR '1'='1'` is true for every row.
```

````exercise
id: ex3
title: A whitelisted dynamic ranking
difficulty: hard
kind: dml
concepts: [dynamic-sql, identifier-quoting, sql-injection]
setup: |
  CREATE TABLE report_rows (metric VARCHAR(20) NOT NULL, customer_id INT NOT NULL, value DECIMAL(14,2) NOT NULL);
prompt: |
  The setup created `report_rows(metric, customer_id, value)`. Create
  **`top_customers(IN p_metric VARCHAR(20), IN p_limit INT)`** that **inserts** into `report_rows` the top `p_limit`
  customers by one of these metrics (ties broken by the smaller `customer_id`):

  - `'orders'` → number of distinct orders that have at least one line item,
  - `'revenue'` → Σ(quantity × unit_price) over their order items.

  The `metric` column holds the metric name. Any other metric must raise `SQLSTATE '45000'` (and insert nothing).
  Build the query dynamically from a **whitelist** — the text of `p_metric` must never be concatenated into SQL —
  and bind the metric name and the limit with placeholders.

  The check calls `top_customers('orders', 3)`, `top_customers('revenue', 3)` and, through a handler, an injection
  attempt, then reads `report_rows` and the captured error message.
hints:
  - "Map the metric with CASE to a hard-coded expression string: 'COUNT(DISTINCT o.order_id)' or 'SUM(oi.quantity * oi.unit_price)'; SIGNAL if it is NULL."
  - "SET @m = p_metric, @lim = p_limit; then build 'INSERT INTO report_rows SELECT ?, o.customer_id, <expr> FROM orders o JOIN order_items oi … GROUP BY o.customer_id ORDER BY 3 DESC, o.customer_id LIMIT ?' and EXECUTE … USING @m, @lim."
  - "Write it inside DELIMITER // … //."
solution: |
  DELIMITER //
  CREATE PROCEDURE top_customers(IN p_metric VARCHAR(20), IN p_limit INT)
    MODIFIES SQL DATA
  BEGIN
    DECLARE v_expr VARCHAR(100);
    SET v_expr = CASE p_metric
                   WHEN 'orders'  THEN 'COUNT(DISTINCT o.order_id)'
                   WHEN 'revenue' THEN 'SUM(oi.quantity * oi.unit_price)'
                   ELSE NULL END;
    IF v_expr IS NULL THEN
      SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Unsupported metric';
    END IF;
    SET @m = p_metric, @lim = p_limit;
    SET @q = CONCAT('INSERT INTO report_rows SELECT ?, o.customer_id, ', v_expr, ' ',
                    'FROM orders o JOIN order_items oi ON oi.order_id = o.order_id ',
                    'GROUP BY o.customer_id ORDER BY 3 DESC, o.customer_id LIMIT ?');
    PREPARE q FROM @q;
    EXECUTE q USING @m, @lim;
    DEALLOCATE PREPARE q;
  END//
  DELIMITER ;
check: |
  CREATE PROCEDURE harness_ex3()
  BEGIN
    DECLARE CONTINUE HANDLER FOR SQLEXCEPTION GET DIAGNOSTICS CONDITION 1 @err = MESSAGE_TEXT;
    SET @err = NULL;
    CALL top_customers('orders;DROP orders', 3);
  END;
  CALL top_customers('orders', 3);
  CALL top_customers('revenue', 3);
  CALL harness_ex3();
  SELECT metric, customer_id, value, @err AS injection_error,
         (SELECT COUNT(*) FROM orders) AS orders_still_there
  FROM report_rows
  ORDER BY metric, value DESC, customer_id;
explanation: |
  The caller's string selects one of two hard-coded expressions; it is never part of the SQL text, so
  `'orders;DROP orders'` simply hits the `ELSE` branch and the `SIGNAL`. The metric name and the `LIMIT`
  are bound values. `COUNT(DISTINCT o.order_id)` is needed because the join produces one row per line item.
  `ORDER BY 3` refers to the third select-list column (the metric value) — in a generated statement this avoids
  repeating the expression. A version that concatenated `p_metric` after checking it with `IN ('orders','revenue')`
  would also be safe, but mapping to constants makes the safety obvious to a reviewer.
````

```exercise
id: ex4
title: Row counts for every table
difficulty: medium
kind: dml
concepts: [dynamic-sql, identifier-quoting]
prompt: |
  Create a table **`row_counts(table_name VARCHAR(64) PRIMARY KEY, n BIGINT)`** and fill it with the exact row
  count of every **base table** in your schema **other than `row_counts` itself**, using dynamic SQL driven by
  `information_schema.TABLES`. Identifiers must be quoted with backticks.

  You may write a procedure or a sequence of statements. The check reads `row_counts`.
hints:
  - "A cursor over SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE() AND TABLE_TYPE = 'BASE TABLE' AND TABLE_NAME <> 'row_counts'."
  - "For each table: SET @q = CONCAT('INSERT INTO row_counts SELECT ', QUOTE(v_t), ', COUNT(*) FROM `', REPLACE(v_t, '`', '``'), '`'); PREPARE/EXECUTE/DEALLOCATE."
  - "Create the table before the procedure runs, then CALL it."
solution: |
  CREATE TABLE row_counts (table_name VARCHAR(64) PRIMARY KEY, n BIGINT);
  CREATE PROCEDURE fill_row_counts()
  BEGIN
    DECLARE v_t VARCHAR(64);
    DECLARE done BOOLEAN DEFAULT FALSE;
    DECLARE cur CURSOR FOR
      SELECT TABLE_NAME FROM information_schema.TABLES
      WHERE TABLE_SCHEMA = DATABASE() AND TABLE_TYPE = 'BASE TABLE' AND TABLE_NAME <> 'row_counts';
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
    OPEN cur;
    l: LOOP
      FETCH cur INTO v_t;
      IF done THEN
        LEAVE l;
      END IF;
      SET @q = CONCAT('INSERT INTO row_counts SELECT ', QUOTE(v_t), ', COUNT(*) FROM `', REPLACE(v_t, '`', '``'), '`');
      PREPARE q FROM @q;
      EXECUTE q;
      DEALLOCATE PREPARE q;
    END LOOP;
    CLOSE cur;
  END;
  CALL fill_row_counts();
check: |
  SELECT table_name, n FROM row_counts ORDER BY table_name;
explanation: |
  Table names come from trusted metadata, but quoting them is still necessary for correctness (names with spaces,
  reserved words or backticks) and as defence in depth. `information_schema.TABLES.TABLE_ROWS` would be faster but
  is only an estimate for InnoDB; `COUNT(*)` per table is exact. One statement per table is inherently dynamic,
  which is why a cursor is appropriate here.
```

```exercise
id: ex5
title: Explain the attack and the defence
difficulty: challenge
kind: open
concepts: [sql-injection, prepared-statements]
prompt: |
  A product search endpoint builds `"SELECT id, name FROM products WHERE name LIKE '%" + q + "%' ORDER BY " + sort`
  from two query-string parameters, `q` and `sort`, and runs it with a driver whose connection allows
  multi-statements.

  1. Show three different malicious inputs (for `q` or `sort`) and what each achieves.
  2. Rewrite the access logic safely. Explain why parameterization alone is not enough here.
  3. Name two defence-in-depth measures that limit the damage if a similar bug slips through elsewhere.
solution: |
  1. **Attacks.** (a) `q = %' OR '1'='1' -- ` — returns every product (tautology); harmless here, devastating on a
  table of users. (b) `q = %' UNION SELECT id, CONCAT(email, ':', password_hash) FROM users -- ` — reads another
  table through the result set (UNION-based extraction; column count and types must match). (c)
  `sort = name; DROP TABLE products` — with multi-statements enabled, the second statement executes. Also:
  `sort = (SELECT IF(SUBSTRING(password_hash,1,1)='a', SLEEP(2), 0) FROM users LIMIT 1)` — a blind, time-based
  attack that works even without visible output.

  2. **Safe version.** Bind the search value: `… WHERE name LIKE ? ORDER BY <col>`, with the parameter
  `'%' + escape_like(q) + '%'` (escape `%`, `_` and `\` in `q` if they should match literally). `sort` cannot be a
  placeholder — a `?` in `ORDER BY` would sort by a constant — so map it through a whitelist:
  `{"name": "name", "price": "price", "newest": "created_at DESC"}`, defaulting to `name` and rejecting anything
  else. Parameterization covers values only; identifiers and keywords need whitelisting.

  3. **Defence in depth.** Run the endpoint with a least-privilege account (`SELECT` on the needed tables only, no
  access to `users`, no DDL), so (b) and (c) fail with privilege errors; disable multi-statements in the driver; add
  input validation (length limits), a WAF/RASP rule set, query logging with alerts on errors such as 1064 spikes,
  and code scanning (Semgrep/CodeQL) for string-built SQL in CI.
```

## Knowledge Assessment

```quiz
- id: q1
  question: "Why does `EXECUTE s USING @city` with `@city = \"x' OR '1'='1\"` NOT return all rows when s is `SELECT … WHERE city = ?`?"
  options:
    - "MySQL strips quotes from bound values"
    - "The value is bound after parsing, so it is always one literal and can never change the statement's syntax"
    - "PREPARE rejects statements containing OR"
    - "User variables cannot contain apostrophes"
  answer: 1
  explanation: "Binding happens after the syntax tree is fixed; the whole string is compared as a city name. Nothing is stripped or rejected, and user variables can hold any string."
  concepts: [prepared-statements, sql-injection]
- id: q2
  question: "Which of these can be a `?` placeholder in a MySQL prepared statement? (Select all that apply.)"
  type: multi
  options:
    - "The value in `WHERE price > ?`"
    - "The row count in `LIMIT ?`"
    - "The table name in `SELECT * FROM ?`"
    - "The sort column in `ORDER BY ?`, sorting by that column"
    - "A value in `INSERT … VALUES (?, ?)`"
  answer: [0, 1, 4]
  explanation: "Placeholders stand for values. A table name is a syntax error; ORDER BY ? is accepted but sorts by a constant value, not by the named column — the lesson's Mistake 1."
  concepts: [placeholders]
- id: q3
  question: "Inside a stored procedure, `EXECUTE s USING p_id;` (p_id is a parameter) fails with a syntax error. What is the fix?"
  options:
    - "Declare p_id as INOUT"
    - "Copy it into a user variable: SET @id = p_id; EXECUTE s USING @id;"
    - "Use EXECUTE IMMEDIATE"
    - "Prepared statements cannot be used in procedures"
  answer: 1
  explanation: "EXECUTE … USING accepts only user variables. MySQL has no EXECUTE IMMEDIATE, and prepared statements are allowed in procedures (not in functions or triggers)."
  concepts: [prepared-statements]
- id: q4
  question: "The walkthrough executed the same prepared EXPLAIN with 'escalated' (10 rows) and 'closed' (most rows). What did it show about MySQL?"
  options:
    - "The plan from PREPARE time is reused for every execution"
    - "MySQL optimizes at every EXECUTE using the actual bound value, so estimates (and potentially plans) differ"
    - "Prepared statements cannot use indexes"
    - "The second EXECUTE failed because the value changed"
  answer: 1
  explanation: "MySQL saves parsing and resolution, not the plan; each execution is optimized with the real value. That avoids the parameter-sniffing problems of systems that cache one plan."
  concepts: [prepared-statements]
- id: q5
  question: "A report procedure must sort by a column the user chooses. What is the safe implementation?"
  options:
    - "CONCAT the user's column name into ORDER BY after QUOTE()"
    - "Use ORDER BY ? with the column name bound"
    - "Map the user's choice through a whitelist to a hard-coded column name, then build the SQL"
    - "Escape backticks in the user's input and concatenate it"
  answer: 2
  explanation: "QUOTE() makes a string literal (sorting by a constant), ORDER BY ? also sorts by a constant, and escaping user-supplied identifiers still lets the user pick any column (or expression tricks). A whitelist means user text never reaches the SQL."
  concepts: [dynamic-sql, identifier-quoting]
- id: q6
  question: "Which statements about client drivers are true? (Select all that apply.)"
  type: multi
  options:
    - "Some drivers emulate prepared statements by escaping values and sending plain queries"
    - "Emulated prepares are safe only if the driver's escaping matches the connection character set"
    - "Server-side prepared statements avoid the optimizer on every execution"
    - "A client that prepares statements in a loop without closing them can exhaust max_prepared_stmt_count for the whole server"
  answer: [0, 1, 3]
  explanation: "Emulation is common (PyMySQL, PDO default, Connector/J default) and depends on correct escaping. MySQL optimizes server-side prepared statements on each execution. max_prepared_stmt_count is a global cap, so leaks affect everyone (error 1461)."
  concepts: [prepared-statements]
- id: q7
  question: "`PREPARE s FROM 'SELECT 1; DROP TABLE t'` fails. Does that make string concatenation in your application safe from injection?"
  options:
    - "Yes — attackers need a second statement to cause harm"
    - "No — tautologies and UNION-based reads need no second statement, and connections with multi-statements enabled do execute stacked statements"
    - "Yes, if the table has a primary key"
    - "No, because PREPARE ignores everything after the first semicolon and runs it later"
  answer: 1
  explanation: "The lesson's demo leaked all customers and their contact data with a single statement. PREPARE rejects the multi-statement text outright (error 1064) rather than running it later."
  concepts: [sql-injection]
- id: q8
  question: "What does `QUOTE(NULL)` return?"
  options: ["'NULL' (a quoted string)", "The word NULL without quotes", "An empty string ''", "An error"]
  answer: 1
  explanation: "QUOTE(NULL) returns the unquoted word NULL, so a generated statement contains a real NULL rather than the four-character string 'NULL'."
  concepts: [dynamic-sql]
```
