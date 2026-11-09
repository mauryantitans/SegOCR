---
id: m10-connections-pooling
dataset: shop
summary: How a client talks to mysqld — the packet-based client/server protocol, the handshake and authentication, the life of a connection and its dedicated thread — and why production systems put connection pools (application-side, ProxySQL, MySQL Router) between applications and the server.
concepts: [client-server-protocol, connection-lifecycle, thread-per-connection, max-connections, thread-cache, connection-pooling, proxysql, mysql-router, session-state]
references:
  - title: "MySQL 8.4 Reference Manual — How MySQL Handles Client Connections"
    url: https://dev.mysql.com/doc/refman/8.4/en/connection-interfaces.html
  - title: "MySQL 8.4 Reference Manual — Connection Management Variables (max_connections, thread_cache_size)"
    url: https://dev.mysql.com/doc/refman/8.4/en/server-system-variables.html
  - title: "MySQL 8.4 Reference Manual — Too many connections"
    url: https://dev.mysql.com/doc/refman/8.4/en/too-many-connections.html
  - title: "MySQL 8.4 Reference Manual — Caching SHA-2 Pluggable Authentication"
    url: https://dev.mysql.com/doc/refman/8.4/en/caching-sha2-pluggable-authentication.html
  - title: "MySQL Source Code Documentation — Client/Server Protocol"
    url: https://dev.mysql.com/doc/dev/mysql-server/latest/PAGE_PROTOCOL.html
  - title: "MySQL Router 8.4 — Introduction"
    url: https://dev.mysql.com/doc/mysql-router/8.4/en/mysql-router-general.html
  - title: "ProxySQL Documentation"
    url: https://proxysql.com/documentation/
related: [m10-server-architecture, m10-query-lifecycle, m08-transaction-lifecycle]
---

## Introduction

Every interaction with MySQL starts with a **connection**: a long-lived, stateful conversation between one client and one server thread. Opening one is not free — a TCP handshake, usually a TLS handshake, an authentication exchange, and the creation (or reuse) of a server thread with its own memory. Holding one is not free either — the thread, its buffers, its session variables, and possibly an open transaction with locks.

This lesson answers the questions that matter when you build or run a service on MySQL:

- What actually travels over the wire, and what does the server do for each **packet**?
- What is the **lifecycle** of a connection, from `connect()` to `COM_QUIT` or a timeout?
- Why does MySQL use **one thread per connection**, and what do `max_connections` and the **thread cache** control?
- Why do almost all production deployments use **connection pooling**, and what are the trade-offs between an application-side pool, **ProxySQL** and **MySQL Router**?
- What goes wrong when pooled connections carry **session state** or **open transactions** from one request to the next?

The [previous lesson](/lessons/m10-server-architecture) placed the connection layer at the top of the server. Here we look inside it.

:::note[What the sandbox can show]
Your lab can read connection-related system variables, its own session counters (`Bytes_sent`, `Com_*`, `Ssl_*`) and the server-wide `Threads_*`/`Connections` counters, and it can open up to three concurrent sessions in the scenarios below. It cannot change global settings (`SET GLOBAL max_connections`), kill other threads or see other accounts' connections. All server-wide numbers below are the **build server's** values at capture time.
:::

## DBMS Theory

:::theory[Sessions, workers and the client/server model]
A **session** is the server-side state associated with one client: identity and privileges, current schema, session variables, prepared statements, temporary tables and, at most, one open transaction. A DBMS must map sessions onto **workers** that execute their requests. Three classic models exist:

1. **Process per session** — the OS isolates sessions completely (PostgreSQL, Oracle dedicated server). Robust, but each session costs a process.
2. **Thread per session** — sessions share one address space; each gets a thread (MySQL community server, SQL Server before schedulers). Cheaper to create than a process; isolation is the program's responsibility.
3. **Worker pool** — a bounded set of workers multiplexes many sessions, picking up whichever has a request ready (thread pools, SQL Server's schedulers, MySQL Enterprise Thread Pool).

Independently of the server model, **connection pooling** on the client side keeps a set of established sessions open and lends them to application requests. Its theoretical basis is queueing: by **Little's law**, the number of connections busy on average equals arrival rate × time each request holds a connection (L = λ · W). A service doing 2,000 queries per second at 5 ms each keeps only about 10 connections busy on average — far fewer than the number of application threads that *might* want one.
:::

Terminology used in this lesson:

| Term | Meaning |
|---|---|
| **Connection** | The transport (TCP or Unix socket) plus the authenticated session on it |
| **Thread** (server side) | The OS thread executing that session's commands; visible as `ID` in the processlist and returned by `CONNECTION_ID()` |
| **Command** | One protocol request: `COM_QUERY`, `COM_STMT_EXECUTE`, `COM_PING`, `COM_QUIT`, … |
| **Pool** | A client-side or proxy-side cache of open connections |
| **Multiplexing** | Serving many client sessions over fewer server connections (needs a proxy that understands session state) |

## Why It Exists

Why can't every web request simply open a connection, run its query and close it? Because the fixed cost of a connection dwarfs the cost of a typical OLTP query:

- **Round trips.** TCP setup (1 round trip), TLS 1.3 setup (at least 1 more), MySQL handshake and authentication (1–2 more) — before the first query is sent. Across availability zones at 1 ms per round trip that is several milliseconds for a query that runs in 100 µs.
- **CPU.** TLS key exchange and `caching_sha2_password`'s full authentication path are cryptographic operations; thousands of new connections per second consume real cores.
- **Server resources.** Each connection is a thread with a stack and per-session buffers; thousands of threads compete for CPU and mutexes. MySQL bounds the damage with `max_connections` (default 151) — and the next connection receives error 1040, *Too many connections*.

Pools exist to pay the connection cost once and amortise it over thousands of requests, and to **bound concurrency** at the server: a pool of 20 connections guarantees the application never runs more than 20 concurrent statements, no matter how many requests arrive. Without pooling, traffic spikes turn into *connection storms*: latency rises, requests time out, clients retry by opening even more connections, and the server spends its time creating threads instead of running queries.

But pools introduce their own failure mode: a **pooled connection is reused**, so anything a previous borrower left behind — an open transaction holding locks, a changed `sql_mode`, a user variable, a temporary table — leaks into the next request. Much of the engineering in pools and proxies is about managing that state.

## Internal Working

### The client/server protocol

The classic MySQL protocol is a binary, packet-based, request/response protocol over TCP (default port 3306) or a Unix socket.

:::internals[Packets, phases and commands]
**Packet framing.** Every packet starts with a 4-byte header: a 3-byte little-endian **payload length** and a 1-byte **sequence id** that increments within one command exchange. The largest payload per packet is 2²⁴ − 1 bytes (16 MB − 1); larger messages are split into several packets. A single statement or row value may not exceed `max_allowed_packet` (default 64 MB in 8.0+).

**Connection phase.**
1. The server sends an *Initial Handshake* packet (protocol version 10): server version, the **connection id** (the thread id you later see as `CONNECTION_ID()`), a random *scramble* for authentication, **capability flags** and the default authentication plugin.
2. If both sides support TLS, the client sends an *SSL Request* and both perform a TLS handshake; everything afterwards is encrypted.
3. The client sends a *Handshake Response*: its capability flags, maximum packet size, character set, user name, an authentication response computed from the scramble, optional default schema and **connection attributes** (`_client_name`, `_os`, `program_name`, …).
4. Authentication: with `caching_sha2_password` (the default since 8.0) the server first tries the **fast path** — a SHA-256-based challenge/response checked against an in-memory cache of successful logins. On a cache miss it requires the **full path**, which sends the password over TLS (or RSA-encrypted). The server may also send an *Auth Switch Request* if the account uses a different plugin.
5. The server answers with `OK` (or `ERR`, e.g. 1045 access denied, 1040 too many connections).

**Command phase.** The client sends one command at a time and waits for the response:

| Command | Purpose |
|---|---|
| `COM_QUERY` | Text SQL; results come back in the *text protocol* (every value as a length-encoded string) |
| `COM_STMT_PREPARE` / `COM_STMT_EXECUTE` / `COM_STMT_CLOSE` | Server-side prepared statements; parameters and results in the compact *binary protocol* |
| `COM_PING` | Liveness check (what pools use for validation) |
| `COM_INIT_DB` | Change default schema (`USE`) |
| `COM_RESET_CONNECTION` | Reset session state without re-authenticating (5.7.3+) |
| `COM_CHANGE_USER` | Re-authenticate as another user on the same connection |
| `COM_QUIT` | Close the session |

A result set is a sequence of packets: column count, one column-definition packet per column, one packet per row, and a terminating `OK`/`EOF` packet carrying status flags such as "more results exist" and "in transaction". Because the protocol is strictly request/response per connection, **one connection runs at most one statement at a time** — the reason concurrency requires multiple connections.
:::

### The connection lifecycle and the thread model

```mermaid
sequenceDiagram
  participant C as Client
  participant A as Listener (accept)
  participant T as Connection thread
  participant D as Dictionary/ACL cache
  C->>A: TCP connect :3306
  A->>T: hand socket to cached thread or create one
  T->>C: Initial Handshake (connection id, scramble)
  C->>T: SSL Request + TLS handshake
  C->>T: Handshake Response (user, auth data, attrs)
  T->>D: check account, plugin, privileges
  T-->>C: OK
  loop command phase
    C->>T: COM_QUERY / COM_STMT_EXECUTE
    T-->>C: result set or OK / ERR
  end
  C->>T: COM_QUIT (or wait_timeout expires)
  T->>T: free session; return thread to cache if room
```

:::internals[Threads, the cache and the limits]
- **One thread per connection.** In the community server (`thread_handling = one-thread-per-connection`) the listener hands every accepted connection to a dedicated OS thread that owns a `THD` object — the server's session structure holding everything from the current statement's parse tree to the transaction context.
- **Thread cache.** When a connection ends, its thread is parked in a cache instead of being destroyed, up to `thread_cache_size` threads (default: 8 + `max_connections` / 100, capped at 100). The next connection reuses a parked thread. `Threads_created` counts threads that had to be created because the cache was empty; compare it with `Connections` (all connection attempts) to get the cache miss rate.
- **`max_connections`.** The maximum number of simultaneous client connections (default 151). The server permits **one extra** connection for an account with `CONNECTION_ADMIN` (or the deprecated `SUPER`), so an administrator can still get in — unless an application connects with such an account and uses up the spare. The separate **admin interface** (`admin_address`, `admin_port`) is the more robust escape hatch.
- **Per-account limits.** `max_user_connections` (global) and `CREATE USER … WITH MAX_USER_CONNECTIONS n` cap individual accounts so one misbehaving service cannot exhaust the server.
- **Timeouts.** An idle connection is closed after `wait_timeout` seconds (default 28,800 = 8 hours; `interactive_timeout` for interactive clients). `connect_timeout` bounds the handshake; `net_read_timeout`/`net_write_timeout` bound waiting for packets during a command.
- **Memory.** A connection costs a thread stack (`thread_stack`), a network buffer that starts at `net_buffer_length` (16 KB) and can grow to `max_allowed_packet`, plus per-operation buffers allocated only while needed (`sort_buffer_size`, `join_buffer_size`, `read_buffer_size`, temporary tables). A thousand idle connections are cheap in memory; a thousand connections each running a large sort are not.
- **Concurrency inside InnoDB.** Even with many connections, the number of threads *actively* working inside InnoDB can be limited with `innodb_thread_concurrency` (default 0 = no limit).
:::

:::version[8.0 / 8.4 notes]
- `caching_sha2_password` has been the default authentication plugin since **8.0.4**. In **8.4**, `mysql_native_password` is still shipped but **disabled by default**; old clients or proxies that only speak it fail to authenticate until the server is started with `--mysql-native-password=ON` (and it is removed in 9.0).
- Protocol compression gained **zstd** in 8.0.18 (`protocol_compression_algorithms`).
- `information_schema.PROCESSLIST` is **deprecated** in favour of `performance_schema.processlist` (8.0.22+); you will see deprecation warnings below.
- The thread pool remains a MySQL **Enterprise** feature (`thread_handling = loaded-dynamically` with the thread pool plugin); Percona Server and MariaDB ship their own thread pools.
:::

## MySQL Implementation

The connection-related settings of the build server:

```sql run
SELECT @@max_connections         AS max_connections,
       @@max_user_connections    AS max_user_connections,
       @@thread_cache_size       AS thread_cache_size,
       @@thread_handling         AS thread_handling,
       @@thread_stack            AS thread_stack_bytes,
       @@max_allowed_packet      AS max_allowed_packet,
       @@net_buffer_length       AS net_buffer_length;
```

```sql run
SELECT @@wait_timeout          AS wait_timeout_s,
       @@interactive_timeout   AS interactive_timeout_s,
       @@connect_timeout       AS connect_timeout_s,
       @@net_read_timeout      AS net_read_timeout_s,
       @@net_write_timeout     AS net_write_timeout_s,
       @@protocol_compression_algorithms AS compression;
```

Most of these are the defaults: 151 connections, a thread cache of 9 (8 + 151/100), eight hours of idle time before the server closes a connection. One value is not a server default: `max_user_connections` reads 6 because, when an account has its own `MAX_USER_CONNECTIONS` limit, the *session* value of this variable reports that per-account limit — the platform caps every lab account at a handful of connections, exactly the protection recommended below for production service accounts. That last value surprises many teams — the server will happily keep an abandoned connection, and any transaction it left open, alive for eight hours.

**Server-wide connection counters.** These status variables are **global** — they describe every connection on the build server, not just yours:

```sql run
SHOW GLOBAL STATUS WHERE Variable_name IN
  ('Connections', 'Threads_connected', 'Threads_running', 'Threads_cached',
   'Threads_created', 'Max_used_connections', 'Aborted_clients', 'Aborted_connects');
```

How to read them:

| Counter | Meaning | Healthy pattern |
|---|---|---|
| `Connections` | Connection attempts since startup (successful or not) | Grows slowly when clients pool |
| `Threads_connected` | Currently open connections | Well below `max_connections` |
| `Threads_running` | Connections currently executing a statement (not sleeping) | Roughly ≤ CPU cores under steady load |
| `Threads_cached` | Idle threads parked in the cache | ≤ `thread_cache_size` |
| `Threads_created` | Threads created because the cache was empty | `Threads_created / Connections` close to 0 |
| `Max_used_connections` | High-water mark of `Threads_connected` | Leaves headroom below `max_connections` |
| `Aborted_clients` | Connections dropped without `COM_QUIT` (client died, `wait_timeout`) | Low and stable |
| `Aborted_connects` | Failed handshakes (bad password, timeouts, too many connections) | Near 0; spikes indicate misconfiguration or attacks |

On the build server, compare `Threads_created` with `Connections` in the output: hundreds of connections were served by only a handful of newly created threads — the thread cache absorbed almost every connection, as it should when clients connect and disconnect at a moderate rate.

**Your own session.** `CONNECTION_ID()` returns the thread id assigned in the handshake. It differs on every connection, so the output is not shown:

```sql run volatile
SELECT CONNECTION_ID() AS my_thread_id;
```

The session counters below are **session-scoped**: they describe only this connection.

```sql run
SHOW SESSION STATUS WHERE Variable_name IN ('Ssl_version', 'Ssl_cipher', 'Compression');
```

The lab connection is encrypted with TLS 1.3 and uses no protocol compression. TLS has been on by default since 8.0 when the server has certificates (generated automatically at initialisation); `require_secure_transport = ON` makes it mandatory.

## Syntax Breakdown

### Settings

| Setting | Scope | Default (8.0 / 8.4) | Purpose |
|---|---|---|---|
| `max_connections` | Global, dynamic | 151 | Cap on simultaneous client connections (+1 for `CONNECTION_ADMIN`) |
| `max_user_connections` | Global / per account | 0 (unlimited) | Cap per account |
| `thread_cache_size` | Global, dynamic | 8 + max_connections/100 (≤ 100) | Idle threads kept for reuse |
| `thread_handling` | Global, startup | `one-thread-per-connection` | `loaded-dynamically` with the Enterprise thread pool |
| `wait_timeout` / `interactive_timeout` | Global + session | 28800 s | Close idle connections |
| `connect_timeout` | Global | 10 s | Max time for the handshake |
| `max_allowed_packet` | Global + session (read-only) | 64 MB | Largest packet / statement / row value |
| `net_read_timeout` / `net_write_timeout` | Global + session | 30 s / 60 s | Waiting for packets during a command |
| `admin_address` / `admin_port` | Global, startup | unset / 33062 | Separate admin listener, not limited by `max_connections` |
| `require_secure_transport` | Global | OFF | Reject unencrypted TCP connections |

### Statements and client calls

```sql
-- Administration (not available in the sandbox)
SET GLOBAL max_connections = 500;
SET PERSIST thread_cache_size = 64;
CREATE USER 'reporting'@'%' IDENTIFIED BY '…' WITH MAX_USER_CONNECTIONS 20;
KILL 12345;                 -- terminate a connection
KILL QUERY 12345;           -- abort only its current statement
SELECT * FROM performance_schema.processlist;           -- 8.0.22+ replacement for SHOW PROCESSLIST
SELECT * FROM performance_schema.session_connect_attrs; -- program_name, _client_name, ...
```

| Client-side call | Protocol command | Typical use |
|---|---|---|
| `mysql_real_connect()` / driver `connect()` | handshake | Open a connection |
| `mysql_ping()` / pool validation query | `COM_PING` | Check a pooled connection is alive |
| `mysql_reset_connection()` | `COM_RESET_CONNECTION` | Clear session state when a pool takes a connection back |
| `mysql_change_user()` | `COM_CHANGE_USER` | Switch identity on an existing connection |
| `mysql_close()` | `COM_QUIT` | Close |

`COM_RESET_CONNECTION` rolls back any open transaction, releases table locks and named locks, drops temporary tables, deallocates prepared statements, resets session variables to their global defaults and clears user variables — without a new handshake.

## Practical Examples

### 1. Session state lives on the connection

Everything you `SET` lives on this connection until it closes or is reset. A pool hands the *same* connection to the next request, so state set by one request is visible to the next:

```sql run
-- Request 1 borrows the connection
SET @tenant_id = 42;
SET SESSION sql_mode = 'ANSI_QUOTES';
CREATE TEMPORARY TABLE request_scratch (id INT);
INSERT INTO request_scratch VALUES (1), (2);
```

```sql run
-- Request 2 borrows the same pooled connection later
SELECT @tenant_id AS tenant_id_seen_by_request_2,
       @@SESSION.sql_mode AS sql_mode_seen_by_request_2,
       (SELECT COUNT(*) FROM request_scratch) AS scratch_rows_seen;
```

Request 2 sees request 1's tenant id, its non-default SQL mode (in which `"text"` is now an *identifier*, not a string) and its temporary table. In a multi-tenant application a leaked `@tenant_id` is a data-isolation bug. The fix is for the pool to **reset** the connection on return (`COM_RESET_CONNECTION`), or for the application never to rely on session state. Clean up before moving on:

```sql run
DROP TEMPORARY TABLE request_scratch;
SET @tenant_id = NULL;
SET SESSION sql_mode = DEFAULT;
SELECT @@SESSION.sql_mode = @@GLOBAL.sql_mode AS sql_mode_restored;
```

### 2. A connection's own timeouts

Session copies of timeouts can be changed without privileges, which is how pools and drivers tune idle limits per connection:

```sql run
SET SESSION wait_timeout = 300;
SELECT @@SESSION.wait_timeout AS this_connection, @@GLOBAL.wait_timeout AS server_default;
```

A pool's own *idle timeout* or *max lifetime* must be **shorter** than the server's `wait_timeout` (and any firewall or load-balancer idle timeout), or the pool will hand out connections the server has already closed — the classic "MySQL server has gone away" (client error 2006) or "Lost connection to MySQL server during query" (2013).

### 3. What the protocol carries: bytes on the wire

`Bytes_sent` and `Bytes_received` are **session-scoped** and count protocol bytes on this connection. Fetching many wide rows costs bandwidth even when the query itself is trivial:

```sql run
SHOW SESSION STATUS LIKE 'Bytes_sent';
SELECT order_id FROM orders WHERE order_id = 1;
SHOW SESSION STATUS LIKE 'Bytes_sent';
SELECT * FROM orders;
SHOW SESSION STATUS LIKE 'Bytes_sent';
```

Compare the three snapshots: the single-column, single-row result added a small number of bytes (including the column-definition packet and the snapshot's own result), while `SELECT *` over the 80 orders added several kilobytes. Every column you select becomes a column-definition packet plus a value in every row packet — another reason to avoid `SELECT *` in hot paths, especially across regions.

### 4. Prepared statements use different protocol commands

SQL-level `PREPARE`/`EXECUTE` (shown here) uses `COM_QUERY` to send the statements; drivers that use **server-side** prepared statements send `COM_STMT_PREPARE`/`COM_STMT_EXECUTE` packets and exchange parameters in the binary protocol. Both are counted in session status:

```sql run
PREPARE find_order FROM 'SELECT status FROM orders WHERE order_id = ?';
SET @id = 7;
EXECUTE find_order USING @id;
SET @id = 8;
EXECUTE find_order USING @id;
DEALLOCATE PREPARE find_order;
SHOW SESSION STATUS WHERE Variable_name IN ('Com_prepare_sql', 'Com_execute_sql', 'Com_stmt_prepare', 'Com_stmt_execute');
```

`Com_prepare_sql` and `Com_execute_sql` count the SQL-level statements: one prepare, two executions. The `Com_stmt_*` counters count prepared-statement operations of every kind on the connection — the manual notes that they are incremented for SQL-level `PREPARE`/`EXECUTE` as well as for protocol-level `COM_STMT_*` packets — so they are at least as large. Prepared statements belong to the **session**: a pooled connection that is reset loses them, which is why drivers re-prepare transparently.

## Execution Walkthrough

### A leaked transaction in a pooled connection

The most damaging pooling bug is a connection returned to the pool **with an open transaction**. The next borrower of that connection is not affected — everybody else is, because the transaction's row locks stay held. Watch it with three real sessions: **B** is a request that forgets to commit and returns its connection to the pool; **A** is a monitoring query; **C** is an unrelated request that needs the same row.

```concurrency
title: A pooled connection returned with an open transaction
description: Session B updates a row inside a transaction and "returns to the pool" without committing. Its thread sleeps, holding the lock.
setup: |
  CREATE TABLE inventory (sku VARCHAR(10) PRIMARY KEY, on_hand INT NOT NULL);
  INSERT INTO inventory VALUES ('A-100', 10), ('B-200', 5);
steps:
  - session: B
    sql: START TRANSACTION; UPDATE inventory SET on_hand = on_hand - 1 WHERE sku = 'A-100';
    note: B's request did its work but its error-handling path skipped COMMIT. The connection goes back to the pool, idle, with the transaction open.
  - session: A
    sql: SELECT COMMAND, STATE, INFO FROM information_schema.PROCESSLIST WHERE DB = DATABASE() ORDER BY COMMAND, INFO;
    note: The monitoring session sees B as Sleep with no statement running — nothing looks busy, yet B holds an exclusive row lock.
  - session: C
    sql: UPDATE inventory SET on_hand = on_hand - 1 WHERE sku = 'A-100';
    expect: blocks
    note: An unrelated request needs the same row and waits for B's lock (up to innodb_lock_wait_timeout, 50 s by default).
  - observe: locks
  - session: B
    sql: ROLLBACK;
    note: The pool's reset (COM_RESET_CONNECTION) or an explicit ROLLBACK on check-in releases the lock; C's UPDATE completes immediately.
  - session: C
    sql: SELECT sku, on_hand FROM inventory ORDER BY sku;
```

Three things to take away:

1. The processlist shows the leaking connection as **`Sleep`** — idle connections are not harmless if they are *inside a transaction*. On a real server, `information_schema.INNODB_TRX` (needs `PROCESS`) lists open transactions with `trx_mysql_thread_id` and `trx_started`; an old transaction whose thread is sleeping is the signature of this bug.
2. The lock snapshot shows B's transaction `RUNNING` with one row locked and one modified, holding `IX` on the table and `X,REC_NOT_GAP` on the primary-key record `'A-100'`; C's transaction is in `LOCK WAIT` with an identical `X,REC_NOT_GAP` request on `'A-100'` in status `WAITING`. C's UPDATE completed only after B's ROLLBACK step.
3. The fix belongs in the pool configuration (reset or roll back on return), in the code (always `try/finally` around transactions), and in monitoring (alert on transactions older than N seconds).

## Output

The scenario's final `SELECT` shows `A-100` reduced by exactly one — C's decrement. B's change was rolled back, so the row was never decremented twice. Had the pool *committed* on return instead (some frameworks do), B's half-finished business operation would have been made permanent — which is why pools roll back, never commit, on check-in.

From the earlier blocks, the output to remember:

- The build server runs the defaults: `max_connections` 151, `thread_cache_size` 9, `wait_timeout` 28,800 s, one thread per connection.
- Server-wide counters (`Threads_*`, `Connections`, `Aborted_*`) are global; `Bytes_*`, `Com_*`, `Ssl_*` are per session.
- Session state (user variables, `sql_mode`, temporary tables, prepared statements, open transactions) survives between "requests" on the same connection until it is reset.

## Alternative Approaches

There are four places to manage connections. Large systems often combine them.

| Approach | How it works | Strengths | Weaknesses |
|---|---|---|---|
| **No pooling** (connect per request) | Open, query, close | Simple; no state leakage | Handshake cost on every request; connection storms; only viable for low traffic or batch jobs |
| **Application-side pool** (HikariCP, SQLAlchemy `QueuePool`, Go `database/sql`, Node `mysql2` pool) | Each application process keeps N open connections and lends them to threads | No extra hop; per-service tuning; validation and max-lifetime built in | Total connections = instances × pool size — grows with every autoscaled pod; no cross-service multiplexing |
| **ProxySQL** | A MySQL-protocol proxy between applications and servers; maintains its own backend connection pools | **Multiplexing** (many client sessions over few backend connections), query routing rules, read/write split, query rewriting and caching, failover awareness, connection limits per user | Extra hop and component to operate; multiplexing is disabled for a session that uses state the proxy cannot track (open transaction, user variables, temporary tables, `GET_LOCK`, some `SET`s) |
| **MySQL Router** | Oracle's lightweight router, configured from InnoDB Cluster / ClusterSet / ReplicaSet metadata | Topology-aware routing to the current primary or to replicas, automatic failover handling, official support; newer releases add connection pooling/sharing and read/write splitting | Fewer query-level features than ProxySQL; primarily designed for InnoDB Cluster topologies |
| **Server-side thread pool** (MySQL Enterprise, Percona, MariaDB) | Many connections share a bounded set of worker threads in `mysqld` | Protects the server from too many *active* threads; transparent to clients | Does not reduce handshake cost or connection count; Enterprise-only in Oracle MySQL |

**Choosing.** A handful of application instances: an application-side pool is enough. Hundreds of autoscaled pods or serverless functions: add a proxy (ProxySQL, Router, or a cloud proxy such as Amazon RDS Proxy) so that the *server* sees a stable, small number of connections. High concurrency on a big server: consider a thread pool to limit active threads. Topology changes (failover): Router or ProxySQL hide them from the application.

## Performance Analysis

**Sizing a pool.** Start from Little's law: busy connections ≈ throughput × time a request holds a connection. A pool much larger than that does not add throughput — it adds queueing *inside* MySQL instead of in the pool, where it is harder to see and costs more (context switches, mutex and latch contention, InnoDB lock waits). Commonly cited starting points (HikariCP's guidance, for example) are on the order of `2 × CPU cores` of the *database* server for the *total* active connections across all clients, then adjusted by measurement.

| Symptom | Likely cause | Where to look |
|---|---|---|
| `ERROR 1040 Too many connections` | Sum of pool maxima across instances > `max_connections`; leaked connections; long `wait_timeout` with abandoned clients | `Max_used_connections`, `Threads_connected` by user/host |
| High connect latency, CPU in TLS | Connection churn (no pooling, or pool max-lifetime too short) | `Connections` rate, `Threads_created` |
| `Threads_running` ≫ cores | Too many concurrent active statements | Pool size; thread pool; slow queries holding connections |
| "Server has gone away" (2006/2013) | Pool idle timeout > `wait_timeout`, or a network device dropping idle flows | Pool `maxLifetime`/`idleTimeout`, keepalive |
| Lock waits with idle connections | Leaked open transactions | `INNODB_TRX` with sleeping threads |

:::performance[Theory vs measurement]
Connection cost and pool behaviour cannot be measured meaningfully inside this sandbox — your lab cannot open thousands of connections or change server limits. Measure on a staging system with your real driver and network: connect latency (p50/p99), `Threads_created` per second, `Threads_running` over time, and pool wait time from the pool's own metrics (HikariCP, for instance, exposes pending threads and acquisition time). The numbers above come from queueing theory and widely published pool guidance, not from this server.
:::

## Best Practices

:::best-practice[Connections in production]
1. **Always pool** in long-running services; size the pool from measured throughput × hold time, not from the number of application threads.
2. **Budget connections globally:** Σ(instances × pool max) across all services ≤ `max_connections` minus headroom for administrators, replicas, backups and monitoring. Revisit when autoscaling limits change.
3. **Reset or roll back on check-in.** Configure the pool to roll back open transactions and reset session state (`COM_RESET_CONNECTION`, or `useResetSession`-style options), and never commit on return.
4. **Keep pool idle/lifetime limits below** `wait_timeout` and any network idle timeout; enable TCP keepalive.
5. **Set per-account limits** (`MAX_USER_CONNECTIONS`) and use separate accounts per service, so one service cannot exhaust the server and `processlist` shows who is who.
6. **Never let applications connect as an account with `CONNECTION_ADMIN`/`SUPER`** — they would consume the reserved administrative connection. Configure `admin_address` for operators.
7. **Keep transactions short** and outside of network calls; a connection is "busy" for the whole transaction, not just its statements.
8. **Identify clients** with connection attributes (`program_name`) so `performance_schema.session_connect_attrs` shows which service owns each connection.
:::

## Common Mistakes

**Mistake 1 — relying on a user variable to carry request context in a pooled connection.** As shown above, `@tenant_id` set by one request is visible to the next borrower. Pass context explicitly as query parameters.

**Mistake 2 — treating `max_connections` as a throughput knob.** Raising it from 151 to 5,000 lets more connections in; it does not make the server faster, and it can make overload worse by allowing thousands of concurrently active threads. Fix the pool sizes instead.

**Mistake 3 — trying to change server-wide limits from an application account.** Global connection settings require administrative privileges (`SYSTEM_VARIABLES_ADMIN`). The sandbox rejects the statement before it reaches the server; on a real server an ordinary account receives error 1227 (access denied):

```sql run error
SET GLOBAL max_connections = 1000;
```

On a real server, `SET GLOBAL` is also lost at restart unless you use `SET PERSIST` or the option file — a common reason an emergency fix "disappears".

**Mistake 4 — assuming `Sleep` means harmless.** A sleeping connection inside a transaction holds locks and keeps an old read view alive, blocking purge (see [Redo and Undo Logs](/lessons/m10-redo-undo)).

**Mistake 5 — an `ANSI_QUOTES` (or other `sql_mode`) leak changing query meaning.** If a previous borrower set `ANSI_QUOTES`, a later query using `"double quotes"` for a string literal fails with an unknown-column error:

```sql run
SET SESSION sql_mode = 'ANSI_QUOTES';
```

```sql run error
SELECT order_id FROM orders WHERE status = "delivered" LIMIT 1;
```

```sql run
SET SESSION sql_mode = DEFAULT;
SELECT order_id FROM orders WHERE status = "delivered" ORDER BY order_id LIMIT 1;
```

Under `ANSI_QUOTES`, `"delivered"` is a column name, so the query fails with error 1054; with the default mode the same text works. Use single quotes for strings everywhere, and reset connections on return.

## Industry Application

- **Kubernetes microservices.** A service with 50 pods × a pool of 20 connections = 1,000 potential connections from one service alone. Teams either cap the pools aggressively, or route all pods through ProxySQL or a cloud proxy that multiplexes them onto a few dozen backend connections per MySQL instance.
- **Serverless functions.** AWS Lambda and similar platforms can start thousands of short-lived instances, each wanting its own connection. Managed proxies (Amazon RDS Proxy) or HTTP-based data APIs exist largely to absorb these connection storms.
- **InnoDB Cluster deployments.** MySQL Router runs as a sidecar next to each application instance; it reads the cluster metadata and sends read-write traffic to the current primary, failing over automatically when Group Replication elects a new one.
- **ProxySQL in large fleets.** Organisations running many MySQL replicas use ProxySQL query rules to route `SELECT`s to replicas, mirror traffic for testing, throttle abusive query digests, and drain a backend before maintenance — all without application changes.
- **Incident runbooks.** "Too many connections" runbooks typically: log in through the admin port, inspect `performance_schema.processlist` grouped by user/host to find the offending service, `KILL` its idle connections, cap it with `MAX_USER_CONNECTIONS`, and then fix its pool configuration.

## Practice Exercises

```exercise
id: ex1
title: Memory reserved for thread stacks
difficulty: easy
kind: query
dataset: shop
concepts: [max-connections, thread-per-connection]
prompt: |
  Return one row with three columns:

  1. `max_connections` — the server's `max_connections`
  2. `thread_stack_kb` — the thread stack size in KB (`@@thread_stack / 1024`)
  3. `stack_mb_at_max` — the memory reserved for thread stacks if every allowed connection were open,
     in MB, rounded to 1 decimal place: `max_connections × thread_stack / 1024 / 1024`.
hints:
  - "System variables can be read in a SELECT list with @@name."
  - "ROUND(@@max_connections * @@thread_stack / 1024 / 1024, 1)"
solution: |
  SELECT @@max_connections AS max_connections,
         @@thread_stack / 1024 AS thread_stack_kb,
         ROUND(@@max_connections * @@thread_stack / 1024 / 1024, 1) AS stack_mb_at_max;
explanation: |
  Thread stacks are only one part of per-connection memory (network buffers and per-operation buffers such as
  sort and join buffers come on top), but the calculation shows why thread-per-connection scales poorly to tens of
  thousands of connections: memory and scheduling overhead grow linearly with connections, whether or not they
  are doing anything.
```

```exercise
id: ex2
title: Size pools with Little's law
difficulty: medium
kind: query
dataset: empty
setup: |
  CREATE TABLE service_load (
    service    VARCHAR(30) PRIMARY KEY,
    peak_qps   INT NOT NULL,
    avg_hold_ms DECIMAL(8,2) NOT NULL
  );
  INSERT INTO service_load VALUES
    ('checkout', 1200, 4.50),
    ('catalog',  5000, 1.20),
    ('reports',    20, 850.00),
    ('auth',     3000, 0.80);
concepts: [connection-pooling]
prompt: |
  `service_load` records, for each service, its peak queries per second and the average time (ms) each query
  holds a connection. Using Little's law (busy connections = arrival rate × hold time), return for each service:

  - `service`
  - `busy_connections` — `peak_qps × avg_hold_ms / 1000`, rounded **up** to an integer

  Order by `busy_connections` descending, then `service`.
hints:
  - "Convert milliseconds to seconds by dividing by 1000."
  - "CEIL() rounds up."
solution: |
  SELECT service, CEIL(peak_qps * avg_hold_ms / 1000) AS busy_connections
  FROM service_load
  ORDER BY busy_connections DESC, service;
orderMatters: true
explanation: |
  `reports` needs the most connections despite 20 QPS, because each query holds a connection for 850 ms —
  hold time matters as much as rate. `catalog`, with 5,000 QPS, needs only 6 busy connections on average. Real
  pools add headroom for bursts (a percentile of hold time rather than the mean), but the order of magnitude
  is what Little's law gives you — and it is usually far smaller than teams expect.
```

```exercise
id: ex3
title: Will the fleet exhaust max_connections?
difficulty: hard
kind: query
dataset: empty
setup: |
  CREATE TABLE deployments (
    service   VARCHAR(30) PRIMARY KEY,
    instances INT NOT NULL,
    pool_max  INT NOT NULL
  );
  INSERT INTO deployments VALUES
    ('checkout', 12, 10),
    ('catalog',   8, 15),
    ('reports',   2,  5),
    ('admin-ui',  1,  5);
concepts: [max-connections, connection-pooling]
prompt: |
  Each application instance opens up to `pool_max` connections. Return **one row** with:

  1. `potential_connections` — Σ(instances × pool_max) across all services
  2. `max_connections` — the server's current `max_connections`
  3. `headroom` — `max_connections − potential_connections` (negative means the fleet can exhaust the server)
hints:
  - "SUM(instances * pool_max) gives the total."
  - "You can mix an aggregate with @@max_connections in the same SELECT list."
solution: |
  SELECT SUM(instances * pool_max) AS potential_connections,
         @@max_connections AS max_connections,
         @@max_connections - SUM(instances * pool_max) AS headroom
  FROM deployments;
explanation: |
  The fleet can open 255 connections against a server that allows 151 — a "Too many connections" incident waiting
  for the next traffic peak or deployment (during a rolling deploy, old and new instances briefly coexist, so the
  real peak is even higher). Options: shrink pools (Little's law usually shows they are oversized), put a
  multiplexing proxy in front, or raise max_connections only after checking memory and active-thread limits.
  Always leave headroom for replication, backups, monitoring and administrators.
```

```exercise
id: ex4
title: Spot the leaked transaction
difficulty: medium
kind: open
concepts: [connection-lifecycle, session-state, connection-pooling]
prompt: |
  At 14:02 an on-call engineer sees hundreds of `Lock wait timeout exceeded` errors (1205) on `UPDATE inventory`.
  `SHOW PROCESSLIST` shows no long-running statements; one connection from the `checkout` service has been in
  `Sleep` for 640 seconds. Explain what is probably happening, how you would confirm it, how to stop the incident,
  and two changes that prevent it from recurring.
solution: |
  **Hypothesis.** A `checkout` request opened a transaction, modified (and therefore X-locked) inventory rows, and
  returned its connection to the pool without COMMIT/ROLLBACK — e.g. an exception path skipped the cleanup. The
  connection sits idle (`Sleep`), but its transaction is still open and holds the row locks, so every other
  update of those rows waits up to `innodb_lock_wait_timeout` and fails with 1205.

  **Confirm.** Query `information_schema.INNODB_TRX` (or `performance_schema.data_locks` /
  `sys.innodb_lock_waits`): find a transaction with an old `trx_started`, `trx_rows_locked > 0` and
  `trx_mysql_thread_id` equal to the sleeping connection's id, and lock waits pointing at it as the blocker.

  **Stop the incident.** `KILL <thread_id>` the sleeping connection; InnoDB rolls its transaction back and releases
  the locks (the rollback may take time if the transaction is large). The pool will discover the dead connection
  and replace it.

  **Prevent.** (1) Configure the pool to roll back and reset on return (`COM_RESET_CONNECTION`), and fix the code
  path with try/finally or a framework-managed transaction scope. (2) Monitor and alert on transactions older than
  a threshold, and consider a lower `innodb_lock_wait_timeout` for interactive paths so failures surface quickly.
  Optionally a lower `wait_timeout` limits how long abandoned connections live.
```

```exercise
id: ex5
title: Design the connection architecture
difficulty: challenge
kind: open
concepts: [proxysql, mysql-router, connection-pooling, max-connections]
prompt: |
  A company runs one MySQL 8.4 primary with two replicas (32 cores each). Traffic: 40 Kubernetes services,
  autoscaling between 5 and 60 pods each, plus a fleet of serverless functions for webhooks. Today each pod opens a
  pool of 10 connections directly to the primary, and every Black Friday they hit "Too many connections". Propose a
  connection architecture, justify the choice between ProxySQL, MySQL Router and application pools, and give a
  concrete sizing rule and three monitoring signals.
solution: |
  **Problem.** Worst case 40 × 60 × 10 = 24,000 potential connections plus serverless bursts, against a server
  whose useful active concurrency is a few dozen threads. Direct per-pod pools cannot scale; connection count is
  coupled to pod count.

  **Architecture.** Keep small application-side pools (2–4 connections per pod) but point them at a **ProxySQL**
  layer (deployed as a few highly available instances or as sidecars), which multiplexes client sessions onto a
  bounded backend pool per server, routes `SELECT`s that tolerate replica lag to the replicas through query rules,
  and enforces per-user connection caps. Route serverless functions through the same proxy (or a managed proxy) so
  their bursts never reach `mysqld` as new connections. If the topology were an InnoDB Cluster and query-level
  routing were not needed, **MySQL Router** would be the simpler, officially supported choice for primary/replica
  routing and failover; ProxySQL wins here because of multiplexing at this scale and query rules.

  **Sizing.** Backend connections per server ≈ throughput × mean hold time (Little's law) with headroom, typically
  starting near 2–4 × cores for active work (≈ 64–128 on 32 cores), and `max_connections` set comfortably above the
  proxies' backend maximum plus admin/replication/backup/monitoring. Ensure sessions do not disable multiplexing
  (avoid user variables, temp tables and long transactions in request paths).

  **Monitoring.** (1) `Threads_running` vs cores and `Max_used_connections` vs `max_connections`; (2) proxy-side
  pool wait time and connection errors per backend; (3) `Threads_created`/`Connections` rate and `Aborted_connects`
  for churn and authentication problems — plus an alert on long-open transactions.
```

## Knowledge Assessment

```quiz
- id: q1
  question: "What does the 4-byte header of every MySQL protocol packet contain?"
  options:
    - "A 4-byte connection id"
    - "A 3-byte payload length and a 1-byte sequence id"
    - "A checksum of the payload"
    - "The command type and the statement id"
  answer: 1
  explanation: "Packets carry a 3-byte length (so at most 16 MB − 1 of payload per packet) and a sequence id that increments within one exchange. The connection id is sent once in the initial handshake; the command byte is the first byte of the payload, not of the header."
  concepts: [client-server-protocol]
- id: q2
  question: "`Connections` = 50,000 and `Threads_created` = 49,800 since startup. What does that suggest?"
  options:
    - "The thread cache is too small (or 0) for the connection churn — almost every connection created a new thread"
    - "max_connections is too low"
    - "Clients are using a connection pool efficiently"
    - "Authentication is failing for most clients"
  answer: 0
  explanation: "Threads_created counts threads made because none was cached. A ratio near 1 means the cache is not absorbing churn (and that clients connect and disconnect a lot — pooling would help more). Failed authentications show up in Aborted_connects."
  concepts: [thread-cache]
- id: q3
  question: "A service runs at 800 queries per second and each query holds a connection for 25 ms on average. Roughly how many connections are busy on average?"
  options: ["8", "20", "32", "800"]
  answer: 1
  explanation: "Little's law: L = λ × W = 800/s × 0.025 s = 20. A pool much larger than this adds queueing inside MySQL rather than throughput."
  concepts: [connection-pooling]
- id: q4
  type: multi
  question: "Which of the following does COM_RESET_CONNECTION clear? (Select all that apply.)"
  options:
    - "An open transaction (rolled back)"
    - "User variables such as @tenant_id"
    - "Temporary tables"
    - "Rows the session committed earlier"
    - "Prepared statements"
  answer: [0, 1, 2, 4]
  explanation: "Reset rolls back the transaction and resets session state — user variables, session variables, temporary tables, prepared statements, locks — without re-authenticating. Committed data is durable and unaffected."
  concepts: [session-state, connection-pooling]
- id: q5
  question: "Why can ProxySQL not always multiplex a client session onto a shared backend connection?"
  options:
    - "Because the MySQL protocol does not allow proxies"
    - "Because some session state (an open transaction, user variables, temporary tables, locks) is bound to a specific backend connection, so the proxy must pin the session to it"
    - "Because multiplexing requires the Enterprise thread pool"
    - "Because TLS prevents reading the packets"
  answer: 1
  explanation: "Multiplexing is only safe when the next statement does not depend on state stored in a particular server session. ProxySQL tracks such state and disables multiplexing for the session when it appears. Proxies terminate TLS themselves, and no thread pool is needed."
  concepts: [proxysql, session-state]
- id: q6
  question: "A pool's idle timeout is 10 hours; the server's wait_timeout is the default. What error will applications eventually see?"
  options:
    - "ERROR 1040 Too many connections"
    - "'MySQL server has gone away' / 'Lost connection' when using a connection the server closed after 8 hours idle"
    - "ERROR 1205 Lock wait timeout"
    - "No error; the server extends the timeout"
  answer: 1
  explanation: "The server closes connections idle longer than wait_timeout (28,800 s = 8 h). A pool that keeps them longer hands out dead connections (client errors 2006/2013) unless it validates them first. Keep pool lifetimes below wait_timeout."
  concepts: [connection-lifecycle]
- id: q7
  question: "In a processlist snapshot, a connection shows COMMAND = Sleep for 600 seconds. Which statement is correct?"
  options:
    - "It cannot be holding locks because it is not running a statement"
    - "It may be inside an open transaction holding row locks and an old read view; check INNODB_TRX"
    - "It will be closed after 600 more seconds by default"
    - "Sleep means the thread is in the thread cache"
  answer: 1
  explanation: "Sleep only means no statement is executing. Locks and read views belong to the transaction, which lasts until COMMIT/ROLLBACK. Default wait_timeout is 8 hours, and cached threads are not shown as connections at all."
  concepts: [connection-lifecycle, session-state]
- id: q8
  question: "In MySQL 8.4, a legacy client can connect only with mysql_native_password and now fails to authenticate. Why?"
  options:
    - "8.4 removed TLS support"
    - "mysql_native_password is disabled by default in 8.4; caching_sha2_password is the default plugin"
    - "8.4 requires the X Protocol"
    - "max_connections is 0 in 8.4"
  answer: 1
  explanation: "8.4 ships mysql_native_password disabled unless --mysql-native-password=ON; it is removed in 9.0. The fix is to upgrade the client/driver to support caching_sha2_password."
  concepts: [client-server-protocol]
```
