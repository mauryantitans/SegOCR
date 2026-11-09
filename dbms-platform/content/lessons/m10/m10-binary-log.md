---
id: m10-binary-log
dataset: bank
summary: The server-level binary log — why it exists (replication, point-in-time recovery, change data capture), its ROW, STATEMENT and MIXED formats, events and GTIDs, the internal two-phase commit that keeps it consistent with InnoDB's redo log, group commit and sync_binlog, and retention.
concepts: [binary-log, binlog-formats, row-based-logging, gtid, two-phase-commit, group-commit, sync-binlog, binlog-retention, point-in-time-recovery]
references:
  - title: "MySQL 8.4 Reference Manual — The Binary Log"
    url: https://dev.mysql.com/doc/refman/8.4/en/binary-log.html
  - title: "MySQL 8.4 Reference Manual — Binary Logging Formats"
    url: https://dev.mysql.com/doc/refman/8.4/en/binary-log-formats.html
  - title: "MySQL 8.4 Reference Manual — Determination of Safe and Unsafe Statements in Binary Logging"
    url: https://dev.mysql.com/doc/refman/8.4/en/replication-rbr-safe-unsafe.html
  - title: "MySQL 8.4 Reference Manual — Replication with Global Transaction Identifiers"
    url: https://dev.mysql.com/doc/refman/8.4/en/replication-gtids.html
  - title: "MySQL 8.4 Reference Manual — Binary Logging Options and Variables"
    url: https://dev.mysql.com/doc/refman/8.4/en/replication-options-binary-log.html
  - title: "MySQL 8.4 Reference Manual — Point-in-Time (Incremental) Recovery"
    url: https://dev.mysql.com/doc/refman/8.4/en/point-in-time-recovery.html
  - title: "MySQL 8.4 Reference Manual — mysqlbinlog"
    url: https://dev.mysql.com/doc/refman/8.4/en/mysqlbinlog.html
related: [m08-transaction-logs, m10-redo-undo, m10-crash-recovery, m10-server-architecture]
---

## Introduction

InnoDB's redo log answers one question: *how do I bring my pages back after a crash?* It is physical, private to InnoDB, circular, and useless outside the server that wrote it. Many other questions need a different log:

- How does a **replica** learn every change the primary committed, in commit order?
- How do you restore last night's backup and then **replay every change up to 14:31:59**, one second before someone dropped a table?
- How does a **change-data-capture** pipeline (Debezium, Kafka Connect, a search indexer) stream row changes to other systems?
- How do you audit *what* changed, regardless of storage engine?

The answer is the **binary log** (binlog): a server-level, engine-independent, append-only sequence of **logical** change events, written once per committed transaction, kept as numbered files for days. [Transaction Logs](/lessons/m08-transaction-logs) introduced it next to the redo and undo logs; this lesson covers its internals — formats, events, GTIDs, the two-phase commit with InnoDB, group commit, `sync_binlog`, and retention — and what each means for correctness.

:::note[Sandbox limits]
Reading the binary log requires administrative privileges: `SHOW BINARY LOGS`, `SHOW BINLOG EVENTS`, `SHOW BINARY LOG STATUS` and `mysqlbinlog` are not available in your lab, and the lab account cannot change `binlog_format` (even for its session). What you *can* observe is the binlog's configuration, the server-wide binlog-cache counters, and — through the session's `Handler_prepare` counter — exactly when the binlog takes part in a commit. Server output that cannot be captured (such as `mysqlbinlog` decodes) is shown as clearly labelled **illustrative examples**.
:::

## DBMS Theory

:::theory[Logical logs, replication and atomic commit]
A **logical log** records operations at the level of the data model — "row (id = 7) of table accounts changed balance from 120.00 to 95.00", or the SQL statement itself — rather than byte changes to pages. Logical logs are portable between servers with different physical layouts, which makes them the natural basis for **log-shipping replication**: a replica applies the primary's log in the same order and reaches the same state (*state-machine replication*). For this to work, the log must be:

1. **Complete and ordered** — every committed transaction appears exactly once, in a serialisation-consistent order (commit order).
2. **Deterministic to apply** — applying an entry on another server must yield the same result.
3. **Consistent with the database** — a transaction is in the log *if and only if* it is committed in the database.

When two independent durable components must agree on whether a transaction committed — here, the binlog (server) and the redo log (InnoDB) — the classic solution is **two-phase commit (2PC)**: a *coordinator* asks every participant to *prepare* (make the transaction durable but undecided), then records the decision durably, then tells participants to *commit*. After a crash, prepared participants ask the coordinator's durable record for the outcome.
:::

| Theory | MySQL |
|---|---|
| Logical log | Binary log: row images (ROW) or SQL text (STATEMENT) grouped into transactions |
| Log-shipping replication | Replica receiver thread copies binlog events into its *relay log*; applier threads execute them |
| Coordinator's durable decision | The transaction's events + **Xid event** written (and fsynced) to the binlog |
| Participants | InnoDB (prepare/commit hooks) and the `binlog` pseudo-engine |
| Global transaction ids | **GTIDs**: `source_uuid:transaction_number` (8.4 also allows tagged GTIDs, `uuid:tag:number`) |

## Why It Exists

The binary log exists because the redo log **cannot** do these jobs:

| Requirement | Why redo cannot meet it | How the binlog does |
|---|---|---|
| Replication to other servers | Redo describes page bytes of *this* instance's tablespaces; replicas have different page layouts | Logical row/statement events apply anywhere |
| Point-in-time recovery from a backup | Redo is circular (100 MB by default) and recycled after checkpoints | Binlog files are kept for days (`binlog_expire_logs_seconds`) |
| Engine independence | Redo is InnoDB-only | The binlog records changes for every engine |
| Only committed work | Redo contains uncommitted changes | Events are written only at commit |
| Change data capture | Redo format is internal and version-specific | Documented event format, decoded by `mysqlbinlog` and CDC libraries |

What goes wrong without a correct binlog — or with a mis-configured one — is replication divergence: a replica that silently differs from its primary because a statement behaved differently there (STATEMENT format), or because the primary committed a transaction that never reached its binlog (`sync_binlog ≠ 1` plus an OS crash), or because a replica applied the same transaction twice after failover (no GTIDs). These incidents are rarely loud; they surface weeks later as "the numbers don't match".

## Internal Working

### Events and files

:::internals[What is in a binary log file]
The binlog is a sequence of files `binlog.000001`, `binlog.000002`, … plus an index file `binlog.index` (base name from `log_bin_basename`). A new file is started at server start, at `FLUSH BINARY LOGS`, and when the current file exceeds `max_binlog_size` (1 GB by default; a transaction is never split across files). Each file is a sequence of **events**, each with a header (timestamp, type, server id, size, position of the next event, flags) and a body, optionally protected by a CRC32 checksum (`binlog_checksum`).

A committed InnoDB transaction in ROW format looks like:

| Event | Content |
|---|---|
| `Gtid` (or `Anonymous_Gtid` when GTIDs are off) | The transaction's GTID, plus logical timestamps used by parallel replication |
| `Query` | `BEGIN` |
| `Table_map` | Maps a numeric table id to `schema.table` and column types |
| `Write_rows` / `Update_rows` / `Delete_rows` | Row images: after image for inserts, before + after for updates, before for deletes |
| `Xid` | The commit marker with the internal XA transaction id |

Plus file-level events: `Format_description` (first event of each file), `Previous_gtids`, `Rotate` (pointer to the next file). DDL is always logged as a `Query` event with the SQL text, in every format.
:::

### Formats

| Format | What is logged | Strengths | Weaknesses |
|---|---|---|---|
| **ROW** (default since 5.7.7) | Row images of every changed row | Deterministic replay; replicas need no re-execution of the query; precise CDC | Volume ∝ rows changed (a one-line `UPDATE` of 10 M rows writes 10 M row images); SQL text not visible unless `binlog_rows_query_log_events = ON` |
| **STATEMENT** | The SQL text plus context (current database, timestamps, `@@` variables needed, auto-increment and `RAND()` seeds) | Compact for bulk statements; human-readable | **Unsafe** statements can produce different results on the replica: `UUID()`, `SYSDATE()`, `USER()`, `LIMIT` without `ORDER BY`, `INSERT … SELECT` from tables without a deterministic order, some trigger and stored-function cases, non-transactional + transactional mixes |
| **MIXED** | STATEMENT by default, switching to ROW for statements the server classifies as unsafe | Compromise | Two behaviours to understand; the classification is the server's, not yours |

Row images are controlled by `binlog_row_image`: `FULL` (all columns in both images, the default), `MINIMAL` (before image: only the primary key; after image: only changed columns) or `NOBLOB`. `binlog_row_value_options = PARTIAL_JSON` logs partial JSON updates compactly. `binlog_row_metadata = FULL` adds column names to `Table_map` events (helpful for CDC).

### GTIDs

:::internals[Global transaction identifiers]
With `gtid_mode = ON`, every transaction committed on a server gets a GTID `source_uuid:N`, where `source_uuid` is the originating server's `server_uuid` and N increments per transaction. A server tracks the set of GTIDs it has executed in `gtid_executed` (compactly as ranges, e.g. `3E11FA47-71CA-11E1-9E33-C80AA9429562:1-5291`) and persists it in `mysql.gtid_executed`.

Consequences:

- **Idempotent application.** A replica skips a transaction whose GTID is already in its `gtid_executed`, so the same transaction is never applied twice — even after failover to another source.
- **Auto-positioning.** A replica tells its source "I have this GTID set"; the source sends everything missing. No file names and byte offsets to compute after failover.
- **Consistency checks.** Comparing `gtid_executed` across servers shows exactly which transactions each has.
- **Restrictions.** `enforce_gtid_consistency = ON` rejects statements that cannot be logged as one transaction safely (for example, `CREATE TABLE … SELECT` before 8.0.21, updates mixing transactional and non-transactional tables in one transaction).

**8.3/8.4 tagged GTIDs**: `uuid:tag:N` lets administrators label transactions (e.g. maintenance jobs) with `SET gtid_next = 'AUTOMATIC:tag'`.
:::

### Two-phase commit between the binlog and InnoDB

With binary logging on, the server's binlog acts as the **transaction coordinator**, and every transaction that changes InnoDB data commits through internal XA:

```mermaid
sequenceDiagram
  participant S as Session thread
  participant I as InnoDB
  participant L as Leader of commit group
  participant B as Binary log file
  S->>I: Prepare (state PREPARED, XID in undo; redo written)
  S->>L: enter group commit queue
  Note over L: FLUSH stage — leader writes all members' binlog caches<br/>to the binlog and flushes InnoDB redo for their prepare
  L->>B: write events + Xid for every member
  Note over L: SYNC stage — one fsync of the binlog (sync_binlog = 1)
  L->>B: fsync
  Note over B: commit point: transaction is durable in the binlog
  Note over L: COMMIT stage — InnoDB commit in binlog order (binlog_order_commits)
  L->>I: commit each member (release locks, undo to history list)
  I-->>S: OK to client
```

:::internals[Prepare, commit point and recovery]
1. **Prepare.** InnoDB marks the transaction *prepared*, records its XID in the transaction's undo log header, and its redo is written. From now on, InnoDB can either commit or roll back the transaction after a crash — it no longer decides alone.
2. **Binlog write and sync — the commit point.** The transaction's events, accumulated during execution in the session's **binlog cache** (`binlog_cache_size`, spilling to a temporary file when larger — counted in `Binlog_cache_disk_use`), are appended to the binlog followed by an `Xid` event, and fsynced according to `sync_binlog`. Once the `Xid` event is durable in the binlog, the transaction *will* be committed.
3. **Commit.** InnoDB commits (a small redo record; with group commit, the redo flush for the commit can be lazy because the binlog already decides the outcome), locks are released, and the client receives OK.

**Group commit** pipelines these steps over three queues — *flush*, *sync*, *commit* — each with a leader that performs the stage for all queued sessions. One binlog fsync and one redo fsync then serve dozens of transactions. `binlog_group_commit_sync_delay` (microseconds) and `binlog_group_commit_sync_no_delay_count` can deliberately wait to form larger groups.

**Crash recovery** ([Crash Recovery](/lessons/m10-crash-recovery)): after redo replay, InnoDB reports its *prepared* transactions; the server scans the **last binlog file** for their XIDs. XID found → the transaction reached the binlog (replicas may already have it) → **commit**. XID not found → the client was never told OK and no replica has it → **roll back**. This is what guarantees "in the binlog ⇔ committed in InnoDB".
:::

:::version[Version notes: 8.0 → 8.4]
- **Binary logging is ON by default since 8.0** (`log_bin`), with `binlog_format = ROW` (default since 5.7.7) and `sync_binlog = 1` (since 5.7.7).
- **`binlog_format` is deprecated since 8.0.34** — reading it raises warning 1287 below; ROW is the long-term format.
- `binlog_expire_logs_seconds` (default 30 days) replaces the deprecated `expire_logs_days`; `binlog_expire_logs_auto_purge` (8.0.29+) can turn automatic purging off.
- `binlog_transaction_compression` (zstd) since 8.0.20.
- Terminology: `SHOW MASTER STATUS` → **`SHOW BINARY LOG STATUS`** (8.2+; the old form is removed in 8.4), `RESET MASTER` → **`RESET BINARY LOGS AND GTIDS`**, `log_slave_updates` → `log_replica_updates`.
- **8.4** removes `binlog_transaction_dependency_tracking`: dependency tracking for parallel replication always uses WRITESET. Tagged GTIDs are available (8.3+).
:::

```sql
-- 8.4 syntax (display only — requires REPLICATION CLIENT / admin privileges)
SHOW BINARY LOG STATUS;
SHOW BINARY LOGS;
SHOW BINLOG EVENTS IN 'binlog.000042' LIMIT 20;
PURGE BINARY LOGS BEFORE NOW() - INTERVAL 7 DAY;
RESET BINARY LOGS AND GTIDS;     -- destroys all binlogs: never on a source with replicas
```

## MySQL Implementation

The binary-log configuration of the build server:

```sql run
SELECT @@log_bin                      AS binlog_enabled,
       @@log_bin_basename             AS file_base_name,
       @@binlog_format                AS binlog_format,
       @@binlog_row_image             AS row_image,
       @@sync_binlog                  AS sync_binlog,
       @@max_binlog_size / 1024 / 1024 AS max_file_mb;
```

Binary logging is on with the 8.0 defaults — ROW format with full row images, fsync of every commit group (`sync_binlog = 1`), files rotated at 100 MB here (the build server sets `max_binlog_size` below the 1 GB default; the variable is a common per-installation choice). Note the deprecation warning attached to reading `@@binlog_format`.

```sql run
SELECT @@binlog_expire_logs_seconds          AS retention_seconds,
       @@binlog_expire_logs_seconds / 86400  AS retention_days,
       @@binlog_expire_logs_auto_purge       AS auto_purge,
       @@binlog_cache_size                   AS binlog_cache_bytes,
       @@binlog_checksum                     AS checksum,
       @@binlog_rows_query_log_events        AS log_sql_text_with_rows,
       @@binlog_transaction_compression      AS compression;
```

```sql run
SELECT @@server_id                AS server_id,
       @@server_uuid IS NOT NULL  AS has_server_uuid,
       @@gtid_mode                AS gtid_mode,
       @@enforce_gtid_consistency AS enforce_gtid_consistency,
       @@gtid_next                AS gtid_next,
       @@log_replica_updates      AS log_replica_updates,
       @@binlog_order_commits     AS order_commits,
       @@binlog_group_commit_sync_delay AS group_commit_delay_us;
```

GTIDs are **off** on the build server (the 8.0 and 8.4 default, although most production topologies turn them on), so transactions are logged with anonymous GTID events, and `gtid_next = AUTOMATIC` means the server assigns identifiers itself. (The server's UUID is not shown because it differs per installation.)

## Syntax Breakdown

| Variable | Default (8.0 / 8.4) | Meaning |
|---|---|---|
| `log_bin` | ON | Binary logging enabled (startup option `--log-bin[=base]`, `--skip-log-bin` to disable) |
| `binlog_format` | ROW (deprecated variable since 8.0.34) | ROW, STATEMENT or MIXED |
| `binlog_row_image` | FULL | FULL, MINIMAL, NOBLOB |
| `binlog_rows_query_log_events` | OFF | Also log the SQL text (Rows_query event) with row events |
| `sync_binlog` | 1 | 0 = never fsync explicitly (OS decides); 1 = fsync every commit group; N = every N groups |
| `binlog_cache_size` / `max_binlog_cache_size` | 32 KB / 16 EB | Per-session in-memory cache for a transaction's events; larger transactions spill to disk |
| `max_binlog_size` | 1 GB | Rotation threshold |
| `binlog_expire_logs_seconds` | 2,592,000 (30 days) | Automatic purge age |
| `binlog_expire_logs_auto_purge` | ON | Whether automatic purging runs (8.0.29+) |
| `gtid_mode`, `enforce_gtid_consistency` | OFF, OFF | GTID-based replication |
| `binlog_order_commits` | ON | Commit in binlog order (needed for some backup/replication guarantees) |
| `binlog_group_commit_sync_delay` | 0 µs | Wait to enlarge commit groups |
| `binlog_transaction_compression` | OFF | zstd compression of transaction payloads |

| Status counter (global) | Meaning |
|---|---|
| `Binlog_cache_use` | Transactions that used the binlog cache |
| `Binlog_cache_disk_use` | …of which spilled to a temporary file (cache too small for them) |
| `Binlog_stmt_cache_use` / `_disk_use` | Same for non-transactional statements |

## Practical Examples

### 1. When does the binlog take part in a commit?

`Handler_prepare` (session-scoped) counts prepare calls into transaction participants. An autocommit change to an InnoDB table has **two** participants — the `binlog` pseudo-engine and InnoDB — and therefore two prepare calls:

```sql run
CREATE TABLE audit_note (id INT PRIMARY KEY, note VARCHAR(40)) ENGINE = InnoDB;
CREATE TABLE audit_note_myisam (id INT PRIMARY KEY, note VARCHAR(40)) ENGINE = MyISAM;
CREATE TEMPORARY TABLE scratch_note (id INT PRIMARY KEY, note VARCHAR(40));
SHOW SESSION STATUS WHERE Variable_name = 'Handler_prepare';
INSERT INTO audit_note VALUES (1, 'innodb');
SHOW SESSION STATUS WHERE Variable_name = 'Handler_prepare';
INSERT INTO audit_note_myisam VALUES (1, 'myisam');
SHOW SESSION STATUS WHERE Variable_name = 'Handler_prepare';
INSERT INTO scratch_note VALUES (1, 'temporary');
SHOW SESSION STATUS WHERE Variable_name = 'Handler_prepare';
UPDATE audit_note SET note = 'nothing' WHERE id = 999;
SHOW SESSION STATUS WHERE Variable_name = 'Handler_prepare';
```

Compare consecutive snapshots:

| Statement | Δ `Handler_prepare` | Why |
|---|---|---|
| InnoDB `INSERT` | +2 | Internal XA: binlog + InnoDB both prepare, then the binlog decides |
| MyISAM `INSERT` | 0 | MyISAM is not transactional: its change is already applied; the event is written to the binlog without a prepare phase |
| Temporary-table `INSERT` | 0 | In ROW format, changes to temporary tables are not written to the binlog, so InnoDB is the only participant and commits in one phase |
| `UPDATE` matching no rows | 0 | Nothing changed, nothing to log — no two-phase commit needed |

This is the internal 2PC you saw in the sequence diagram, observed from SQL.

### 2. Large transactions spill the binlog cache

A transaction's row events are collected in the session's binlog cache (32 KB by default) and written to the binlog only at commit. A transaction larger than the cache spills to a temporary file:

```sql run
SET SESSION cte_max_recursion_depth = 10000;
SHOW GLOBAL STATUS LIKE 'Binlog_cache%';
INSERT INTO audit_note
  WITH RECURSIVE s(n) AS (SELECT 10 UNION ALL SELECT n + 1 FROM s WHERE n < 3000)
  SELECT n, REPEAT('z', 40) FROM s;
SHOW GLOBAL STATUS LIKE 'Binlog_cache%';
```

One transaction, about 3,000 row images of ~50 bytes each — well over 32 KB: `Binlog_cache_use` and `Binlog_cache_disk_use` both increased by one (these counters are global, so other activity could add to them). On a production server, a high `Binlog_cache_disk_use / Binlog_cache_use` ratio suggests raising `binlog_cache_size` — or, more often, splitting large transactions.

### 3. Non-deterministic functions and why ROW format is safe

Statement-based replication re-executes SQL on the replica. Some functions cannot give the same answer twice:

```sql run
SELECT UUID() = UUID() AS uuid_repeats,
       NOW() = NOW() AS now_is_statement_constant;
```

Two calls to `UUID()` in the same statement differ, so `INSERT INTO t VALUES (UUID())` logged as text would generate a *different* key on each replica. `SYSDATE()` returns the time of each call, so it changes during a long statement and differs on the replica; `NOW()` is fixed per statement and its value is shipped with statement events, so it is safe. In ROW format none of this matters: the replica receives the generated value itself. The lab cannot switch `binlog_format`, but on a server you administer, executing such a statement under `binlog_format = STATEMENT` produces warning 1592 ("Unsafe statement written to the binary log using statement format…").

### 4. What a ROW event looks like (illustrative)

The following is an **illustrative example** of `mysqlbinlog --base64-output=DECODE-ROWS --verbose binlog.000042` output for an `UPDATE accounts SET balance = balance - 25 WHERE account_id = 7` — not captured from the build server (the lab cannot read binlogs); positions, timestamps and the GTID are invented:

```text
# at 1534
#261109 14:31:58 server id 1  end_log_pos 1613 CRC32 0x5b1e2a0c  GTID  last_committed=41  sequence_number=42
SET @@SESSION.GTID_NEXT= '3e11fa47-71ca-11e1-9e33-c80aa9429562:5292'/*!*/;
# at 1613
#261109 14:31:58 server id 1  end_log_pos 1697 CRC32 0x0f1d9b11  Query  thread_id=88  exec_time=0  error_code=0
BEGIN
# at 1697
#261109 14:31:58 server id 1  end_log_pos 1780 CRC32 0x7a3c5e21  Table_map: `bank`.`accounts` mapped to number 112
# at 1780
#261109 14:31:58 server id 1  end_log_pos 1902 CRC32 0x2c9d0b44  Update_rows: table id 112 flags: STMT_END_F
### UPDATE `bank`.`accounts`
### WHERE
###   @1=7
###   @5=120.00
### SET
###   @1=7
###   @5=95.00
# at 1902
#261109 14:31:58 server id 1  end_log_pos 1933 CRC32 0x91aa3e07  Xid = 6512
COMMIT/*!*/;
```

(With `binlog_row_image = FULL` all columns would be listed in both images; they are abbreviated here to the key and the changed column.) `last_committed` and `sequence_number` are the logical clock that lets replica applier threads run non-conflicting transactions in parallel.

## Execution Walkthrough

### One transfer, from SQL to the binlog

```walkthrough
title: The binlog's view of a money transfer
steps:
  - title: "1. The data before"
    explain: Two accounts in the bank dataset. Nothing has been logged yet for this transfer.
    sql: SELECT account_id, balance FROM accounts WHERE account_id IN (1, 2) ORDER BY account_id;
  - title: "2. Execute inside a transaction"
    explain: >-
      Each UPDATE changes one InnoDB row (undo + redo inside InnoDB) and appends an Update_rows event — with the
      before and after image of the row — to this session's binlog cache, in memory. The binary log file itself is
      untouched until COMMIT, so a replica cannot see a half-finished transfer.
    sql: |
      START TRANSACTION;
      UPDATE accounts SET balance = balance - 25.00 WHERE account_id = 1;
      UPDATE accounts SET balance = balance + 25.00 WHERE account_id = 2;
  - title: "3. Commit — two-phase"
    explain: >-
      COMMIT runs the internal XA protocol: InnoDB prepare, then the group-commit leader writes the cached events
      (GTID/anonymous GTID, BEGIN, Table_map, two Update_rows, Xid) to the binlog file and fsyncs it
      (sync_binlog = 1) — the commit point — then InnoDB commits and the client gets OK.
    sql: COMMIT;
  - title: "4. The data after"
    explain: >-
      The new balances are committed. The same change now exists in three forms: redo records (for this server's
      crash recovery), update undo (until purge), and row events in the binlog (for replicas, point-in-time
      recovery and CDC, kept for binlog_expire_logs_seconds).
    sql: SELECT account_id, balance FROM accounts WHERE account_id IN (1, 2) ORDER BY account_id;
```

## Output

The walkthrough moved 25.00 between accounts 1 and 2 in one atomic transaction; the earlier blocks showed:

- the build server's binlog configuration: enabled, ROW format, full images, `sync_binlog = 1`, 30-day retention, GTIDs off;
- `Handler_prepare` rising by 2 for an InnoDB write (two-phase commit) and by 0 for MyISAM, temporary-table and no-op writes;
- a 3,000-row transaction spilling the 32 KB binlog cache to disk (`Binlog_cache_disk_use` + 1);
- `UUID()` returning different values within one statement — the kind of non-determinism that ROW format makes harmless.

## Alternative Approaches

**Choosing a format.**

| Situation | Choice |
|---|---|
| General OLTP, replication, CDC | **ROW** (default) with `binlog_row_image = FULL` for CDC tools, or `MINIMAL` to cut volume when consumers do not need full images |
| Very large bulk statements on a dedicated pipeline | Prefer batching under ROW; STATEMENT only if every statement is deterministic and you accept the risks — and note the variable is deprecated |
| Legacy topologies | MIXED is a historical compromise; migrate to ROW |

**Disabling the binlog.** `--skip-log-bin` saves an fsync per commit group and disk space, and removes the internal 2PC. It also removes replication, point-in-time recovery and CDC. It is reasonable for disposable instances (CI, scratch analytics, a replica that will never be promoted and has no downstream consumers).

**Other change-capture designs.**

| Approach | Notes |
|---|---|
| Binlog-based CDC (Debezium, Maxwell, Canal, AWS DMS) | Reads the binlog as a replica would; needs ROW format and enough retention |
| Trigger-based audit tables | Works without binlog access, but adds write cost inside every transaction (Module 9) |
| PostgreSQL logical decoding | The analogous facility in PostgreSQL, built on its WAL rather than a separate log |
| Application outbox table | The application writes an "event" row in the same transaction; a relay publishes it — consistent with business transactions, independent of binlog formats |

## Performance Analysis

| Cost | Driver | Mitigation |
|---|---|---|
| Commit latency | One binlog fsync per commit group (`sync_binlog = 1`) in addition to the redo fsync | Group commit (concurrency), fast storage; `binlog_group_commit_sync_delay` trades latency for throughput |
| Volume | ROW: rows × row-image size | `binlog_row_image = MINIMAL`, `binlog_transaction_compression = ON`, avoid no-op mass updates |
| Large transactions | Cache spill to disk; one huge event group written at commit; replica applies it as a unit (lag) | Batch changes; raise `binlog_cache_size` only for legitimately large transactions |
| Disk space | Retention × write rate | Size `binlog_expire_logs_seconds` to backup frequency + safety; monitor free space |
| Replica throughput | Single-threaded apply of dependent transactions | Multi-threaded appliers with WRITESET dependency tracking (default in 8.4) |

:::performance[Theory vs measurement]
The lab cannot measure binlog fsync latency or volume: it cannot read the binlog files or `SHOW BINARY LOG STATUS` positions, and the binlog counters it can read are server-wide. On a server you administer, measure binlog bytes per second from successive `SHOW BINARY LOG STATUS` positions, the spill ratio `Binlog_cache_disk_use / Binlog_cache_use`, and commit latency from `performance_schema.events_transactions_summary_global_by_event_name`, before and after changing `sync_binlog`, `binlog_row_image` or compression.
:::

## Best Practices

:::best-practice[Binary log in production]
1. **Keep `log_bin = ON`, `binlog_format = ROW` and `sync_binlog = 1`** together with `innodb_flush_log_at_trx_commit = 1` on any server whose data matters or that has replicas.
2. **Enable GTIDs** (`gtid_mode = ON`, `enforce_gtid_consistency = ON`) for replication topologies: failover without file/position arithmetic and protection against double-applying transactions.
3. **Set retention from your recovery plan**: binlogs must cover at least the interval between full backups plus a margin; with daily backups, 7 days is common; CDC consumers may need more.
4. **Batch large changes** so that binlog transactions stay small, caches do not spill, and replicas do not lag.
5. **Use `binlog_row_image = FULL` when CDC consumers need complete rows**; `MINIMAL` only after checking every consumer.
6. **Never `RESET BINARY LOGS AND GTIDS` or delete binlog files by hand on a source**; use `PURGE BINARY LOGS` after verifying replicas and backups no longer need them.
7. **Back up binlogs continuously** (e.g. `mysqlbinlog --read-from-remote-server --raw --stop-never` to another host) so point-in-time recovery survives the loss of the primary.
:::

## Common Mistakes

**Mistake 1 — `sync_binlog = 0` "for performance".** After an OS crash or power loss, InnoDB may have committed transactions (redo fsynced) whose events never reached the binlog file. The primary then contains data its replicas will never receive. Keep `sync_binlog = 1` whenever replicas or PITR matter.

**Mistake 2 — assuming the binlog contains uncommitted or rolled-back work.** It does not; events are written at commit only. A rolled-back transaction leaves no trace in the binlog (except for changes to non-transactional tables, which are logged because they cannot be undone).

**Mistake 3 — trying to change the format per session as an application user.**

```sql run error
SET SESSION binlog_format = 'STATEMENT';
```

Changing `binlog_format` even for a session requires `SYSTEM_VARIABLES_ADMIN` or `SESSION_VARIABLES_ADMIN`: the format affects replicas, not just the session. (And the variable itself is deprecated.)

**Mistake 4 — mixing InnoDB and MyISAM tables in one transaction.** The MyISAM change is logged immediately (it cannot be rolled back), the InnoDB change at commit; after a rollback the replica gets the MyISAM part only, and with `enforce_gtid_consistency = ON` such statements are rejected.

**Mistake 5 — a retention period shorter than the backup interval.** Weekly full backups with 3-day binlog retention make point-in-time recovery impossible for most of the week.

## Industry Application

- **Replication everywhere.** Every MySQL replication topology — classic asynchronous replicas, semi-synchronous replication, Group Replication/InnoDB Cluster (which certifies transactions using the binlog's write sets), managed read replicas in the cloud — is built on the binary log.
- **Point-in-time recovery.** Runbooks restore the latest full backup, then replay binlogs with `mysqlbinlog --start-position/--stop-datetime` or `--exclude-gtids/--include-gtids` up to just before the incident (an accidental `DROP TABLE` or bad `UPDATE`). Cloud services expose this as "restore to any second within the retention window".
- **Change data capture.** Debezium and similar tools read ROW-format binlogs to stream changes to Kafka for search indexes, caches, data warehouses and microservices; they require `binlog_row_image = FULL` (or careful handling of MINIMAL) and enough retention to survive consumer outages.
- **Online schema change.** gh-ost reads the binlog to apply ongoing changes to its shadow table while copying rows, instead of using triggers.
- **Auditing and forensics.** After an incident, engineers decode binlogs to find exactly which transaction changed a row, from which thread, at what time.

## Practice Exercises

```exercise
id: ex1
title: Binlog configuration summary
difficulty: easy
kind: query
dataset: bank
concepts: [binary-log, binlog-retention, sync-binlog]
prompt: |
  Return one row with:

  1. `binlog_on` — the value of `log_bin`
  2. `retention_days` — `binlog_expire_logs_seconds` converted to days (decimal result is fine)
  3. `sync_binlog` — the value of `sync_binlog`
  4. `fully_durable_binlog` — `'yes'` if `sync_binlog = 1`, otherwise `'no'`
hints:
  - "Read system variables with @@name."
  - "Use CASE or IF() for the last column."
solution: |
  SELECT @@log_bin AS binlog_on,
         @@binlog_expire_logs_seconds / 86400 AS retention_days,
         @@sync_binlog AS sync_binlog,
         IF(@@sync_binlog = 1, 'yes', 'no') AS fully_durable_binlog;
explanation: |
  On default 8.0/8.4 servers: binlog on, 30 days retention, sync_binlog = 1. Only sync_binlog = 1 guarantees that a
  transaction acknowledged to the client is in the binary log after an OS crash; any other value can leave the
  primary with committed transactions its replicas never receive.
```

```exercise
id: ex2
title: Which binlog files would be purged?
difficulty: medium
kind: query
dataset: empty
setup: |
  CREATE TABLE binlog_files (
    file_name  VARCHAR(20) PRIMARY KEY,
    size_mb    INT NOT NULL,
    closed_at  DATETIME NULL
  );
  INSERT INTO binlog_files VALUES
    ('binlog.000101', 1024, '2026-09-20 03:10:00'),
    ('binlog.000102', 1024, '2026-10-02 11:45:00'),
    ('binlog.000103', 1024, '2026-10-11 08:00:00'),
    ('binlog.000104',  980, '2026-10-25 17:20:00'),
    ('binlog.000105',  512, '2026-11-03 09:30:00'),
    ('binlog.000106',  130, NULL);
concepts: [binlog-retention]
prompt: |
  `binlog_files` lists binary log files and when each was closed (NULL = the current file). Automatic purging removes
  files older than `binlog_expire_logs_seconds`. Taking **`2026-11-09 12:00:00`** as "now" and using the server's
  actual `binlog_expire_logs_seconds`, return the `file_name` and `size_mb` of the files that are eligible for
  purging, ordered by `file_name`.
hints:
  - "A file is eligible when closed_at < now − retention."
  - "TIMESTAMP '2026-11-09 12:00:00' - INTERVAL @@binlog_expire_logs_seconds SECOND"
  - "The current file (closed_at IS NULL) is never purged; a NULL comparison excludes it automatically."
solution: |
  SELECT file_name, size_mb
  FROM binlog_files
  WHERE closed_at < TIMESTAMP '2026-11-09 12:00:00' - INTERVAL @@binlog_expire_logs_seconds SECOND
  ORDER BY file_name;
orderMatters: true
explanation: |
  With the default 30-day retention the cutoff is 2026-10-10 12:00:00, so binlog.000101 and binlog.000102 are
  eligible. Before purging in production, also check that no replica still needs them (each replica's received
  GTIDs/positions) and that the latest backup is newer than the cutoff — otherwise point-in-time recovery from that
  backup becomes impossible. MySQL's own purge only checks age (and refuses files in use by a connected replica).
```

```exercise
id: ex3
title: How far behind is each replica (in transactions)?
difficulty: hard
kind: query
dataset: empty
setup: |
  CREATE TABLE gtid_status (
    server_name   VARCHAR(20) PRIMARY KEY,
    role          ENUM('source', 'replica') NOT NULL,
    gtid_executed VARCHAR(100) NOT NULL
  );
  INSERT INTO gtid_status VALUES
    ('db-primary', 'source',  '3e11fa47-71ca-11e1-9e33-c80aa9429562:1-52930'),
    ('db-r1',      'replica', '3e11fa47-71ca-11e1-9e33-c80aa9429562:1-52930'),
    ('db-r2',      'replica', '3e11fa47-71ca-11e1-9e33-c80aa9429562:1-52811'),
    ('db-r3',      'replica', '3e11fa47-71ca-11e1-9e33-c80aa9429562:1-49002');
concepts: [gtid]
prompt: |
  Each server reports a single-range GTID set of the form `uuid:1-N` for the same source UUID. For every **replica**,
  return `server_name` and `missing_trx` — the number of the source's transactions the replica has not executed
  yet (source N − replica N). Order by `missing_trx` descending.
hints:
  - "The last number of the range is SUBSTRING_INDEX(gtid_executed, '-', -1)."
  - "CAST it to UNSIGNED and join (or cross join) each replica with the source row."
solution: |
  SELECT r.server_name,
         CAST(SUBSTRING_INDEX(s.gtid_executed, '-', -1) AS UNSIGNED)
           - CAST(SUBSTRING_INDEX(r.gtid_executed, '-', -1) AS UNSIGNED) AS missing_trx
  FROM gtid_status r
  JOIN gtid_status s ON s.role = 'source'
  WHERE r.role = 'replica'
  ORDER BY missing_trx DESC;
orderMatters: true
explanation: |
  db-r3 is 3,928 transactions behind, db-r2 119, db-r1 none. This is the GTID advantage: lag and catch-up are
  expressed in transactions, independent of file names and byte offsets, and a promoted replica can ask any other
  server for exactly the missing set. Real GTID sets can contain several UUIDs and gaps (e.g. `uuid:1-100:105-200`);
  in production use the built-in functions `GTID_SUBTRACT(source_set, replica_set)` and `GTID_SUBSET()` instead of
  string parsing.
```

```exercise
id: ex4
title: Crash at every point of the two-phase commit
difficulty: hard
kind: open
concepts: [two-phase-commit, sync-binlog]
prompt: |
  With `innodb_flush_log_at_trx_commit = 1` and `sync_binlog = 1`, a transaction commits through: (1) InnoDB prepare
  with redo flushed, (2) binlog write, (3) binlog fsync, (4) InnoDB commit, (5) OK to the client. For a crash
  **between each pair of steps** (after 1, after 2, after 3, after 4), state whether the transaction is committed or
  rolled back after recovery, whether replicas will receive it, and whether the client could have seen OK. Then
  describe what changes if `sync_binlog = 0`.
solution: |
  - **After 1 (prepared, nothing in binlog):** recovery finds the transaction PREPARED in InnoDB, scans the last binlog,
    does not find its XID → **rolled back**. Not in the binlog → replicas never get it. The client never saw OK. Consistent.
  - **After 2 (written but not fsynced):** if the OS survived (mysqld crash only), the events are in the file → XID found
    → **committed**, replicas will receive it. If the OS crashed, the unsynced write may be lost → XID not found →
    **rolled back**, not on replicas. Either way the client never saw OK, and binlog and InnoDB agree.
  - **After 3 (binlog durable):** XID found → **committed**; replicas receive it. The client did not see OK, so the
    application might retry: idempotency (or checking the result) is the application's responsibility.
  - **After 4 (InnoDB committed, OK not yet sent):** committed, in the binlog, replicated. Client did not see OK — same
    retry caveat.

  The invariant is "in the binlog ⇔ committed in InnoDB", and "OK ⇒ durable in both".

  **With `sync_binlog = 0`:** step 3 is skipped; the binlog write sits in the OS cache. After an **OS crash or power
  loss**, transactions may have been acknowledged (OK sent, InnoDB committed with a durable redo) while their binlog
  events were lost — and because they are already committed in InnoDB, recovery cannot roll them back. The primary
  keeps them, replicas never receive them: silent divergence, and PITR from binlogs misses them. A mysqld-only crash is
  still safe because the OS cache survives.
```

```exercise
id: ex5
title: Plan a point-in-time recovery
difficulty: challenge
kind: open
concepts: [point-in-time-recovery, binary-log, gtid]
prompt: |
  At 14:32:05 a developer ran `DELETE FROM transactions WHERE created_at < '2026-01-01'` on the production primary
  (MySQL 8.4, GTIDs on, ROW format, sync_binlog = 1, 7-day binlog retention, nightly full backup at 02:00 with the
  binlog position/GTID set recorded). The business wants all data back as of just before the mistake, including the
  legitimate transactions committed after it. Write the recovery plan step by step, including how you find the bad
  transaction, how you replay the binlogs, and the alternatives if only the `transactions` table must be repaired.
solution: |
  1. **Stop the damage and preserve evidence.** Block writes if needed; copy all current binlog files off the server
     (or ensure the binlog backup stream is up to date). Do not purge or reset anything.
  2. **Find the bad transaction.** Decode the binlogs around 14:32 with
     `mysqlbinlog --base64-output=DECODE-ROWS -v --start-datetime='… 14:31:00' --stop-datetime='… 14:33:00'` and locate the
     large `Delete_rows` group on `bank.transactions`; note its GTID (e.g. `uuid:88123`). With
     `binlog_rows_query_log_events = ON` the original SQL text would be visible too.
  3. **Restore the backup** (02:00) to a new instance (never over the primary), and set `gtid_purged` to the backup's
     GTID set (backup tools do this).
  4. **Replay binlogs excluding the bad transaction:** `mysqlbinlog --exclude-gtids='uuid:88123' --skip-gtids=false
     binlog.000231 binlog.000232 … | mysql` on the restored instance — replaying everything from the backup's position to
     the end except the DELETE. (Without GTIDs: replay with `--stop-position` just before the DELETE, then
     `--start-position` just after it.)
  5. **Validate** (row counts, checksums of key tables, application smoke tests), then **switch** traffic to the
     restored instance or use it as the source to rebuild replicas.

  **Repairing only the table.** Restore the backup to a scratch instance, replay binlogs up to just before the DELETE
  (`--include-gtids`/`--stop-position`, or filter with `--database`/a replica with `replicate-do-table`), export the
  deleted rows (`created_at < '2026-01-01'`) and re-insert them on the primary — keeping the 14:32–now changes intact.
  Alternatively, because the binlog is ROW format with full images, tools can *invert* the `Delete_rows` events into
  `INSERT`s ("flashback" tools such as binlog2sql/MyFlash) and apply them directly. Afterwards: add guardrails
  (restricted privileges on production, `sql_safe_updates` for interactive sessions, delayed replicas).
```

## Knowledge Assessment

```quiz
- id: q1
  question: "Which of these can the binary log do that the InnoDB redo log cannot?"
  options:
    - "Recover dirty pages after a crash"
    - "Feed replicas and point-in-time recovery with logical, engine-independent change events kept for days"
    - "Store uncommitted changes for rollback"
    - "Detect torn pages"
  answer: 1
  explanation: "The binlog is logical, retained as files and written at commit — ideal for replication, PITR and CDC. Page recovery is redo's job, rollback is undo's, and torn pages are handled by the doublewrite buffer."
  concepts: [binary-log]
- id: q2
  question: "An autocommit INSERT into an InnoDB table raises Handler_prepare by 2, but an INSERT into a temporary table raises it by 0. Why?"
  options:
    - "Temporary tables are MyISAM"
    - "In ROW format changes to temporary tables are not binlogged, so InnoDB is the only participant and no two-phase commit is needed"
    - "Temporary tables are never committed"
    - "The counter ignores temporary tables by design"
  answer: 1
  explanation: "The two prepares are the binlog pseudo-engine and InnoDB in the internal XA commit. With only one participant (InnoDB), the server commits in one phase. Temporary tables use InnoDB by default and are committed normally."
  concepts: [two-phase-commit]
- id: q3
  question: "What is the commit point of a transaction when binary logging is enabled?"
  options:
    - "When InnoDB writes the prepare record"
    - "When the transaction's Xid event is durably written to the binary log"
    - "When the dirty pages are flushed"
    - "When the replica acknowledges the transaction"
  answer: 1
  explanation: "After a crash, a prepared transaction is committed if and only if its XID is in the binlog. Page flushing is unrelated to commit, and asynchronous replicas acknowledge nothing (semi-sync adds an acknowledgement, but the binlog decision comes first)."
  concepts: [two-phase-commit]
- id: q4
  type: multi
  question: "Which statements are unsafe for STATEMENT-based logging (they may produce different results on a replica)? (Select all that apply.)"
  options:
    - "INSERT INTO t (id) VALUES (UUID())"
    - "UPDATE t SET flag = 1 ORDER BY id LIMIT 10"
    - "DELETE FROM t LIMIT 100 (no ORDER BY)"
    - "INSERT INTO log (created) VALUES (SYSDATE())"
    - "UPDATE accounts SET balance = balance - 25 WHERE account_id = 7"
  answer: [0, 2, 3]
  explanation: "UUID() and SYSDATE() are non-deterministic; LIMIT without ORDER BY may pick different rows on the replica. With ORDER BY on a unique key the row set is deterministic, and an update by primary key is deterministic."
  concepts: [binlog-formats]
- id: q5
  question: "With GTIDs enabled, why can a replica safely be pointed at a different source after failover?"
  options:
    - "Because binlog file names are identical on all servers"
    - "Because it sends its gtid_executed set and receives exactly the transactions it is missing; transactions it already has are skipped"
    - "Because GTIDs disable the relay log"
    - "Because the new source replays its entire binlog from the beginning"
  answer: 1
  explanation: "GTID auto-positioning removes file/offset arithmetic, and GTID-based skipping makes application idempotent. File names differ between servers; relay logs are still used."
  concepts: [gtid]
- id: q6
  question: "A server runs with innodb_flush_log_at_trx_commit = 1 but sync_binlog = 0. The host loses power. What can happen?"
  options:
    - "Nothing — redo protects everything"
    - "Transactions acknowledged to clients may be committed in InnoDB but missing from the binlog, so replicas and PITR never see them"
    - "All transactions since the last checkpoint are rolled back"
    - "The binlog contains transactions InnoDB rolled back"
  answer: 1
  explanation: "Without a binlog fsync, binlog writes in the OS cache are lost on power failure, while InnoDB's redo was fsynced; those transactions stay committed on the primary but are absent from the binlog — silent divergence."
  concepts: [sync-binlog]
- id: q7
  question: "Binlog_cache_disk_use is a large fraction of Binlog_cache_use. What does it indicate?"
  options:
    - "The binary log files are on a slow disk"
    - "Many transactions are larger than binlog_cache_size and spill their events to temporary files before commit"
    - "Replicas are lagging"
    - "Binary log compression is enabled"
  answer: 1
  explanation: "The binlog cache holds a transaction's events until commit; overflow goes to a temporary file. Raise binlog_cache_size for legitimately large transactions, or split them. It says nothing direct about replica lag or compression."
  concepts: [binary-log]
- id: q8
  question: "In MySQL 8.4, which statement shows the current binary log file and position?"
  options: ["SHOW MASTER STATUS", "SHOW BINARY LOG STATUS", "SHOW BINLOG POSITION", "SELECT @@binlog_position"]
  answer: 1
  explanation: "SHOW BINARY LOG STATUS (introduced in 8.2) replaces SHOW MASTER STATUS, which was removed in 8.4. The other two do not exist."
  concepts: [binary-log]
```
