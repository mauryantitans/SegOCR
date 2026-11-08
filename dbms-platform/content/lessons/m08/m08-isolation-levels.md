---
id: m08-isolation-levels
dataset: bank
summary: The four SQL isolation levels as InnoDB actually implements them — read views, gap locks, semi-consistent reads and SERIALIZABLE's shared locks — where InnoDB deviates from the SQL standard, how to set a level, and how to choose one deliberately.
concepts: [isolation-levels, read-uncommitted, read-committed, repeatable-read, serializable, snapshot-isolation, gap-locks]
references:
  - title: "MySQL 8.4 Reference Manual — Transaction Isolation Levels"
    url: https://dev.mysql.com/doc/refman/8.4/en/innodb-transaction-isolation-levels.html
  - title: "MySQL 8.4 Reference Manual — SET TRANSACTION Statement"
    url: https://dev.mysql.com/doc/refman/8.4/en/set-transaction.html
  - title: "MySQL 8.4 Reference Manual — Consistent Nonlocking Reads"
    url: https://dev.mysql.com/doc/refman/8.4/en/innodb-consistent-read.html
  - title: "MySQL 8.4 Reference Manual — Locks Set by Different SQL Statements in InnoDB"
    url: https://dev.mysql.com/doc/refman/8.4/en/innodb-locks-set.html
  - title: "MySQL 8.4 Reference Manual — Server System Variables (transaction_isolation)"
    url: https://dev.mysql.com/doc/refman/8.4/en/server-system-variables.html#sysvar_transaction_isolation
related: [m08-read-phenomena, m08-mvcc, m08-locking]
---

## Introduction

The previous lesson catalogued the anomalies that concurrency can cause. An **isolation level** is a contract that says which of those anomalies a transaction is protected from. The SQL standard defines four levels; InnoDB implements all four, with REPEATABLE READ as its default.

Choosing a level is an engineering decision with real consequences:

- **READ COMMITTED** is the default of PostgreSQL, Oracle and SQL Server, and many large MySQL deployments switch to it to reduce gap-lock contention and deadlocks.
- **REPEATABLE READ**, MySQL's default, gives each transaction a stable snapshot and blocks phantoms for locking reads with gap locks — but, as you will see, it is **not** the same as snapshot isolation in PostgreSQL, and it lets lost updates through without an error.
- **SERIALIZABLE** in InnoDB is implemented with shared locks on every read, not with optimistic certification. It is correct but blocks more and deadlocks more.
- **READ UNCOMMITTED** is almost never the right choice.

This lesson explains, for each level, **what InnoDB does internally**, demonstrates the behaviour with real sessions, shows exactly where InnoDB departs from the standard, and ends with a practical decision guide.

:::note[Dataset]
The scenarios use the **bank** dataset (accounts 1 and 2 belong to customer 1 and hold 492.57 and 4428.20; account 7 is frozen; `status` has no index, `customer_id` has `idx_accounts_customer`). The read-phenomena lesson already reproduced dirty reads, non-repeatable reads, phantoms, lost updates and write skew; this lesson focuses on what *changes* between levels.
:::

## DBMS Theory

:::theory[Isolation levels in the SQL standard]
SQL-92 defines isolation levels by the phenomena they must **not** exhibit:

| Level | Dirty read (P1) | Non-repeatable read (P2) | Phantom (P3) |
|---|---|---|---|
| READ UNCOMMITTED | possible | possible | possible |
| READ COMMITTED | — | possible | possible |
| REPEATABLE READ | — | — | possible |
| SERIALIZABLE | — | — | — |

SERIALIZABLE additionally requires that concurrent execution be equivalent to *some* serial execution. The standard says nothing about how a DBMS achieves this, and allows an implementation to provide a stronger level than requested.
:::

Two families of implementation exist, and they behave differently even under the same level names:

1. **Lock-based isolation** (classic two-phase locking). Writers take exclusive locks until commit. Readers take shared locks: released immediately after the read (READ COMMITTED), held until commit (REPEATABLE READ), or held until commit on rows **and ranges** (SERIALIZABLE, via predicate or key-range locks). Readers and writers block each other.
2. **Multiversion (snapshot) isolation.** Readers read a consistent snapshot of committed data and take no locks; writers still lock. Snapshot isolation (SI) as defined by Berenson et al. also requires **first-committer-wins**: if two concurrent transactions write the same item, the second must abort. SI prevents P1, P2, P3 (strict interpretation) and lost updates, but allows **write skew**, so it is not serializable. *Serializable snapshot isolation* (SSI, used by PostgreSQL's SERIALIZABLE) adds tracking of read-write dependencies to abort the transactions that could cause write skew.

InnoDB is a **hybrid**: consistent reads use multiversioning (like SI), but writes and locking reads use two-phase locking on the **latest** version, with no first-committer-wins check. That hybrid explains every InnoDB-specific behaviour in this lesson.

:::standard[Standard SQL vs InnoDB — the deviations]
- **REPEATABLE READ prevents phantoms for consistent reads** (the standard permits them) using the transaction's snapshot, and **for locking reads** using next-key/gap locks.
- **REPEATABLE READ is not snapshot isolation**: `UPDATE`, `DELETE` and locking reads operate on the latest committed rows, not the snapshot, and there is no first-committer-wins abort. A transaction can therefore overwrite a change it never saw (a lost update) without an error — PostgreSQL's REPEATABLE READ would fail with a serialization error. After such an update, the transaction's plain `SELECT`s see the updated row, mixing snapshot and current data.
- **SERIALIZABLE** is REPEATABLE READ plus implicit `FOR SHARE` on plain `SELECT`s in non-autocommit transactions. With autocommit enabled, a plain `SELECT` is a consistent read even at SERIALIZABLE.
- **READ UNCOMMITTED** still never permits dirty *writes*: every write locks.
- **READ COMMITTED** in InnoDB disables gap locking (except for foreign-key and duplicate-key checks) and uses *semi-consistent reads* for `UPDATE`, which changes locking behaviour, not just read visibility.
:::

```viz
type: isolation
```

## Why It Exists

A single, fully serializable level for everything would be simplest to reason about, and it is what every developer implicitly assumes when they write a transaction. It also costs concurrency: readers block writers, range reads block inserters, and deadlocks multiply. Isolation levels let each transaction pay only for the guarantees it needs:

- A monitoring query that estimates queue depth does not care about a few uncommitted rows.
- A web request that reads a product page and writes an order line needs committed data, but not a snapshot frozen across statements.
- A month-end report that runs fifty queries must see one consistent state of the books.
- A seat-booking transaction that checks availability and inserts a booking must not race with another booking — it needs serializable behaviour for that check, or explicit locking.

Different defaults across products also exist for historical reasons. MySQL chose REPEATABLE READ partly because, for many years, its **statement-based** binary log required it: replaying statements on a replica only reproduces the primary's result if concurrent transactions could not insert into ranges a statement had read. Gap locks at REPEATABLE READ guaranteed that. With row-based binary logging (the default since 5.7.7), READ COMMITTED is safe for replication, and many large deployments use it.

## Internal Working

:::internals[What each level changes inside InnoDB]
| Level | Plain `SELECT` (consistent read) | Locking reads, `UPDATE`, `DELETE` | Gap / next-key locks |
|---|---|---|---|
| READ UNCOMMITTED | No read view: returns the newest version of each row, committed or not | Lock the latest version | Not used for searches (as in READ COMMITTED) |
| READ COMMITTED | **New read view for every statement** | Lock latest committed version; rows that do not match the `WHERE` are unlocked after evaluation; `UPDATE` uses **semi-consistent reads** | Disabled, except for foreign-key and duplicate-key checks |
| REPEATABLE READ | **One read view per transaction**, created by the first consistent read (or by `START TRANSACTION WITH CONSISTENT SNAPSHOT`) | Lock every index record the scan touches, for the whole transaction | **Yes**: next-key locks on range scans, gap locks where no record matches; a unique search for one existing row locks only that record |
| SERIALIZABLE | Becomes `SELECT … FOR SHARE` (when autocommit is off / inside a transaction) | As REPEATABLE READ | Yes |
:::

**Read views.** A read view records, at the moment it is created, which transactions are still active (not committed). A row version is visible if the transaction that wrote it committed before the read view was created, or is the reader itself. READ COMMITTED's per-statement read view is why a second `SELECT` sees new commits; REPEATABLE READ's per-transaction read view is why it does not. Both are cheap — creating a read view copies the list of active transaction IDs — and are explained fully in *MVCC*.

**Semi-consistent reads (READ COMMITTED only).** When an `UPDATE` scans rows that another transaction has locked, InnoDB first reads the latest **committed** version of the row to decide whether it matches the `WHERE` clause. If it does not match, the row is skipped without waiting. If it does match, InnoDB reads it again, this time waiting for the lock. Together with unlocking non-matching rows after evaluation, this means an `UPDATE` with a `WHERE` on an unindexed column does not block (or get blocked by) unrelated rows at READ COMMITTED — while at REPEATABLE READ the same statement locks every row it scans.

**SERIALIZABLE**. InnoDB does not implement predicate locks; it gets serializability from next-key locking applied to *every* read in the transaction. Two transactions that read the same rows (shared locks, compatible) and then try to write them (exclusive locks, incompatible with the other's shared lock) deadlock — and the deadlock detector rolls one back. That is how write skew is turned into an error you can retry.

## MySQL Implementation

The level can be set globally (server default for new sessions), per session, or for the next transaction only:

```sql
-- Display only: server-wide default (forbidden in the sandbox; usually set in my.cnf)
SET GLOBAL TRANSACTION ISOLATION LEVEL READ COMMITTED;
-- my.cnf:  [mysqld]  transaction_isolation = READ-COMMITTED
```

```sql run
SELECT @@GLOBAL.transaction_isolation  AS server_default,
       @@SESSION.transaction_isolation AS this_session;
```

```sql run
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;
SELECT @@transaction_isolation AS this_session;
SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;
```

`SET TRANSACTION` without `SESSION` applies only to the **next** transaction. `@@transaction_isolation` keeps showing the session value, so the effect is easiest to see in InnoDB's own transaction list, which the lock snapshot below includes:

```concurrency
title: SET TRANSACTION affects only the next transaction
description: Session A sets READ COMMITTED for its next transaction, runs one, then runs another.
steps:
  - session: A
    sql: SET TRANSACTION ISOLATION LEVEL READ COMMITTED; START TRANSACTION; SELECT COUNT(*) AS n FROM branches;
  - observe: locks
  - session: A
    sql: COMMIT; START TRANSACTION; SELECT COUNT(*) AS n FROM branches;
  - observe: locks
  - session: A
    sql: COMMIT;
```

The first snapshot lists A's transaction with isolation `READ COMMITTED`; the second, after the commit, shows `REPEATABLE READ` again — the one-shot setting was consumed by the first transaction.

`START TRANSACTION WITH CONSISTENT SNAPSHOT` only makes sense at REPEATABLE READ. At other levels InnoDB ignores it and says so:

```sql run
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;
START TRANSACTION WITH CONSISTENT SNAPSHOT;
COMMIT;
SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;
```

## Syntax Breakdown

```sql
SET [GLOBAL | SESSION] TRANSACTION
    ISOLATION LEVEL { READ UNCOMMITTED | READ COMMITTED | REPEATABLE READ | SERIALIZABLE }
    [, { READ WRITE | READ ONLY }];
SET @@SESSION.transaction_isolation = 'READ-COMMITTED';   -- variable form (hyphenated value)
SELECT @@transaction_isolation;
```

| Form | Scope | Notes |
|---|---|---|
| `SET GLOBAL TRANSACTION ISOLATION LEVEL …` | Sessions that connect **afterwards** | Needs `SYSTEM_VARIABLES_ADMIN`; existing sessions are unaffected. Forbidden in the sandbox. |
| `SET PERSIST transaction_isolation = '…'` | Global, survives restarts | Written to `mysqld-auto.cnf`. |
| `SET SESSION TRANSACTION ISOLATION LEVEL …` | All later transactions in this session | Allowed inside a transaction; takes effect from the next one. |
| `SET TRANSACTION ISOLATION LEVEL …` | The next transaction only | Error 1568 if a transaction is active. |
| `transaction_isolation` in the option file | Server default at startup | Value uses hyphens: `READ-COMMITTED`. |
| `tx_isolation` | — | Old name, **removed in MySQL 8.0**; use `transaction_isolation`. |
| `START TRANSACTION …` | — | Cannot set the isolation level (only `READ ONLY`/`READ WRITE`/`WITH CONSISTENT SNAPSHOT`). |

:::warning[READ COMMITTED and statement-based binary logging]
With `binlog_format = STATEMENT`, InnoDB refuses writes at READ COMMITTED or READ UNCOMMITTED, because statement replay on a replica would not be deterministic without gap locks. Use row-based logging (the default) with READ COMMITTED. In MySQL 8.0.34 and later the `binlog_format` variable itself is deprecated, and row-based logging is the long-term direction.
:::

## Practical Examples

### 1. SERIALIZABLE turns plain reads into shared locks

```concurrency
title: SERIALIZABLE — a plain SELECT takes a shared lock and blocks writers
description: Session A reads account 4 at SERIALIZABLE inside a transaction. Session B (REPEATABLE READ) reads it too, then tries to update it.
steps:
  - session: A
    sql: SET SESSION TRANSACTION ISOLATION LEVEL SERIALIZABLE; START TRANSACTION; SELECT balance FROM accounts WHERE account_id = 4;
  - observe: locks
  - session: B
    sql: START TRANSACTION; SELECT balance FROM accounts WHERE account_id = 4;
    note: B's plain read is a consistent read at REPEATABLE READ — it does not wait for anything.
  - session: B
    sql: UPDATE accounts SET balance = balance + 10.00 WHERE account_id = 4;
    expect: blocks
    note: B needs an exclusive lock, which conflicts with A's shared lock from a plain SELECT.
  - session: A
    sql: COMMIT;
  - session: B
    sql: COMMIT; SELECT balance FROM accounts WHERE account_id = 4;
```

The snapshot shows A's transaction at isolation `SERIALIZABLE` holding an `IS` table lock and an `S,REC_NOT_GAP` record lock on account 4 — from a statement with no `FOR SHARE` in it. B's update waited until A committed. At SERIALIZABLE, every read-only transaction that runs for a while holds back writers on everything it has read.

### 2. ...except in autocommit mode

```concurrency
title: SERIALIZABLE plain SELECT in autocommit mode is a consistent read
description: Session B holds an uncommitted update on account 5. Session A at SERIALIZABLE reads account 5 first in autocommit mode, then inside a transaction.
steps:
  - session: B
    sql: START TRANSACTION; UPDATE accounts SET balance = balance - 1000.00 WHERE account_id = 5;
  - session: A
    sql: SET SESSION TRANSACTION ISOLATION LEVEL SERIALIZABLE; SELECT balance FROM accounts WHERE account_id = 5;
    note: Autocommit — a single read-only statement can be serialised as a non-locking consistent read, so A does not wait.
  - session: A
    sql: START TRANSACTION;
  - session: A
    sql: SELECT balance FROM accounts WHERE account_id = 5;
    expect: blocks
    note: Inside a transaction the same SELECT becomes FOR SHARE and waits for B's exclusive lock.
  - session: B
    sql: ROLLBACK;
  - session: A
    sql: COMMIT;
```

In autocommit mode A returned the committed balance immediately; inside a transaction the identical statement waited for B. The manual documents this: with autocommit enabled, a `SELECT` at SERIALIZABLE is its own transaction, known to be read-only, and can be serialised as a consistent read.

### 3. SERIALIZABLE turns write skew into a deadlock

The linked-account rule from the previous lesson — customer 1's combined balance must stay ≥ 1,000.00 — broke at REPEATABLE READ. At SERIALIZABLE:

```concurrency
title: Write skew at SERIALIZABLE — one transaction is rolled back
description: The same two withdrawals as in the read-phenomena lesson (3,500.00 from account 2 and 450.00 from account 1), both sessions at SERIALIZABLE.
steps:
  - session: A
    sql: SET SESSION TRANSACTION ISOLATION LEVEL SERIALIZABLE; START TRANSACTION; SELECT SUM(balance) AS combined FROM accounts WHERE customer_id = 1;
  - session: B
    sql: SET SESSION TRANSACTION ISOLATION LEVEL SERIALIZABLE; START TRANSACTION; SELECT SUM(balance) AS combined FROM accounts WHERE customer_id = 1;
    note: Both hold shared locks on customer 1's accounts — shared locks are compatible.
  - observe: locks
  - session: A
    sql: UPDATE accounts SET balance = balance - 3500.00 WHERE account_id = 2;
    expect: blocks
    note: A needs an exclusive lock on account 2, but B holds a shared lock on it.
  - session: B
    sql: UPDATE accounts SET balance = balance - 450.00 WHERE account_id = 1;
    expect: error
    errorCode: 1213
    note: B needs account 1, which A has share-locked, while A waits for B — a cycle. InnoDB rolls B back.
  - session: A
    sql: COMMIT; SELECT SUM(balance) AS combined FROM accounts WHERE customer_id = 1;
    note: A's withdrawal went through; B must retry and will then see the reduced combined balance.
```

The lock snapshot shows A and B holding identical, compatible shared locks: `S` next-key locks on the two `idx_accounts_customer` entries for customer 1 (`1, 1` and `1, 2`), an `S,GAP` lock on the gap before the next entry (`2, 3`, customer 2's first account) so that no new account for customer 1 can appear, and `S,REC_NOT_GAP` locks on primary-key records 1 and 2. When both tried to upgrade to exclusive locks, InnoDB detected the cycle and rolled B back with error 1213. The final combined balance is 1420.77 — the rule held. SERIALIZABLE did not make the conflict disappear; it converted a silent integrity violation into an error that the application **must** handle by retrying B.

### 4. READ COMMITTED does not lock gaps

```concurrency
title: READ COMMITTED — a locking range read does not block inserts into the range
description: Session A locks the accounts with id > 20 FOR UPDATE at READ COMMITTED. Session B opens account 26.
steps:
  - session: A
    sql: SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED; START TRANSACTION; SELECT account_id FROM accounts WHERE account_id > 20 FOR UPDATE;
  - observe: locks
  - session: B
    sql: INSERT INTO accounts (account_id, customer_id, branch_id, account_type, balance, opened_at) VALUES (26, 15, 2, 'savings', 500.00, '2024-12-31 09:00:00');
    note: No gap lock — the insert proceeds immediately.
  - session: A
    sql: SELECT account_id FROM accounts WHERE account_id > 20 FOR UPDATE; COMMIT;
    note: A's second locking read in the same transaction sees a sixth row — a phantom.
```

At READ COMMITTED the snapshot contains only `X,REC_NOT_GAP` locks on records 21–25 — no `X` next-key locks and no supremum lock, unlike the same statement at REPEATABLE READ in the read-phenomena lesson. B's insert succeeded at once, and A's repeated locking read returned account 26 as well. Fewer locks, fewer deadlocks, and phantoms for locking reads.

### 5. READ COMMITTED releases locks on non-matching rows

`status` has no index, so `UPDATE … WHERE status = 'frozen'` must scan the whole table. At REPEATABLE READ that scan locks every row and every gap — the *UPDATE and DELETE* lesson in Module 3 shows the resulting blocking. At READ COMMITTED:

```concurrency
title: READ COMMITTED — an unindexed UPDATE keeps locks only on matching rows
description: Session A bumps the version of every frozen account (only account 7) at READ COMMITTED, without committing. Session B updates account 1.
steps:
  - session: A
    sql: SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED; START TRANSACTION; UPDATE accounts SET version = version + 1 WHERE status = 'frozen';
  - observe: locks
  - session: B
    sql: UPDATE accounts SET balance = balance + 5.00 WHERE account_id = 1;
    note: Account 1 does not match A's WHERE clause, so A released its lock on it during the scan.
  - session: A
    sql: COMMIT;
```

The snapshot shows a single record lock, on account 7. InnoDB examined all 25 rows, but at READ COMMITTED it unlocked each row that failed the `WHERE` condition, so B's update of account 1 did not wait.

### 6. InnoDB REPEATABLE READ is not snapshot isolation

```concurrency
title: An UPDATE at REPEATABLE READ acts on data its snapshot cannot see
description: Session A (REPEATABLE READ) reads account 9. Session B deposits 1,000.00 into account 9 and commits. A then adds 10.00 interest and reads the account again.
steps:
  - session: A
    sql: START TRANSACTION; SELECT balance, version FROM accounts WHERE account_id = 9;
    note: A's snapshot is created here.
  - session: B
    sql: UPDATE accounts SET balance = balance + 1000.00, version = version + 1 WHERE account_id = 9;
    note: Committed immediately (autocommit).
  - session: A
    sql: SELECT balance, version FROM accounts WHERE account_id = 9;
    note: Still the snapshot — B's deposit is invisible.
  - session: A
    sql: UPDATE accounts SET balance = balance + 10.00, version = version + 1 WHERE account_id = 9;
    note: No error. The UPDATE reads the latest committed version, which includes B's deposit.
  - session: A
    sql: SELECT balance, version FROM accounts WHERE account_id = 9; COMMIT;
    note: Now A sees B's deposit too — because the newest version of the row is A's own.
```

A's snapshot showed 9169.88 twice, yet its `UPDATE` succeeded on top of B's committed deposit, and A's next `SELECT` showed 10179.88 with version 2 — a value that includes a change A's snapshot never contained. Under snapshot isolation with first-committer-wins (PostgreSQL's REPEATABLE READ), A's update would have failed with a serialization error because the row changed after A's snapshot was taken. Here the outcome happens to be correct because `balance = balance + 10.00` is an atomic update; had A written an absolute value computed from its snapshot (9169.88 + 10.00), B's deposit would have been lost without any error.

## Execution Walkthrough

How a single plain `SELECT` is executed at each level, traced for the query `SELECT balance FROM accounts WHERE account_id = 9` issued while another transaction holds an uncommitted change to that row:

```mermaid
sequenceDiagram
  participant Q as SELECT (plain)
  participant RV as Read view
  participant Row as Clustered index row
  participant U as Undo log
  Note over Q: READ UNCOMMITTED
  Q->>Row: read newest version (uncommitted)
  Note over Q: READ COMMITTED / REPEATABLE READ
  Q->>RV: RC - create new view now; RR - reuse first view
  Q->>Row: read newest version, check DB_TRX_ID against view
  Row-->>Q: not visible (writer still active)
  Q->>U: follow DB_ROLL_PTR to previous version
  U-->>Q: last committed version
  Note over Q: SERIALIZABLE (in a transaction)
  Q->>Row: request S lock (waits for writer's X lock)
```

The walkthrough below performs the visible part of this in your lab: it creates a scratch table, then shows the result of the same query under each level in one session — and why, without a concurrent writer, the levels give identical answers. Isolation levels only differ when transactions overlap; the concurrency scenarios above are the real evidence.

```walkthrough
title: The same reads at four isolation levels, without concurrency
steps:
  - title: "1. A scratch table"
    explain: One row, so the results are easy to compare.
    sql: |
      CREATE TABLE iso_demo (id INT PRIMARY KEY, v INT);
      INSERT INTO iso_demo VALUES (1, 100);
  - title: "2. READ UNCOMMITTED"
    explain: Reads the newest version. With no concurrent writer that is simply the committed value.
    sql: |
      SET SESSION TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
      START TRANSACTION; SELECT v FROM iso_demo WHERE id = 1; COMMIT;
  - title: "3. READ COMMITTED"
    explain: Creates a read view for the statement; the committed value is visible.
    sql: |
      SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;
      START TRANSACTION; SELECT v FROM iso_demo WHERE id = 1; COMMIT;
  - title: "4. REPEATABLE READ"
    explain: Creates the transaction's read view at the first read and keeps it until COMMIT.
    sql: |
      SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;
      START TRANSACTION; SELECT v FROM iso_demo WHERE id = 1; COMMIT;
  - title: "5. SERIALIZABLE"
    explain: >-
      The SELECT is executed as SELECT … FOR SHARE and holds a shared lock until COMMIT. With nobody else
      writing, nothing waits — and the answer is the same.
    sql: |
      SET SESSION TRANSACTION ISOLATION LEVEL SERIALIZABLE;
      START TRANSACTION; SELECT v FROM iso_demo WHERE id = 1; COMMIT;
      SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ;
```

## Output

Summarising the captured behaviour:

| Behaviour | RU | RC | RR (default) | SERIALIZABLE |
|---|---|---|---|---|
| Plain `SELECT` sees uncommitted data | yes (previous lesson) | no | no | no — it waits for the lock |
| Plain `SELECT` re-read sees new commits | yes | yes | no | n/a (rows are share-locked) |
| Plain `SELECT` takes locks | no | no | no | yes, inside a transaction (Example 1); not in autocommit (Example 2) |
| Locking range read blocks inserts into the range | no | no (Example 4) | yes (previous lesson) | yes |
| Unindexed `UPDATE` locks non-matching rows | no | no (Example 5) | yes (see Module 3) | yes |
| Write skew | possible | possible | possible | becomes deadlock 1213 (Example 3) |
| Update of a row changed after the snapshot | — | — | succeeds silently (Example 6) | succeeds after waiting |

## Alternative Approaches

Instead of raising the isolation level for a whole transaction, you can strengthen **individual statements**:

| Goal | Raise the level | Targeted alternative |
|---|---|---|
| Consistent multi-query report | REPEATABLE READ for the report transaction | Single statement; or `START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY` |
| Prevent lost update | SERIALIZABLE | Atomic `UPDATE … SET x = x + ?`; `SELECT … FOR UPDATE`; optimistic version column |
| Prevent write skew on a rule | SERIALIZABLE for those transactions | `FOR UPDATE` on all rows (and range) the rule reads; materialised conflict row; unique constraint |
| Avoid phantoms in a check-then-insert | REPEATABLE READ locking read or SERIALIZABLE | Unique constraint so the second insert fails with 1062 |
| Fewer deadlocks / less gap locking | READ COMMITTED | Better indexes (fewer rows scanned and locked); smaller transactions |

Targeted locking keeps the rest of the application at a cheaper level, but it must be applied **everywhere** the rule is touched — one forgotten code path reopens the anomaly. SERIALIZABLE is "secure by default" for the transactions that use it, at the cost of more blocking and retries.

## Performance Analysis

| Level | Read cost | Locking footprint | Typical side effects |
|---|---|---|---|
| READ UNCOMMITTED | Slightly cheaper (no version reconstruction) | Same as RC for writes | Wrong answers; rarely worth it |
| READ COMMITTED | A read view per statement (cheap); short version chains because snapshots are short-lived | Record locks only on matching rows; no gap locks | Fewer deadlocks and lock waits; anomalies P2/P3 possible |
| REPEATABLE READ | One read view per transaction; long transactions keep old versions alive | Next-key locks on everything scanned by locking reads and DML | More lock waits and deadlocks around range scans and inserts; purge lag from long transactions |
| SERIALIZABLE | Plain reads become locking reads | Shared next-key locks on everything read | Readers block writers; many more deadlocks; requires retry logic |

Two practical observations:

1. **Index quality matters more than level choice for locking cost.** At REPEATABLE READ an `UPDATE` with an unindexed predicate locks the whole table; with a good index it locks a few records. Switching to READ COMMITTED hides the symptom; adding the index fixes the cause.
2. **Long transactions hurt at every level, but most at REPEATABLE READ**, where one read view lives as long as the transaction and prevents purge of every newer version.

:::performance[Theory vs measurement]
Claims that "READ COMMITTED is faster" are workload-specific. To decide, run your real transaction mix at both levels on a staging system and compare throughput, `Innodb_row_lock_waits`, `Innodb_row_lock_time` and the deadlock count (`SELECT COUNT FROM information_schema.INNODB_METRICS WHERE NAME = 'lock_deadlocks'`, which requires the PROCESS privilege) — and verify that the application is still correct at the weaker level.
:::

## Best Practices

:::best-practice[Choosing and using isolation levels]
1. **Choose the level per workload, deliberately.** Keep REPEATABLE READ (the default) unless you have measured gap-lock contention; then consider READ COMMITTED with row-based binary logging — and re-check the application for read skew and phantom-dependent logic.
2. **Do not rely on the level to prevent lost updates or write skew.** Use atomic updates, locking reads on the full read set, optimistic version checks, or constraints.
3. **Use SERIALIZABLE surgically**, for the few transactions that enforce cross-row rules and cannot easily lock explicitly — and always with a retry loop for 1213/1205.
4. **Set the level at session or connection-pool level**, not ad hoc in the middle of code paths; log it in your service configuration.
5. **Keep REPEATABLE READ transactions short.** A forgotten open transaction at RR is the most common cause of purge lag.
6. **Never use READ UNCOMMITTED for data that drives decisions.**
7. **Remember the hybrid**: at REPEATABLE READ, snapshot reads and current reads (locking reads, DML) can disagree inside one transaction. Base writes on locking reads.
:::

## Common Mistakes

**Mistake 1 — expecting `SET TRANSACTION` to change the current transaction.** It applies to the next one, and is refused inside an active transaction:

```sql run error
START TRANSACTION;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

```sql run
ROLLBACK;
```

**Mistake 2 — using the removed variable name.** `tx_isolation` was removed in MySQL 8.0:

```sql run error
SELECT @@tx_isolation;
```

**Mistake 3 — assuming MySQL's REPEATABLE READ behaves like PostgreSQL's.** Code ported from PostgreSQL that relies on serialization failures to detect concurrent updates (first-committer-wins) silently loses that protection on MySQL (Example 6). Add explicit version checks or locking reads.

**Mistake 4 — switching to READ COMMITTED to "fix deadlocks" without reviewing logic.** Fewer gap locks also means phantoms for locking reads (Example 4). Any check-then-insert that relied on gap locks for correctness is now a race.

**Mistake 5 — believing SERIALIZABLE removes the need for error handling.** It makes conflicts *visible* as deadlocks and timeouts; the application must roll back and retry (Example 3).

**Mistake 6 — setting the level globally and expecting existing connections to change.** `SET GLOBAL` affects only sessions created afterwards; pooled connections keep their old setting until reconnected.

## Industry Application

- **Large-scale MySQL users** (for example, many high-traffic web companies) commonly run READ COMMITTED with row-based replication to reduce lock contention and deadlocks from gap locks on hot insert ranges, accepting non-repeatable reads in exchange.
- **Financial and inventory systems** often keep REPEATABLE READ and add explicit `FOR UPDATE` on the rows involved in money or stock movements, with deterministic lock ordering.
- **Backups and exports**: `mysqldump --single-transaction` issues `SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ` and `START TRANSACTION WITH CONSISTENT SNAPSHOT`, giving a consistent logical backup of InnoDB tables without blocking writers.
- **Cross-database portability**: teams migrating between MySQL and PostgreSQL review every transaction that depends on isolation behaviour — PostgreSQL RR aborts on concurrent updates, MySQL RR does not; PostgreSQL SERIALIZABLE uses SSI (aborts at commit), MySQL SERIALIZABLE uses shared locks (blocks and deadlocks).
- **ORMs and drivers** expose the level per transaction (`@Transactional(isolation = …)` in Spring, `isolation_level` in SQLAlchemy); setting it there issues the same `SET TRANSACTION` statement you used above.

## Practice Exercises

```exercise
id: ex1
title: Configure a session for READ COMMITTED
difficulty: easy
kind: dml
concepts: [isolation-levels, read-committed]
prompt: |
  Configure **your session** so that every subsequent transaction runs at READ COMMITTED. (The checker reads
  `@@SESSION.transaction_isolation`.)
hints:
  - "SET TRANSACTION without SESSION affects only the next transaction."
  - "SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;"
solution: |
  SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;
check: |
  SELECT @@SESSION.transaction_isolation AS level;
explanation: |
  `SET SESSION TRANSACTION …` (or `SET @@SESSION.transaction_isolation = 'READ-COMMITTED'`) changes the session
  default. `SET TRANSACTION …` alone would leave the session value at REPEATABLE-READ and affect only one
  transaction; `SET GLOBAL` would affect only future connections (and is not allowed in the lab).
```

```exercise
id: ex2
title: A consistent branch report without REPEATABLE READ
difficulty: medium
concepts: [isolation-levels, read-committed]
prompt: |
  Your service runs at READ COMMITTED. You need a branch summary whose numbers are consistent with each other
  even while transfers commit. Write it as **a single statement** returning `branch_name`, `accounts`
  (number of accounts), and `total_balance`, for all branches, ordered by `branch_name`.
hints:
  - "A single statement uses a single read view even at READ COMMITTED."
  - "Join branches to accounts and GROUP BY the branch."
solution: |
  SELECT b.name AS branch_name, COUNT(a.account_id) AS accounts, SUM(a.balance) AS total_balance
  FROM branches b
  LEFT JOIN accounts a ON a.branch_id = b.branch_id
  GROUP BY b.branch_id, b.name
  ORDER BY b.name;
orderMatters: true
explanation: |
  At READ COMMITTED every statement gets its own read view, so a report built from several queries can suffer
  read skew (a transfer between branches could be counted on both sides or neither). One statement reads
  everything from one read view and is internally consistent. If several statements are unavoidable, run them in
  a REPEATABLE READ transaction (`SET TRANSACTION ISOLATION LEVEL REPEATABLE READ; START TRANSACTION WITH
  CONSISTENT SNAPSHOT, READ ONLY;`).
```

```exercise
id: ex3
title: Predict the timeline at each level
difficulty: medium
kind: open
concepts: [isolation-levels, read-committed, repeatable-read, serializable]
prompt: |
  Account 9 holds 9169.88. Fill in what T1 sees at steps 2 and 4 (or whether it waits) when T1 runs at each of
  READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ and SERIALIZABLE. T2 is always at REPEATABLE READ.

  ~~~
  1. T1: START TRANSACTION;
  2. T1: SELECT balance FROM accounts WHERE account_id = 9;
  3. T2: START TRANSACTION; UPDATE accounts SET balance = balance + 1000 WHERE account_id = 9;
  4. T1: SELECT balance FROM accounts WHERE account_id = 9;
  5. T2: COMMIT;
  6. T1: SELECT balance FROM accounts WHERE account_id = 9;
  ~~~

  Also say what happens at step 3 when T1 is SERIALIZABLE.
solution: |
  | Level | Step 2 | Step 4 | Step 6 |
  |---|---|---|---|
  | READ UNCOMMITTED | 9169.88 | 10169.88 (dirty) | 10169.88 |
  | READ COMMITTED | 9169.88 | 9169.88 (new read view, T2 uncommitted) | 10169.88 (new read view sees the commit) |
  | REPEATABLE READ | 9169.88 | 9169.88 | 9169.88 (snapshot from step 2) |
  | SERIALIZABLE | 9169.88, with an S lock on the row | — | — |

  At SERIALIZABLE, T1's step-2 SELECT holds a shared lock until T1 ends, so **T2's UPDATE at step 3 blocks**.
  Steps 4–6 therefore run differently: T1's step-4 read returns 9169.88 immediately (T1 already holds the lock and
  no change has happened), T2 cannot commit at step 5 because it is still waiting, and only when T1 commits does
  T2's update proceed (or time out after innodb_lock_wait_timeout). SERIALIZABLE serialises T2 after T1.
```

```exercise
id: ex4
title: Choose a level for each workload
difficulty: hard
kind: open
concepts: [isolation-levels]
prompt: |
  Recommend an isolation level (and any additional technique) for each, with a one-paragraph justification:

  1. A high-volume order-insert API on MySQL with frequent deadlocks caused by gap locks on a secondary index.
  2. A nightly regulatory report of 40 queries over the bank ledger that must be mutually consistent.
  3. A meeting-room booking service that must never double-book a room for overlapping times.
  4. A real-time dashboard estimating the number of jobs in a queue table.
solution: |
  1. **READ COMMITTED with row-based binary logging**, after checking the indexes. RC disables gap locking for
     searches, which removes the insert-vs-gap deadlocks; review the code for logic that relied on gap locks
     (check-then-insert) and protect it with unique constraints instead.
  2. **REPEATABLE READ** in one `START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY` transaction — ideally on a
     replica, so the long-lived read view does not delay purge on the primary. All 40 queries see one snapshot.
  3. Overlap checks are a write-skew/phantom problem. Options: lock the room row (`SELECT … FROM rooms WHERE
     room_id = ? FOR UPDATE`) at the start of every booking transaction, so bookings for a room serialise;
     or SERIALIZABLE for the booking transaction with retries; or discretise time into slots with a unique key on
     (room_id, slot). A plain SELECT at any level below SERIALIZABLE is not enough.
  4. **READ COMMITTED** (or simply autocommit SELECTs at the default). An approximate count tolerates
     non-repeatable reads; READ UNCOMMITTED adds no performance benefit in InnoDB because consistent reads take no
     locks anyway.
```

```exercise
id: ex5
title: Port a PostgreSQL concurrency assumption
difficulty: challenge
kind: open
concepts: [repeatable-read, snapshot-isolation]
prompt: |
  A service was developed on PostgreSQL at REPEATABLE READ. Its "apply interest" transaction reads an account
  with a plain SELECT, computes the new balance in Java, writes it with `UPDATE accounts SET balance = :new`,
  and relies on PostgreSQL raising a serialization failure if another transaction modified the account
  concurrently, in which case it retries. The service is being moved to MySQL 8.4 with default settings.

  Explain precisely what changes, demonstrate the failure scenario as a timeline, and propose two MySQL-correct
  designs.
solution: |
  **What changes:** PostgreSQL's REPEATABLE READ is snapshot isolation with first-committer-wins: an UPDATE of a
  row modified by a transaction that committed after the snapshot fails with "could not serialize access due to
  concurrent update". InnoDB's REPEATABLE READ has no such check — UPDATE operates on the latest committed version
  and succeeds (this lesson's Example 6). The retry trigger never fires, so the service silently overwrites
  concurrent changes.

  **Timeline:** T1 reads balance 1000 (snapshot). T2 deposits 500 and commits (1500). T1 computes 1000 × 1.01 =
  1010 and runs `SET balance = 1010` — succeeds on MySQL, erasing the deposit (lost update). On PostgreSQL, T1's
  UPDATE would have failed and been retried.

  **Designs for MySQL:**
  1. *Atomic update:* `UPDATE accounts SET balance = balance * 1.01, version = version + 1 WHERE account_id = ?` —
     computed in the database on the current version under an exclusive lock.
  2. *Optimistic check:* read `balance, version`, compute, then `UPDATE … SET balance = :new, version = version + 1
     WHERE account_id = ? AND version = :read_version`; if the affected-row count is 0, retry. This reproduces the
     first-committer-wins behaviour explicitly and works identically on both databases.
  (A third option is `SELECT … FOR UPDATE` before computing.) SERIALIZABLE on MySQL would also prevent the lost
  update, but via shared locks and deadlocks rather than PostgreSQL-style serialization failures.
```

## Knowledge Assessment

```quiz
- id: q1
  question: "What is MySQL's default transaction isolation level?"
  options: ["READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE", "READ UNCOMMITTED"]
  answer: 1
  explanation: "InnoDB's default is REPEATABLE READ (shown as REPEATABLE-READ in @@transaction_isolation). PostgreSQL, Oracle and SQL Server default to READ COMMITTED, which is a frequent source of confusion when porting applications."
  concepts: [isolation-levels]
- id: q2
  question: "At SERIALIZABLE, inside an explicit transaction, how does InnoDB execute `SELECT balance FROM accounts WHERE account_id = 4`?"
  options: ["As a consistent read from a snapshot", "As SELECT … FOR SHARE, taking a shared lock held until commit", "As SELECT … FOR UPDATE", "As a dirty read"]
  answer: 1
  explanation: "InnoDB implements SERIALIZABLE by converting plain SELECTs to FOR SHARE when autocommit is off / inside a transaction; the captured lock snapshot showed S,REC_NOT_GAP on account 4. It is not FOR UPDATE (exclusive), and in autocommit mode it stays a consistent read."
  concepts: [serializable]
- id: q3
  question: "Which statements describe InnoDB deviations from the SQL standard or from textbook snapshot isolation? (Select all that apply.)"
  type: multi
  options:
    - "REPEATABLE READ prevents phantoms for consistent reads"
    - "REPEATABLE READ aborts a transaction that updates a row changed after its snapshot"
    - "UPDATE at REPEATABLE READ operates on the latest committed row version, not the snapshot"
    - "READ UNCOMMITTED allows dirty writes"
    - "SERIALIZABLE is implemented with shared locks on reads rather than serializable snapshot isolation"
  answer: [0, 2, 4]
  explanation: "InnoDB RR prevents phantoms for snapshot reads (stronger than the standard), updates the current version without a first-committer-wins abort (Example 6) — so option B is false — and implements SERIALIZABLE by locking. Dirty writes are prevented at every level."
  concepts: [snapshot-isolation, repeatable-read]
- id: q4
  question: "A session runs `SET TRANSACTION ISOLATION LEVEL READ COMMITTED;` and then two transactions. At which level does the second transaction run?"
  options: ["READ COMMITTED", "The session's level (REPEATABLE READ by default)", "SERIALIZABLE", "It depends on the global level only"]
  answer: 1
  explanation: "Without SESSION or GLOBAL, SET TRANSACTION applies only to the next transaction; the captured scenario showed READ COMMITTED for the first transaction and REPEATABLE READ for the second."
  concepts: [isolation-levels]
- id: q5
  question: "Two transactions at SERIALIZABLE each read customer 1's accounts and then each update a different one of those accounts. What happens in InnoDB?"
  options: ["Both commit; the rule may be violated (write skew)", "One blocks and the other receives error 1213, so it must retry", "Both fail with error 1205 immediately", "The second read blocks until the first transaction commits"]
  answer: 1
  explanation: "Both hold compatible shared locks after reading; each UPDATE needs an exclusive lock that conflicts with the other's shared lock, forming a cycle. The deadlock detector rolls one back with 1213 (Example 3). Shared reads do not block each other, and no timeout is involved."
  concepts: [serializable, gap-locks]
- id: q6
  question: "Why does an `UPDATE … WHERE status = 'frozen'` on an unindexed column block an unrelated update of account 1 at REPEATABLE READ but not at READ COMMITTED?"
  options:
    - "READ COMMITTED does not lock rows during UPDATE"
    - "At READ COMMITTED, InnoDB releases locks on rows that do not match the WHERE clause (and uses semi-consistent reads); at REPEATABLE READ it keeps locks on every row the scan touched"
    - "REPEATABLE READ uses table locks for UPDATE"
    - "READ COMMITTED uses a different storage engine"
  answer: 1
  explanation: "Both levels lock rows during the scan, but RC unlocks non-matching rows after evaluating the WHERE clause, leaving only the matching row locked (Example 5 showed a single lock on account 7). RR keeps next-key locks on everything scanned. Neither uses table locks for this."
  concepts: [read-committed, repeatable-read, gap-locks]
- id: q7
  question: "A team switches from REPEATABLE READ to READ COMMITTED. Which code path is most likely to become incorrect?"
  options:
    - "A single-row UPDATE … SET stock = stock - 1 WHERE id = ? AND stock > 0"
    - "A check-then-insert that relied on a locking range read to stop concurrent inserts into the range"
    - "A SELECT by primary key"
    - "An INSERT with a unique constraint violation handler"
  answer: 1
  explanation: "READ COMMITTED does not take gap locks, so a locking range read no longer blocks inserts into the range (Example 4) — the check-then-insert becomes a race. Atomic guarded updates, primary-key reads and unique constraints behave correctly at both levels."
  concepts: [read-committed, gap-locks]
```
