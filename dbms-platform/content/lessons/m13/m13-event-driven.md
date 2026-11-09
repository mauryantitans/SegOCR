---
id: m13-event-driven
dataset: shop
summary: What changes when a MySQL database publishes events to other services — the dual-write problem, the transactional outbox implemented in SQL, a relay that claims work with FOR UPDATE SKIP LOCKED, idempotent consumers, per-aggregate ordering with version numbers, binlog-based change data capture with Debezium, and evolving event schemas without breaking consumers.
concepts: [event-driven-architecture, transactional-outbox, dual-write, change-data-capture, skip-locked, idempotent-consumer, event-ordering, event-schema-evolution]
references:
  - title: "MySQL 8.4 Reference Manual — Locking Read Concurrency with NOWAIT and SKIP LOCKED"
    url: https://dev.mysql.com/doc/refman/8.4/en/innodb-locking-reads.html
  - title: "MySQL 8.4 Reference Manual — The Binary Log"
    url: https://dev.mysql.com/doc/refman/8.4/en/binary-log.html
  - title: "MySQL 8.4 Reference Manual — Replication Formats (row-based logging)"
    url: https://dev.mysql.com/doc/refman/8.4/en/replication-formats.html
  - title: "Debezium documentation — MySQL connector"
    url: https://debezium.io/documentation/reference/stable/connectors/mysql.html
  - title: "Debezium documentation — Outbox Event Router"
    url: https://debezium.io/documentation/reference/stable/transformations/outbox-event-router.html
  - title: "Chris Richardson — Pattern: Transactional outbox"
    url: https://microservices.io/patterns/data/transactional-outbox.html
---

## Introduction

In a microservice or event-driven architecture, the service that owns orders does not call the loyalty, shipping and analytics services directly. It **publishes events** — `OrderPlaced`, `OrderPaid`, `OrderShipped` — to a broker such as Apache Kafka, and other services react to them. The database is no longer only a place where state is stored; it is the **source of a stream of facts** that other systems depend on.

That raises database questions that ordinary CRUD applications never face:

- How do we **update the database and publish an event atomically**, when the database and the broker are different systems?
- How do several publisher workers share the work **without blocking each other** or publishing the same event twice?
- How does a consumer survive **duplicate deliveries**, which every practical broker produces?
- In what **order** do events arrive, and what happens when they arrive out of order?
- How can the **shape of an event** change over the years without breaking consumers?
- Can we get events straight from the **binary log** instead of writing them ourselves?

This lesson answers them with a **transactional outbox** built in the lab, a relay that claims work with `FOR UPDATE SKIP LOCKED` (demonstrated with real concurrent sessions), an idempotent consumer, version-based ordering, and schema evolution — and explains binlog-based **change data capture (CDC)** with Debezium, which runs outside MySQL and is shown display-only. It builds on [The Binary Log](/lessons/m10-binary-log) and [Data Integrity in Distributed Applications](/lessons/m12-distributed-integrity).

:::note[Dataset]
The lab uses the **shop** dataset (orders, order items, customers). The lesson adds `outbox`, `processed_events`, `loyalty_points` and `order_projection`. A message broker is not available in the sandbox, so "publishing" is represented by marking outbox rows as published; consumers are stored procedures in the same schema, standing in for other services' databases.
:::

## DBMS Theory

:::theory[State, events and the dual-write problem]
- An **event** is an immutable record that something happened (`OrderPlaced` for order 81 at 10:00). A **state change** in the database and the **event** describing it are two representations of the same fact; consistency requires that one exists **if and only if** the other does.
- **Dual write**: the application commits to MySQL and then sends to Kafka (or the reverse). These are two independent systems with no shared transaction; a crash, timeout or network partition between the two steps leaves one done and the other not. Distributed transactions (2PC, see [Distributed Database Fundamentals](/lessons/m13-distributed-databases)) are rarely supported by brokers and couple availability.
- **Transactional outbox**: write the event into an `outbox` table **in the same local ACID transaction** as the state change. Atomicity is then guaranteed by the database alone. A separate **relay** (polling publisher or CDC) later copies outbox rows to the broker — at least once.
- **Delivery semantics**: *at-most-once* (may lose), *at-least-once* (may duplicate), *exactly-once* (neither). End-to-end exactly-once *effects* are achieved by at-least-once delivery plus **idempotent consumers** (deduplication by event id), not by the transport alone.
- **Ordering**: brokers order messages within a partition. Events for the same **aggregate** (one order) must go to the same partition (key = aggregate id) to be consumed in order; there is no global order across aggregates, and consumers should not need one.
:::

```mermaid
sequenceDiagram
  participant App as Order service
  participant DB as MySQL (orders + outbox)
  participant Relay as Relay / CDC
  participant K as Kafka topic (key = order_id)
  participant C as Loyalty service
  App->>DB: BEGIN; INSERT orders, order_items, outbox; COMMIT
  Note over DB: state and event committed atomically
  Relay->>DB: claim unpublished events (SKIP LOCKED) or read binlog
  Relay->>K: publish (at least once)
  Relay->>DB: mark published (or advance binlog offset)
  K->>C: deliver (possibly twice)
  C->>C: BEGIN; record event_id; apply effect; COMMIT (duplicate → skip)
```

## Why It Exists

Every alternative to the outbox fails in a specific way:

| Approach | Failure |
|---|---|
| Commit to MySQL, then publish | Crash after commit → state changed, **event lost**; downstream never knows |
| Publish, then commit | Commit fails (deadlock, constraint) → **phantom event** for something that never happened |
| Publish inside the DB transaction, before commit | Event visible before the data is committed (and even if it later rolls back) |
| Distributed transaction (XA) across DB and broker | Most brokers do not support it; coordinator failures block both systems |
| Consumers query the database for changes | Tight coupling, load on the OLTP database, no history of intermediate states |

The outbox replaces a distributed consistency problem with a **local** one (one database transaction) plus an **at-least-once** delivery problem, which idempotent consumers solve. CDC goes one step further: the binary log *is* already a durable, ordered outbox of every committed change.

## Internal Working

:::internals[The MySQL mechanisms behind each piece]
- **Atomicity of the outbox write** comes from InnoDB: the order rows and the outbox row are in one transaction, written to the redo log and the binary log together; InnoDB and the binary log are kept consistent by the server's internal two-phase commit ([The Binary Log](/lessons/m10-binary-log)).
- **`FOR UPDATE SKIP LOCKED`** (8.0+): a locking read that **skips rows already locked by other transactions** instead of waiting. Several relay workers can each run `SELECT … WHERE published_at IS NULL ORDER BY event_id LIMIT n FOR UPDATE SKIP LOCKED` and receive **disjoint** batches. `NOWAIT` instead fails immediately (error 3572) if a row is locked. Both apply to row locks only and are not safe for statement-based replication — irrelevant for a relay that only reads and marks rows.
- **Ordering in the outbox**: `AUTO_INCREMENT` ids are assigned at insert time, **not at commit time** — a transaction that inserts event 10 and commits after event 11's transaction makes 11 visible first. A polling relay can therefore publish 11 before 10. Per-aggregate ordering is preserved if each aggregate's events are written by transactions that serialize on the aggregate's row (they do, because they update the same order row) and the relay publishes in id order per aggregate; a per-aggregate version number makes the order explicit and checkable.
- **Binlog CDC**: with `binlog_format = ROW` and `binlog_row_image = FULL` (defaults), each committed transaction is written to the binary log as row events with before and after images, **in commit order**, with GTIDs. A CDC connector registers like a replica (it needs a unique `server_id` and replication privileges), takes an initial consistent snapshot, then streams events and stores its binlog position/GTID set as its offset.
:::

## MySQL Implementation

The outbox table: one row per event, with the aggregate it belongs to, a per-aggregate version, the event type, a payload with an explicit schema version, and publication bookkeeping. The index serves the relay's "next unpublished events" query; the unique key makes each aggregate version unique:

```sql run
CREATE TABLE outbox (
  event_id          BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  aggregate_type    VARCHAR(30)  NOT NULL,
  aggregate_id      BIGINT UNSIGNED NOT NULL,
  aggregate_version INT UNSIGNED NOT NULL,
  event_type        VARCHAR(40)  NOT NULL,
  schema_version    TINYINT UNSIGNED NOT NULL DEFAULT 1,
  payload           JSON NOT NULL,
  created_at        DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  published_at      DATETIME(6) NULL,
  KEY idx_outbox_unpublished (published_at, event_id),
  UNIQUE KEY uq_outbox_aggregate_version (aggregate_type, aggregate_id, aggregate_version)
);
```

**Placing an order** writes the state and the event in one transaction:

```sql run
START TRANSACTION;
INSERT INTO orders (customer_id, order_date, status) VALUES (3, '2024-12-31 10:00:00', 'pending');
SET @order_id = LAST_INSERT_ID();
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (@order_id, 5, 2, 29.99);
INSERT INTO outbox (aggregate_type, aggregate_id, aggregate_version, event_type, payload)
VALUES ('order', @order_id, 1, 'OrderPlaced',
        JSON_OBJECT('order_id', @order_id, 'customer_id', 3, 'total', 59.98,
                    'items', JSON_ARRAY(JSON_OBJECT('product_id', 5, 'qty', 2))));
COMMIT;
SELECT event_id, aggregate_id, aggregate_version, event_type, payload, published_at FROM outbox;
```

If anything fails before `COMMIT`, **both** the order and its event disappear. Prove it with a rollback:

```sql run
START TRANSACTION;
INSERT INTO orders (customer_id, order_date, status) VALUES (7, '2024-12-31 11:00:00', 'pending');
SET @order_id = LAST_INSERT_ID();
INSERT INTO outbox (aggregate_type, aggregate_id, aggregate_version, event_type, payload)
VALUES ('order', @order_id, 1, 'OrderPlaced', JSON_OBJECT('order_id', @order_id, 'customer_id', 7, 'total', 10.00));
ROLLBACK;
SELECT (SELECT COUNT(*) FROM orders WHERE order_date = '2024-12-31 11:00:00') AS orders_left,
       (SELECT COUNT(*) FROM outbox WHERE aggregate_id = @order_id)            AS events_left;
```

Neither row survives: there is no window in which the event exists without the order or vice versa. (The `AUTO_INCREMENT` value consumed by the rolled-back insert is not reused — gaps in ids are normal.)

## Syntax Breakdown

| Element | Role |
|---|---|
| `outbox.event_id` | Global, unique event identifier — the consumer's deduplication key |
| `aggregate_type`, `aggregate_id` | Which entity the event is about; becomes the broker's **message key** (partitioning → per-aggregate order) |
| `aggregate_version` | 1, 2, 3, … per aggregate; lets consumers detect gaps, duplicates and reordering |
| `schema_version` | Version of the payload's structure, for evolution |
| `published_at` | `NULL` until the relay has published it (polling relay); not needed with CDC |
| `SELECT … FOR UPDATE SKIP LOCKED` | Claim unclaimed rows without waiting for other workers |
| `SELECT … FOR UPDATE NOWAIT` | Claim or fail immediately (error 3572) |
| `INSERT IGNORE` / `INSERT … ON DUPLICATE KEY UPDATE` into `processed_events` | Consumer-side deduplication |
| `UPDATE … WHERE version = expected` | Optimistic, order-checking application of an event ([Optimistic and Pessimistic Locking](/lessons/m08-optimistic-pessimistic)) |

The relay's claim query uses the `(published_at, event_id)` index — a `ref` on `published_at IS NULL`, already sorted by `event_id`:

```sql run
EXPLAIN SELECT event_id FROM outbox WHERE published_at IS NULL ORDER BY event_id LIMIT 100 FOR UPDATE SKIP LOCKED;
```

## Practical Examples

### 1. A polling relay batch

The relay claims a batch, publishes it to the broker (outside the database), then marks it published and commits. If the relay crashes after publishing but before committing, the rows are still unpublished and will be **published again** by the next run — at-least-once, by design.

```sql run
INSERT INTO outbox (aggregate_type, aggregate_id, aggregate_version, event_type, payload) VALUES
  ('order', 81, 2, 'OrderPaid',   JSON_OBJECT('order_id', 81, 'amount', 59.98, 'method', 'card')),
  ('order', 12, 1, 'OrderPlaced', JSON_OBJECT('order_id', 12, 'customer_id', 18, 'total', 120.00)),
  ('order', 12, 2, 'OrderPaid',   JSON_OBJECT('order_id', 12, 'amount', 120.00, 'method', 'paypal'));
START TRANSACTION;
SELECT event_id, aggregate_id, aggregate_version, event_type
FROM outbox WHERE published_at IS NULL ORDER BY event_id LIMIT 2 FOR UPDATE SKIP LOCKED;
-- … the relay sends these two events to the broker here …
UPDATE outbox SET published_at = '2024-12-31 12:00:00'
WHERE published_at IS NULL ORDER BY event_id LIMIT 2;
COMMIT;
SELECT event_id, aggregate_id, event_type, published_at FROM outbox ORDER BY event_id;
```

Two details in the output. First, there is **no event 2**: the rolled-back transaction consumed that `AUTO_INCREMENT` value, so ids have gaps — consumers must never treat a missing id as a lost event (that is what per-aggregate versions are for). Second, the `UPDATE` marks exactly the rows the `SELECT` claimed — the same predicate and order, and this transaction holds their locks, so no other worker can have taken them in between.

### 2. An idempotent consumer

The loyalty service awards one point per dollar for `OrderPlaced`. Its deduplication table records every event id it has processed, **in the same transaction** as the effect:

```sql run
CREATE TABLE processed_events (
  consumer     VARCHAR(30) NOT NULL,
  event_id     BIGINT UNSIGNED NOT NULL,
  processed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (consumer, event_id)
);
CREATE TABLE loyalty_points (customer_id INT UNSIGNED PRIMARY KEY, points INT NOT NULL);
CREATE PROCEDURE loyalty_on_order_placed(IN p_event_id BIGINT UNSIGNED)
BEGIN
  DECLARE v_customer INT UNSIGNED;
  DECLARE v_total DECIMAL(10,2);
  START TRANSACTION;
  INSERT IGNORE INTO processed_events (consumer, event_id) VALUES ('loyalty', p_event_id);
  IF ROW_COUNT() = 1 THEN
    SELECT payload->>'$.customer_id', payload->>'$.total' INTO v_customer, v_total
    FROM outbox WHERE event_id = p_event_id AND event_type = 'OrderPlaced';
    INSERT INTO loyalty_points (customer_id, points) VALUES (v_customer, FLOOR(v_total)) AS new
    ON DUPLICATE KEY UPDATE points = loyalty_points.points + new.points;
  END IF;
  COMMIT;
END;
CALL loyalty_on_order_placed(1);
CALL loyalty_on_order_placed(1);
CALL loyalty_on_order_placed(1);
SELECT customer_id, points FROM loyalty_points;
SELECT consumer, event_id FROM processed_events;
```

The event was delivered three times; the points were awarded once. `INSERT IGNORE` turns the duplicate-key conflict into a warning, and `ROW_COUNT() = 1` tells the procedure whether this delivery is new. Because the dedup row and the effect commit together, a crash in between cannot leave "marked processed but not applied" (or the reverse). Without the dedup table, the same three deliveries would award three times the points — the classic duplicate-processing bug.

### 3. Ordering with aggregate versions

A consumer maintains a projection of order states. Each event carries `aggregate_version`; the consumer applies an event only if it is the **next** version, using an optimistic `UPDATE … WHERE version = expected`:

```sql run
CREATE TABLE order_projection (order_id INT UNSIGNED PRIMARY KEY, status VARCHAR(12) NOT NULL, version INT UNSIGNED NOT NULL);
INSERT INTO order_projection VALUES (12, 'placed', 1);
-- OrderShipped (version 3) arrives BEFORE OrderPaid (version 2):
UPDATE order_projection SET status = 'shipped', version = 3 WHERE order_id = 12 AND version = 2;
SELECT order_id, status, version FROM order_projection;
```

Zero rows were updated: version 3 cannot be applied on top of version 1. The consumer **parks** the event (retry later, or buffer it) instead of overwriting newer state with older state. When version 2 arrives:

```sql run
UPDATE order_projection SET status = 'paid',    version = 2 WHERE order_id = 12 AND version = 1;
UPDATE order_projection SET status = 'shipped', version = 3 WHERE order_id = 12 AND version = 2;
SELECT order_id, status, version FROM order_projection;
```

Both apply in order. The same check makes **duplicates** harmless (re-applying version 2 updates zero rows) — versioning gives ordering *and* idempotency for state projections.

### 4. Evolving the event schema

Version 2 of `OrderPlaced` replaces `total` (a bare number) with `amount` (value + currency). Old events stay in the outbox (and in the broker's retention) with `schema_version = 1`. Consumers handle both with an **upcaster** — here a view that normalizes every version to the latest shape:

```sql run
INSERT INTO outbox (aggregate_type, aggregate_id, aggregate_version, event_type, schema_version, payload)
VALUES ('order', 40, 1, 'OrderPlaced', 2,
        JSON_OBJECT('order_id', 40, 'customer_id', 7, 'amount', JSON_OBJECT('value', 75.00, 'currency', 'EUR')));
CREATE VIEW v_order_placed_v2 AS
SELECT event_id, aggregate_id AS order_id,
       CAST(payload->>'$.customer_id' AS UNSIGNED) AS customer_id,
       CASE schema_version
         WHEN 1 THEN JSON_OBJECT('value', payload->'$.total', 'currency', 'USD')
         ELSE payload->'$.amount'
       END AS amount,
       schema_version AS original_schema_version
FROM outbox WHERE event_type = 'OrderPlaced';
SELECT event_id, order_id, customer_id, amount, original_schema_version FROM v_order_placed_v2 ORDER BY event_id;
```

The v1 events are presented as `{"value": …, "currency": "USD"}` (the documented default for v1), so downstream logic has one shape to handle. The rules that make this possible are the event-schema equivalent of [online schema migrations](/lessons/m12-schema-evolution): add optional fields freely; never change the meaning or type of an existing field in place; introduce a new version (or a new event type) for incompatible changes; keep consumers **tolerant readers** that ignore unknown fields; and keep upcasters for as long as old events can be replayed.

## Execution Walkthrough

Two relay workers run the claim query at the same time. With `SKIP LOCKED` they receive disjoint batches and neither waits; a third worker using a plain `FOR UPDATE` blocks behind them. Replay it with real sessions.

```concurrency
title: Relay workers claiming outbox events with SKIP LOCKED
description: Workers A and B claim batches concurrently; C uses FOR UPDATE without SKIP LOCKED and has to wait.
setup: |
  CREATE TABLE outbox (event_id BIGINT UNSIGNED PRIMARY KEY, aggregate_id INT UNSIGNED NOT NULL,
    event_type VARCHAR(20) NOT NULL, published_at DATETIME NULL, KEY idx_unpub (published_at, event_id));
  INSERT INTO outbox VALUES (1, 81, 'OrderPlaced', NULL), (2, 81, 'OrderPaid', NULL), (3, 12, 'OrderPlaced', NULL),
                            (4, 12, 'OrderPaid', NULL), (5, 40, 'OrderPlaced', NULL), (6, 40, 'OrderPaid', NULL);
steps:
  - session: A
    sql: START TRANSACTION; SELECT event_id, aggregate_id, event_type FROM outbox WHERE published_at IS NULL ORDER BY event_id LIMIT 3 FOR UPDATE SKIP LOCKED;
    note: Worker A claims events 1–3 and holds their row locks while it publishes them.
  - session: B
    sql: START TRANSACTION; SELECT event_id, aggregate_id, event_type FROM outbox WHERE published_at IS NULL ORDER BY event_id LIMIT 3 FOR UPDATE SKIP LOCKED;
    note: Worker B does not wait for A — it skips the locked rows and claims events 4–6.
  - observe: locks
  - session: C
    sql: SELECT event_id FROM outbox WHERE published_at IS NULL ORDER BY event_id LIMIT 3 FOR UPDATE;
    expect: blocks
    note: A worker using a plain locking read waits on event 1, held by A — this is the contention SKIP LOCKED avoids.
  - session: A
    sql: UPDATE outbox SET published_at = '2024-12-31 12:00:00' WHERE event_id IN (1, 2, 3); COMMIT;
    note: A marks its batch published and commits; C wakes up, re-reads the now-published rows, and reaches event 4, which is locked by B.
  - session: B
    sql: UPDATE outbox SET published_at = '2024-12-31 12:00:01' WHERE event_id IN (4, 5, 6); COMMIT;
    note: B commits; C's statement completes and finds no unpublished events.
  - session: C
    sql: SELECT event_id, published_at FROM outbox ORDER BY event_id;
    note: Every event was claimed by exactly one worker.
```

Notice what `SKIP LOCKED` does **not** guarantee — and this very run shows it. Worker A's batch ended with event 3 (`OrderPlaced` for order 12) and worker B's batch started with event 4 (`OrderPaid` for order 12): one aggregate's events were split across two workers, so B could publish `OrderPaid` before A publishes `OrderPlaced`. Production relays prevent that by claiming per aggregate (or per key range of `aggregate_id`), by relying on the broker key plus consumer-side version checks (Practical Example 3), or by using a single CDC reader that follows commit order.

## Output

After the relay and consumers have run, the bookkeeping tables tell the full story:

```sql run
SELECT event_id, aggregate_id, aggregate_version, event_type, schema_version,
       IF(published_at IS NULL, 'pending', 'published') AS state
FROM outbox ORDER BY event_id;
SELECT 'loyalty points' AS what, COUNT(*) AS n FROM loyalty_points
UNION ALL SELECT 'processed events', COUNT(*) FROM processed_events
UNION ALL SELECT 'unpublished outbox rows', COUNT(*) FROM outbox WHERE published_at IS NULL;
```

Read it as an operations dashboard: unpublished rows are the relay's backlog (alert if it grows or the oldest one gets old), `processed_events` is each consumer's dedup memory (prune it after the broker's retention period), and the per-aggregate versions document the order consumers must respect.

## Alternative Approaches

**Log-based CDC with Debezium** (display-only). Instead of a polling relay, a Debezium MySQL connector reads the binary log and emits a change event for every committed row change. Combined with Debezium's **Outbox Event Router** transformation, it reads inserts into the `outbox` table and routes each one to a topic derived from `aggregate_type`, keyed by `aggregate_id`. An illustrative connector configuration (see the Debezium documentation for the authoritative property list):

```json
{
  "name": "shop-outbox",
  "config": {
    "connector.class": "io.debezium.connector.mysql.MySqlConnector",
    "database.hostname": "mysql-primary.internal",
    "database.port": "3306",
    "database.user": "debezium",
    "database.server.id": "5501",
    "topic.prefix": "shop",
    "table.include.list": "shop.outbox",
    "transforms": "outbox",
    "transforms.outbox.type": "io.debezium.transforms.outbox.EventRouter",
    "transforms.outbox.table.field.event.id": "event_id",
    "transforms.outbox.table.field.event.key": "aggregate_id",
    "transforms.outbox.table.field.event.payload": "payload",
    "transforms.outbox.route.by.field": "aggregate_type"
  }
}
```

The server-side prerequisites are the ones replicas need (shown display-only; the lab cannot change them):

```sql
-- my.cnf (defaults in 8.0/8.4 already satisfy most of these)
--   log_bin = ON, binlog_format = ROW, binlog_row_image = FULL, gtid_mode = ON (recommended)
CREATE USER 'debezium'@'%' IDENTIFIED BY '…';
GRANT SELECT, RELOAD, SHOW DATABASES, REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'debezium'@'%';
```

| Approach | Ordering | Latency | Load on MySQL | Operational cost | Notes |
|---|---|---|---|---|---|
| Polling relay over the outbox | By `event_id`, caveat: id ≠ commit order | Poll interval (e.g. 100 ms–1 s) | Queries + updates on `outbox` | Low: a small service | Needs cleanup of published rows |
| **CDC on the outbox table** (Debezium + Outbox Event Router) | Commit order (binlog) | Sub-second typically | Reads the binlog like a replica | Kafka Connect cluster, offsets, monitoring | Outbox rows can be deleted right after insert (the binlog keeps the event) |
| CDC on business tables directly (no outbox) | Commit order | Sub-second | Same | Same | Events expose the internal schema; every column change breaks consumers — prefer an outbox for public events |
| Triggers writing to an outbox | Insert order | — | Trigger overhead | Low | Events derived from table changes, not business intent |
| Listen/notify style push | — | — | — | — | MySQL has no `LISTEN/NOTIFY` (PostgreSQL has); polling or CDC is used instead |

## Performance Analysis

- **Outbox write cost** is one extra row insert (plus its index entries) per business transaction — small next to the order rows themselves, and it shares the same commit (one redo flush, one binlog group commit).
- **Relay polling** must be index-driven: `published_at IS NULL ORDER BY event_id LIMIT n` on `(published_at, event_id)` reads only the backlog. Without the index, every poll scans the whole table.
- **Table growth**: published rows accumulate. Delete them in small batches (`DELETE … WHERE published_at < NOW() - INTERVAL 7 DAY ORDER BY event_id LIMIT 5000`), or partition the outbox by day and drop old partitions. With CDC, the outbox can be emptied immediately after insert (insert + delete in the same transaction, as Debezium's documentation describes), because the binlog already holds the event.
- **SKIP LOCKED throughput**: workers never wait for each other, so throughput scales with workers until the broker or the `UPDATE … published_at` becomes the bottleneck. Keep claim transactions short; a worker that stalls holds its batch locked.
- **Consumers**: the dedup insert is a primary-key insert — cheap — but the table must be pruned. Version checks are single-row updates by primary key.
- **CDC**: reading the binlog adds little load; the cost is in operating Kafka Connect, handling schema changes, and ensuring binlog retention (`binlog_expire_logs_seconds`) exceeds the longest connector outage, or the connector must re-snapshot.

## Best Practices

:::best-practice[Event-driven MySQL]
1. **Never dual-write.** Persist state and event in one local transaction (outbox) or derive events from the binlog (CDC).
2. **Design for at-least-once**: every consumer deduplicates by event id in the same transaction as its effect, or applies events with version checks.
3. **Key by aggregate**: broker message key = aggregate id; include a per-aggregate version; never rely on global order.
4. **Publish business events, not table diffs**: the outbox payload is a public contract, decoupled from your internal schema.
5. **Version event schemas explicitly** (`schema_version`), evolve them additively, validate payloads (`JSON_SCHEMA_VALID` on the outbox) and keep upcasters for old versions.
6. **Use `FOR UPDATE SKIP LOCKED`** for competing relay workers and job queues; keep claim transactions short.
7. **Operate it**: monitor outbox backlog and age, CDC lag (binlog position), dedup-table size; prune both outbox and dedup tables.
:::

## Common Mistakes

**Mistake 1 — publishing after commit in application code** (`commit(); kafka.send(event)`) — lost events on crash. There is no SQL to show because the bug is the absence of the outbox row.

**Mistake 2 — non-idempotent consumers.** Re-running the loyalty effect without the dedup check awards points again:

```sql run
INSERT INTO loyalty_points (customer_id, points) VALUES (3, 59) AS new
ON DUPLICATE KEY UPDATE points = loyalty_points.points + new.points;
SELECT customer_id, points FROM loyalty_points;
```

One redelivery, and customer 3 now has double the points. That is why the effect must sit behind the dedup insert.

**Mistake 3 — `FOR UPDATE` without `SKIP LOCKED` in a worker pool.** Workers serialize on the first locked row (the blocked session C in the walkthrough), so adding workers adds waiting, not throughput. `NOWAIT` is the other option when "fail fast" is preferable:

```concurrency
title: NOWAIT fails instead of waiting
setup: |
  CREATE TABLE jobs (job_id INT PRIMARY KEY, state VARCHAR(10) NOT NULL);
  INSERT INTO jobs VALUES (1, 'new'), (2, 'new');
steps:
  - session: A
    sql: START TRANSACTION; SELECT job_id FROM jobs WHERE job_id = 1 FOR UPDATE;
    note: A holds job 1.
  - session: B
    sql: SELECT job_id FROM jobs WHERE job_id = 1 FOR UPDATE NOWAIT;
    expect: error
    errorCode: 3572
    note: B is refused immediately with error 3572 instead of waiting up to innodb_lock_wait_timeout.
  - session: A
    sql: COMMIT;
```

**Mistake 4 — assuming `event_id` order equals commit order.** `AUTO_INCREMENT` values are allocated at insert; a slow transaction can commit a smaller id after a larger one was published (the same late-commit effect as in [ETL and ELT Pipelines](/lessons/m13-etl-elt)). Use per-aggregate versions for correctness, or CDC for commit order.

**Mistake 5 — breaking changes to event payloads.** Renaming `total` to `amount` in place breaks every consumer that has not been deployed yet — and every replay of old events. Add a new schema version and upcast.

## Industry Application

- **Microservices** at retailers, banks and SaaS companies use the outbox pattern as the standard way to publish domain events from relational services; Debezium's Outbox Event Router exists precisely for this.
- **CDC into analytics**: the same binlog stream feeds warehouses and search indexes (Elasticsearch/OpenSearch) — MySQL remains the system of record while other systems are derived views ([Full-Text Search](/lessons/m13-full-text-search), [ETL and ELT Pipelines](/lessons/m13-etl-elt)).
- **Job queues in MySQL**: `SELECT … FOR UPDATE SKIP LOCKED` turns an ordinary table into a reliable work queue (email sending, webhook delivery, report generation) without a separate queueing system, at moderate throughput.
- **Schema registries** (Confluent Schema Registry, Apicurio) enforce compatibility rules (backward, forward, full) for Avro/Protobuf/JSON Schema events — the governance version of `schema_version` and upcasters.
- **Event sourcing** goes further and stores events as the primary record, deriving state from them; MySQL-based event stores use the same building blocks (append-only table, per-aggregate version with a unique key, optimistic concurrency).

## Practice Exercises

````exercise
id: ex1
title: Backlog of the relay
difficulty: easy
concepts: [transactional-outbox]
setup: |
  CREATE TABLE outbox (event_id BIGINT UNSIGNED PRIMARY KEY, aggregate_type VARCHAR(30) NOT NULL, aggregate_id BIGINT UNSIGNED NOT NULL,
    event_type VARCHAR(40) NOT NULL, created_at DATETIME NOT NULL, published_at DATETIME NULL);
  INSERT INTO outbox VALUES
   (1, 'order', 81, 'OrderPlaced', '2024-12-31 10:00:00', '2024-12-31 10:00:01'),
   (2, 'order', 81, 'OrderPaid',   '2024-12-31 10:02:00', NULL),
   (3, 'order', 12, 'OrderPlaced', '2024-12-31 10:01:00', NULL),
   (4, 'customer', 7, 'CustomerUpdated', '2024-12-31 10:03:00', NULL),
   (5, 'order', 12, 'OrderPaid',   '2024-12-31 10:04:00', '2024-12-31 10:04:02');
prompt: |
  For monitoring, return one row per `aggregate_type` that has unpublished events, with columns
  `aggregate_type`, `pending` (number of unpublished events) and `oldest_pending` (the earliest `created_at`
  among them).
hints:
  - "Unpublished means published_at IS NULL."
  - "GROUP BY aggregate_type with COUNT(*) and MIN(created_at)."
solution: |
  SELECT aggregate_type, COUNT(*) AS pending, MIN(created_at) AS oldest_pending
  FROM outbox
  WHERE published_at IS NULL
  GROUP BY aggregate_type;
explanation: |
  The age of the oldest pending event is the most useful alert signal: a growing backlog with an old
  oldest_pending means the relay is stuck. `WHERE published_at = NULL` would return nothing (three-valued
  logic).
````

````exercise
id: ex2
title: Write an order and its event atomically
difficulty: medium
kind: dml
concepts: [transactional-outbox, dual-write]
setup: |
  CREATE TABLE outbox (event_id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY, aggregate_type VARCHAR(30) NOT NULL,
    aggregate_id BIGINT UNSIGNED NOT NULL, aggregate_version INT UNSIGNED NOT NULL, event_type VARCHAR(40) NOT NULL,
    payload JSON NOT NULL, published_at DATETIME NULL,
    UNIQUE KEY uq_agg_version (aggregate_type, aggregate_id, aggregate_version));
prompt: |
  Cancel order **44** in one transaction: set its `status` to `cancelled` in `orders` **and** write an outbox
  event with `aggregate_type = 'order'`, `aggregate_id = 44`, `aggregate_version = 2`,
  `event_type = 'OrderCancelled'` and a payload `{"order_id": 44, "reason": "customer_request"}`.

  The checker returns the order's status and the outbox row's `aggregate_id`, `aggregate_version`,
  `event_type` and `payload`.
hints:
  - "START TRANSACTION; UPDATE orders …; INSERT INTO outbox …; COMMIT;"
  - "Build the payload with JSON_OBJECT('order_id', 44, 'reason', 'customer_request')."
solution: |
  START TRANSACTION;
  UPDATE orders SET status = 'cancelled' WHERE order_id = 44;
  INSERT INTO outbox (aggregate_type, aggregate_id, aggregate_version, event_type, payload)
  VALUES ('order', 44, 2, 'OrderCancelled', JSON_OBJECT('order_id', 44, 'reason', 'customer_request'));
  COMMIT;
check: |
  SELECT o.status, x.aggregate_id, x.aggregate_version, x.event_type, x.payload
  FROM orders o JOIN outbox x ON x.aggregate_id = o.order_id
  WHERE o.order_id = 44;
explanation: |
  Both writes commit or neither does, so a crash can never leave a cancelled order without its event (or an
  event for an order that was not cancelled). The unique key on (aggregate_type, aggregate_id,
  aggregate_version) also stops two concurrent cancellations from both writing version 2.
````

````exercise
id: ex3
title: Find events that arrived out of order
difficulty: hard
concepts: [event-ordering]
setup: |
  CREATE TABLE received_events (receive_seq INT PRIMARY KEY, event_id BIGINT UNSIGNED NOT NULL,
    aggregate_id BIGINT UNSIGNED NOT NULL, aggregate_version INT UNSIGNED NOT NULL);
  INSERT INTO received_events VALUES
   (1, 101, 12, 1), (2, 102, 12, 2), (3, 201, 40, 1), (4, 104, 12, 4), (5, 103, 12, 3),
   (6, 202, 40, 2), (7, 202, 40, 2), (8, 301, 81, 2), (9, 300, 81, 1);
prompt: |
  A consumer logged every delivery in `received_events` in arrival order (`receive_seq`). Return the
  deliveries whose `aggregate_version` is **not exactly one more** than the version of the **previous
  delivery for the same aggregate** (the first delivery of each aggregate is in order only if its version
  is 1). Columns: `receive_seq`, `aggregate_id`, `aggregate_version`, `previous_version` (NULL for the first
  delivery). Order by `receive_seq`.
hints:
  - "LAG(aggregate_version) OVER (PARTITION BY aggregate_id ORDER BY receive_seq) gives the previous delivery's version."
  - "Out of order: previous_version IS NULL AND version <> 1, or version <> previous_version + 1."
  - "Filter the window result in an outer query."
solution: |
  SELECT receive_seq, aggregate_id, aggregate_version, previous_version
  FROM (
    SELECT receive_seq, aggregate_id, aggregate_version,
           LAG(aggregate_version) OVER (PARTITION BY aggregate_id ORDER BY receive_seq) AS previous_version
    FROM received_events
  ) AS r
  WHERE (previous_version IS NULL AND aggregate_version <> 1)
     OR aggregate_version <> previous_version + 1
  ORDER BY receive_seq;
orderMatters: true
explanation: |
  Order 12 received version 4 before 3 (both deliveries are flagged: 4 after 2, then 3 after 4), order 40
  received version 2 twice (a duplicate: 2 after 2), and order 81's first delivery was version 2. A consumer
  applying `UPDATE … WHERE version = new_version - 1` would have rejected every flagged delivery
  automatically — parking the early ones and ignoring the duplicate.
````

````exercise
id: ex4
title: Outbox or CDC on business tables?
difficulty: challenge
kind: open
concepts: [change-data-capture, transactional-outbox, event-schema-evolution]
prompt: |
  A team proposes to skip the outbox and stream the `orders`, `order_items` and `payments` tables directly
  with Debezium to Kafka; consumers would reconstruct "OrderPlaced" from the row changes. Evaluate the
  proposal: what works, what breaks (think about schema changes, transactions spanning several tables,
  business intent, and data exposure), and what you would recommend instead. When is direct table CDC the
  right choice?
solution: |
  - **What works**: CDC gives commit-ordered, complete change capture with no application changes and no
    dual-write risk; deletes are captured; latency is low.
  - **What breaks**:
    - *Coupling to the internal schema*: every column rename, split table or type change in OLTP becomes a
      breaking change for all consumers. The database schema becomes a public API.
    - *Multi-table transactions*: an order is one row in `orders` plus N rows in `order_items` plus payments;
      consumers receive separate events per row and must reassemble a transaction (Debezium can emit
      transaction metadata, but consumers become complex and stateful).
    - *Business intent is lost*: an `UPDATE orders SET status = 'cancelled'` does not say *why* (customer
      request vs fraud); a domain event can.
    - *Data exposure*: internal columns (cost, fraud scores, PII) flow to every consumer unless filtered.
  - **Recommendation**: publish **domain events via an outbox** (written in the same transaction as the
    state change) and stream **only the outbox** with Debezium's Outbox Event Router. Version the payloads
    (`schema_version`), key by aggregate id, keep consumers idempotent.
  - **Direct table CDC is right** for *internal* replication-like uses: feeding the team's own warehouse or
    search index, cache invalidation, migrations between databases — cases where the consumer is owned by
    the same team and is expected to mirror the schema.
````

## Knowledge Assessment

```quiz
- id: q1
  question: "What problem does the transactional outbox solve?"
  options:
    - "Slow Kafka producers"
    - "The dual-write problem: making a database change and the corresponding event atomic without a distributed transaction"
    - "Exactly-once delivery by the broker"
    - "Ordering events across all aggregates"
  answer: 1
  explanation: "The outbox row is committed in the same local transaction as the state change, so both exist or neither does. Delivery is still at-least-once (consumers deduplicate), and ordering is only per aggregate."
  concepts: [transactional-outbox, dual-write]
- id: q2
  question: "Two relay workers run `SELECT … WHERE published_at IS NULL ORDER BY event_id LIMIT 3 FOR UPDATE SKIP LOCKED` concurrently on 6 unpublished events. What happens?"
  options:
    - "The second worker waits until the first commits"
    - "Both receive events 1–3"
    - "They receive disjoint batches (1–3 and 4–6) without waiting"
    - "The second worker gets error 3572"
  answer: 2
  explanation: "SKIP LOCKED skips rows locked by other transactions, as the concurrency scenario shows. A plain FOR UPDATE would wait; NOWAIT would raise error 3572."
  concepts: [skip-locked]
- id: q3
  question: "A consumer receives the same `OrderPlaced` event twice. Which design guarantees the effect is applied once? (Select all that apply.)"
  type: multi
  options:
    - "Insert the event_id into a processed_events table with a primary key in the same transaction as the effect, and skip if it already exists"
    - "Apply state changes with UPDATE … WHERE version = new_version - 1"
    - "Ask the broker for exactly-once delivery and do nothing else"
    - "Process the event and record the event_id in a separate, later transaction"
  answer: [0, 1]
  explanation: "Deduplication in the same transaction, or version-checked updates, make processing idempotent. Broker features cannot cover side effects in the consumer's own database, and recording the id in a separate transaction leaves a crash window in which the effect is applied but not recorded."
  concepts: [idempotent-consumer]
- id: q4
  question: "Why can a polling relay that publishes in `event_id` order still publish events out of commit order?"
  options:
    - "Because AUTO_INCREMENT values are assigned at insert time, so a transaction holding a smaller id can commit after one with a larger id"
    - "Because InnoDB reorders rows on disk"
    - "Because ORDER BY event_id is not deterministic"
    - "Because SKIP LOCKED randomizes results"
  answer: 0
  explanation: "Id allocation and commit are different moments. Binlog-based CDC follows commit order; per-aggregate version numbers make order checkable by consumers. ORDER BY on a primary key is deterministic, and SKIP LOCKED only skips locked rows."
  concepts: [event-ordering]
- id: q5
  question: "What does a Debezium MySQL connector read to produce change events?"
  options:
    - "The InnoDB redo log"
    - "The general query log"
    - "The row-based binary log, as a replica would, after an initial consistent snapshot"
    - "Triggers installed on every table"
  answer: 2
  explanation: "Debezium registers like a replica and streams row events (before/after images) from the binary log in commit order, storing its binlog position or GTID set as the offset. The redo log is InnoDB-internal, the general log is a diagnostic text log, and Debezium does not use triggers."
  concepts: [change-data-capture]
- id: q6
  question: "Version 2 of an event renames `total` to `amount`. What is the safest evolution strategy?"
  options:
    - "Rename the field in place and redeploy all consumers at the same time"
    - "Publish the new shape with schema_version = 2, keep old events readable, and upcast v1 to v2 in consumers (or a view) while tolerant readers ignore unknown fields"
    - "Delete all old events"
    - "Store events as plain text"
  answer: 1
  explanation: "Coordinated big-bang deployments fail in practice and replays of old events would still break. Explicit versions plus upcasting keep every historical event processable. Deleting history removes replay ability; plain text loses structure without solving compatibility."
  concepts: [event-schema-evolution]
- id: q7
  question: "In the ordering example, `OrderShipped` (version 3) arrives before `OrderPaid` (version 2) and `UPDATE order_projection … WHERE version = 2` affects 0 rows. What should the consumer do?"
  options:
    - "Overwrite the projection with version 3 anyway"
    - "Park or retry the event until version 2 has been applied"
    - "Discard the event permanently"
    - "Lower the version number to 2"
  answer: 1
  explanation: "An early event must not overwrite state with a gap, and discarding it would lose a real state change. Parking/retrying (with an alert if the gap persists) preserves both order and completeness."
  concepts: [event-ordering]
```
