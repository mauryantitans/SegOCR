---
id: m12-distributed-integrity
dataset: empty
summary: How to keep data correct when one business operation spans a database, other services and a message broker — idempotency keys that turn duplicate requests into a handled error 1062, the transactional outbox and its relay, at-least-once delivery with idempotent consumers, sagas with compensating actions, why "exactly once" is a property you build rather than buy, and reconciliation jobs that catch what slips through.
concepts: [idempotency-keys, transactional-outbox, sagas, compensating-actions, at-least-once, idempotent-consumers, reconciliation, dual-write-problem]
references:
  - title: "MySQL 8.4 Reference Manual — INSERT … ON DUPLICATE KEY UPDATE"
    url: https://dev.mysql.com/doc/refman/8.4/en/insert-on-duplicate.html
  - title: "MySQL 8.4 Reference Manual — DECLARE … HANDLER Statement"
    url: https://dev.mysql.com/doc/refman/8.4/en/declare-handler.html
  - title: "MySQL 8.4 Reference Manual — Locking Reads (SKIP LOCKED)"
    url: https://dev.mysql.com/doc/refman/8.4/en/innodb-locking-reads.html
  - title: "Garcia-Molina & Salem — Sagas (SIGMOD 1987)"
    url: https://doi.org/10.1145/38713.38742
  - title: "Debezium Documentation — Outbox Event Router"
    url: https://debezium.io/documentation/reference/stable/transformations/outbox-event-router.html
  - title: "Stripe API Reference — Idempotent requests"
    url: https://docs.stripe.com/api/idempotent_requests
  - title: "microservices.io — Pattern: Transactional outbox"
    url: https://microservices.io/patterns/data/transactional-outbox.html
related: [m08-optimistic-pessimistic, m08-acid-in-innodb, m12-web-app-architecture, m12-sharding, m12-caching]
---

## Introduction

Inside one MySQL transaction, ACID makes correctness almost easy: either every change commits or none does ([ACID in InnoDB](/lessons/m08-acid-in-innodb)). Real business operations rarely stay inside one database. Placing an order writes the order to MySQL, charges a card through a payment provider's API, publishes an `OrderPlaced` event to Kafka for the warehouse and the e-mail service, and perhaps updates a search index and a cache. Each of those systems commits independently, networks drop messages and connections, clients retry, and processes crash between any two steps.

The resulting failure modes are concrete and expensive:

- A customer double-clicks "Pay" — or the mobile app retries after a timeout — and is **charged twice**.
- The order commits but the process crashes before publishing the event: the warehouse **never ships** it.
- The event is published but the order transaction rolls back: the warehouse ships an order that **does not exist**.
- The payment succeeds, the inventory reservation fails, and nothing undoes the payment.
- A message is delivered twice and a loyalty-points consumer **credits twice**.

This lesson presents the patterns that professional systems use to stay correct anyway — **idempotency keys**, the **transactional outbox**, **idempotent consumers**, **sagas with compensating actions**, and **reconciliation jobs** — and runs the MySQL half of each for real: a duplicate request rejected by a unique key (error 1062) and handled, a concurrent duplicate that waits and then fails, an outbox whose events can never disagree with the data, a relay that publishes with `SKIP LOCKED`, a saga that compensates, and a reconciliation report against a provider's settlement data.

:::note[Prerequisites]
This lesson builds on [Optimistic vs Pessimistic Locking](/lessons/m08-optimistic-pessimistic) (conditional writes, version checks), on unique constraints as concurrency control, and on the request-handler patterns of [Database Architecture for Web Applications](/lessons/m12-web-app-architecture).
:::

## DBMS Theory

:::theory[Why distributed operations lose atomicity]
**The dual-write problem.** An operation that writes to two independent systems (a database and a broker, or two databases) cannot make both writes atomic without an atomic commitment protocol. Whatever order the writes happen in, a crash between them leaves one done and the other not.

**Two-phase commit (2PC)** solves atomicity across participants that support it (MySQL implements the participant side with XA transactions), but it blocks if the coordinator fails after the participants have prepared, it adds round trips to every operation, and most external services (payment APIs, SaaS endpoints, Kafka in its usual configuration) cannot take part in it.

**Delivery semantics.** Over an unreliable network, a sender that waits for an acknowledgement and retries on timeout guarantees **at-least-once** delivery; a sender that never retries gives **at-most-once**. **Exactly-once delivery** is impossible in general — a lost acknowledgement makes "processed and the ack was lost" indistinguishable from "never arrived" (a consequence of the Two Generals problem). What systems actually provide is **exactly-once effect**: at-least-once delivery combined with **idempotent processing**, so that a repeated message changes nothing.

**Idempotence.** An operation f is idempotent if f(f(x)) = f(x): applying it twice has the same effect as once. `SET balance = 100` is idempotent; `SET balance = balance + 10` is not — but it becomes idempotent if each application is recorded under a unique key and duplicates are rejected.

**Sagas** (Garcia-Molina and Salem, 1987) replace one long atomic transaction with a sequence of local transactions T₁ … Tₙ, each with a **compensating transaction** Cᵢ that semantically undoes it. If Tₖ fails, the saga runs Cₖ₋₁ … C₁. Sagas give up isolation (other operations can see intermediate states) in exchange for never holding locks across services.
:::

| Problem | Pattern | Guarantee obtained |
|---|---|---|
| Client retries a non-idempotent request | Idempotency key + unique constraint | Each logical request takes effect at most once; retries get the original response |
| Database write + message publish must agree | Transactional outbox + relay | Event published if and only if the transaction committed (at least once) |
| Broker delivers a message twice | Idempotent consumer (processed-message table) | Exactly-once *effect* |
| Business operation spans services | Saga with compensations | Eventually all steps done, or all completed steps compensated |
| Something still went wrong | Reconciliation job | Discrepancies detected and repaired, with an audit trail |

## Why It Exists

These patterns exist because the alternatives fail in production in predictable ways:

- **"Just don't retry"** loses requests whenever a network timeout happens after the server committed — the client cannot know.
- **"Check before insert"** (`SELECT … WHERE request_id = ?`, then `INSERT`) lets two concurrent duplicates both pass the check; only a unique constraint serialises them ([Optimistic vs Pessimistic Locking](/lessons/m08-optimistic-pessimistic)).
- **"Publish after commit"** loses events when the process dies between commit and publish; **"publish before commit"** emits events for transactions that later roll back.
- **"Use a distributed transaction"** is unavailable for HTTP APIs and brokers, and where available (XA) it couples the availability of all participants.
- **"Our broker has exactly-once"** — broker features such as Kafka's idempotent producer and transactions give exactly-once *within Kafka* (producer to topic to Kafka Streams); the moment a consumer writes to MySQL or calls an API, the end-to-end guarantee again depends on the consumer being idempotent.
- **Assuming the patterns are perfect** misses bugs, manual interventions and partner-side errors; reconciliation is the safety net that turns silent divergence into an alert.

## Internal Working

:::internals[How MySQL makes these patterns safe]
The patterns rely on a few precise InnoDB behaviours:

1. **Unique indexes serialise duplicates.** When a transaction inserts a key into a unique index, InnoDB locks that index record. A concurrent transaction inserting the **same** key must wait for the first to commit or roll back (it requests a shared lock on the duplicate record). If the first commits, the second fails with **error 1062** (duplicate entry); if the first rolls back, the second succeeds. There is no window in which both can succeed — unlike any check done with a plain SELECT.
2. **One transaction covers several tables.** The business row, the idempotency record and the outbox event are inserted in the **same** InnoDB transaction, so they commit or roll back together, and the redo log makes the commit durable as a unit.
3. **`SELECT … FOR UPDATE SKIP LOCKED`** lets several relay workers claim different unpublished outbox rows without blocking each other or publishing the same row concurrently.
4. **Conditional updates** (`UPDATE … WHERE status = 'pending'` + affected-row count) implement state-machine transitions that cannot happen twice — the basis of idempotent saga steps and compensations.
5. **The binary log** records committed changes in commit order. Change-data-capture tools (Debezium) can read outbox inserts from it, which turns the outbox relay into log tailing instead of polling.
:::

```mermaid
sequenceDiagram
  participant C as Client
  participant API as Order service
  participant DB as MySQL
  participant R as Outbox relay
  participant K as Broker (Kafka)
  participant W as Warehouse consumer
  C->>API: POST /orders (Idempotency-Key: k1)
  API->>DB: BEGIN; INSERT idempotency_keys(k1); INSERT orders; INSERT outbox(OrderPlaced); COMMIT
  API-->>C: 201 (stored as k1's response)
  C->>API: retry POST /orders (k1)
  API->>DB: INSERT idempotency_keys(k1) → 1062
  API-->>C: 201 (same stored response)
  R->>DB: SELECT … FROM outbox WHERE published_at IS NULL FOR UPDATE SKIP LOCKED
  R->>K: publish OrderPlaced (may repeat after a crash)
  R->>DB: UPDATE outbox SET published_at = …
  K->>W: deliver (at least once)
  W->>W: INSERT processed_events(event_id) → duplicate ignored
```

## MySQL Implementation

### Idempotency keys

The client generates a unique key per *logical* request (a UUID) and sends it with every retry (Stripe's `Idempotency-Key` header is the best-known example). The server records the key with a fingerprint of the request, in the same transaction as the business effect:

```sql run
CREATE TABLE idempotency_keys (
  idem_key      CHAR(36)     NOT NULL PRIMARY KEY,
  request_hash  CHAR(64)     NOT NULL,
  status        ENUM('in_progress', 'completed') NOT NULL,
  response_code SMALLINT     NULL,
  response_body JSON         NULL,
  created_at    DATETIME     NOT NULL
);
CREATE TABLE payments (
  payment_id  INT AUTO_INCREMENT PRIMARY KEY,
  order_ref   VARCHAR(20)   NOT NULL,
  amount      DECIMAL(10,2) NOT NULL,
  idem_key    CHAR(36)      NOT NULL UNIQUE
);
-- First request: record the key, perform the effect, store the response — one transaction
START TRANSACTION;
INSERT INTO idempotency_keys VALUES ('7f1c2a9e-0d2b-4c55-9a41-2b6f0b1d9c01', SHA2('order=A-1001&amount=59.90', 256),
                                     'in_progress', NULL, NULL, '2024-12-01 10:00:00');
INSERT INTO payments (order_ref, amount, idem_key) VALUES ('A-1001', 59.90, '7f1c2a9e-0d2b-4c55-9a41-2b6f0b1d9c01');
UPDATE idempotency_keys
   SET status = 'completed', response_code = 201,
       response_body = JSON_OBJECT('payment_id', LAST_INSERT_ID(), 'status', 'succeeded')
 WHERE idem_key = '7f1c2a9e-0d2b-4c55-9a41-2b6f0b1d9c01';
COMMIT;
SELECT idem_key, status, response_code, response_body FROM idempotency_keys;
```

The client's connection drops before it sees the response, so it retries with the **same** key. The insert of the key fails — this is the database enforcing "at most once":

```sql run error
INSERT INTO idempotency_keys VALUES ('7f1c2a9e-0d2b-4c55-9a41-2b6f0b1d9c01', SHA2('order=A-1001&amount=59.90', 256),
                                     'in_progress', NULL, NULL, '2024-12-01 10:00:07');
```

Error 1062 is not a failure to report to the client; it is the signal to **replay the stored response**. A handler wraps the logic (in an application this is middleware; here a stored procedure with a condition handler for 1062):

```sql run
DELIMITER //
CREATE PROCEDURE pay(IN p_key CHAR(36), IN p_order VARCHAR(20), IN p_amount DECIMAL(10,2), IN p_now DATETIME)
BEGIN
  DECLARE duplicate BOOLEAN DEFAULT FALSE;
  DECLARE CONTINUE HANDLER FOR 1062 SET duplicate = TRUE;
  START TRANSACTION;
  INSERT INTO idempotency_keys VALUES (p_key, SHA2(CONCAT('order=', p_order, '&amount=', p_amount), 256),
                                       'in_progress', NULL, NULL, p_now);
  IF duplicate THEN
    ROLLBACK;
    SELECT IF(request_hash = SHA2(CONCAT('order=', p_order, '&amount=', p_amount), 256),
              'replayed stored response', 'rejected: key reused with a different request (HTTP 422)') AS outcome,
           response_code, response_body
    FROM idempotency_keys WHERE idem_key = p_key;
  ELSE
    INSERT INTO payments (order_ref, amount, idem_key) VALUES (p_order, p_amount, p_key);
    UPDATE idempotency_keys SET status = 'completed', response_code = 201,
           response_body = JSON_OBJECT('payment_id', LAST_INSERT_ID(), 'status', 'succeeded')
     WHERE idem_key = p_key;
    COMMIT;
    SELECT 'executed' AS outcome, response_code, response_body FROM idempotency_keys WHERE idem_key = p_key;
  END IF;
END //
DELIMITER ;
```

```sql run
CALL pay('7f1c2a9e-0d2b-4c55-9a41-2b6f0b1d9c01', 'A-1001', 59.90, '2024-12-01 10:00:07');
CALL pay('7f1c2a9e-0d2b-4c55-9a41-2b6f0b1d9c01', 'A-1001', 99.00, '2024-12-01 10:00:09');
CALL pay('3b8e7c41-5a6f-4e2d-8c19-0f2a7d6e5b42', 'A-1002', 15.00, '2024-12-01 10:01:00');
SELECT payment_id, order_ref, amount FROM payments ORDER BY payment_id;
```

The retry replayed the stored response; a reuse of the key with a *different* amount was rejected instead of silently returning a response for a different request; a new key executed normally. There is exactly one payment per logical request.

## Syntax Breakdown

| Element | Role |
|---|---|
| `PRIMARY KEY` / `UNIQUE` on the idempotency key | The concurrency control: serialises duplicates; second insert gets 1062 |
| `request_hash` (e.g. `SHA2(canonical_request, 256)`) | Detects key reuse with a different payload |
| `status` `in_progress` / `completed` | Distinguishes "being processed" (tell the retry to wait / 409) from "done" (replay) |
| `DECLARE CONTINUE HANDLER FOR 1062` | Stored-program handling of the duplicate; in Python, `except mysql.connector.IntegrityError as e: if e.errno == 1062` |
| `INSERT IGNORE` | Inserts or silently skips duplicates — convenient for dedup tables, but also downgrades *other* errors to warnings |
| `INSERT … ON DUPLICATE KEY UPDATE` | Upsert; with `id = id` it becomes an "insert if absent" that reports affected rows 1 (inserted) or 0 (existed) |
| Outbox table (`event_id`, `aggregate_type`, `aggregate_id`, `event_type`, `payload`, `created_at`, `published_at`) | Events written in the same transaction as the state change |
| `SELECT … FOR UPDATE SKIP LOCKED` | Relay workers claim disjoint batches |
| Processed-messages table (`consumer`, `event_id` PK) | Consumer-side deduplication |
| Saga log / state table + conditional `UPDATE … WHERE state = ?` | Durable saga progress; transitions that cannot run twice |
| Expiry job for idempotency keys | Keys are kept for a bounded period (Stripe documents 24 hours) and then purged |

```python
# Idempotency middleware sketch (display only) — key insert, effect and response in ONE transaction
def handle_payment(request):
    key = request.headers["Idempotency-Key"]
    fingerprint = sha256(canonical_json(request.json))
    with primary.begin() as conn:
        try:
            conn.execute(text("INSERT INTO idempotency_keys (idem_key, request_hash, status, created_at) "
                              "VALUES (:k, :h, 'in_progress', NOW())"), {"k": key, "h": fingerprint})
        except IntegrityError as e:
            if e.orig.args[0] != 1062:
                raise
            conn.rollback()
            row = load_key(key)
            if row.request_hash != fingerprint:
                return error(422, "Idempotency-Key reused with a different request")
            if row.status == "in_progress":
                return error(409, "Request with this key is still being processed")
            return Response(row.response_body, status=row.response_code)   # replay
        result = charge_and_record(conn, request.json, key)   # local effect (see outbox for external calls)
        store_response(conn, key, 201, result)
    return Response(result, status=201)
```

## Practical Examples

### 1. The transactional outbox

The order service must publish `OrderPlaced` exactly when an order commits. It writes the event into an **outbox** table in the same transaction; a separate relay publishes it later.

```sql run
CREATE TABLE orders (order_id INT AUTO_INCREMENT PRIMARY KEY, customer_id INT NOT NULL, total DECIMAL(10,2) NOT NULL, status VARCHAR(12) NOT NULL);
CREATE TABLE outbox (
  event_id       BIGINT AUTO_INCREMENT PRIMARY KEY,
  aggregate_type VARCHAR(20) NOT NULL,
  aggregate_id   VARCHAR(20) NOT NULL,
  event_type     VARCHAR(30) NOT NULL,
  payload        JSON        NOT NULL,
  created_at     DATETIME    NOT NULL,
  published_at   DATETIME    NULL,
  KEY idx_outbox_unpublished (published_at, event_id)
);
-- A committed order: state change and event together
START TRANSACTION;
INSERT INTO orders (customer_id, total, status) VALUES (42, 120.00, 'placed');
INSERT INTO outbox (aggregate_type, aggregate_id, event_type, payload, created_at)
VALUES ('order', LAST_INSERT_ID(), 'OrderPlaced', JSON_OBJECT('order_id', LAST_INSERT_ID(), 'customer_id', 42, 'total', 120.00), '2024-12-01 10:00:00');
COMMIT;
-- A failed order: the transaction rolls back, and so does its event
START TRANSACTION;
INSERT INTO orders (customer_id, total, status) VALUES (43, 80.00, 'placed');
INSERT INTO outbox (aggregate_type, aggregate_id, event_type, payload, created_at)
VALUES ('order', LAST_INSERT_ID(), 'OrderPlaced', JSON_OBJECT('order_id', LAST_INSERT_ID(), 'customer_id', 43, 'total', 80.00), '2024-12-01 10:00:05');
ROLLBACK;
INSERT INTO orders (customer_id, total, status) VALUES (44, 35.50, 'placed');
INSERT INTO outbox (aggregate_type, aggregate_id, event_type, payload, created_at)
VALUES ('order', LAST_INSERT_ID(), 'OrderPlaced', JSON_OBJECT('order_id', LAST_INSERT_ID(), 'customer_id', 44, 'total', 35.50), '2024-12-01 10:00:09');
SELECT o.order_id, o.customer_id, e.event_id, e.event_type
FROM orders o LEFT JOIN outbox e ON e.aggregate_id = o.order_id ORDER BY o.order_id;
SELECT event_id, aggregate_id, event_type, published_at FROM outbox ORDER BY event_id;
```

Every committed order has its event and the rolled-back order has none — the outbox can never disagree with the data, because they share one transaction. (The third order shows the trap of *autocommit*: the order and its outbox insert ran as two separate transactions; a crash between them would have produced an order without an event. Always put both in one explicit transaction.) Note also that the rolled-back transaction consumed an `AUTO_INCREMENT` value for both tables — gaps in ids are normal and must never be used as a signal of missing events.

### 2. The relay: claim, publish, mark

A relay worker claims a batch of unpublished events, publishes them to the broker, and marks them published. With several workers, `SKIP LOCKED` keeps them from colliding:

```sql run
START TRANSACTION;
SELECT event_id, event_type, payload FROM outbox
WHERE published_at IS NULL ORDER BY event_id LIMIT 10 FOR UPDATE SKIP LOCKED;
-- … the relay publishes these events to Kafka and waits for the broker's acknowledgement …
UPDATE outbox SET published_at = '2024-12-01 10:00:15' WHERE published_at IS NULL AND event_id IN (1, 3);
COMMIT;
SELECT event_id, aggregate_id, published_at FROM outbox ORDER BY event_id;
```

The ordering of "publish" and "mark" decides the semantics. Publishing first and marking second is **at-least-once**: a crash after publishing but before the `UPDATE` commits makes the next relay run publish the same events again. Marking first would be at-most-once (events lost on a crash). Outbox relays always choose at-least-once and push deduplication to the consumers — which is why every event carries a unique `event_id`.

:::industry[Polling relay vs log-based relay]
The relay above **polls** the outbox table. The alternative is **change data capture**: Debezium's MySQL connector reads outbox inserts from the binary log and its *Outbox Event Router* transformation turns each row into a Kafka message keyed by `aggregate_id`, so no polling query runs against the primary. Either way, the outbox table needs a retention job (delete published events older than N days) so it does not grow forever.
:::

### 3. Idempotent consumers

The warehouse consumer receives `OrderPlaced` at least once. It records each processed `event_id` in the **same transaction** as its own effect, so a redelivery is detected and skipped:

```sql run
CREATE TABLE processed_events (consumer VARCHAR(20) NOT NULL, event_id BIGINT NOT NULL, processed_at DATETIME NOT NULL,
                               PRIMARY KEY (consumer, event_id));
CREATE TABLE shipments (shipment_id INT AUTO_INCREMENT PRIMARY KEY, order_id INT NOT NULL, created_at DATETIME NOT NULL);
-- Delivery of event 1
START TRANSACTION;
INSERT INTO processed_events VALUES ('warehouse', 1, '2024-12-01 10:00:20');
INSERT INTO shipments (order_id, created_at) VALUES (1, '2024-12-01 10:00:20');
COMMIT;
-- Redelivery of event 1 (the relay crashed after publishing): the dedup insert affects 0 rows
INSERT INTO processed_events VALUES ('warehouse', 1, '2024-12-01 10:00:31') AS new
  ON DUPLICATE KEY UPDATE processed_at = processed_events.processed_at;
SELECT ROW_COUNT() AS newly_recorded;
SELECT order_id, COUNT(*) AS shipments FROM shipments GROUP BY order_id;
```

`ROW_COUNT()` returned 0 — the event was already processed — so the consumer acknowledges the message without creating a second shipment. This is what "exactly once" means in a real system: **at-least-once delivery + idempotent processing = exactly-once effect**.

### 4. A saga with compensating actions

Placing an order across services: (1) reserve stock in the inventory service, (2) charge the customer in the payment service, (3) confirm the order. Each step is a local transaction in its owner's database; if a step fails, the completed steps are compensated in reverse order. The orchestrator stores the saga's state durably so that it can resume after a crash.

```sql run
CREATE TABLE inventory (sku VARCHAR(10) PRIMARY KEY, available INT NOT NULL, reserved INT NOT NULL);
INSERT INTO inventory VALUES ('SKU-1', 5, 0);
CREATE TABLE saga_steps (saga_id INT NOT NULL, step_no INT NOT NULL, action VARCHAR(30) NOT NULL,
                         state ENUM('done', 'failed', 'compensated') NOT NULL, at DATETIME NOT NULL,
                         PRIMARY KEY (saga_id, step_no));
-- Step 1 (inventory service): reserve 2 units — a conditional, idempotent-by-state transition
UPDATE inventory SET available = available - 2, reserved = reserved + 2 WHERE sku = 'SKU-1' AND available >= 2;
INSERT INTO saga_steps VALUES (501, 1, 'reserve SKU-1 x2', IF(ROW_COUNT() = 1, 'done', 'failed'), '2024-12-01 11:00:00');
-- Step 2 (payment service): the provider declines the card
INSERT INTO saga_steps VALUES (501, 2, 'charge 99.80', 'failed', '2024-12-01 11:00:02');
-- Compensation of step 1, only if step 1 is done and not yet compensated (safe to retry)
UPDATE inventory i JOIN saga_steps s ON s.saga_id = 501 AND s.step_no = 1 AND s.state = 'done'
   SET i.available = i.available + 2, i.reserved = i.reserved - 2
 WHERE i.sku = 'SKU-1';
UPDATE saga_steps SET state = 'compensated', at = '2024-12-01 11:00:03' WHERE saga_id = 501 AND step_no = 1 AND state = 'done';
-- A retried compensation (the orchestrator restarted) changes nothing
UPDATE inventory i JOIN saga_steps s ON s.saga_id = 501 AND s.step_no = 1 AND s.state = 'done'
   SET i.available = i.available + 2, i.reserved = i.reserved - 2
 WHERE i.sku = 'SKU-1';
SELECT ROW_COUNT() AS rows_changed_by_retry;
SELECT * FROM inventory;
SELECT step_no, action, state FROM saga_steps WHERE saga_id = 501 ORDER BY step_no;
```

The reservation was made, the payment failed, the reservation was released, and a duplicate compensation (common after an orchestrator restart) changed nothing because it was guarded by the step's recorded state. (In a real system the guard and the inventory change would be in the inventory service's own database, often as "release reservation R-501 if it is still active".) Two properties make this saga safe: every step and every compensation is **idempotent**, and the saga's progress is **durable**. What sagas cannot provide is isolation: between steps 1 and 3, other customers see 2 fewer units available, and a report could count a reservation that is later released — semantic locks ("reserved" rather than "sold") make that intermediate state explicit.

## Execution Walkthrough

### Two concurrent duplicates

The most important property of an idempotency key is that it works even when the duplicates arrive *at the same time* — two app instances receiving the same retried request in parallel. A SELECT-based check would let both through; the unique index does not.

```concurrency
title: Two simultaneous requests with the same idempotency key
description: Sessions A and B are two app instances handling the same retried payment request.
setup: |
  CREATE TABLE idempotency_keys (idem_key CHAR(36) PRIMARY KEY, status VARCHAR(12) NOT NULL, response_body JSON NULL);
  CREATE TABLE payments (payment_id INT AUTO_INCREMENT PRIMARY KEY, order_ref VARCHAR(20) NOT NULL, amount DECIMAL(10,2) NOT NULL);
steps:
  - session: A
    sql: START TRANSACTION; INSERT INTO idempotency_keys VALUES ('c0ffee00-0000-4000-8000-000000000001', 'in_progress', NULL); INSERT INTO payments (order_ref, amount) VALUES ('A-2001', 42.00);
    note: Instance A records the key and performs the payment insert; it has not committed yet.
  - session: B
    sql: START TRANSACTION;
  - session: B
    sql: INSERT INTO idempotency_keys VALUES ('c0ffee00-0000-4000-8000-000000000001', 'in_progress', NULL);
    expect: blocks
    note: Instance B tries to record the same key. InnoDB makes it wait for A's uncommitted insert of that key.
  - observe: locks
  - session: A
    sql: UPDATE idempotency_keys SET status = 'completed', response_body = JSON_OBJECT('payment_id', 1) WHERE idem_key = 'c0ffee00-0000-4000-8000-000000000001'; COMMIT;
    note: A completes and commits. B's waiting insert now fails with 1062 — the request is a duplicate.
  - session: B
    sql: ROLLBACK; SELECT status, response_body FROM idempotency_keys WHERE idem_key = 'c0ffee00-0000-4000-8000-000000000001';
    note: B rolls back and replays A's stored response. Exactly one payment exists.
  - session: B
    sql: SELECT COUNT(*) AS payments FROM payments;
```

The lock snapshot shows the mechanism: A holds `X,REC_NOT_GAP` on the new `idempotency_keys` PRIMARY record — originally an *implicit* lock of A's insert, converted into an explicit one when B needed to check the key (which is why the snapshot notes it was reported by B's thread) — and B waits with an `S,REC_NOT_GAP` request on the same key: the duplicate-key check. When A commits, B's insert completes with error 1062 (*Duplicate entry … for key 'idempotency_keys.PRIMARY'*). Had A rolled back instead, B's insert would have succeeded and B would have performed the payment — still exactly once.

## Output

- The first request with key `7f1c2a9e-…` created payment 1 and stored a 201 response; the retry's key insert failed with error 1062.
- The `pay` procedure replayed the stored response for the retry, rejected the same key with a different amount, and executed a new key — two payments for two logical requests.
- The outbox contained an event for every committed order and none for the rolled-back order; the relay claimed unpublished events with `SKIP LOCKED` and marked them published.
- The warehouse consumer's dedup insert recorded 0 new rows on redelivery, so order 1 has exactly one shipment.
- The saga reserved stock, recorded the payment failure, released the reservation, and ignored a repeated compensation (0 rows changed).
- In the concurrency scenario the second instance's key insert waited for the first and then failed with 1062; one payment was created.

## Alternative Approaches

| Need | Pattern here | Alternatives | Trade-off |
|---|---|---|---|
| Safe retries | Idempotency-key table | Natural idempotency (`PUT` with full state, conditional updates with versions); unique business keys (`UNIQUE (order_ref)` on payments) | Natural keys are simpler when they exist; generic keys work for any POST |
| DB + event consistency | Transactional outbox | CDC on the business tables directly (no outbox; consumers infer events from row changes); event sourcing (the event log *is* the source of truth); XA/2PC with a transactional broker | CDC on raw tables couples consumers to your schema; event sourcing changes the whole architecture; XA is rarely available end to end |
| Cross-service operation | Orchestrated saga | **Choreographed** saga (services react to each other's events, no orchestrator); 2PC (rare); redesign service boundaries so the operation is local | Choreography avoids a central coordinator but makes the flow harder to see and debug |
| Duplicate messages | Processed-events table | Natural idempotence of the effect (upsert by key); broker-side dedup windows | Broker dedup covers only the broker's scope |
| Divergence detection | Reconciliation job | Continuous verification (checksums, `pt-table-checksum` between replicas), audit logs | Reconciliation is batch and catches *business* divergence, not just replication drift |

**Reconciliation** deserves a runnable example of its own, because it is how organisations actually find the cases the patterns missed. Compare the orders the system believes are paid with the payment provider's settlement report:

```sql run
CREATE TABLE our_payments (order_ref VARCHAR(20) PRIMARY KEY, amount DECIMAL(10,2) NOT NULL, status VARCHAR(12) NOT NULL);
CREATE TABLE provider_settlements (provider_ref VARCHAR(20) PRIMARY KEY, order_ref VARCHAR(20) NOT NULL, amount DECIMAL(10,2) NOT NULL, settled_on DATE NOT NULL);
INSERT INTO our_payments VALUES ('A-3001', 20.00, 'paid'), ('A-3002', 45.00, 'paid'), ('A-3003', 12.50, 'paid'), ('A-3004', 99.00, 'failed');
INSERT INTO provider_settlements VALUES ('ch_1', 'A-3001', 20.00, '2024-12-02'), ('ch_2', 'A-3002', 54.00, '2024-12-02'),
                                        ('ch_3', 'A-3004', 99.00, '2024-12-02'), ('ch_4', 'A-3005', 15.00, '2024-12-02');
SELECT order_ref, our_status, our_amount, provider_amount,
       CASE WHEN our_amount IS NULL THEN 'charged by provider, unknown to us'
            WHEN provider_amount IS NULL AND our_status = 'paid' THEN 'marked paid, never settled'
            WHEN our_status <> 'paid' THEN 'provider charged a payment we consider failed'
            WHEN our_amount <> provider_amount THEN 'amount mismatch'
       END AS discrepancy
FROM (
  SELECT o.order_ref, o.status AS our_status, o.amount AS our_amount, p.amount AS provider_amount
  FROM our_payments o LEFT JOIN provider_settlements p ON p.order_ref = o.order_ref
  UNION ALL
  SELECT p.order_ref, NULL, NULL, p.amount
  FROM provider_settlements p LEFT JOIN our_payments o ON o.order_ref = p.order_ref
  WHERE o.order_ref IS NULL
) AS r
WHERE our_amount IS NULL OR provider_amount IS NULL OR our_status <> 'paid' OR our_amount <> provider_amount
ORDER BY order_ref;
```

MySQL has no `FULL OUTER JOIN`, so the report is a `LEFT JOIN` plus the anti-join of the other direction. Four kinds of discrepancy surfaced — each one a real class of incident: an amount mismatch (A-3002), an order marked paid that never settled (A-3003), a charge for a payment we recorded as failed (A-3004, typically a timeout where the provider actually succeeded — the exact case idempotency keys with the *provider* prevent), and a charge we have no record of (A-3005). A production reconciliation job runs daily, writes its findings to a table, alerts above a threshold, and feeds a repair workflow with an audit trail.

## Performance Analysis

- **Idempotency keys** add one indexed insert (and one update) per request, plus storage until expiry. With random UUIDv4 keys as a primary key, inserts land at random positions in the B+ tree; a time-ordered key (UUIDv7) or an AUTO_INCREMENT surrogate PK with a UNIQUE key on the idempotency key keeps inserts sequential. A purge job (batched, see [Handling Millions of Records](/lessons/m12-large-tables)) deletes keys after the retention window.
- **Outbox writes** add one insert per event in the business transaction — usually negligible next to the business writes. The polling relay's query must be an index range scan (`(published_at, event_id)` here) or it degrades as the table grows; published rows must be purged. CDC relays avoid polling load entirely and deliver events with lower latency.
- **Consumer dedup** adds one indexed insert per message in the consumer's transaction. The processed-events table can be pruned by time once the broker's redelivery window has passed.
- **Sagas** trade latency (several sequential steps across the network) and complexity for not holding locks across services; each step's local transaction is short.
- **Unique-key waits** under concurrent duplicates are brief (the duration of the first request's transaction) — another reason to keep that transaction short and free of external calls.
- **Reconciliation** is a batch scan; run it against replicas or a warehouse and index the join keys.

## Best Practices

:::best-practice[Correctness across system boundaries]
1. **Require idempotency keys on every non-idempotent API** that clients may retry (payments, orders, transfers); store the key, a request fingerprint and the response in the **same transaction** as the effect.
2. **Let the unique index decide**, never a preceding SELECT; treat 1062 on the key as "replay", and a fingerprint mismatch as a client error.
3. **Never call external services inside the database transaction**; when you must call them, pass your idempotency key to *their* API so retries are safe on their side too.
4. **Write events through an outbox** in the same transaction as the state change; relay at least once; give every event a unique id.
5. **Make every consumer idempotent** — dedup table or naturally idempotent upserts — and do the dedup insert in the same transaction as the effect.
6. **Model sagas explicitly**: durable state, idempotent steps and compensations, timeouts, and a terminal state for every saga; alert on sagas stuck in intermediate states.
7. **Reconcile** every important cross-system invariant (payments vs orders, stock vs reservations, ledger vs balances) on a schedule, and record repairs.
8. **Expire and purge** idempotency keys, outbox rows and processed-event records with batched jobs.
:::

## Common Mistakes

**Mistake 1 — checking for duplicates with SELECT.** `SELECT 1 FROM payments WHERE request_id = ?` followed by `INSERT` lets two concurrent duplicates both see "no row". Only a unique constraint closes the race, as the concurrency scenario showed.

**Mistake 2 — recording the idempotency key in a separate transaction.** If the key commits but the effect fails (or vice versa), retries are either blocked forever or executed twice. Key, effect and stored response belong in one transaction.

**Mistake 3 — publishing to the broker inside or after the transaction without an outbox.** Inside: events for rolled-back transactions. After: lost events on a crash. Use the outbox.

**Mistake 4 — non-idempotent consumers.** `UPDATE loyalty SET points = points + 10` on every delivery double-credits on redelivery. Record the event id first, or make the effect an upsert keyed by the event.

**Mistake 5 — compensations that are not guarded.** A compensation retried after a crash must not release a reservation twice; guard it on recorded state, exactly like the saga example.

**Mistake 6 — `INSERT IGNORE` hiding real errors.** It turns duplicates into warnings — but also data-truncation and other errors:

```sql run
CREATE TABLE dedup_demo (event_id INT PRIMARY KEY, note VARCHAR(5) NOT NULL);
INSERT IGNORE INTO dedup_demo VALUES (1, 'first'), (1, 'again'), (2, 'this note is far too long');
SHOW WARNINGS;
SELECT * FROM dedup_demo ORDER BY event_id;
```

The duplicate was skipped as intended, but the second row's note was silently truncated to `this ` — a data-quality bug hidden by the same keyword. Prefer `INSERT … ON DUPLICATE KEY UPDATE pk = pk` (or catch 1062 explicitly) for deduplication.

## Industry Application

- **Payment APIs** (Stripe, Adyen, PayPal) accept idempotency keys on mutating requests and replay the original response for retries; integrators pass their own order or attempt id as the key, which prevents double charges across timeouts.
- **Microservice platforms** standardise the outbox: libraries for Spring (e.g. via Debezium or framework modules), Rails and Django; Debezium's Outbox Event Router is widely used to stream outbox rows from MySQL's binlog to Kafka.
- **Sagas** are implemented with workflow engines (Temporal, AWS Step Functions, Camunda) or in-house orchestrators that persist saga state in MySQL; choreographed sagas use events between services.
- **Finance teams** run daily reconciliation between internal ledgers, payment providers' settlement files and bank statements; discrepancies feed operational queues, and the reports are audited.
- **Kafka exactly-once** (idempotent producers, transactions, `read_committed` consumers) is used within streaming pipelines; at the boundary to MySQL, consumers still deduplicate by event id or use transactional upserts.

## Practice Exercises

```exercise
id: ex1
title: Find reused idempotency keys
difficulty: easy
setup: |
  CREATE TABLE request_log (id INT AUTO_INCREMENT PRIMARY KEY, idem_key VARCHAR(36) NOT NULL, request_hash CHAR(8) NOT NULL, received_at DATETIME NOT NULL);
  INSERT INTO request_log (idem_key, request_hash, received_at) VALUES
    ('k1','aaaa1111','2024-12-01 10:00:00'),('k1','aaaa1111','2024-12-01 10:00:03'),('k2','bbbb2222','2024-12-01 10:01:00'),
    ('k3','cccc3333','2024-12-01 10:02:00'),('k3','dddd4444','2024-12-01 10:02:30'),('k4','eeee5555','2024-12-01 10:03:00'),
    ('k1','aaaa1111','2024-12-01 10:00:09'),('k5','ffff6666','2024-12-01 10:04:00'),('k5','ffff7777','2024-12-01 10:05:00');
concepts: [idempotency-keys]
prompt: |
  A key sent several times with the **same** fingerprint is a normal retry; a key sent with **different**
  fingerprints is a client bug. Return `idem_key`, `requests` and `distinct_payloads` for keys used with more
  than one distinct `request_hash`.
hints:
  - "GROUP BY idem_key, then HAVING COUNT(DISTINCT request_hash) > 1."
solution: |
  SELECT idem_key, COUNT(*) AS requests, COUNT(DISTINCT request_hash) AS distinct_payloads
  FROM request_log GROUP BY idem_key HAVING COUNT(DISTINCT request_hash) > 1;
explanation: |
  k3 and k5 were reused with different payloads and must be rejected (HTTP 422), not replayed. k1 was sent
  three times with the same fingerprint — legitimate retries that should receive the stored response.
```

```exercise
id: ex2
title: Idempotent consumer
difficulty: medium
kind: dml
setup: |
  CREATE TABLE processed_events (consumer VARCHAR(20) NOT NULL, event_id BIGINT NOT NULL, PRIMARY KEY (consumer, event_id));
  CREATE TABLE loyalty (customer_id INT PRIMARY KEY, points INT NOT NULL);
  INSERT INTO loyalty VALUES (7, 100);
  INSERT INTO processed_events VALUES ('loyalty', 900);
concepts: [idempotent-consumers, at-least-once]
prompt: |
  The loyalty consumer receives these deliveries in order: event **900** (+10 points for customer 7, already
  processed earlier), event **901** (+25 points for customer 7), and event **901** again (a redelivery).
  Process all three deliveries so that each event's points are applied **exactly once**: for each delivery,
  record the event in `processed_events` for consumer `'loyalty'` and apply the points only if the event was
  not processed before. Use one transaction per delivery.
hints:
  - "Insert into processed_events first with INSERT … ON DUPLICATE KEY UPDATE event_id = event_id, then check ROW_COUNT()."
  - "Capture ROW_COUNT() in a user variable immediately after the INSERT, then make the UPDATE conditional on that variable."
solution: |
  START TRANSACTION;
  INSERT INTO processed_events VALUES ('loyalty', 900) ON DUPLICATE KEY UPDATE event_id = event_id;
  SET @new = ROW_COUNT();
  UPDATE loyalty SET points = points + 10 WHERE customer_id = 7 AND @new = 1;
  COMMIT;
  START TRANSACTION;
  INSERT INTO processed_events VALUES ('loyalty', 901) ON DUPLICATE KEY UPDATE event_id = event_id;
  SET @new = ROW_COUNT();
  UPDATE loyalty SET points = points + 25 WHERE customer_id = 7 AND @new = 1;
  COMMIT;
  START TRANSACTION;
  INSERT INTO processed_events VALUES ('loyalty', 901) ON DUPLICATE KEY UPDATE event_id = event_id;
  SET @new = ROW_COUNT();
  UPDATE loyalty SET points = points + 25 WHERE customer_id = 7 AND @new = 1;
  COMMIT;
check: |
  SELECT (SELECT points FROM loyalty WHERE customer_id = 7) AS points,
         (SELECT GROUP_CONCAT(event_id ORDER BY event_id) FROM processed_events WHERE consumer = 'loyalty') AS processed;
explanation: |
  The final balance is 125: event 900 was already applied before (no change), 901 is applied once, its
  redelivery is skipped. `ON DUPLICATE KEY UPDATE event_id = event_id` reports 1 affected row for a new
  event and 0 for an existing one, without raising an error. Recording the event and applying the effect in
  one transaction is essential: if they were separate, a crash between them would either lose or duplicate
  the points.
```

```exercise
id: ex3
title: Stuck outbox events
difficulty: medium
setup: |
  CREATE TABLE outbox (event_id BIGINT PRIMARY KEY, aggregate_id VARCHAR(10) NOT NULL, event_type VARCHAR(20) NOT NULL, created_at DATETIME NOT NULL, published_at DATETIME NULL);
  INSERT INTO outbox VALUES
    (1,'o1','OrderPlaced','2024-12-01 10:00:00','2024-12-01 10:00:02'),
    (2,'o2','OrderPlaced','2024-12-01 10:01:00',NULL),
    (3,'o1','OrderShipped','2024-12-01 10:03:00',NULL),
    (4,'o3','OrderPlaced','2024-12-01 10:09:30',NULL),
    (5,'o2','OrderCancelled','2024-12-01 10:02:00','2024-12-01 10:02:01');
concepts: [transactional-outbox]
prompt: |
  It is now `'2024-12-01 10:10:00'`. Monitoring must alert on outbox events that are **unpublished and older
  than 5 minutes**. Return `event_id`, `aggregate_id`, `event_type` and `age_minutes` (whole minutes) for those
  events.
hints:
  - "published_at IS NULL AND created_at < '2024-12-01 10:10:00' - INTERVAL 5 MINUTE"
  - "TIMESTAMPDIFF(MINUTE, created_at, '2024-12-01 10:10:00')"
solution: |
  SELECT event_id, aggregate_id, event_type,
         TIMESTAMPDIFF(MINUTE, created_at, '2024-12-01 10:10:00') AS age_minutes
  FROM outbox
  WHERE published_at IS NULL AND created_at < '2024-12-01 10:10:00' - INTERVAL 5 MINUTE;
explanation: |
  Events 2 (9 minutes) and 3 (7 minutes) are stuck; event 4 is only 30 seconds old. Note event 5:
  `OrderCancelled` for o2 was published while o2's earlier `OrderPlaced` (event 2) is still pending —
  consumers could see the cancellation before the placement. Relays that must preserve per-aggregate order
  publish in `event_id` order per `aggregate_id` (Kafka partitioning by aggregate id) and stop at the first
  failure for that aggregate.
```

```exercise
id: ex4
title: Reconcile orders and settlements
difficulty: hard
setup: |
  CREATE TABLE our_payments (order_ref VARCHAR(10) PRIMARY KEY, amount DECIMAL(10,2) NOT NULL, status VARCHAR(10) NOT NULL);
  CREATE TABLE settlements (order_ref VARCHAR(10) PRIMARY KEY, amount DECIMAL(10,2) NOT NULL);
  INSERT INTO our_payments VALUES ('B-1', 10.00, 'paid'), ('B-2', 25.00, 'paid'), ('B-3', 30.00, 'paid'), ('B-4', 5.00, 'failed');
  INSERT INTO settlements VALUES ('B-1', 10.00), ('B-2', 20.00), ('B-4', 5.00), ('B-9', 7.50);
concepts: [reconciliation]
prompt: |
  Return every discrepancy between `our_payments` and `settlements` with columns `order_ref`, `our_amount`,
  `settled_amount` and `issue`, where `issue` is one of:
  `'missing settlement'` (we have it as paid, provider has nothing), `'unknown charge'` (provider has it, we
  have no row), `'charged but failed'` (we have it as failed, provider settled it), `'amount mismatch'`.
  Matching rows with no issue must not appear.
hints:
  - "MySQL has no FULL OUTER JOIN: combine a LEFT JOIN from our_payments with an anti-join from settlements."
  - "Wrap the union in a derived table and compute issue with CASE, filtering out rows where issue IS NULL."
solution: |
  SELECT order_ref, our_amount, settled_amount, issue
  FROM (
    SELECT order_ref, our_amount, settled_amount,
           CASE WHEN our_amount IS NULL THEN 'unknown charge'
                WHEN settled_amount IS NULL AND status = 'paid' THEN 'missing settlement'
                WHEN status = 'failed' AND settled_amount IS NOT NULL THEN 'charged but failed'
                WHEN our_amount <> settled_amount THEN 'amount mismatch' END AS issue
    FROM (
      SELECT o.order_ref, o.amount AS our_amount, s.amount AS settled_amount, o.status
      FROM our_payments o LEFT JOIN settlements s ON s.order_ref = o.order_ref
      UNION ALL
      SELECT s.order_ref, NULL, s.amount, NULL
      FROM settlements s LEFT JOIN our_payments o ON o.order_ref = s.order_ref
      WHERE o.order_ref IS NULL
    ) AS joined
  ) AS r
  WHERE issue IS NOT NULL;
explanation: |
  B-2 is an amount mismatch, B-3 is missing a settlement, B-4 was charged although we recorded a failure
  (the classic "timeout, but the provider succeeded" case), and B-9 is a charge we know nothing about. B-1
  matches and is excluded. Using only a LEFT JOIN from our side would silently miss B-9.
```

```exercise
id: ex5
title: Design a booking saga
difficulty: hard
kind: open
concepts: [sagas, compensating-actions, idempotency-keys]
prompt: |
  A travel site books a trip as: (1) hold a flight seat with an airline API (holds expire after 15 minutes),
  (2) reserve a hotel room with a hotel API, (3) charge the customer via a payment provider, (4) confirm the
  flight and hotel, (5) e-mail the itinerary. Design the saga: the order of steps, each step's compensation,
  how retries and crashes are handled, where state is stored, and what the customer sees during and after
  failures.
solution: |
  **Order.** Put the steps that are easiest to undo and most likely to fail first; put the irreversible
  money movement late: hold seat → reserve room → authorise payment (not capture) → confirm flight → confirm
  hotel → capture payment → e-mail. Authorise/capture splits the charge into a reversible hold and a final
  capture.

  **Compensations.** Seat hold → release hold (or let it expire); room reservation → cancel reservation;
  payment authorisation → void authorisation; after capture → refund (a different, visible operation —
  hence capture last). Confirmation steps that fail after partial confirmation require compensation of the
  confirmed part (cancel the confirmed flight per fare rules) — model these explicitly.

  **Retries and crashes.** The orchestrator stores the saga in MySQL (`trips`, `trip_steps` with state per
  step), updates state with conditional transitions, and passes a stable idempotency key per step to every
  external API (e.g. `trip-123-step-2`), so a retry after a timeout never double-reserves or double-charges.
  Timeouts are treated as "unknown" and resolved by querying the provider (or retrying with the same key),
  never as failure. A recovery worker resumes sagas from their last durable state; sagas stuck longer than
  the seat-hold expiry are compensated.

  **Customer view.** "Booking in progress" while running; on failure, a clear message and either automatic
  release (nothing charged — authorisation voided) or a refund notice; the e-mail step is retried
  independently (outbox) and never undoes the booking.

  **Safety net.** A daily reconciliation compares trips with airline, hotel and payment provider reports to
  catch holds, reservations or charges without a matching completed or compensated saga.
explanation: |
  Strong answers sequence steps by reversibility, use authorise/capture for money, give each external call an
  idempotency key, store saga state durably with idempotent transitions, treat timeouts as unknown, and add
  reconciliation.
```

```exercise
id: ex6
title: The exactly-once claim
difficulty: challenge
kind: open
concepts: [at-least-once, idempotent-consumers, transactional-outbox, dual-write-problem]
prompt: |
  A design document says: "We use Kafka with exactly-once semantics enabled, so the inventory service will
  decrement stock exactly once per OrderPlaced event. The order service publishes the event right after
  committing the order to MySQL." Identify every way this can still produce wrong stock or missing events,
  and specify the changes needed for a correct exactly-once *effect*.
solution: |
  **Producer side (dual write).** "Publish right after commit" loses the event if the order service crashes
  or the broker is unavailable between the commit and the publish; retries inside the service do not survive
  a process crash. Fix: transactional outbox in the order transaction, relayed by a poller or Debezium. The
  relay is at-least-once, so duplicates can appear even with an idempotent Kafka producer (a relay that
  crashes after publishing but before marking will publish again — Kafka's idempotent producer only
  deduplicates retries within one producer session).

  **Consumer side.** Kafka's exactly-once (transactions + `read_committed`) covers consume-transform-produce
  *within Kafka*. The inventory service writes to MySQL: if it commits the stock decrement and then crashes
  before committing its consumer offset, the message is redelivered and stock is decremented twice; if it
  commits the offset first, a crash loses the decrement. Fix: an idempotent consumer — insert the `event_id`
  into a processed-events table **in the same MySQL transaction** as the decrement (or store the consumed
  offset in MySQL in that transaction), and make the decrement conditional (`WHERE available >= qty`).

  **Ordering.** Events for the same order (Placed, Cancelled) must be processed in order; partition by
  `order_id` and publish per-aggregate in order from the outbox.

  **Safety net.** Reconcile stock against orders and reservations daily.

  The correct claim is: at-least-once delivery from the outbox + idempotent, transactional consumers =
  exactly-once effect on stock.
explanation: |
  "Exactly once" is an end-to-end property. Broker features cover only the broker's part; the database
  boundaries on both sides need the outbox and idempotent consumers.
```

## Knowledge Assessment

```quiz
- id: q1
  question: "Two app instances process the same retried request concurrently. Both run `SELECT 1 FROM idempotency_keys WHERE idem_key = ?` and find nothing. What prevents a double charge?"
  options:
    - "Nothing; the SELECT check is sufficient"
    - "A UNIQUE/PRIMARY KEY on idem_key: the second INSERT waits for the first transaction and then fails with 1062"
    - "REPEATABLE READ isolation"
    - "The binary log"
  answer: 1
  explanation: "Plain SELECTs don't lock and both see no row; only the unique index serialises the inserts (the lab scenario showed the wait and the 1062). Isolation level and the binlog do not prevent the race."
  concepts: [idempotency-keys]
- id: q2
  question: "Why must the idempotency key, the business effect and the stored response be written in the same transaction?"
  options:
    - "For performance"
    - "So that either all of them exist or none does; otherwise a crash could leave a key without an effect (retries blocked) or an effect without a key (retries executed twice)"
    - "Because MySQL requires it for unique keys"
    - "To allow the key to be reused"
  answer: 1
  explanation: "Atomicity of key + effect is what makes the at-most-once guarantee hold across crashes. It is a correctness requirement, not a performance one."
  concepts: [idempotency-keys]
- id: q3
  question: "What problem does the transactional outbox solve?"
  options:
    - "Slow queries on the orders table"
    - "The dual-write problem: making 'state committed' and 'event published' agree without a distributed transaction"
    - "Exactly-once delivery to consumers"
    - "Replication lag"
  answer: 1
  explanation: "Writing the event in the same local transaction guarantees an event exists iff the state change committed; a relay publishes it at least once. Consumers still need dedup — the outbox does not give exactly-once delivery."
  concepts: [transactional-outbox, dual-write-problem]
- id: q4
  question: "An outbox relay publishes a batch to Kafka and then marks the rows published. It crashes after publishing but before the UPDATE commits. What happens?"
  options:
    - "The events are lost"
    - "The events are published again on the next run — at-least-once — so consumers must be idempotent"
    - "Kafka rejects the duplicates automatically in all cases"
    - "The order transactions are rolled back"
  answer: 1
  explanation: "The rows are still unpublished, so they are sent again. An idempotent producer deduplicates only within one producer session, not across a crashed relay; the orders committed long ago and are unaffected."
  concepts: [transactional-outbox, at-least-once]
- id: q5
  question: "Which statements about 'exactly once' are accurate? (Select all that apply.)"
  type: multi
  options:
    - "Exactly-once delivery over an unreliable network is impossible in general"
    - "At-least-once delivery plus idempotent processing gives an exactly-once effect"
    - "Enabling a broker's exactly-once feature makes database consumers idempotent automatically"
    - "Recording the processed event id in the same transaction as the consumer's effect is a standard way to achieve idempotence"
  answer: [0, 1, 3]
  explanation: "Lost acknowledgements make delivery guarantees at-least-once at best; the effect becomes exactly-once through idempotent, transactional consumers. Broker features do not extend to writes in an external database."
  concepts: [at-least-once, idempotent-consumers]
- id: q6
  question: "In a saga, step 3 (charge payment) fails after steps 1 and 2 succeeded. What should happen?"
  options:
    - "Roll back steps 1 and 2 with ROLLBACK"
    - "Run the compensating actions for step 2 and then step 1, each idempotent and recorded durably"
    - "Retry step 3 forever"
    - "Ignore the failure; eventual consistency will fix it"
  answer: 1
  explanation: "Steps 1 and 2 were separate, committed local transactions — ROLLBACK cannot undo them. Sagas compensate completed steps in reverse order. Retrying may be appropriate for transient errors, but not forever and not for a declined card."
  concepts: [sagas, compensating-actions]
- id: q7
  question: "A reconciliation report finds an order recorded as 'failed' that the payment provider settled. What is the most likely cause?"
  options:
    - "A duplicate key error"
    - "A timeout: the provider succeeded but the response was lost, and the system recorded a failure (and possibly retried without an idempotency key)"
    - "Replication lag on a replica"
    - "A deadlock"
  answer: 1
  explanation: "Ambiguous outcomes (timeouts) must be treated as unknown and resolved by querying the provider or retrying with the same idempotency key. Duplicate keys, lag and deadlocks would not make the provider settle a charge."
  concepts: [reconciliation, idempotency-keys]
- id: q8
  question: "Why is `INSERT IGNORE` a risky way to implement a dedup table?"
  options:
    - "It is slower than INSERT"
    - "It also downgrades other errors (such as data truncation) to warnings, silently storing wrong data"
    - "It does not detect duplicates"
    - "It locks the whole table"
  answer: 1
  explanation: "As the lab showed, the duplicate was skipped but an over-long value was silently truncated. INSERT … ON DUPLICATE KEY UPDATE pk = pk, or catching 1062 explicitly, deduplicates without hiding other errors."
  concepts: [idempotent-consumers]
```
