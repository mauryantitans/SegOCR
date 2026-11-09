---
id: m12-caching
dataset: shop
summary: How caches in front of MySQL trade freshness for latency and load — cache-aside, read-through, write-through and write-behind, TTLs and invalidation, the stale-set race and how version checks prevent it, cache stampedes reproduced with three concurrent sessions and fixed with a lock row and SKIP LOCKED, Redis in practice, and materialized summary tables as a cache inside the database.
concepts: [caching, cache-aside, write-through, cache-invalidation, ttl, cache-stampede, skip-locked, redis, summary-tables]
references:
  - title: "MySQL 8.4 Reference Manual — Locking Read Concurrency with NOWAIT and SKIP LOCKED"
    url: https://dev.mysql.com/doc/refman/8.4/en/innodb-locking-reads.html#innodb-locking-reads-nowait-skip-locked
  - title: "MySQL 8.0 Release Notes — query cache removal (MySQL Server Blog: Retiring Support for the Query Cache)"
    url: https://dev.mysql.com/blog-archive/mysql-8-0-retiring-support-for-the-query-cache/
  - title: "Redis Documentation — SET (EX, PX, NX options)"
    url: https://redis.io/docs/latest/commands/set/
  - title: "Redis Documentation — Key eviction policies"
    url: https://redis.io/docs/latest/develop/reference/eviction/
  - title: "Nishtala et al. — Scaling Memcache at Facebook (NSDI 2013)"
    url: https://www.usenix.org/conference/nsdi13/technical-sessions/presentation/nishtala
  - title: "Vattani, Chierichetti, Lowenstein — Optimal Probabilistic Cache Stampede Prevention (VLDB 2015)"
    url: https://doi.org/10.14778/2757807.2757813
related: [m12-web-app-architecture, m12-read-replicas, m06-denormalization, m10-buffer-pool]
---

## Introduction

The fastest query is the one you do not send. A product page that shows a product's name, price, rating and review count runs the same handful of queries thousands of times per minute, and the answer changes perhaps once a day. Storing that answer in memory close to the application — a **cache** — turns a database round trip of a millisecond or more into a lookup of tens of microseconds, and removes the load from MySQL entirely.

Caching is also where many subtle production bugs live. The famous quip that there are only two hard things in computer science — cache invalidation and naming things — survives because invalidation really is hard: as soon as data exists in two places, they can disagree, and every caching design is a choice about **how stale** the cache may be, **for how long**, and **what happens under concurrency**.

This lesson covers the caching patterns professionals use in front of MySQL — **cache-aside**, **read-through**, **write-through**, **write-behind** — and the failure modes that matter in practice: stale data after updates, the race that writes stale data back into the cache after an invalidation, and the **cache stampede** (thundering herd) when a popular entry expires and hundreds of requests recompute it at once. The cache in the lab is a MySQL table standing in for Redis, so every step can be executed and observed — including a real stampede with three concurrent sessions and its fix with a lock row and `SKIP LOCKED`. The lesson ends with the cache that lives *inside* the database: **materialized summary tables**.

:::note[Dataset and simulation]
The examples use the **shop** dataset. The cache is the table `cache_entries`; in production it would be Redis or Memcached, and the SQL here corresponds to `GET`, `SET … EX`, `DEL` and `SET … NX`. Time is passed explicitly (a fixed "now") so the TTL behaviour is reproducible.
:::

## DBMS Theory

:::theory[Caches as replicas with weaker guarantees]
A cache is a **partial, derived, volatile copy** of data whose source of truth is elsewhere:

- *Partial* — it holds only the hot subset (the working set).
- *Derived* — an entry may be a transformation (a rendered JSON document, an aggregate) rather than a copy of a row.
- *Volatile* — entries may disappear at any time (eviction, restart), so the system must be correct with an empty cache.

Its consistency with the source is governed by two mechanisms: **expiry** (time-to-live, TTL — staleness bounded by time) and **invalidation** (the writer removes or updates the entry when the source changes — staleness bounded by event propagation). Neither is transactional with the database in general: the database commit and the cache update are two separate operations on two separate systems, so between them — or if one fails — the copies differ.

Cache effectiveness is measured by the **hit ratio** h = hits / (hits + misses). The expected latency is h × t_cache + (1 − h) × (t_cache + t_db + t_fill), and the load on the database is proportional to (1 − h). Going from h = 0.90 to h = 0.99 cuts database load by a factor of 10.
:::

The four write/read patterns differ in who loads the cache and when:

| Pattern | Read path | Write path | Staleness risk |
|---|---|---|---|
| **Cache-aside** (lazy loading) | App checks cache; on miss, reads DB and fills cache | App writes DB, then deletes (invalidates) the key | Stale-set race; stale until TTL if invalidation is missed |
| **Read-through** | Cache library loads from DB on miss (the app only talks to the cache) | Same as cache-aside or write-through | Same as cache-aside |
| **Write-through** | Always hit (after first write) | App (or cache layer) writes DB and cache synchronously | Partial failure between the two writes |
| **Write-behind** (write-back) | Hit | Write to cache; flush to DB asynchronously | Data loss if the cache fails before flushing; DB is not the source of truth for a while |

**MySQL's own caches** are internal: the InnoDB **buffer pool** caches pages, not query results ([The Buffer Pool](/lessons/m10-buffer-pool)). MySQL once had a **query cache** for result sets; it was deprecated in 5.7.20 and **removed in MySQL 8.0** because its global invalidation (any write to a table flushed all cached results for that table) and a global mutex made it a scalability bottleneck on multi-core servers. Application-level caching replaced it.

## Why It Exists

- **Latency.** An in-memory key lookup in Redis on the same network is typically well under a millisecond end to end; a query that joins products, reviews and categories and serialises the result costs more, and much more under load.
- **Database load.** Read traffic that the cache absorbs never reaches MySQL, which is cheaper than adding replicas and keeps the primary's CPU for writes.
- **Expensive derived data.** Some values — a recommendation list, a dashboard aggregate, a rendered page fragment — cost far more to compute than to store.
- **Spikes.** A flash sale or a viral link multiplies traffic on a few keys; a cache absorbs the spike, a database alone may not.

What goes wrong without a disciplined design: users see old prices after an update (missing invalidation), a deleted product keeps appearing (negative results never invalidated), a cache restart sends 100% of traffic to the database (cold cache), and a popular entry expiring at peak time triggers hundreds of identical expensive queries at once (stampede), which can take the database down — after which *every* entry expires and the outage feeds itself.

## Internal Working

:::internals[What happens on a cache-aside read and write]
**Read (`GET product:7`)**

1. The application computes a **cache key** that identifies the data and its shape, e.g. `product:7:v3` (entity, id, schema version of the cached document).
2. `GET` from the cache. On a **hit**, deserialise and return — no database access.
3. On a **miss**, query MySQL, build the value, `SET key value EX ttl`, return it.

**Write (update product 7's price)**

1. `UPDATE products … WHERE product_id = 7` and `COMMIT` on the primary.
2. `DEL product:7` (invalidate) — *after* the commit, so a concurrent reader cannot re-fill the cache from the old committed data after the delete… which, as the race below shows, is still not enough on its own.

**Where the data lives in Redis.** Redis keeps all data in memory in a single-threaded event loop (I/O threads exist since 6.0, command execution remains single-threaded), with optional persistence (RDB snapshots, AOF log). When `maxmemory` is reached it **evicts** keys according to `maxmemory-policy` (`allkeys-lru`, `allkeys-lfu`, `volatile-ttl`, …); expired keys are removed lazily on access and by a periodic sampler. Memcached is similar but simpler: a multi-threaded, non-persistent LRU key–value store.

**Concurrency.** Cache operations are individually atomic, but a cache-aside read is a *sequence* (GET, query, SET) and a write is another sequence (UPDATE, COMMIT, DEL). Interleavings of these sequences produce the classic races: the **stale set** (a reader fills the cache with a value read before a concurrent write) and the **stampede** (many readers miss at once and all recompute).
:::

```mermaid
sequenceDiagram
  participant R as Reader
  participant C as Cache
  participant DB as MySQL
  participant W as Writer
  R->>C: GET product:7 (miss)
  R->>DB: SELECT … (reads price 49.99)
  W->>DB: UPDATE price = 44.99; COMMIT
  W->>C: DEL product:7
  R->>C: SET product:7 = {price: 49.99} EX 300
  Note over C: stale value cached until TTL expires
```

## MySQL Implementation

The cache table mirrors a key–value store with TTL. A `version` column on `products` (bumped on every update) will be used later to make cache fills safe.

```sql run
CREATE TABLE cache_entries (
  cache_key  VARCHAR(64) PRIMARY KEY,
  payload    JSON        NOT NULL,
  version    INT         NOT NULL,
  expires_at DATETIME(3) NOT NULL
);
CREATE TABLE cache_stats (
  id        INT AUTO_INCREMENT PRIMARY KEY,
  cache_key VARCHAR(64) NOT NULL,
  outcome   ENUM('hit', 'miss') NOT NULL,
  at        DATETIME(3) NOT NULL
);
ALTER TABLE products ADD COLUMN version INT NOT NULL DEFAULT 1;
-- cache-aside read: GET; on miss, load from the database and SET with a TTL
DELIMITER //
CREATE PROCEDURE get_product(IN pid INT, IN now_ts DATETIME(3), IN ttl_s INT)
BEGIN
  DECLARE k VARCHAR(64) DEFAULT CONCAT('product:', pid);
  DECLARE v JSON;
  SET v = (SELECT payload FROM cache_entries WHERE cache_key = k AND expires_at > now_ts);
  IF v IS NOT NULL THEN
    INSERT INTO cache_stats (cache_key, outcome, at) VALUES (k, 'hit', now_ts);
    SELECT 'cache' AS served_from, v AS payload;
  ELSE
    INSERT INTO cache_stats (cache_key, outcome, at) VALUES (k, 'miss', now_ts);
    SET v = (SELECT JSON_OBJECT('id', p.product_id, 'name', p.name, 'price', p.price,
                                'reviews', COUNT(r.review_id), 'avg_rating', ROUND(AVG(r.rating), 2))
             FROM products p LEFT JOIN reviews r ON r.product_id = p.product_id
             WHERE p.product_id = pid GROUP BY p.product_id);
    INSERT INTO cache_entries (cache_key, payload, version, expires_at)
      SELECT k, v, version, now_ts + INTERVAL ttl_s SECOND FROM products WHERE product_id = pid
      ON DUPLICATE KEY UPDATE payload = v, version = products.version, expires_at = now_ts + INTERVAL ttl_s SECOND;
    SELECT 'database' AS served_from, v AS payload;
  END IF;
END //
DELIMITER ;
```

The first request misses and fills the cache; the second, a few seconds later, hits:

```sql run
CALL get_product(25, '2024-12-01 10:00:00.000', 60);
CALL get_product(25, '2024-12-01 10:00:05.000', 60);
SELECT cache_key, payload, version, expires_at FROM cache_entries;
```

After the TTL has passed, the entry is expired and the next request goes to the database again:

```sql run
CALL get_product(25, '2024-12-01 10:01:30.000', 60);
SELECT outcome, COUNT(*) AS n FROM cache_stats GROUP BY outcome ORDER BY outcome;
```

Two misses (the cold start and the expiry) and one hit so far.

## Syntax Breakdown

The sandbox uses SQL against a table; the same operations in Redis (display only):

| Operation | Lab SQL equivalent | Redis | Notes |
|---|---|---|---|
| Read | `SELECT payload … WHERE cache_key = ? AND expires_at > now` | `GET key` | Miss = NULL / nil |
| Write with TTL | `INSERT … ON DUPLICATE KEY UPDATE …, expires_at = now + ttl` | `SET key value EX 300` (or `PX` ms) | Always set a TTL — it bounds staleness and memory |
| Invalidate | `DELETE FROM cache_entries WHERE cache_key = ?` | `DEL key` / `UNLINK key` | `UNLINK` frees memory asynchronously |
| Set if absent (lock) | `INSERT IGNORE` / unique key, or a lock row `FOR UPDATE SKIP LOCKED` | `SET lock:key token NX PX 10000` | Basis of stampede protection |
| Atomic counter | `UPDATE … SET n = n + 1` | `INCR key` | Rate limits, counters |
| Bulk expiry | `DELETE … WHERE expires_at <= now` | automatic (lazy + active expiry) | |
| Eviction | — (table grows) | `maxmemory` + `maxmemory-policy` | `allkeys-lru` is the usual cache policy |

```python
# Cache-aside with redis-py (display only)
import json, redis
r = redis.Redis(host="cache.internal", port=6379, socket_timeout=0.05)   # fail fast: the cache is optional

def get_product(pid: int) -> dict:
    key = f"product:{pid}:v3"                      # include a schema version in the key
    try:
        cached = r.get(key)
        if cached is not None:
            return json.loads(cached)
    except redis.RedisError:
        pass                                       # cache down → fall through to the database
    row = db_fetch_product(pid)                    # parameterised SELECT on a replica or the primary
    try:
        r.set(key, json.dumps(row), ex=300 + random.randint(0, 30))   # TTL with jitter
    except redis.RedisError:
        pass
    return row

def update_price(pid: int, price) -> None:
    with primary.begin() as conn:                  # commit first …
        conn.execute(text("UPDATE products SET price = :p, version = version + 1 WHERE product_id = :id"),
                     {"p": price, "id": pid})
    r.delete(f"product:{pid}:v3")                  # … then invalidate
```

## Practical Examples

### 1. Stale data after an update — and invalidation

Change the price of product 25 without touching the cache. Readers keep getting the old price until the TTL expires:

```sql run
UPDATE products SET price = price - 10.00, version = version + 1 WHERE product_id = 25;
CALL get_product(25, '2024-12-01 10:01:40.000', 60);
SELECT price AS price_in_database, version FROM products WHERE product_id = 25;
```

The cache served the pre-update price while the database already has the new one. Invalidate on write — in cache-aside, **delete** the key after the commit rather than writing the new value into it (deleting is idempotent and avoids racing writers overwriting each other's values):

```sql run
DELETE FROM cache_entries WHERE cache_key = 'product:25';
CALL get_product(25, '2024-12-01 10:01:41.000', 60);
```

The next read missed, reloaded and cached the new price.

### 2. Invalidation with a trigger

Invalidation code scattered across the application is easy to forget in one code path (an admin tool, a batch job). Putting it next to the data guarantees it runs for every update. Inside MySQL a trigger can delete the cache row; with an external cache, the equivalent is a change-data-capture consumer (Debezium reading the binlog) or a transactional outbox that publishes invalidation events ([Data Integrity in Distributed Applications](/lessons/m12-distributed-integrity)).

```sql run
CREATE TRIGGER products_cache_invalidate AFTER UPDATE ON products FOR EACH ROW
  DELETE FROM cache_entries WHERE cache_key = CONCAT('product:', NEW.product_id);
CALL get_product(5, '2024-12-01 11:00:00.000', 60);
UPDATE products SET price = 27.99, version = version + 1 WHERE product_id = 5;
SELECT COUNT(*) AS cached_entries_for_product_5 FROM cache_entries WHERE cache_key = 'product:5';
CALL get_product(5, '2024-12-01 11:00:02.000', 60);
```

Here the trigger's DELETE is part of the same transaction as the UPDATE — an advantage of keeping the "cache" in the database that an external cache cannot have. With Redis, the database commit and the `DEL` are two steps, and the race in the next section becomes possible.

### 3. The stale-set race and version-checked fills

The sequence diagram above shows the race: a reader loads the old value, a writer updates and invalidates, then the reader writes the old value into the cache *after* the invalidation. The fix is to make the fill conditional: the reader remembers the **version** it read, and the fill is skipped if the source has moved on since. In the lab, reproduce the interleaving step by step (one session playing both roles in the bad order):

```sql run
-- Reader: cache miss for product 9; reads the row (version and price) — think of this as the reader's SELECT
SET @read_version = (SELECT version FROM products WHERE product_id = 9);
SET @read_payload = (SELECT JSON_OBJECT('id', product_id, 'price', price) FROM products WHERE product_id = 9);
-- Writer: updates the product (the trigger invalidates the cache; nothing cached yet anyway)
UPDATE products SET price = 35.00, version = version + 1 WHERE product_id = 9;
-- Reader, naive fill: writes what it read, unconditionally
INSERT INTO cache_entries VALUES ('product:9', @read_payload, @read_version, '2024-12-01 12:05:00.000') AS new
  ON DUPLICATE KEY UPDATE payload = new.payload;
SELECT c.payload AS cached, p.price AS in_database
FROM cache_entries c JOIN products p ON p.product_id = 9 WHERE c.cache_key = 'product:9';
```

The cache now holds the old price and will keep it until its TTL. Now the version-checked fill — the same race, but the fill only succeeds if the version it read is still current:

```sql run
DELETE FROM cache_entries WHERE cache_key = 'product:9';
SET @read_version = (SELECT version FROM products WHERE product_id = 9);
SET @read_payload = (SELECT JSON_OBJECT('id', product_id, 'price', price) FROM products WHERE product_id = 9);
UPDATE products SET price = 33.00, version = version + 1 WHERE product_id = 9;
INSERT INTO cache_entries (cache_key, payload, version, expires_at)
  SELECT 'product:9', @read_payload, @read_version, '2024-12-01 12:05:00.000'
  FROM products WHERE product_id = 9 AND version = @read_version;
SELECT ROW_COUNT() AS rows_cached, (SELECT COUNT(*) FROM cache_entries WHERE cache_key = 'product:9') AS entries;
```

The conditional fill inserted nothing, because the product's version had changed after the reader's snapshot; the next request will miss and load the fresh value. In Redis the same idea is implemented with a version inside the cached value plus a compare-and-set script, or with **leases** as described in *Scaling Memcache at Facebook*: on a miss, the cache hands out a lease token, a `DEL` invalidates outstanding leases, and a `SET` with an invalidated lease is rejected.

### 4. Write-through

Write-through updates the cache in the same code path as the database, so reads after a write hit fresh data. Because our cache is a table, the two writes can even be one transaction:

```sql run
START TRANSACTION;
UPDATE products SET price = 239.00, version = version + 1 WHERE product_id = 10;
REPLACE INTO cache_entries (cache_key, payload, version, expires_at)
  SELECT 'product:10', JSON_OBJECT('id', product_id, 'name', name, 'price', price), version, '2024-12-01 13:00:00.000'
  FROM products WHERE product_id = 10;
COMMIT;
SELECT c.payload, c.version AS cached_version, p.version AS db_version
FROM cache_entries c JOIN products p ON p.product_id = 10 WHERE c.cache_key = 'product:10';
```

(The trigger from Example 2 deleted any old entry during the UPDATE; `REPLACE` then wrote the new one — for a cache row, "delete if present, then insert" is exactly the semantics wanted.) With Redis there is no shared transaction: if the database commit succeeds and the cache write fails, the cache is stale; if the order is reversed, the cache can hold a value that was never committed. Write-through therefore still needs a TTL as a safety net, and is mostly used for data that is read immediately after being written (sessions, user settings).

### 5. Materialized summary tables: a cache inside the database

A **summary table** stores a pre-computed aggregate in MySQL itself, maintained either synchronously (triggers or the same transaction as the base write) or asynchronously (a periodic job). Unlike an external cache it is durable, queryable with SQL, and can be kept transactionally consistent ([Denormalization](/lessons/m06-denormalization)):

```sql run
CREATE TABLE product_rating_summary (
  product_id   INT UNSIGNED PRIMARY KEY,
  review_count INT UNSIGNED NOT NULL,
  rating_sum   INT UNSIGNED NOT NULL
);
INSERT INTO product_rating_summary
SELECT product_id, COUNT(*), SUM(rating) FROM reviews GROUP BY product_id;
CREATE TRIGGER reviews_summary_ai AFTER INSERT ON reviews FOR EACH ROW
  INSERT INTO product_rating_summary VALUES (NEW.product_id, 1, NEW.rating)
  ON DUPLICATE KEY UPDATE review_count = review_count + 1, rating_sum = rating_sum + NEW.rating;
INSERT INTO reviews (product_id, customer_id, rating, comment, created_at)
VALUES (25, 2, 1, 'Stopped working after a week', '2024-12-20 10:00:00');
SELECT s.product_id, s.review_count, ROUND(s.rating_sum / s.review_count, 2) AS avg_rating,
       (SELECT ROUND(AVG(rating), 2) FROM reviews WHERE product_id = 25) AS recomputed
FROM product_rating_summary s WHERE s.product_id = 25;
```

The summary stores the additive pieces (count and sum) so that each new review updates it in O(1), and the average is computed on read; the recomputed value from the base table agrees. The cost is a write on a hot summary row for every review — for very hot counters, teams split them into several slot rows and sum them, or update them asynchronously.

## Execution Walkthrough

### A cache stampede, live

A dashboard aggregate is cached with a TTL. It expires at 10:00:00. Three requests arrive at 10:00:05, each in its own connection. With naive cache-aside, every request checks, misses, and recomputes — the expensive query runs three times (in production: hundreds of times, all at once).

```concurrency
title: Three requests miss the same expired key and all recompute
description: report_cache holds an expired entry; recompute_log records every execution of the expensive query.
setup: |
  CREATE TABLE report_cache (cache_key VARCHAR(32) PRIMARY KEY, payload JSON NOT NULL, expires_at DATETIME NOT NULL);
  CREATE TABLE recompute_log (id INT AUTO_INCREMENT PRIMARY KEY, session_label CHAR(1) NOT NULL);
  INSERT INTO report_cache VALUES ('dashboard', JSON_OBJECT('revenue', 1000), '2024-12-01 10:00:00');
steps:
  - session: A
    sql: SELECT COUNT(*) AS fresh_hits FROM report_cache WHERE cache_key = 'dashboard' AND expires_at > '2024-12-01 10:00:05';
    note: Request A — miss (the entry expired 5 seconds ago).
  - session: B
    sql: SELECT COUNT(*) AS fresh_hits FROM report_cache WHERE cache_key = 'dashboard' AND expires_at > '2024-12-01 10:00:05';
    note: Request B — miss as well; A has not refilled the cache yet.
  - session: C
    sql: SELECT COUNT(*) AS fresh_hits FROM report_cache WHERE cache_key = 'dashboard' AND expires_at > '2024-12-01 10:00:05';
    note: Request C — miss.
  - session: A
    sql: INSERT INTO recompute_log (session_label) VALUES ('A'); SELECT SUM(amount) INTO @rev FROM payments WHERE status = 'succeeded'; UPDATE report_cache SET payload = JSON_OBJECT('revenue', @rev), expires_at = '2024-12-01 10:01:05' WHERE cache_key = 'dashboard';
    note: A runs the expensive query and refills the cache.
  - session: B
    sql: INSERT INTO recompute_log (session_label) VALUES ('B'); SELECT SUM(amount) INTO @rev FROM payments WHERE status = 'succeeded'; UPDATE report_cache SET payload = JSON_OBJECT('revenue', @rev), expires_at = '2024-12-01 10:01:05' WHERE cache_key = 'dashboard';
    note: B does the same work again, because it decided on its own (stale) observation.
  - session: C
    sql: INSERT INTO recompute_log (session_label) VALUES ('C'); SELECT SUM(amount) INTO @rev FROM payments WHERE status = 'succeeded'; UPDATE report_cache SET payload = JSON_OBJECT('revenue', @rev), expires_at = '2024-12-01 10:01:05' WHERE cache_key = 'dashboard';
    note: And C. Three identical recomputations for one expiry.
  - session: A
    sql: SELECT session_label FROM recompute_log ORDER BY id;
```

Now the fix. A **lock row** per cache key elects one recompute leader. Every request that misses tries to lock the row with `FOR UPDATE SKIP LOCKED`: the first gets it and recomputes; the others get an empty result immediately — no waiting — and serve the **stale** value they already have (*stale-while-revalidate*). Exactly one recomputation happens.

```concurrency
title: One leader recomputes; the others skip the lock and serve the stale value
setup: |
  CREATE TABLE report_cache (cache_key VARCHAR(32) PRIMARY KEY, payload JSON NOT NULL, expires_at DATETIME NOT NULL);
  CREATE TABLE recompute_log (id INT AUTO_INCREMENT PRIMARY KEY, session_label CHAR(1) NOT NULL);
  CREATE TABLE cache_locks (cache_key VARCHAR(32) PRIMARY KEY);
  INSERT INTO report_cache VALUES ('dashboard', JSON_OBJECT('revenue', 1000), '2024-12-01 10:00:00');
  INSERT INTO cache_locks VALUES ('dashboard');
steps:
  - session: A
    sql: START TRANSACTION; SELECT cache_key FROM cache_locks WHERE cache_key = 'dashboard' FOR UPDATE SKIP LOCKED;
    note: A missed and tries to become the leader. The lock row is free, so A gets it (one row returned).
  - session: B
    sql: START TRANSACTION; SELECT cache_key FROM cache_locks WHERE cache_key = 'dashboard' FOR UPDATE SKIP LOCKED;
    note: B missed too. SKIP LOCKED returns no row instead of waiting — B knows someone else is recomputing.
  - session: B
    sql: SELECT payload AS stale_payload FROM report_cache WHERE cache_key = 'dashboard'; COMMIT;
    note: B serves the expired (stale) value immediately and ends its transaction.
  - observe: locks
  - session: C
    sql: START TRANSACTION; SELECT cache_key FROM cache_locks WHERE cache_key = 'dashboard' FOR UPDATE SKIP LOCKED; SELECT payload AS stale_payload FROM report_cache WHERE cache_key = 'dashboard'; COMMIT;
    note: C behaves like B — no lock, stale value, no recomputation, no waiting.
  - session: A
    sql: INSERT INTO recompute_log (session_label) VALUES ('A'); SELECT SUM(amount) INTO @rev FROM payments WHERE status = 'succeeded'; UPDATE report_cache SET payload = JSON_OBJECT('revenue', @rev), expires_at = '2024-12-01 10:01:05' WHERE cache_key = 'dashboard'; COMMIT;
    note: The leader recomputes once, refreshes the cache and releases the lock by committing.
  - session: B
    sql: SELECT session_label FROM recompute_log ORDER BY id; SELECT payload FROM report_cache WHERE cache_key = 'dashboard';
    note: One recomputation in total, and the cache now holds the fresh value for everyone.
```

The lock snapshot shows A's transaction holding an `X,REC_NOT_GAP` lock on the `cache_locks` record `'dashboard'` while B and C hold nothing — `SKIP LOCKED` never queues. If serving stale data is not acceptable (there is no old value, or it must be fresh), replace `SKIP LOCKED` with a plain `FOR UPDATE`: the followers then *wait* for the leader, and after acquiring the lock they re-check the cache (double-checked locking) and find it fresh. In Redis the leader election is `SET lock:dashboard <token> NX PX 10000`; followers that fail to set the key serve stale data or poll briefly.

## Output

- The cache-aside procedure missed on the cold start, hit five seconds later, and missed again after the 60-second TTL: hit ratio 1/3 at that point.
- After a price update without invalidation, the cache served the old price; after the delete, the next read reloaded the new one. The trigger-based invalidation removed product 5's entry inside the update transaction.
- The naive fill after a concurrent update cached a stale price; the version-checked fill inserted 0 rows.
- The write-through transaction left cache and database at the same version.
- The summary table's trigger-maintained average for product 25 matched the value recomputed from `reviews`.
- In the stampede scenario the expensive query ran three times (A, B, C); with the lock row and `SKIP LOCKED` it ran once (A), and B and C were served the stale value without waiting.

## Alternative Approaches

| Technique | Where | Consistency | Best for |
|---|---|---|---|
| No cache, better queries/indexes | MySQL | Strong | Always try first — a 0.2 ms indexed lookup may not need a cache |
| InnoDB buffer pool sizing | MySQL | Strong | Hot data that fits in RAM; caches pages, not results |
| Read replicas | MySQL | Lagging | Broad read scaling with SQL flexibility |
| Summary / materialized tables | MySQL | Transactional (sync) or bounded (async) | Aggregates, counters, denormalised read models |
| Application in-process cache (LRU dict, Caffeine, Guava) | App memory | Per instance; TTL only | Tiny, hot, rarely changing reference data (country lists, feature flags) |
| Distributed cache (Redis, Memcached) | Separate tier | TTL + invalidation | Hot objects, sessions, rate limits, rendered fragments |
| HTTP / CDN caching | Edge | TTL, `Cache-Control`, purge APIs | Public pages and assets |

**Stampede protection options:**

| Method | Idea | Trade-off |
|---|---|---|
| Lock / single flight (lock row, `SET NX`, in-process request coalescing) | One recompute per key; others wait or serve stale | Needs lock timeouts; a slow leader delays everyone (if they wait) |
| Stale-while-revalidate | Keep serving the old value while one worker refreshes | Must store values past their logical expiry |
| TTL jitter | Randomise TTLs so keys do not expire together | Does not help a single very hot key |
| Probabilistic early expiration (XFetch, Vattani et al.) | Each reader refreshes early with a probability rising as expiry nears | No coordination; occasional extra recomputes |
| Background refresh | A job refreshes hot keys before expiry | Only for known hot keys; extra infrastructure |

## Performance Analysis

**Hit ratio arithmetic.** With t_cache = 0.3 ms and a miss path costing 0.3 + 4 ms (cache check + database work + fill), expected latency is 0.3 + (1 − h) × 4 ms: 0.7 ms at h = 0.9, 0.34 ms at h = 0.99. Database load is (1 − h) × request rate: at 20,000 req/s, h = 0.9 sends 2,000 queries/s to MySQL and h = 0.99 sends 200. These are illustrative numbers, but the shape is general: **the last few percent of hit ratio matter most for the database**.

Compute the lab's hit ratio from the stats table:

```sql run
SELECT SUM(outcome = 'hit') AS hits, SUM(outcome = 'miss') AS misses,
       ROUND(100 * AVG(outcome = 'hit'), 1) AS hit_ratio_pct
FROM cache_stats;
```

In the lab, 2 of 7 lookups hit (28.6%): every update and expiry in the examples forced a miss — a realistic illustration of how write-heavy data and short TTLs erode a cache's value.

**What determines the hit ratio:** the working-set size relative to cache memory (eviction), TTL length relative to how often each key is read, key design (a timestamp inside a key makes every key unique — hit ratio near zero), and invalidation frequency.

**Cold cache and failure.** When the cache restarts or a new region comes up, the hit ratio starts at 0 and the database receives the full read load. Capacity plans must say whether MySQL can survive that (often it cannot), and mitigations include warming the cache from a list of hot keys, rate-limiting misses, and replicating the cache.

**Cost of invalidation.** Each write adds a cache round trip (a `DEL`), and over-broad invalidation (deleting every key of a category when one product changes) drives the hit ratio down. Fine-grained keys and versioned keys (`product:7:v42`, where the version comes from the row) make invalidation precise.

## Best Practices

:::best-practice[Caching that stays correct]
1. **Measure before caching.** Cache only what is hot and measurably expensive; record hit ratio, latency and database load per cache.
2. **Always set a TTL**, with jitter. It bounds staleness when invalidation fails and prevents synchronized expiry.
3. **In cache-aside, commit then delete** — do not write the new value from the writer; let the next reader fill it.
4. **Make fills conditional on a version** (or use leases) to prevent the stale-set race on hot, frequently updated keys.
5. **Protect expensive keys against stampedes**: single-flight with a lock (`SET NX` / lock row with `SKIP LOCKED`), stale-while-revalidate, or probabilistic early refresh.
6. **Treat the cache as optional**: short client timeouts, fall back to the database on cache errors, and make sure the database can survive a cold cache (or that the application degrades gracefully).
7. **Version your cache keys by schema** (`…:v3`) so a deployment that changes the cached document's shape does not read old-format values.
8. **Never cache authorisation decisions or money balances without a strict invalidation story**; prefer the database (or a replica with read-your-writes) for them.
9. **Prefer summary tables over caches** for aggregates that must be durable, queryable or transactionally consistent.
:::

## Common Mistakes

**Mistake 1 — writing to the cache before committing.** If the transaction rolls back, the cache holds a value that never existed. Delete (or write) only after a successful commit.

**Mistake 2 — no TTL.** A missed invalidation then lasts forever, and the cache fills with dead keys. Every entry gets a TTL, even when invalidation exists.

**Mistake 3 — caching negative results without invalidation.** Caching "product 9999 not found" is useful against repeated lookups of missing IDs, but creating product 9999 must invalidate that negative entry.

**Mistake 4 — keys that never repeat.** Building keys from request timestamps, unsorted parameter lists or session IDs for shared data gives a near-zero hit ratio while consuming memory.

**Mistake 5 — expecting the database to cache result sets.** The query cache no longer exists in MySQL 8.0/8.4:

```sql run error
SELECT SQL_CACHE name FROM products WHERE product_id = 1;
```

The `SQL_CACHE` modifier was removed together with the query cache. It is no longer even a keyword: MySQL parses it as a column name and fails with error 1054 (*Unknown column 'SQL_CACHE' in 'field list'*) — a confusing message to meet in legacy code. Result caching is the application's job; MySQL caches *pages* in the buffer pool.

**Mistake 6 — summary tables that store averages.** Store counts and sums (as `product_rating_summary` does), so updates stay O(1) and roll-ups stay exact.

## Industry Application

- **Redis and Memcached** are the standard distributed caches in front of MySQL; managed versions (Amazon ElastiCache, Google Memorystore, Azure Cache for Redis) are common. Facebook's NSDI 2013 paper describes memcache at very large scale, including leases against stale sets and stampedes, and cross-region invalidation driven by the MySQL commit log (mcsqueal).
- **CDC-driven invalidation**: teams publish invalidations from the binlog (Debezium, Maxwell) so every write path — including batch jobs and manual fixes — invalidates the cache, instead of relying on application code.
- **Rails, Django and Spring** provide cache abstractions (`Rails.cache.fetch` with `race_condition_ttl` for stale-while-revalidate, Django's cache framework, Spring's `@Cacheable`), and HTTP-level caching with ETags and CDNs.
- **Counters and leaderboards** move to Redis data structures (`INCR`, sorted sets) with periodic persistence to MySQL, a write-behind pattern accepted where losing a few seconds of counts is tolerable.
- **Summary tables and read models** (CQRS) live in MySQL itself: order totals per customer, inventory snapshots, denormalised search documents — maintained in the same transaction for correctness or asynchronously from an outbox for throughput.

## Practice Exercises

```exercise
id: ex1
title: Hit ratio per key prefix
difficulty: easy
setup: |
  CREATE TABLE cache_stats (id INT AUTO_INCREMENT PRIMARY KEY, cache_key VARCHAR(64) NOT NULL, outcome ENUM('hit','miss') NOT NULL);
  INSERT INTO cache_stats (cache_key, outcome) VALUES
    ('product:1','miss'),('product:1','hit'),('product:1','hit'),('product:2','miss'),('product:2','hit'),
    ('user:7','miss'),('user:7','miss'),('user:8','miss'),('user:7','hit'),
    ('dashboard','miss'),('dashboard','hit'),('dashboard','hit'),('dashboard','hit');
concepts: [caching]
prompt: |
  Return, per key **prefix** (the part before the first `:`, or the whole key if there is no `:`), the
  columns `prefix`, `requests`, `hits` and `hit_ratio_pct` (percentage of hits, rounded to 1 decimal).
hints:
  - "SUBSTRING_INDEX(cache_key, ':', 1) returns the part before the first colon (or the whole string)."
  - "SUM(outcome = 'hit') counts hits; 100 * AVG(outcome = 'hit') is the percentage."
solution: |
  SELECT SUBSTRING_INDEX(cache_key, ':', 1) AS prefix,
         COUNT(*) AS requests,
         SUM(outcome = 'hit') AS hits,
         ROUND(100 * AVG(outcome = 'hit'), 1) AS hit_ratio_pct
  FROM cache_stats
  GROUP BY prefix;
explanation: |
  Breaking the hit ratio down by key family shows where the cache works (dashboard 75%, product 60%) and
  where it does not (user 25%) — typically because the keys are too fine-grained, the TTL is too short, or
  the data is invalidated too often to be worth caching.
```

```exercise
id: ex2
title: Warm the cache
difficulty: medium
kind: dml
setup: |
  CREATE TABLE cache_entries (cache_key VARCHAR(64) PRIMARY KEY, payload JSON NOT NULL, expires_at DATETIME NOT NULL);
  INSERT INTO cache_entries VALUES
    ('product:1', JSON_OBJECT('id', 1, 'price', 1.00), '2024-12-01 09:00:00'),
    ('product:2', JSON_OBJECT('id', 2, 'price', 999.00), '2024-12-01 12:00:00');
concepts: [cache-aside, ttl]
prompt: |
  Before a sale starts at `'2024-12-01 10:00:00'`, warm the cache for every **active** product: each must end
  up with a cache entry `product:<id>` whose payload is `JSON_OBJECT('id', product_id, 'price', price)` from
  `products` and whose `expires_at` is `'2024-12-01 10:10:00'`. Replace existing entries, including ones that
  have not expired yet.
hints:
  - "INSERT INTO cache_entries SELECT CONCAT('product:', product_id), JSON_OBJECT(…), '2024-12-01 10:10:00' FROM products WHERE is_active"
  - "Existing keys must be overwritten: use REPLACE INTO, or INSERT … SELECT … ON DUPLICATE KEY UPDATE."
solution: |
  REPLACE INTO cache_entries (cache_key, payload, expires_at)
  SELECT CONCAT('product:', product_id), JSON_OBJECT('id', product_id, 'price', price), '2024-12-01 10:10:00'
  FROM products WHERE is_active;
check: |
  SELECT cache_key, JSON_EXTRACT(payload, '$.price') AS price, expires_at FROM cache_entries ORDER BY cache_key;
explanation: |
  Warming avoids a cold-cache spike at the start of an event. The stale entry for product 2 (price 999.00,
  still valid until 12:00) must be overwritten — a warm-up that only fills *missing* keys would leave it in
  place. Inactive product 4 is not warmed. In production, warm-ups use TTL jitter so the warmed keys do not
  all expire in the same second.
```

```exercise
id: ex3
title: Invalidate on update with a trigger
difficulty: medium
kind: dml
setup: |
  CREATE TABLE cache_entries (cache_key VARCHAR(64) PRIMARY KEY, payload JSON NOT NULL);
  INSERT INTO cache_entries VALUES ('product:5', '{"price": 29.99}'), ('product:6', '{"price": 89.99}'), ('category:2', '{"n": 4}');
concepts: [cache-invalidation]
prompt: |
  Create an `AFTER UPDATE` trigger named `products_invalidate` on `products` that deletes the cache entry
  `product:<id>` of the updated product. Then raise the price of product **6** by 10.00.
  (Leave the other cache entries untouched.)
hints:
  - "CREATE TRIGGER products_invalidate AFTER UPDATE ON products FOR EACH ROW DELETE FROM cache_entries WHERE cache_key = CONCAT('product:', NEW.product_id);"
solution: |
  CREATE TRIGGER products_invalidate AFTER UPDATE ON products FOR EACH ROW
    DELETE FROM cache_entries WHERE cache_key = CONCAT('product:', NEW.product_id);
  UPDATE products SET price = price + 10.00 WHERE product_id = 6;
check: |
  SELECT 'cache' AS kind, cache_key AS k, NULL AS price FROM cache_entries
  UNION ALL
  SELECT 'product', product_id, price FROM products WHERE product_id = 6
  ORDER BY kind, k;
explanation: |
  Only `product:6` disappears; the next read reloads the new price (99.99). The trigger runs inside the
  UPDATE's transaction, so the invalidation can never be skipped by a code path that forgets it. With an
  external cache the equivalent is a CDC consumer or an outbox event, which is asynchronous — hence TTLs
  remain necessary.
```

```exercise
id: ex4
title: Maintain an additive summary table
difficulty: hard
kind: dml
setup: |
  CREATE TABLE product_sales_summary (product_id INT UNSIGNED PRIMARY KEY, units INT UNSIGNED NOT NULL, revenue DECIMAL(12,2) NOT NULL);
  INSERT INTO product_sales_summary
  SELECT product_id, SUM(quantity), SUM(quantity * unit_price) FROM order_items GROUP BY product_id;
concepts: [summary-tables]
prompt: |
  `product_sales_summary` is a materialized summary of `order_items`. Create an `AFTER INSERT` trigger
  `order_items_summary` on `order_items` that keeps it up to date (including for products that have no summary
  row yet), then insert a new order line: order **1**, product **4**, quantity **2**, unit price **249.00**.
hints:
  - "INSERT INTO product_sales_summary VALUES (NEW.product_id, NEW.quantity, NEW.quantity * NEW.unit_price) ON DUPLICATE KEY UPDATE …"
  - "In ON DUPLICATE KEY UPDATE, add to the existing values: units = units + NEW.quantity."
solution: |
  CREATE TRIGGER order_items_summary AFTER INSERT ON order_items FOR EACH ROW
    INSERT INTO product_sales_summary VALUES (NEW.product_id, NEW.quantity, NEW.quantity * NEW.unit_price)
    ON DUPLICATE KEY UPDATE units = units + NEW.quantity, revenue = revenue + NEW.quantity * NEW.unit_price;
  INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (1, 4, 2, 249.00);
check: |
  SELECT s.product_id, s.units, s.revenue,
         (SELECT SUM(quantity) FROM order_items i WHERE i.product_id = s.product_id) AS units_recomputed
  FROM product_sales_summary s WHERE s.product_id IN (4, 5) ORDER BY s.product_id;
explanation: |
  Product 4 (the inactive Chromebook) had never been sold, so it had no summary row: the upsert inserts it.
  The summary is updated in the same transaction as the order line, so it is always consistent — at the cost
  of serialising concurrent inserts for the same product on that summary row. A complete solution also needs
  UPDATE and DELETE triggers, or a periodic reconciliation that recomputes the summary and compares.
```

```exercise
id: ex5
title: Design the product-page cache
difficulty: challenge
kind: open
concepts: [cache-aside, cache-stampede, cache-invalidation, redis]
prompt: |
  A marketplace product page needs: product details (change a few times a day), price (changes up to every
  few minutes during sales), stock level (changes on every purchase), seller rating (changes daily), and the
  top 10 reviews. Peak traffic is 50,000 page views/s, with 70% of traffic on 1,000 products during flash
  sales. MySQL has a primary and replicas; Redis is available.

  Design the caching: what to cache and with which keys and TTLs, how each item is invalidated, what is
  never cached, how you prevent stampedes on the flash-sale products, and what happens when Redis is down.
solution: |
  **Split by volatility.** Cache the page as separate fragments rather than one blob:
  - `product:{id}:v3` — details and seller rating; TTL 1 h ± jitter; invalidated by a CDC consumer on
    product updates (commit → binlog → `DEL`).
  - `price:{id}` — TTL 30–60 s; invalidated on price change via CDC; version-checked fills (price row
    version) to prevent stale sets during rapid repricing.
  - `reviews:top:{id}` — TTL 10 min, refreshed in the background for hot products.
  - **Stock: not cached for decisions.** Display an approximate "in stock / few left" from a short-TTL cache
    (5 s) or a replica, but the purchase itself is a conditional `UPDATE … WHERE stock >= ?` on the primary.

  **Stampedes.** For the 1,000 flash-sale products: pre-warm all fragments before the sale (with jittered
  TTLs), refresh them in the background before expiry, and use single-flight (`SET lock:{key} NX PX 5000`)
  with stale-while-revalidate for any miss, so a hot key is recomputed by one worker at a time. Also
  coalesce identical in-flight requests inside each app instance.

  **Redis down.** Short client timeouts (≤ 50 ms) and a circuit breaker; fall back to replicas with
  rate-limiting of misses; serve a degraded page (no reviews widget) rather than overload the primary. Run
  Redis with replicas (and failover) so a node loss does not empty the cache; size the database so it can
  survive a partial cold cache.

  **Never cache:** authorisation decisions, checkout totals, payment state.
explanation: |
  A strong design caches by volatility, ties invalidation to the commit log so no write path is missed,
  keeps correctness-critical decisions on the primary, protects hot keys explicitly, and plans the failure
  mode of the cache itself.
```

## Knowledge Assessment

```quiz
- id: q1
  question: "In cache-aside, what should the writer do after updating a product in MySQL?"
  options:
    - "Write the new value into the cache before committing"
    - "Commit, then delete the cache key so the next reader reloads it"
    - "Do nothing; the TTL will handle it"
    - "Flush the entire cache"
  answer: 1
  explanation: "Deleting after the commit is the standard cache-aside write path. Writing before commit can cache a value that is rolled back; relying only on the TTL serves stale data for up to the TTL; flushing everything destroys the hit ratio and invites a stampede."
  concepts: [cache-aside, cache-invalidation]
- id: q2
  question: "A reader misses, reads price 49.99 from MySQL, a writer then updates the price to 44.99 and deletes the key, and finally the reader stores 49.99 in the cache. What is this called and how is it prevented?"
  options:
    - "A cache stampede; prevented with TTL jitter"
    - "A stale-set race; prevented by version-checked fills or leases"
    - "A dirty read; prevented by READ COMMITTED"
    - "Eviction; prevented by more memory"
  answer: 1
  explanation: "The fill happened after the invalidation using data read before the update. Making the fill conditional on the version read (or using memcache-style leases) rejects it. It is not a stampede (that is many recomputes), not a dirty read (the reader read committed data) and not eviction."
  concepts: [cache-invalidation]
- id: q3
  question: "In the lab's second concurrency scenario, why did sessions B and C not wait for A's lock?"
  options:
    - "Because the lock row was not indexed"
    - "Because `FOR UPDATE SKIP LOCKED` skips rows locked by other transactions and returns immediately"
    - "Because B and C ran at READ UNCOMMITTED"
    - "Because A had already committed"
  answer: 1
  explanation: "SKIP LOCKED never waits; it omits locked rows from the result. B and C therefore learned that a leader existed and served the stale value. A had not committed yet; isolation level does not affect locking reads this way."
  concepts: [cache-stampede, skip-locked]
- id: q4
  question: "Which techniques reduce cache stampedes? (Select all that apply.)"
  type: multi
  options:
    - "Single-flight recomputation with a lock (SET NX or a lock row)"
    - "Serving stale values while one worker refreshes"
    - "Giving every key exactly the same TTL"
    - "Probabilistic early expiration"
  answer: [0, 1, 3]
  explanation: "Locks, stale-while-revalidate and probabilistic early refresh all limit concurrent recomputation. Identical TTLs make keys expire together and cause synchronized stampedes; jitter is the fix."
  concepts: [cache-stampede]
- id: q5
  question: "Why was MySQL's query cache removed in 8.0?"
  options:
    - "Because Redis replaced it inside MySQL"
    - "Because any write to a table invalidated all its cached results and a global mutex limited scalability on multi-core servers"
    - "Because it returned incorrect results"
    - "Because InnoDB does not support caching"
  answer: 1
  explanation: "Coarse invalidation and lock contention made it a bottleneck. It did not return wrong results, Redis is not part of MySQL, and InnoDB caches pages in the buffer pool."
  concepts: [caching]
- id: q6
  question: "A cache has a 95% hit ratio at 10,000 requests/s. Engineers raise it to 99%. How does the read load on MySQL change?"
  options:
    - "It drops by about 4%"
    - "It drops from 500 to 100 queries/s — a factor of 5"
    - "It is unchanged"
    - "It drops from 9,500 to 9,900"
  answer: 1
  explanation: "Database load is proportional to the miss ratio: 5% → 1% of 10,000 is 500 → 100 queries/s. The 4% difference in hit ratio is a 5× reduction in misses."
  concepts: [caching]
- id: q7
  question: "Why does `product_rating_summary` store `review_count` and `rating_sum` rather than `avg_rating`?"
  options:
    - "To save space"
    - "Because counts and sums can be updated incrementally in O(1) and rolled up exactly; an average cannot be updated correctly without the count"
    - "Because MySQL cannot store decimals in summary tables"
    - "Because triggers cannot compute averages"
  answer: 1
  explanation: "Additive components allow incremental maintenance and exact aggregation; the average is derived on read. Space, decimal support and trigger capabilities are not the reason."
  concepts: [summary-tables]
- id: q8
  question: "Redis becomes unreachable. What should a well-designed cache-aside client do?"
  options:
    - "Retry the cache until it responds, blocking the request"
    - "Fail the request with an error"
    - "Time out quickly and fall back to the database, with protections so the database is not overwhelmed"
    - "Serve empty responses"
  answer: 2
  explanation: "The cache is an optimisation; correctness must not depend on it. Short timeouts, a circuit breaker and rate-limited fallback keep requests working without overloading MySQL. Blocking or failing requests turns a cache incident into an outage."
  concepts: [caching, redis]
```
