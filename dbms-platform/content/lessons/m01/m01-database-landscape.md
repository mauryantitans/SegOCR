---
id: m01-database-landscape
dataset: shop
summary: The major data models — relational, key-value, document, wide-column, graph and time-series — what each optimises for, how to choose between them, how MySQL's JSON and recursive-query features relate to them (and where MySQL is not a substitute), and who the people are that design, build, operate and use databases.
concepts: [relational-model, key-value, document-model, wide-column, graph-model, time-series, polyglot-persistence, database-users, dba-roles]
related: [m01-data-database-dbms]
references:
  - title: "MySQL 8.4 Reference Manual — The JSON Data Type"
    url: https://dev.mysql.com/doc/refman/8.4/en/json.html
  - title: "MySQL 8.4 Reference Manual — JSON Table Functions (JSON_TABLE)"
    url: https://dev.mysql.com/doc/refman/8.4/en/json-table-functions.html
  - title: "MySQL 8.4 Reference Manual — Multi-Valued Indexes"
    url: https://dev.mysql.com/doc/refman/8.4/en/create-index.html#create-index-multi-valued
  - title: "MySQL 8.4 Reference Manual — WITH (Common Table Expressions), Recursive CTEs"
    url: https://dev.mysql.com/doc/refman/8.4/en/with.html
  - title: "MySQL 8.4 Reference Manual — Using MySQL as a Document Store"
    url: https://dev.mysql.com/doc/refman/8.4/en/document-store.html
  - title: "MySQL 8.4 Reference Manual — Using Roles"
    url: https://dev.mysql.com/doc/refman/8.4/en/roles.html
---

## Introduction

"Should we use MySQL or MongoDB?" "Do we need a graph database for recommendations?" "Isn't Cassandra what you use at scale?" These questions come up in every engineering team, often asked as if databases were interchangeable brands. They are not. Each family of database systems is built around a **data model** — a way of structuring data and the operations allowed on it — and each model makes some workloads easy and others painful.

This lesson gives you a working map of the landscape:

1. **The six families you will meet in industry** — relational, key-value, document, wide-column, graph and time-series — what each stores, how it is queried, and what it gives up.
2. **How to choose a model** for a workload, and why most real systems use several (polyglot persistence).
3. **How MySQL relates to the non-relational models.** MySQL 8.4 has a native `JSON` type, JSON indexing and recursive queries, so it can *serve* some document, key-value and graph-shaped workloads — but it is still a relational DBMS, and you should know exactly where the analogy ends.
4. **The people around a database** — end users, application developers, analysts, data engineers, and database administrators — and what each of them needs from it.

:::note[Dataset]
The examples use the **shop** dataset (customers, a 10-node category hierarchy, products, orders, line items, payments, reviews), plus small tables created in your lab to illustrate key-value, document, and time-series shapes.
:::

## DBMS Theory

:::theory[Data model]
A **data model** is a collection of concepts for describing (1) the **structure** of data, (2) the **operations** for retrieving and updating it, and (3) the **constraints** it must satisfy. The relational model, for example, has relations (structure), relational algebra/SQL (operations), and keys and integrity constraints (constraints). A DBMS implements one primary data model, sometimes with extensions toward others ("multi-model").
:::

Non-relational systems are often grouped under the label **NoSQL** ("not only SQL"), a term popularised around 2009 for systems that relaxed relational features — fixed schemas, joins, multi-row ACID transactions — to gain horizontal scalability, flexible structure or a specific access pattern. The label is now fuzzy: many NoSQL products added SQL-like languages and transactions, and relational systems added JSON and horizontal scaling. Think in terms of **models and trade-offs**, not camps.

| Model | Unit of data | Typical query | Strengths | Gives up (typically) | Examples |
|---|---|---|---|---|---|
| **Relational** | Rows in tables (relations) with a declared schema | Declarative SQL: filters, joins, aggregates over any attributes | Ad hoc queries, integrity constraints, multi-row ACID transactions, mature tooling | Horizontal write scaling is harder; schema changes need migration | MySQL, PostgreSQL, Oracle, SQL Server, SQLite |
| **Key-value** | An opaque value addressed by a unique key | `GET key`, `PUT key value`, `DELETE key` | Very low latency, simple to partition and scale | Queries on anything except the key; relationships; usually rich transactions | Redis, Memcached, Amazon DynamoDB (key-value mode), etcd |
| **Document** | Self-contained, nested documents (JSON/BSON) in collections | Find documents by field values, nested paths, arrays; aggregation pipelines | Flexible, evolving structure; an aggregate read in one fetch | Joins and cross-document constraints are weak or absent; duplication across documents | MongoDB, Couchbase, Amazon DocumentDB, Firestore |
| **Wide-column** | Rows identified by a partition key, with many sparse columns sorted by a clustering key | Read a partition (or a range within it) by key | Massive write throughput, linear horizontal scaling, multi-datacentre replication | Ad hoc queries (you design one table per query), joins, usually multi-partition transactions | Apache Cassandra, ScyllaDB, Apache HBase, Google Bigtable |
| **Graph** | Nodes and edges, both with properties | Pattern matching and traversal: "friends of friends who bought X" | Variable-length path queries, relationship-heavy domains | Bulk aggregation and tabular analytics are less natural; scaling traversals across machines is hard | Neo4j, Amazon Neptune, TigerGraph; SQL/PGQ in SQL:2023 |
| **Time-series** | Timestamped measurements per series (tags + fields) | Range scans by time, downsampling, windowed aggregates | High ingest rate, compression, retention policies, time functions | General-purpose updates, joins, transactions | InfluxDB, TimescaleDB (on PostgreSQL), Prometheus, QuestDB |

### Consistency and scaling — the vocabulary you will hear

Relational DBMSs traditionally provide **ACID** transactions (atomicity, consistency, isolation, durability). Many distributed NoSQL systems were designed around **BASE** ("basically available, soft state, eventually consistent"): replicas may briefly disagree, and the system favours availability during network partitions. The formal trade-off is the **CAP theorem**, covered in [CAP theorem and distributed consistency](/lessons/m01-cap-theorem). Two facts matter now:

- The model and the consistency guarantee are **separate choices**. DynamoDB and MongoDB both offer transactions today; MySQL with asynchronous replication can serve stale reads from a replica.
- Scaling out (adding machines) is easiest when every request touches **one partition**. Key-value, wide-column and document systems are designed so that the natural unit of access (a key, a partition, a document) is the unit of distribution. Relational systems can be sharded too, but joins and transactions across shards become the application's problem (see [Sharding](/lessons/m12-sharding)).

## Why It Exists

For roughly three decades (1980s–2000s) the relational model was the default for almost everything. Different models grew out of specific pressures:

- **Web scale and availability (2004–2010).** Amazon's *Dynamo* paper (2007) and Google's *Bigtable* paper (2006) described systems that partition data over thousands of commodity machines and remain available when some fail. They inspired DynamoDB, Cassandra and HBase. The workloads — shopping carts, user sessions, web crawl data, feeds — needed **predictable single-key access at huge scale** more than ad hoc joins.
- **Impedance mismatch.** Object-oriented application code works with nested objects (an order with its line items and shipping address). Mapping them to normalised tables and back requires joins and ORM layers. Document databases store the aggregate as one document, matching the application's shape.
- **Relationship-centric questions.** "Shortest path", "all accounts within four transfers of this fraudulent one", "people you may know" are recursive traversals. SQL can express them (recursive CTEs since SQL:1999), but graph databases store adjacency directly and optimise for traversals.
- **Machine and sensor data.** Metrics, IoT readings and financial ticks arrive at millions of points per second, are almost never updated, and are queried by time range with downsampling. Time-series databases specialise in compression, time-partitioned storage and retention.
- **Caching.** Even when the system of record is relational, an in-memory key-value store (Redis, Memcached) absorbs repeated reads with sub-millisecond latency.

At the same time, relational systems absorbed the most useful ideas: JSON types (MySQL 5.7, 2015; PostgreSQL's `jsonb`, 2014), recursive queries, horizontal sharding layers (Vitess for MySQL), and distributed SQL databases (Google Spanner, CockroachDB, TiDB) that offer SQL and ACID across machines. The result is a landscape where the right answer depends on the workload, not on fashion.

## Internal Working

:::internals[Storage structures behind each model]
- **Relational (InnoDB).** Each table is a **B+ tree clustered on the primary key**; secondary indexes are further B+ trees whose leaves store primary-key values. Updates happen in place on 16 KB pages, protected by redo/undo logs. Good for mixed reads and writes and for range scans on any indexed attribute.
- **Key-value.** In memory, a **hash table** (Redis, Memcached) gives O(1) lookups but no ordering. Persistent stores often use an **LSM tree** (log-structured merge tree; e.g. RocksDB): writes go to an in-memory table and an append-only log, are flushed as sorted immutable files, and are merged in the background (compaction). LSM trees turn random writes into sequential I/O.
- **Wide-column (Cassandra, HBase, Bigtable).** Also LSM-based. The **partition key** is hashed (Cassandra) or range-partitioned (HBase, Bigtable) to choose the node; within a partition, rows are kept sorted by the **clustering key**, so "the last 100 events for device X" is one sequential read.
- **Document (MongoDB).** Documents are stored in a binary JSON format (BSON) inside a storage engine (WiredTiger, using B-trees), with secondary B-tree indexes on document paths, including array elements ("multikey" indexes).
- **Graph (native graph stores like Neo4j).** Nodes store direct references to their relationships (**index-free adjacency**), so following an edge costs a pointer dereference rather than an index lookup — the cost of a traversal depends on the size of the neighbourhood visited, not on the total graph size.
- **Time-series.** Data is partitioned into **time chunks**; within a chunk it is often stored **column-wise** and compressed (delta-of-delta encoding for timestamps, XOR or similar for floats). Retention drops whole old chunks, which is far cheaper than deleting rows.
:::

**What MySQL does with JSON.** MySQL stores a `JSON` column in an optimised **binary format** (not as text): the document is parsed and validated on insert, and the binary layout includes a lookup table so the server can find a key or array element without scanning the whole document. Since MySQL 8.0, `JSON_SET`, `JSON_REPLACE` and `JSON_REMOVE` can perform **partial in-place updates** under certain conditions, rather than rewriting the whole value. JSON values cannot be indexed directly; you index them through **generated columns**, **functional indexes**, or **multi-valued indexes** (for arrays, since 8.0.17). Underneath, a JSON column is still a column of a row in an InnoDB B+ tree — the transactional and locking behaviour is InnoDB's.

## MySQL Implementation

**Relational — the baseline.** The shop schema is relational: facts are split into tables and recombined by joins at query time. Any attribute can be queried without having planned for it:

```sql run
SELECT c.name AS category, COUNT(*) AS products, MIN(p.price) AS cheapest, MAX(p.price) AS priciest
FROM products AS p
JOIN categories AS c ON c.category_id = p.category_id
GROUP BY c.name
ORDER BY products DESC, category
LIMIT 5;
```

**Key-value shape in MySQL.** A key-value workload is "store a value under a key; get it back by key". In MySQL that is a two-column table whose primary key is the key. A shopping-cart session store:

```sql run
CREATE TABLE session_store (
  session_key VARCHAR(64) NOT NULL PRIMARY KEY,
  payload     JSON        NOT NULL,
  expires_at  DATETIME    NOT NULL
);
INSERT INTO session_store VALUES
  ('sess-41', '{"customer_id": 7,  "cart": [{"product_id": 5, "qty": 2}]}', '2024-12-31 23:00:00'),
  ('sess-42', '{"customer_id": 12, "cart": [{"product_id": 1, "qty": 1}, {"product_id": 6, "qty": 1}]}', '2024-12-31 23:00:00');
ANALYZE TABLE session_store;
EXPLAIN SELECT payload FROM session_store WHERE session_key = 'sess-42';
```

The plan shows access type `const`: a single primary-key lookup in the clustered B+ tree. This is a perfectly good key-value store for moderate load, with the bonus of transactions. What it is *not* is Redis: every read goes through SQL parsing and InnoDB, values are not kept purely in memory with expiry handled automatically, and there are no native data structures such as sorted sets. Teams use MySQL for durable key-value data and Redis or Memcached as a cache in front.

**Document shape in MySQL.** Products in a real catalog have category-specific attributes — laptops have RAM and CPU, coffee has roast and origin. A document model stores each product's attributes as a nested document. MySQL can do that with a `JSON` column:

```sql run
CREATE TABLE product_docs (
  product_id INT UNSIGNED NOT NULL PRIMARY KEY,
  doc        JSON NOT NULL
);
INSERT INTO product_docs VALUES
  (1,  '{"name": "Ultrabook 14 Pro", "brand": "Nimbus", "specs": {"ram_gb": 16, "cpu": "8-core"}, "colors": ["silver", "black"]}'),
  (3,  '{"name": "Gaming Laptop 16", "brand": "Vortex", "specs": {"ram_gb": 32, "gpu": "RTX"},    "colors": ["black"]}'),
  (5,  '{"name": "Wireless Mouse",   "brand": "Nimbus", "specs": {"dpi": 1600},                    "colors": ["black", "white", "red"]}'),
  (17, '{"name": "Single-Origin Beans 1kg", "brand": "Roastery", "specs": {"roast": "dark", "origin": "Brazil"}}');

SELECT product_id,
       doc->>'$.name'         AS name,
       doc->>'$.brand'        AS brand,
       doc->'$.specs.ram_gb'  AS ram_gb
FROM product_docs
WHERE JSON_CONTAINS(doc->'$.colors', '"black"')
ORDER BY product_id;
```

`->>` extracts a value and unquotes it; `->` returns a JSON value. The mouse has no `specs.ram_gb` path, so its `ram_gb` is SQL `NULL`. `JSON_CONTAINS` tests array membership. Different documents carry different attributes — exactly the flexibility document databases offer.

`JSON_TABLE` goes the other way: it turns a nested array into relational rows, so the document can be joined with ordinary tables:

```sql run
SELECT d.product_id, jt.color
FROM product_docs AS d,
     JSON_TABLE(d.doc, '$.colors[*]' COLUMNS (color VARCHAR(20) PATH '$')) AS jt
ORDER BY d.product_id, jt.color;
```

Product 17 (coffee beans) has no `colors` array, so it produces no rows — just as an inner join would drop it.

:::mysql[MySQL Document Store]
MySQL also offers a **Document Store** interface: with the X Plugin (enabled by default) and MySQL Shell or an X DevAPI connector, applications work with *collections* of JSON documents using CRUD calls such as `db.products.find("brand = 'Nimbus'")`, without writing SQL. Under the hood a collection is an InnoDB table with a JSON `doc` column and a generated `_id` primary key. It cannot run in this sandbox, which speaks only classic SQL.
:::

## Syntax Breakdown

The MySQL features that relate to non-relational models:

| Feature | Syntax | Relation to other models |
|---|---|---|
| `JSON` data type | `doc JSON NOT NULL` | Store a document per row (document model) |
| Path extraction | `doc->'$.a.b'` (JSON result), `doc->>'$.a.b'` (unquoted text) — shorthand for `JSON_EXTRACT` / `JSON_UNQUOTE(JSON_EXTRACT(...))` | Navigate nested fields like a document query |
| Containment / search | `JSON_CONTAINS(target, candidate[, path])`, `JSON_OVERLAPS(a, b)`, `value MEMBER OF (json_array)` | Array and sub-document predicates |
| Modification | `JSON_SET`, `JSON_INSERT`, `JSON_REPLACE`, `JSON_REMOVE`, `JSON_ARRAY_APPEND` | Partial document updates |
| Shredding | `JSON_TABLE(doc, path COLUMNS (...))` | Turn documents into relational rows for joins |
| Indexing JSON | Generated column + index; functional index `((CAST(doc->>'$.brand' AS CHAR(30))))`; multi-valued index `((CAST(doc->'$.colors' AS CHAR(20) ARRAY)))` | Secondary indexes on document paths |
| Primary-key lookup | `WHERE pk = ?` → access type `const` | Key-value access |
| Recursive CTE | `WITH RECURSIVE r AS (anchor UNION ALL recursive_step) SELECT …` | Graph traversal over an edge table |
| Composite primary key | `PRIMARY KEY (series_id, ts)` | Clustering by series then time, like a wide-column partition + clustering key or a time-series layout |

:::standard[Standard SQL vs MySQL]
SQL:2016 added JSON functions (`JSON_VALUE`, `JSON_QUERY`, `JSON_TABLE`, `JSON_EXISTS`); MySQL implements `JSON_TABLE` and (since 8.0.21) `JSON_VALUE`, but the `->` / `->>` operators, `JSON_CONTAINS`, `MEMBER OF` and multi-valued indexes are **MySQL-specific**. SQL:2023 added **property-graph queries (SQL/PGQ)** with a `GRAPH_TABLE` operator; MySQL 8.4 does **not** implement SQL/PGQ, so graph traversals are written with recursive CTEs.
:::

## Practical Examples

**1. Graph shape: a hierarchy as nodes and edges.** The shop's `categories` table is a tree: each row is a **node** and `parent_id` is an **edge** to its parent. "Every category under Electronics, at any depth" is a traversal. In SQL it is a recursive CTE (taught in [Recursive CTEs](/lessons/m04-recursive-ctes)):

```sql run
WITH RECURSIVE subtree AS (
  SELECT category_id, name, parent_id, 0 AS depth, CAST(name AS CHAR(200)) AS path
  FROM categories
  WHERE name = 'Electronics'
  UNION ALL
  SELECT c.category_id, c.name, c.parent_id, s.depth + 1, CONCAT(s.path, ' > ', c.name)
  FROM categories AS c
  JOIN subtree AS s ON c.parent_id = s.category_id
)
SELECT category_id, depth, path
FROM subtree
ORDER BY path;
```

Each iteration of the recursive part follows one more level of edges. In a graph database the same question is a one-line pattern; in Neo4j's Cypher (display only):

```sql
-- Cypher (Neo4j), not SQL: all categories reachable from Electronics through CHILD_OF edges
MATCH (root:Category {name: 'Electronics'})<-[:CHILD_OF*0..]-(c:Category)
RETURN c.name;
```

For a 10-node tree the difference is cosmetic. For a social graph with hundreds of millions of edges and queries that traverse 3–6 hops with filters on edge properties, a native graph store's adjacency structures and query language matter.

**2. Time-series shape.** Sensor readings keyed by (sensor, time). Clustering the primary key as `(sensor_id, ts)` stores each sensor's readings contiguously and in time order in InnoDB's B+ tree — the same idea as a wide-column partition key + clustering key:

```sql run
CREATE TABLE readings (
  sensor_id SMALLINT UNSIGNED NOT NULL,
  ts        DATETIME          NOT NULL,
  celsius   DECIMAL(5,2)      NOT NULL,
  PRIMARY KEY (sensor_id, ts)
);
INSERT INTO readings VALUES
  (1, '2024-06-01 10:00:00', 21.10), (1, '2024-06-01 10:20:00', 21.40), (1, '2024-06-01 10:40:00', 21.90),
  (1, '2024-06-01 11:00:00', 22.30), (1, '2024-06-01 11:30:00', 22.80),
  (2, '2024-06-01 10:10:00', 18.00), (2, '2024-06-01 10:50:00', 18.40), (2, '2024-06-01 11:10:00', 18.90);

SELECT sensor_id,
       DATE_FORMAT(ts, '%Y-%m-%d %H:00') AS hour_bucket,
       COUNT(*)                          AS samples,
       ROUND(AVG(celsius), 2)            AS avg_celsius
FROM readings
GROUP BY sensor_id, hour_bucket
ORDER BY sensor_id, hour_bucket;
```

This is **downsampling** — reducing raw points to hourly aggregates — the bread-and-butter operation of time-series databases. MySQL can do it, and for modest volumes (millions of rows, with range [partitioning](/lessons/m12-partitioning) by time to drop old data cheaply) it is a reasonable choice. At billions of points with high ingest rates, a specialised time-series store's compression and retention handling usually wins.

**3. Indexing a document path.** To make document lookups fast, MySQL indexes a JSON path through a functional index. A multi-valued index covers every element of an array:

```sql run
ALTER TABLE product_docs
  ADD INDEX idx_brand ((CAST(doc->>'$.brand' AS CHAR(30)) COLLATE utf8mb4_bin)),
  ADD INDEX idx_colors ((CAST(doc->'$.colors' AS CHAR(20) ARRAY)));
SELECT INDEX_NAME, EXPRESSION
FROM information_schema.STATISTICS
WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'product_docs' AND INDEX_NAME <> 'PRIMARY'
ORDER BY INDEX_NAME;
```

The catalog shows each index's defining expression. A query such as `WHERE 'red' MEMBER OF (doc->'$.colors')` can now use `idx_colors` on larger tables. Indexing is covered in Module 7; the point here is that MySQL makes document fields **indexable**, which is what separates a usable document store from a blob column.

**4. Mixed model — the pragmatic middle.** Many production schemas keep stable, queried, constrained attributes as relational columns, and genuinely variable attributes in a JSON column:

```sql run
SELECT p.product_id, p.name, p.price, d.doc->'$.specs' AS specs
FROM products AS p
JOIN product_docs AS d ON d.product_id = p.product_id
ORDER BY p.product_id;
```

`price` stays a `DECIMAL` column with a `CHECK` constraint; `specs` can differ per category without schema migrations.

## Execution Walkthrough

How does MySQL evaluate a document-style query? Trace the colour search from the implementation section step by step.

```walkthrough
title: SELECT … FROM product_docs WHERE JSON_CONTAINS(doc->'$.colors', '"black"')
steps:
  - title: "1. The stored documents"
    explain: >-
      Each row holds one document in MySQL's binary JSON format. The server validated each document on
      INSERT; invalid JSON would have been rejected with an error.
    sql: SELECT product_id, JSON_TYPE(doc) AS type, JSON_LENGTH(doc) AS top_level_keys, JSON_KEYS(doc) AS keys_present FROM product_docs ORDER BY product_id;
  - title: "2. Evaluate the path for every row"
    explain: >-
      doc->'$.colors' navigates the binary document to the colors array. For product 17 the path does not
      exist, so the result is SQL NULL — and JSON_CONTAINS(NULL, …) is NULL, which WHERE treats as false.
    sql: SELECT product_id, doc->'$.colors' AS colors, JSON_CONTAINS(doc->'$.colors', '"black"') AS has_black FROM product_docs ORDER BY product_id;
  - title: "3. Filter and project"
    explain: >-
      Only rows where has_black = 1 survive. The SELECT list then extracts name and brand with ->> so they
      come back as plain strings rather than JSON-quoted values.
    sql: SELECT product_id, doc->>'$.name' AS name, doc->>'$.brand' AS brand FROM product_docs WHERE JSON_CONTAINS(doc->'$.colors', '"black"') ORDER BY product_id;
  - title: "4. The access path"
    explain: >-
      Because the multi-valued index idx_colors was created in Practical Examples, the optimizer can serve
      JSON_CONTAINS on the colors array from that index: the plan shows an index range scan using idx_colors,
      with a residual filter on top. The same applies to MEMBER OF and JSON_OVERLAPS. Module 7 explains how
      to read plans.
    sql: EXPLAIN FORMAT=TREE SELECT product_id FROM product_docs WHERE JSON_CONTAINS(doc->'$.colors', '"black"');
```

Compare with a native document database: MongoDB would evaluate `{colors: "black"}` against BSON documents, using a multikey index if one exists. The logical steps — navigate path, test membership, project fields — are the same. The difference lies in what surrounds them: MySQL lets you join the result to relational tables inside one ACID transaction; MongoDB lets you shard the collection by a document field and scale writes across many nodes with its own replication.

## Output

The document query returns the three products whose `colors` array contains `"black"` — products 1, 3 and 5 — with name and brand as unquoted strings. Product 17 is absent because it has no `colors` path at all, not because it has a non-black colour: in JSON-land, a **missing path** and an **empty array** are different things, and both differ from a relational `NULL`. That distinction is one of the practical costs of schema flexibility: every query must cope with documents that lack fields, and nothing in the database guarantees which fields exist unless you add constraints.

MySQL can add such guarantees. A `CHECK` constraint can require a path, and `JSON_SCHEMA_VALID` (8.0.17+) can validate documents against a JSON Schema:

```sql run error
ALTER TABLE product_docs
  ADD CONSTRAINT chk_doc_has_name CHECK (JSON_SCHEMA_VALID(
    '{"type": "object", "required": ["name", "brand", "colors"]}', doc));
```

The `ALTER` fails with error 3819 because the existing product 17 has no `colors` key: MySQL checks existing rows when a `CHECK` constraint is added. This is the relational instinct — declare the rule, and the DBMS refuses data that violates it — applied to documents.

## Alternative Approaches

Choosing a model is choosing which access patterns are cheap. A practical decision guide:

| If the workload is mainly… | Prefer | Why | Watch out for |
|---|---|---|---|
| Transactions over related entities, ad hoc queries, integrity rules | Relational | Joins, constraints, ACID | Scale-out of writes needs sharding |
| Lookups by a single key at very high rate, cache, sessions | Key-value | O(1) access, easy partitioning | No queries by value; data modelling around keys |
| Aggregates read and written as a unit, varying structure | Document | One read per aggregate, schema flexibility | Duplication across documents, weak cross-document constraints |
| Huge write volumes, known query patterns, multi-region | Wide-column | Linear scaling, tunable consistency | One table per query pattern, no joins |
| Many-hop relationship queries | Graph | Traversal performance, expressive patterns | Aggregations over everything, sharding |
| Append-only measurements queried by time | Time-series | Compression, retention, time functions | Updates, relational joins |
| Analytics over large history | Columnar warehouse (not covered here) | Column storage, vectorised scans | Not for OLTP |

**Polyglot persistence** — using several stores, each for what it does best — is normal in industry: MySQL as the system of record for orders, Redis as a cache and session store, Elasticsearch/OpenSearch for full-text search, a warehouse for analytics. Its cost is **operational complexity** and **consistency between stores**: data must be propagated (usually via change-data-capture from the system of record), and each extra technology needs expertise, monitoring and backups. A common senior-engineer heuristic is to **start relational** and add a specialised store only when a measured requirement demands it — MySQL's JSON support exists precisely so that moderate document needs do not force a second database.

## Performance Analysis

Performance differences between models are about **access patterns**, not about one model being "faster":

| Operation | Relational (InnoDB) | Key-value (in-memory) | Wide-column (LSM) | Graph (native) |
|---|---|---|---|---|
| Get by primary key | B+ tree descent, O(log n), a few page reads (cached in buffer pool) | Hash lookup, O(1), memory only | Partition lookup + possible multi-file read (bloom filters help) | Index lookup for the start node |
| Range scan on key | Excellent (leaf pages linked in key order) | Not supported by hash stores | Excellent within a partition | n/a |
| Query on non-key attribute | Secondary index or scan | Not supported (full scan, if at all) | Needs a separate table or index, often discouraged | Property index |
| Many-hop traversal | One join (index lookup per row) per hop | n/a | n/a | Pointer chasing per edge; cost ∝ neighbourhood |
| Write throughput | Bounded by in-place page updates, locks, log flushes | Very high | Very high (sequential appends) | Moderate |

:::performance[Theory vs measurement]
These are structural tendencies, not benchmark results. Real performance depends on data size versus memory, hardware, configuration, and how well the schema matches the queries. Vendor benchmarks comparing models are notoriously workload-specific. When choosing, build a prototype with your real access patterns and data volumes, and measure.
:::

For MySQL specifically: JSON path evaluation costs CPU per row; without an index on the path, a predicate on a JSON field means a full scan. With a functional or multi-valued index, document-field lookups become ordinary B+ tree lookups. Large JSON documents also increase row size and may be stored off-page, which makes reading them more expensive than reading narrow relational columns.

## Best Practices

:::best-practice[Choosing and combining models]
1. **Start from access patterns.** List the top queries and writes with their frequency and latency targets before naming a technology.
2. **Default to relational for the system of record** when data is interrelated and must be consistent; add other stores for measured needs.
3. **Use JSON columns for genuinely variable attributes**, not to avoid schema design. Keep identifiers, foreign keys, money and anything you filter or join on frequently as typed relational columns.
4. **Index JSON paths you query** (functional or multi-valued indexes), and validate documents with `CHECK` / `JSON_SCHEMA_VALID` where structure matters.
5. **Know the guarantees** of each store you add (transactions? consistency level? durability on acknowledge?) and document them.
6. **Have one system of record per entity** and propagate changes to caches and search indexes, never the reverse.
7. **Grant each user type exactly the privileges its role needs** — analysts read-only, applications DML on their schema, only DBAs DDL in production.
:::

## Common Mistakes

**Mistake 1 — storing relational data as JSON to "save time".** Putting a customer ID inside a document instead of a column loses the foreign key, and the database will happily accept a reference to a customer that does not exist:

```sql run
INSERT INTO session_store VALUES ('sess-99', '{"customer_id": 9999, "cart": []}', '2024-12-31 23:00:00');
SELECT session_key, payload->>'$.customer_id' AS customer_id,
       EXISTS (SELECT 1 FROM customers c WHERE c.customer_id = payload->>'$.customer_id') AS customer_exists
FROM session_store
ORDER BY session_key;
```

Session `sess-99` references customer 9999, who does not exist, and nothing stopped it. If the reference matters, store it as a column with a `FOREIGN KEY`.

**Mistake 2 — expecting `->` to return plain text.** `->` returns a JSON value, so strings keep their quotes; comparisons and display then surprise you:

```sql run
SELECT doc->'$.brand' AS with_arrow, doc->>'$.brand' AS with_double_arrow
FROM product_docs WHERE product_id = 1;
```

**Mistake 3 — invalid JSON.** MySQL validates JSON on write, so malformed documents fail loudly (good) — but code that builds JSON by string concatenation will hit this in production:

```sql run error
INSERT INTO product_docs VALUES (30, '{"name": "Yoga Mat", brand: "Flex"}');
```

Build JSON with `JSON_OBJECT()` / `JSON_ARRAY()` or your language's JSON library, never by concatenating strings.

**Mistake 4 — choosing a NoSQL database for scale you do not have.** A store with a few million rows and hundreds of requests per second fits comfortably on one MySQL server. Adopting a distributed database early trades a non-problem (scale) for real ones (no joins, eventual consistency, operational learning curve).

**Mistake 5 — treating "the DBA" as one job.** Assuming the developer who wrote the schema will also handle backups, upgrades, capacity and security is how backups go untested. The roles below exist because the responsibilities are distinct.

## Industry Application

### Who works with databases

| Role | Interacts with the database by… | Needs from it | Typical MySQL privileges |
|---|---|---|---|
| **End users** (naive/parametric) | Using applications: a checkout page, a banking app, a registration form | Correctness, speed, availability; they never see SQL | None directly — the application's account acts for them |
| **Application developers** | Writing code that issues SQL through drivers/ORMs; designing tables with the team | A clear schema, constraints that catch bugs, predictable performance, migrations | DML (and DDL in development) on the application schema |
| **Data analysts / scientists** (sophisticated users) | Writing ad hoc SQL, notebooks, BI dashboards | Read access to consistent data, documentation of meaning, no impact on production | `SELECT`, ideally on a replica or warehouse |
| **Data engineers** | Building pipelines that extract, transform and load data between systems; change-data-capture; data quality checks | Catalog metadata, binary log access for CDC, stable schemas, bulk interfaces | `SELECT`, `REPLICATION SLAVE`/`REPLICATION CLIENT` for CDC tools, rights on staging schemas |
| **Database designers / data architects** | Modelling entities and relationships, choosing technologies, setting standards | Expressive modelling features, constraint support | DDL in design environments |
| **Database administrators (DBAs)** | Installing, configuring, securing, backing up, monitoring, tuning and upgrading the DBMS | Administrative interfaces, observability (Performance Schema), backup tools | Administrative privileges (dynamic privileges such as `BACKUP_ADMIN`, `SYSTEM_VARIABLES_ADMIN`), user management |
| **Tool developers / DBMS engineers** | Building drivers, monitoring tools, or the DBMS itself | APIs, protocols, plugin interfaces | Varies |

DBA work is commonly split further:

- **System / operational DBA** — servers, installation, configuration, upgrades, backups and restores, replication and high availability, capacity planning, patching.
- **Application / development DBA** — schema design reviews, query tuning, index strategy, migration review, helping developers with transactions and locking.
- **Database reliability engineer (DBRE)** — the SRE-style evolution of the DBA: automation (infrastructure as code, automated failover), observability, SLOs, incident response, often across many database technologies and managed cloud services.
- **Security / governance roles** — access reviews, auditing, encryption, data classification and retention (often shared with a data-governance team).

In MySQL, these responsibilities map onto **roles** (named bundles of privileges, MySQL 8.0+). An organisation might define them like this (display only — the sandbox cannot create roles or users):

```sql
CREATE ROLE 'app_rw', 'analyst_ro', 'cdc_reader';
GRANT SELECT, INSERT, UPDATE, DELETE ON shop.* TO 'app_rw';
GRANT SELECT ON shop.* TO 'analyst_ro';
GRANT SELECT, REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'cdc_reader';
CREATE USER 'checkout_svc'@'10.0.%' IDENTIFIED BY '…' DEFAULT ROLE 'app_rw';
```

:::version[Terminology in MySQL 8.4]
MySQL 8.4 removed the old `MASTER`/`SLAVE` *statements* in favour of `SOURCE`/`REPLICA` (e.g. `SHOW REPLICA STATUS`). The privilege that a replication or CDC client needs is still named `REPLICATION SLAVE` in the privilege system, alongside `REPLICATION CLIENT`. Check the 8.4 manual's privilege list rather than older blog posts when setting up CDC accounts.
:::

### Model choices in practice

- **E-commerce:** relational system of record (orders, payments, inventory) in MySQL; Redis for sessions and carts; a search engine for the product catalog; a warehouse for analytics.
- **Social and messaging platforms:** relational stores for accounts and billing, wide-column stores (Cassandra, HBase) historically used for message and feed storage at massive write rates, graph stores or graph-processing systems for social graph queries.
- **Fraud detection:** graph queries over accounts, devices and transfers to find rings; results fed back to the relational transaction system.
- **Observability:** time-series databases (Prometheus, InfluxDB) store metrics; MySQL's own metrics are typically scraped into such a system by an exporter.

## Practice Exercises

```exercise
id: ex1
title: A key-value lookup
difficulty: easy
setup: |
  CREATE TABLE feature_flags (flag_key VARCHAR(64) NOT NULL PRIMARY KEY, flag_value JSON NOT NULL);
  INSERT INTO feature_flags VALUES
    ('checkout.new_ui', '{"enabled": true, "rollout_pct": 25}'),
    ('search.synonyms', '{"enabled": false}'),
    ('reviews.photos',  '{"enabled": true, "rollout_pct": 100}');
concepts: [key-value]
prompt: |
  The table `feature_flags` is used as a key-value store. Return the value of flag
  `checkout.new_ui`. One column: `flag_value`.
hints:
  - "A key-value GET is a primary-key equality lookup."
  - "SELECT flag_value FROM feature_flags WHERE flag_key = '…'"
solution: |
  SELECT flag_value FROM feature_flags WHERE flag_key = 'checkout.new_ui';
explanation: |
  An equality predicate on the primary key is the relational form of `GET key`; EXPLAIN would show access
  type `const`. A dedicated key-value store would do the same with a hash lookup in memory, without SQL
  parsing — faster, but without transactions spanning other tables.
```

```exercise
id: ex2
title: Query documents by a nested field
difficulty: medium
setup: |
  CREATE TABLE product_docs (product_id INT UNSIGNED NOT NULL PRIMARY KEY, doc JSON NOT NULL);
  INSERT INTO product_docs VALUES
    (1,  '{"name": "Ultrabook 14 Pro", "brand": "Nimbus", "specs": {"ram_gb": 16}, "colors": ["silver", "black"]}'),
    (2,  '{"name": "Ultrabook 13 Air", "brand": "Nimbus", "specs": {"ram_gb": 8},  "colors": ["gold"]}'),
    (3,  '{"name": "Gaming Laptop 16", "brand": "Vortex", "specs": {"ram_gb": 32}, "colors": ["black"]}'),
    (5,  '{"name": "Wireless Mouse",   "brand": "Nimbus", "specs": {"dpi": 1600},  "colors": ["black", "white"]}');
concepts: [document-model]
prompt: |
  Return the products whose document has `specs.ram_gb` of **at least 16**.
  Columns: `product_id`, `name` (unquoted text), `ram_gb` (a number).
hints:
  - "doc->>'$.specs.ram_gb' extracts the value as text; doc->'$.specs.ram_gb' as JSON. Both compare numerically with 16 here, but casting makes intent explicit."
  - "Documents without the path yield NULL, which fails the comparison and is filtered out automatically."
solution: |
  SELECT product_id,
         doc->>'$.name' AS name,
         CAST(doc->>'$.specs.ram_gb' AS UNSIGNED) AS ram_gb
  FROM product_docs
  WHERE CAST(doc->>'$.specs.ram_gb' AS UNSIGNED) >= 16;
explanation: |
  Products 1 and 3 qualify. The mouse has no `ram_gb`, so the path is missing, the extraction is NULL and
  `NULL >= 16` is unknown — the row is dropped. Using `->` for `name` would return `"Ultrabook 14 Pro"` with
  JSON quotes, which graders and users usually do not want.
```

```exercise
id: ex3
title: Graph traversal over the category tree
difficulty: hard
concepts: [graph-model]
prompt: |
  Treat `categories` as a graph in which `parent_id` is an edge to the parent. Return **all categories
  below 'Computers'** at any depth (not including Computers itself).
  Columns: `category_id`, `name`.
hints:
  - "Use WITH RECURSIVE: the anchor selects the direct children of Computers; the recursive part joins categories to the CTE on parent_id."
  - "Anchor: SELECT category_id, name FROM categories WHERE parent_id = (SELECT category_id FROM categories WHERE name = 'Computers')."
solution: |
  WITH RECURSIVE below AS (
    SELECT category_id, name
    FROM categories
    WHERE parent_id = (SELECT category_id FROM categories WHERE name = 'Computers')
    UNION ALL
    SELECT c.category_id, c.name
    FROM categories AS c
    JOIN below AS b ON c.parent_id = b.category_id
  )
  SELECT category_id, name FROM below;
explanation: |
  In this dataset Computers has two children (Laptops, Accessories) and no grandchildren, but the recursive
  form keeps working if the tree grows deeper — a single self-join would not. This is the relational way
  to express a variable-length path; a graph database expresses it as `-[:CHILD_OF*1..]->`.
```

```exercise
id: ex4
title: Downsample a time series
difficulty: hard
setup: |
  CREATE TABLE cpu_metrics (host VARCHAR(20) NOT NULL, ts DATETIME NOT NULL, cpu_pct DECIMAL(5,2) NOT NULL, PRIMARY KEY (host, ts));
  INSERT INTO cpu_metrics VALUES
    ('db1', '2024-06-01 09:05:00', 35.0), ('db1', '2024-06-01 09:35:00', 55.0),
    ('db1', '2024-06-01 10:05:00', 80.0), ('db1', '2024-06-01 10:25:00', 90.0), ('db1', '2024-06-01 10:45:00', 70.0),
    ('db2', '2024-06-01 09:15:00', 20.0), ('db2', '2024-06-01 10:15:00', 25.0);
concepts: [time-series]
prompt: |
  Downsample `cpu_metrics` to **hourly** buckets per host. Return `host`, `hour_start` (a DATETIME at the start
  of the hour, e.g. `2024-06-01 09:00:00`), `samples`, `avg_cpu` (rounded to 1 decimal) and `max_cpu`.
hints:
  - "Truncate the timestamp to the hour. One way: CAST(DATE_FORMAT(ts, '%Y-%m-%d %H:00:00') AS DATETIME)."
  - "GROUP BY host and the truncated hour; aggregate with COUNT(*), AVG and MAX."
solution: |
  SELECT host,
         CAST(DATE_FORMAT(ts, '%Y-%m-%d %H:00:00') AS DATETIME) AS hour_start,
         COUNT(*)              AS samples,
         ROUND(AVG(cpu_pct), 1) AS avg_cpu,
         MAX(cpu_pct)          AS max_cpu
  FROM cpu_metrics
  GROUP BY host, hour_start;
explanation: |
  This is the core operation of a time-series database ("downsampling" or a "rollup"). The primary key
  `(host, ts)` keeps each host's points contiguous and ordered in InnoDB, so a range scan per host is cheap.
  Time-series databases add automatic rollups, compression and retention on top of the same idea. Grouping
  only by `DATE_FORMAT(ts, '%H')` would wrongly merge the same hour on different days.
```

```exercise
id: ex5
title: Design the data layer for a ride-sharing app
difficulty: challenge
kind: open
concepts: [polyglot-persistence, relational-model, key-value, time-series, graph-model, database-users, dba-roles]
prompt: |
  A ride-sharing start-up needs to store: (a) riders, drivers, payment methods and completed trips with fares;
  (b) each driver's live GPS position, updated every 4 seconds; (c) a 90-day history of GPS points for
  dispute resolution and ETA models; (d) referral relationships ("who invited whom") used to detect
  referral-fraud rings.

  For each of (a)–(d), choose a data model (and an example product), and justify the choice in terms of access
  patterns and guarantees. Then say whether you would start with fewer systems, and name which people/roles
  would own and use each store.
solution: |
  - **(a) Accounts, payments, trips — relational (MySQL 8.4 / PostgreSQL).** Interrelated entities, money,
    integrity (a trip must reference an existing rider and driver), multi-row transactions (complete trip +
    charge + driver earnings), ad hoc analytics. ACID matters most here.
  - **(b) Live positions — key-value in memory (Redis, possibly with its geospatial commands).** One key per
    driver, overwritten every few seconds; only the latest value matters; read by "nearby drivers" lookups.
    Durability is unimportant (positions are regenerated within seconds), latency is critical.
  - **(c) GPS history — time-series (TimescaleDB, InfluxDB) or wide-column (Cassandra) keyed by
    (driver_id, day) with time clustering.** Append-only, enormous volume (drivers × 21,600 points/day),
    queried by driver and time range, 90-day retention by dropping old partitions/chunks.
  - **(d) Referral graph — start with a relational edge table and recursive CTEs**; move to a graph
    database (Neo4j, Neptune) only if multi-hop fraud queries over a large graph become too slow.

  **Start smaller:** a single MySQL instance can hold (a), (d) and initially (c) with a `(driver_id, ts)`
  primary key and monthly range partitions; Redis for (b) is justified from day one because the update rate and
  latency needs differ fundamentally. Add specialised stores when measurements demand it.

  **People:** application developers own the schemas and access code; a DBA/DBRE owns MySQL operations
  (backups, replication, upgrades) and Redis configuration; data engineers build CDC pipelines from MySQL into
  the warehouse and the GPS history into ML feature stores; analysts and data scientists get read-only access
  to the warehouse; the fraud team queries the referral graph; end users (riders/drivers) only ever touch the
  apps.
```

## Knowledge Assessment

```quiz
- id: q1
  question: "Which data model is the best fit for storing user sessions that are always read and written by session ID, at very high request rates, where losing a session on a crash is acceptable?"
  options: ["Graph", "In-memory key-value", "Relational with many joins", "Time-series"]
  answer: 1
  explanation: "Single-key GET/PUT at high rate with relaxed durability is the classic key-value use case (Redis, Memcached). A graph model targets relationships, joins are unnecessary for a single-key access, and time-series stores target timestamped measurements."
  concepts: [key-value]
- id: q2
  question: "What does MySQL's `JSON` type give you? (Select all that apply.)"
  type: multi
  options:
    - "Validation of documents on insert and a binary storage format"
    - "Path extraction with -> and ->>, and shredding with JSON_TABLE"
    - "Automatic horizontal sharding of documents across servers"
    - "Indexing of document fields through functional or multi-valued indexes"
    - "Native support for SQL/PGQ graph queries"
  answer: [0, 1, 3]
  explanation: "MySQL validates JSON and stores it in a binary format, offers path operators and JSON_TABLE, and indexes paths via generated/functional/multi-valued indexes. It does not shard a table across servers by itself (that needs tooling such as Vitess or application sharding), and MySQL 8.4 does not implement SQL/PGQ."
  concepts: [document-model]
- id: q3
  question: "In a wide-column store such as Cassandra, a table is designed with partition key (device_id) and clustering key (event_time). Which query is efficient?"
  options:
    - "All events for device 42 between 10:00 and 11:00"
    - "All devices whose firmware version is 3.1"
    - "A join between events and a customers table"
    - "The average of a column across all partitions, returned in milliseconds"
  answer: 0
  explanation: "A partition-key equality plus a clustering-key range reads one partition sequentially. Queries on non-key attributes, joins, and global aggregates require scanning many or all partitions — wide-column stores are designed around known query patterns, one table per pattern."
  concepts: [wide-column]
- id: q4
  question: "`SELECT doc->'$.brand' FROM product_docs WHERE product_id = 1` returns `\"Nimbus\"` with quotes. Why?"
  options:
    - "The document was stored with escaped quotes"
    - "-> returns a JSON value (a JSON string keeps its quotes); ->> unquotes it"
    - "MySQL always quotes VARCHAR values"
    - "The column has the wrong collation"
  answer: 1
  explanation: "-> is JSON_EXTRACT and returns JSON; ->> is JSON_UNQUOTE(JSON_EXTRACT(...)) and returns text. The stored document is fine, and neither VARCHAR nor collation adds quotes."
  concepts: [document-model]
- id: q5
  question: "A team with 3 million rows and 200 requests per second proposes moving from MySQL to a distributed NoSQL database 'for scalability'. What is the most professional response?"
  options:
    - "Agree — relational databases cannot scale"
    - "Ask for the measured bottleneck and access patterns first; this load fits comfortably on a single relational server, and the move would trade joins and transactions for scale that is not needed"
    - "Agree, but keep both databases synchronised by hand"
    - "Disagree because NoSQL databases do not support indexes"
  answer: 1
  explanation: "Model choice should follow measured requirements. This workload is small for MySQL. Relational systems do scale (replicas, sharding), many NoSQL systems do support secondary indexes, and manual synchronisation creates the redundancy problems of file systems."
  concepts: [polyglot-persistence]
- id: q6
  question: "Which role is primarily responsible for configuring replication, testing backup restores, and applying MySQL upgrades?"
  options: ["Data analyst", "Application developer", "System/operational DBA (or DBRE)", "End user"]
  answer: 2
  explanation: "Operational DBAs and database reliability engineers own availability, backups, replication and upgrades. Developers own application schemas and queries, analysts consume data, and end users interact only through applications."
  concepts: [dba-roles, database-users]
- id: q7
  question: "How does MySQL 8.4 express a variable-depth traversal such as 'all categories below Electronics'?"
  options:
    - "With a MATCH clause in SQL/PGQ"
    - "With a recursive common table expression (WITH RECURSIVE)"
    - "It cannot; you must use a graph database"
    - "With a FOREIGN KEY ... CASCADE"
  answer: 1
  explanation: "Recursive CTEs (SQL:1999, MySQL 8.0+) iterate a join until no new rows appear. SQL/PGQ is not implemented in MySQL 8.4, a graph database is optional rather than required, and foreign key actions propagate deletes/updates, not queries."
  concepts: [graph-model]
- id: q8
  question: "A data engineer sets up a change-data-capture tool that streams MySQL changes into a warehouse. Which MySQL facility does such a tool typically read?"
  options: ["The slow query log", "The binary log", "The InnoDB doublewrite buffer", "information_schema.PROCESSLIST"]
  answer: 1
  explanation: "CDC tools (e.g. Debezium) connect like a replica and read row-based binary log events. The slow query log records slow statements, the doublewrite buffer protects against torn pages, and PROCESSLIST shows current connections — none of them is a change stream."
  concepts: [database-users]
```
