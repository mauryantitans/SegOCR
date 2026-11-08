---
id: m01-relational-model
dataset: library
summary: Codd's relational model made precise — domains, attributes, tuples, relations, degree and cardinality, the properties every relation has — and how MySQL tables map onto it, including the places where SQL deliberately departs from the theory (duplicate rows, column order, NULL and three-valued logic).
concepts: [relation, tuple, attribute, domain, degree, cardinality, relation-properties, null, three-valued-logic, bag-semantics]
related: [m01-schemas-three-schema-architecture]
references:
  - title: "MySQL 8.4 Reference Manual — Data Types"
    url: https://dev.mysql.com/doc/refman/8.4/en/data-types.html
  - title: "MySQL 8.4 Reference Manual — Working with NULL Values"
    url: https://dev.mysql.com/doc/refman/8.4/en/working-with-null.html
  - title: "MySQL 8.4 Reference Manual — Problems with NULL Values"
    url: https://dev.mysql.com/doc/refman/8.4/en/problems-with-null.html
  - title: "MySQL 8.4 Reference Manual — The ENUM Type"
    url: https://dev.mysql.com/doc/refman/8.4/en/enum.html
  - title: "MySQL 8.4 Reference Manual — VALUES Statement"
    url: https://dev.mysql.com/doc/refman/8.4/en/values.html
  - title: "MySQL 8.4 Reference Manual — CHECK Constraints"
    url: https://dev.mysql.com/doc/refman/8.4/en/create-table-check-constraints.html
---

## Introduction

In June 1970, Edgar F. Codd, a mathematician at IBM's San Jose Research Laboratory, published *"A Relational Model of Data for Large Shared Data Banks."* It proposed something radical for its time: represent **all** data as **relations** — mathematical sets of tuples — and manipulate it with operations on whole relations, without exposing pointers, record order or storage paths to users. Every relational DBMS you will use, MySQL included, descends from that paper.

When you write `CREATE TABLE books (...)` you are, informally, defining a relation. But "table" and "relation" are not perfect synonyms, and the gaps between them explain a surprising number of real bugs: duplicate rows that inflate a report, a `WHERE` clause that silently skips rows with `NULL`, a query that "always" returned rows in a certain order until it didn't.

This lesson makes the model precise:

- the vocabulary — **domain**, **attribute**, **tuple**, **relation**, **relation schema**, **degree**, **cardinality**;
- the **properties** every relation has by definition (no duplicate tuples, no tuple order, no attribute order, atomic values);
- what **NULL** means in the relational model and the **three-valued logic** SQL uses to handle it;
- exactly how MySQL tables **map** to relations, and where SQL **departs** from the model on purpose.

:::note[Dataset]
The examples use the **library** dataset: 16 authors, 12 books, a many-to-many `book_authors` table, 20 members, 24 physical copies and 52 loans. Author 16 ("Unpublished Author") has no recorded country or birth year.
:::

## DBMS Theory

:::theory[Formal definitions]
- A **domain** *D* is a set of atomic values, with a name, a data type and a meaning. Example: *ISBN13* = the set of 13-digit strings that are valid ISBNs; *Year* = integers 1450–2100.
- An **attribute** *A* is the name of a role played by a domain in a relation. `published_year` is an attribute whose domain is *Year*; write dom(published_year) = *Year*.
- A **relation schema** R(A₁, A₂, …, Aₙ) is a relation name plus a list of attributes. Example: BOOK(book_id, isbn, title, genre, published_year, publisher).
- The **degree** (or *arity*) of a relation is the number of attributes *n*.
- A **tuple** *t* is an ordered list of values ⟨v₁, v₂, …, vₙ⟩ with each vᵢ ∈ dom(Aᵢ) — or, in the alternative definition, a mapping from attribute names to values, which makes attribute order irrelevant.
- A **relation** (or **relation state**) r(R) is a **set** of tuples. Mathematically, r(R) ⊆ dom(A₁) × dom(A₂) × … × dom(Aₙ): a subset of the Cartesian product of the domains.
- The **cardinality** of a relation is the number of tuples it contains, |r(R)|.
- A **relational database schema** is a set of relation schemas plus integrity constraints; a **relational database state** is a set of relation states, one per schema, that satisfies those constraints.
:::

The definition "a subset of the Cartesian product of the domains" is worth pausing on. If dom(genre) = {Fiction, Technology, Textbook} and dom(published_year) has 651 values, then the Cartesian product contains every possible (genre, year) pair — 1,953 of them. A relation over those two attributes is one particular **subset**: the pairs that are true facts about the world ("there is a Fiction book from 1969"). This is the deep idea of the model: **a relation is a set of true propositions**, and each tuple asserts one fact. This is called the **closed-world assumption**: a tuple that is absent from the relation is taken to be false.

### Properties of relations

Because a relation is a *set* of tuples, several properties follow directly from the definition:

| Property | Statement | Consequence |
|---|---|---|
| **No duplicate tuples** | A set cannot contain the same element twice | Every relation has at least one **key** (the set of all attributes, if nothing smaller) |
| **Tuples are unordered** | Sets have no order | "The first row" has no meaning without an explicit ordering |
| **Attributes are unordered** (with the mapping definition of tuples) | Tuples are identified by attribute names, not positions | `R(a, b)` and `R(b, a)` describe the same relation |
| **Values are atomic** | Each value is a single element of its domain, not a set or list | First Normal Form (1NF); nested lists are not relational values |
| **Each attribute has a distinct name** within the relation | Attributes are referred to by name | Query results also need unique column names to be relations |

### Terminology mapping

| Formal relational term | SQL / MySQL term | Informal term |
|---|---|---|
| Relation | Table (more precisely: the rows of a base table at a moment) | File, sheet |
| Relation schema | Table definition (`CREATE TABLE`) | Header, layout |
| Tuple | Row | Record |
| Attribute | Column | Field |
| Domain | Data type + constraints (`CHECK`, `ENUM`, `NOT NULL`) | Allowed values |
| Degree | Number of columns | Width |
| Cardinality | Number of rows (`COUNT(*)`) | Length |
| Derived relation | Query result, view | Report |

### NULL in the relational model

Real data is incomplete: an author's birth year may be **unknown**, a book may have **no** second author, a member's phone number may **not apply**. Codd introduced a special marker, **NULL**, meaning "value absent" — later distinguishing *missing but applicable* (unknown) from *inapplicable*, though SQL implements only a single NULL. NULL is **not a value in any domain**: it is not zero, not the empty string, not false. Comparing anything with NULL yields neither true nor false but **UNKNOWN**, so SQL uses a **three-valued logic** (3VL):

| p | q | p AND q | p OR q | NOT p |
|---|---|---|---|---|
| TRUE | UNKNOWN | UNKNOWN | TRUE | FALSE |
| FALSE | UNKNOWN | FALSE | UNKNOWN | TRUE |
| UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |

A `WHERE` clause keeps only rows for which the condition is **TRUE**; rows where it is FALSE **or UNKNOWN** are discarded. NULL is controversial among theorists — C. J. Date and Hugh Darwen argue it breaks the model's logical foundation and advocate designs that avoid it — but every SQL DBMS supports it, so a professional must master its rules. The full treatment is in [NULL and three-valued logic](/lessons/m02-null-three-valued-logic).

## Why It Exists

Before the relational model, databases were **navigational**. In IBM's IMS (hierarchical) and CODASYL (network) systems, records were linked by physical pointers, and programs retrieved data by walking those links: "find the first ORDER under this CUSTOMER, then the next ORDER, then its first LINE-ITEM". Three problems followed:

1. **Programs depended on storage structure.** Reorganising links or adding an access path broke programs — the opposite of data independence.
2. **Only anticipated questions were easy.** If the links ran from customers to orders, asking "which customers ordered product X?" meant traversing everything.
3. **There was no mathematical basis** for reasoning about queries, proving equivalences, or optimizing them automatically.

Codd's model fixed all three. Relations contain only values — relationships are represented by **matching values** (a book's `book_id` appears in `book_authors`), not pointers. Queries are expressed as operations on relations (relational algebra) or as logical predicates (relational calculus), so any question that can be stated over the data can be asked, and the DBMS — not the programmer — decides how to execute it. Because the operations have precise mathematical meaning, an optimizer can rewrite a query into an equivalent, cheaper one. This is why SQL databases can answer questions their designers never anticipated, and why the relational model has outlived every model that preceded it.

## Internal Working

:::internals[From relation to InnoDB storage]
The relational model deliberately says nothing about storage — that is the point of data independence. But it helps to know how MySQL represents each concept:

- **Relation schema → data dictionary entries.** `CREATE TABLE` stores the table's columns (name, type, nullability, default, ordinal position), indexes and constraints in the transactional data dictionary.
- **Domain → data type + constraints.** MySQL has no `CREATE DOMAIN`; a domain is approximated by a column's data type (`SMALLINT`, `CHAR(13)`, `ENUM(...)`), `NOT NULL`, and `CHECK` constraints, enforced on every write in strict SQL mode.
- **Tuple → row in a clustered B+ tree.** InnoDB stores each row in the leaf pages of the table's **clustered index**, ordered by primary key. A row's physical record also contains hidden system fields: the ID of the transaction that last modified it (`DB_TRX_ID`) and a pointer to its previous version in the undo log (`DB_ROLL_PTR`), used for MVCC. If a table has no primary key and no suitable unique `NOT NULL` key, InnoDB adds a hidden 6-byte row ID (`DB_ROW_ID`) to cluster on.
- **NULL → a bit in the record header.** In the `COMPACT` and `DYNAMIC` row formats, each record has a **NULL bitmap** with one bit per nullable column; a NULL value occupies no space in the variable-length data area. NULL is therefore not "an empty string" at the storage level either.
- **No tuple order.** Rows are physically ordered by primary key inside the clustered index, but the relational model and SQL promise **no** order for a query result unless `ORDER BY` is given — the optimizer may read a secondary index, use a hash join, or parallelise, and the "natural" order changes.
:::

That last point is the most common practical consequence of the model's internals: physical order is an implementation detail. Code that relies on it works until an index is added or a plan changes.

## MySQL Implementation

Look at the `books` relation through the catalog. Its schema — attribute names, domains (types), and positions — is metadata:

```sql run
SELECT ORDINAL_POSITION AS pos, COLUMN_NAME AS attribute, COLUMN_TYPE AS domain_type, IS_NULLABLE
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'books'
ORDER BY ORDINAL_POSITION;
```

**Degree and cardinality.** The degree is the number of attributes; the cardinality is the number of tuples:

```sql run
SELECT (SELECT COUNT(*) FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'books') AS degree,
       (SELECT COUNT(*) FROM books) AS cardinality;
```

Degree belongs to the **schema** and changes only through DDL. Cardinality belongs to the **state** and changes with every insert or delete.

**Tuples as facts.** Each row of `books` asserts one proposition: *"There is a book with id 8, ISBN 9780441478125, titled 'The Left Hand of Darkness', genre Fiction, published in 1969 by …"*:

```sql run
SELECT book_id, isbn, title, genre, published_year
FROM books
WHERE genre = 'Fiction'
ORDER BY published_year;
```

**Domains enforced.** The domain of `members.membership` is `ENUM('standard','student','premium')`. A value outside the domain is rejected in strict SQL mode:

```sql run error
INSERT INTO members (member_id, name, email, joined_on, membership)
VALUES (21, 'Rina Patel', 'rina.patel@reader.example', '2024-09-01', 'platinum');
```

Error 1265 ("Data truncated") is how MySQL reports an `ENUM` value outside the list. The domain of `loans.due_at` vs `loans.loaned_at` is further restricted by `CHECK (due_at >= loaned_at)`:

```sql run error
INSERT INTO loans (loan_id, copy_id, member_id, loaned_at, due_at)
VALUES (53, 1, 1, '2024-09-10', '2024-09-01');
```

**NULL: absent, not empty.** Author 16 has no country and no birth year. Look at how comparisons treat that row:

```sql run
SELECT author_id, name, country,
       country = 'Germany'          AS eq_germany,
       country <> 'Germany'         AS ne_germany,
       country IS NULL              AS country_is_null
FROM authors
WHERE author_id IN (1, 7, 16)
ORDER BY author_id;
```

For author 16, both `country = 'Germany'` and `country <> 'Germany'` evaluate to NULL (UNKNOWN) — neither true nor false. Only `IS NULL` gives a definite answer.

## Syntax Breakdown

How the formal concepts are expressed in MySQL DDL:

```sql
CREATE TABLE book (                                   -- relation schema name
  book_id        INT UNSIGNED NOT NULL,               -- attribute + domain (type) + no NULLs
  isbn           CHAR(13)     NOT NULL,
  title          VARCHAR(200) NOT NULL,
  genre          ENUM('Fiction','Technology','Textbook') NOT NULL,   -- enumerated domain
  published_year SMALLINT     NOT NULL
                 CHECK (published_year BETWEEN 1450 AND 2100),       -- restricted domain
  subtitle       VARCHAR(200) NULL,                   -- attribute that may be absent
  PRIMARY KEY (book_id),                              -- guarantees no duplicate tuples
  UNIQUE (isbn)                                       -- another key
);
```

| Relational concept | MySQL construct | Notes |
|---|---|---|
| Relation schema R(A₁…Aₙ) | `CREATE TABLE R (A₁ type, …, Aₙ type)` | Columns have an ordinal position (a departure from the model) |
| Domain | Data type, `ENUM`/`SET`, `NOT NULL`, `CHECK`, `DEFAULT` | No `CREATE DOMAIN` (standard SQL feature not implemented) |
| Tuple | Row (`INSERT … VALUES (…)`, `VALUES ROW(…)`) | `VALUES ROW(...)` (8.0.19+) constructs tuples directly |
| Set of tuples (no duplicates) | `PRIMARY KEY` or `UNIQUE NOT NULL` on the table; `SELECT DISTINCT` on results | Without a key, a table is a **bag** (multiset) |
| Unordered tuples | No order unless `ORDER BY` | Never rely on insertion or physical order |
| Degree | `COUNT(*)` over `information_schema.COLUMNS` | Schema property |
| Cardinality | `SELECT COUNT(*) FROM R` | State property |
| Missing value | `NULL`; tested with `IS NULL`, `IS NOT NULL`, `<=>` | Comparisons with `=`/`<>` return NULL |

:::standard[Standard SQL vs MySQL]
Standard SQL has `CREATE DOMAIN` (supported by PostgreSQL, for example); **MySQL does not**. MySQL's `ENUM` and `SET` types are **MySQL extensions**, as is the NULL-safe equality operator `<=>` (the standard spelling is `IS NOT DISTINCT FROM`, which MySQL does not accept). The `VALUES` table constructor with `ROW(...)` is MySQL's form of the standard `VALUES (…), (…)` table value constructor.
:::

Trying the standard syntax confirms it:

```sql run error
CREATE DOMAIN year_domain AS SMALLINT CHECK (VALUE BETWEEN 1450 AND 2100);
```

## Practical Examples

**1. Tables are bags unless you make them sets.** The relational model forbids duplicate tuples; SQL tables allow them unless a key forbids them. Create a table without a key and insert the same fact twice:

```sql run
CREATE TABLE genre_tags (
  book_id INT UNSIGNED NOT NULL,
  tag     VARCHAR(30)  NOT NULL
);
INSERT INTO genre_tags VALUES (8, 'sci-fi'), (8, 'sci-fi'), (8, 'classic'), (11, 'sci-fi');
SELECT book_id, tag FROM genre_tags ORDER BY book_id, tag;
```

The fact *"book 8 is tagged sci-fi"* appears twice. As a relation that is meaningless — a proposition is not "more true" for being stated twice — but it changes SQL results:

```sql run
SELECT COUNT(*) AS rows_in_table,
       (SELECT COUNT(*) FROM (SELECT DISTINCT book_id, tag FROM genre_tags) AS s) AS distinct_tuples,
       SUM(tag = 'sci-fi') AS sci_fi_rows
FROM genre_tags;
```

Four rows, three distinct tuples, and a "sci-fi count" of 3 although only two books carry that tag. Declaring a key turns the bag into a set — and MySQL refuses to do so while the duplicate exists:

```sql run error
ALTER TABLE genre_tags ADD PRIMARY KEY (book_id, tag);
```

Remove the duplicate and the key can be added:

```sql run
CREATE TABLE genre_tags_clean AS SELECT DISTINCT book_id, tag FROM genre_tags;
ALTER TABLE genre_tags_clean ADD PRIMARY KEY (book_id, tag);
SELECT book_id, tag FROM genre_tags_clean ORDER BY book_id, tag;
```

**2. Attribute order is a SQL artefact.** In the model, ⟨title, genre⟩ and ⟨genre, title⟩ tuples with named attributes are the same. SQL gives columns positions, which matter for `SELECT *` and `INSERT` without a column list. Positional inserts break when columns are reordered or added; named ones do not:

```sql run
INSERT INTO genre_tags_clean (tag, book_id) VALUES ('magical-realism', 12);
SELECT book_id, tag FROM genre_tags_clean WHERE book_id = 12;
```

Naming the attributes makes the statement independent of column order — closer to the relational ideal.

**3. Atomic values.** The model requires each attribute value to be a single domain element. A column holding `'sci-fi,classic'` violates that: the DBMS sees one opaque string, so it cannot enforce a domain on each tag, index them, or join on them. Compare how hard it is to ask "which books are tagged classic?" with a comma list:

```sql run
CREATE TABLE book_tags_csv (book_id INT UNSIGNED PRIMARY KEY, tags VARCHAR(200));
INSERT INTO book_tags_csv VALUES (8, 'sci-fi,classic'), (11, 'sci-fi'), (12, 'magical-realism,classic-ish');
SELECT book_id, tags,
       tags LIKE '%classic%'         AS like_match,
       FIND_IN_SET('classic', tags) > 0 AS exact_match
FROM book_tags_csv ORDER BY book_id;
```

`LIKE '%classic%'` wrongly matches book 12's `classic-ish`; `FIND_IN_SET` (a MySQL-specific function) works but cannot use an index and breaks if anyone adds a space after a comma. The relational design — one row per (book, tag), as in `genre_tags_clean` — answers the question with a plain, indexable equality. This is First Normal Form, studied in [1NF](/lessons/m06-1nf).

**4. NULL in aggregates and filters.** How many authors are *not* from the United States?

```sql run
SELECT COUNT(*) AS all_authors,
       SUM(country = 'United States') AS us,
       SUM(country <> 'United States') AS not_us_by_comparison,
       SUM(country <> 'United States' OR country IS NULL) AS not_us_or_unknown
FROM authors;
```

The comparison `country <> 'United States'` is UNKNOWN for author 16, so the naive count misses that author: the three "categories" do not add up to the total. Whether author 16 should be counted depends on the question — "authors known not to be American" versus "authors not known to be American" — and SQL makes you choose explicitly.

## Execution Walkthrough

Build a relation from first principles: domains, their Cartesian product, and a relation as a subset of it. The `VALUES ROW(…)` table constructor lets you write tuples literally.

```walkthrough
title: A relation is a subset of the Cartesian product of its domains
steps:
  - title: "1. Two small domains"
    explain: >-
      Take dom(membership) = {standard, student, premium} and dom(joined_year) = {2021, 2022, 2023, 2024}.
      Each is a set of atomic values; VALUES builds them as one-column relations.
    sql: |
      SELECT * FROM (VALUES ROW('standard'), ROW('student'), ROW('premium')) AS d1;
  - title: "2. The Cartesian product of the domains"
    explain: >-
      dom(membership) × dom(joined_year) contains every possible pair — 3 × 4 = 12 tuples. It is the universe
      of all tuples that could ever appear in a relation over these two attributes.
    sql: |
      SELECT d1.column_0 AS membership, d2.column_0 AS joined_year
      FROM (VALUES ROW('standard'), ROW('student'), ROW('premium')) AS d1
      CROSS JOIN (VALUES ROW(2021), ROW(2022), ROW(2023), ROW(2024)) AS d2
      ORDER BY membership, joined_year;
  - title: "3. The relation — the subset that is true"
    explain: >-
      The relation MEMBERSHIP_COHORT(membership, joined_year) contains only the pairs that are facts: "some member
      with this membership level joined in this year". It is derived from the members table, and DISTINCT makes it a
      set. Pairs of the product that are absent are, under the closed-world assumption, false.
    sql: |
      SELECT DISTINCT membership, YEAR(joined_on) AS joined_year
      FROM members
      ORDER BY membership, joined_year;
  - title: "4. Degree and cardinality of the derived relation"
    explain: >-
      The derived relation has degree 2 (two attributes). Its cardinality is the number of distinct true pairs —
      at most 12, the size of the product.
    sql: |
      SELECT 2 AS degree, COUNT(*) AS cardinality
      FROM (SELECT DISTINCT membership, YEAR(joined_on) AS joined_year FROM members) AS r;
```

Note the system-generated column name `column_0` in steps 1 and 2: MySQL names the columns of a `VALUES` constructor `column_0`, `column_1`, … — a reminder that SQL needs **unique attribute names** for a result to be a proper relation, which is why the query renames them with `AS`.

## Output

The walkthrough's final relation lists the (membership, joined_year) pairs that actually occur among the library's 20 members: **7** of the 12 tuples in the Cartesian product. Every pair *missing* from the result is a combination that does not occur — no member of any level joined in 2024, and student members joined only in 2022. Notice also the order of step 3's output: standard, student, premium. Because `members.membership` is an `ENUM`, `ORDER BY` sorts by the position of each value in the ENUM list, not alphabetically as in step 2 (where the values came from a plain string column) — another reminder that "order" is something a query imposes, by rules you must know. Reading it relationally:

- each output row is a **true proposition** ("a premium member joined in 2022");
- the output is a **set** — `DISTINCT` removed the many members who share the same pair;
- the output has **no inherent order** — `ORDER BY` imposed one for display only;
- its **degree (2)** was fixed by the query's SELECT list, while its **cardinality** depends on the data.

## Alternative Approaches

The relational model is one way to structure data; others trade its guarantees for different conveniences.

| Approach | How it differs from relations | When it is preferable |
|---|---|---|
| **Bags (SQL tables without keys)** | Duplicates allowed | Staging/raw-ingest tables, logs where identical events can legitimately repeat — but add a surrogate key or deduplicate before analysis |
| **Ordered sequences (arrays, lists, spreadsheets)** | Position carries meaning | Time series or ranked lists — in SQL, model the position as an explicit attribute (`author_order` in `book_authors`) |
| **Nested / non-atomic values (JSON, arrays)** | Values can be collections | Semi-structured attributes that are read as a unit; MySQL's `JSON` type supports this, at the cost of weaker constraints and indexing |
| **Avoiding NULL by design** (Date/Darwen) | Optional facts moved to separate relations: `author_birth_years(author_id, birth_year)` holds a row only when known | When correctness matters more than convenience; fewer 3VL surprises, more joins |
| **Default values instead of NULL** (`''`, `0`, `'1900-01-01'`) | "Unknown" encoded as a real domain value | Rarely — sentinel values corrupt aggregates (`AVG(birth_year)` with 1900 for unknown) and must be filtered everywhere |

## Performance Analysis

The relational model is logical; performance belongs to the internal level. But modelling choices made at this level have predictable performance effects:

- **Keys make tables sets — and give InnoDB a clustered index.** Without a primary key, InnoDB clusters on a hidden row ID that you cannot use in queries, and replicas applying row-based events may need full table scans to find rows. Every table should have a primary key (the `sql_require_primary_key` server variable can enforce this).
- **Duplicates cost more than storage.** Duplicate tuples inflate joins (each duplicate multiplies the matching rows on the other side) and force `DISTINCT`, which requires sorting or hashing.
- **Atomic values are indexable.** An attribute holding one value can be indexed with a B+ tree; a comma-separated list forces `LIKE '%…%'` or `FIND_IN_SET`, which scan every row.
- **NULL is cheap to store** (one bit in the NULL bitmap of `COMPACT`/`DYNAMIC` rows) and indexes do contain NULL entries in InnoDB, so `IS NULL` predicates can use an index.
- **Degree matters for row width.** Very wide relations mean fewer rows per 16 KB page and more I/O per scan; splitting rarely used attributes into another relation (vertical partitioning) is a design-level optimization.

:::performance[Theory vs measurement]
These effects are structural and follow from how InnoDB stores rows and indexes; on the tiny library dataset they are invisible. You will measure them on generated data in Module 7 ([Why indexes](/lessons/m07-why-indexes)).
:::

## Best Practices

:::best-practice[Keeping tables relational]
1. **Give every table a primary key** so it is a set, not a bag. If no natural key exists, add a surrogate key and still declare `UNIQUE` on the natural identifying attributes.
2. **Model domains explicitly** with the narrowest suitable type, `NOT NULL` wherever a value is required, and `CHECK` constraints for ranges and formats.
3. **Store one value per attribute**; represent multi-valued facts as separate rows in another table.
4. **Never rely on row order** — always use `ORDER BY` with a unique tie-breaker when order matters.
5. **Name columns in `INSERT` statements** and avoid `SELECT *` in application code; attribute order is a SQL artefact.
6. **Decide what NULL means for each nullable column** (unknown? not applicable?) and document it; use `IS NULL`/`IS NOT NULL`, never `= NULL`.
7. **Represent position, if meaningful, as an attribute** (`author_order`), not as physical order.
:::

## Common Mistakes

**Mistake 1 — testing for NULL with `=`.** `country = NULL` is UNKNOWN for every row, so it returns nothing:

```sql run
SELECT COUNT(*) AS wrong_count FROM authors WHERE country = NULL;
```

```sql run
SELECT COUNT(*) AS right_count FROM authors WHERE country IS NULL;
```

**Mistake 2 — forgetting NULLs in negative conditions.** `WHERE country <> 'Germany'` silently drops authors with unknown country. If they belong in the answer, say so: `WHERE country <> 'Germany' OR country IS NULL` (or the MySQL-specific `NOT (country <=> 'Germany')`).

**Mistake 3 — `NOT IN` with a NULL in the list.** Because `x <> NULL` is UNKNOWN, `x NOT IN (…, NULL)` is never TRUE:

```sql run
SELECT COUNT(*) AS authors_not_in_list
FROM authors
WHERE country NOT IN ('Germany', NULL);
```

The result is 0 even though most authors are not from Germany. The full explanation, and the `NOT EXISTS` alternative, are in [EXISTS and IN](/lessons/m04-exists-in).

**Mistake 4 — assuming the order you inserted is the order you get.** A query without `ORDER BY` may return rows in primary-key order today and in a different order after an index is added or the optimizer changes plan. That is not a bug in MySQL; tuples are unordered.

**Mistake 5 — tables without keys.** As Practical Example 1 showed, duplicates creep in and later prevent adding the key. Declare keys when the table is created.

## Industry Application

- **Schema design reviews.** Senior engineers check new tables against relational properties: Is there a primary key? Are there natural keys that should be `UNIQUE`? Are any columns lists or JSON blobs that should be rows? Are nullable columns justified?
- **Data quality in pipelines.** Data engineers routinely test for "relation-ness" in ingested data: duplicate detection (`GROUP BY key HAVING COUNT(*) > 1`), domain checks (values outside allowed sets), and NULL-rate monitoring. Tools such as dbt express these as tests (`unique`, `not_null`, `accepted_values`) — which are exactly the relational constraints of this lesson.
- **Analytics correctness.** Most reporting errors trace back to bag semantics (duplicate rows inflating counts after a join) or 3VL (NULLs silently excluded by a filter). Analysts who think in relations catch them early.
- **Interoperability.** Because the relational model is mathematical and SQL is standardised, relations move between systems — MySQL to a warehouse, to a dataframe (`pandas.read_sql`), to Parquet — with their structure intact, which is far harder for pointer-based or application-specific formats.
- **Interviews.** "What's the difference between a table and a relation?", "Why can't you compare with NULL using `=`?", and "What is the degree/cardinality of this result?" are standard database interview questions.

## Practice Exercises

```exercise
id: ex1
title: Degree and cardinality
difficulty: easy
concepts: [degree, cardinality]
prompt: |
  Return one row describing the `loans` relation, with two columns in this order: `degree` (the number of
  attributes) and `cardinality` (the number of tuples).
hints:
  - "Degree comes from the catalog: count the rows of information_schema.COLUMNS for this table."
  - "Cardinality is COUNT(*) over the table itself. Combine the two with scalar subqueries."
solution: |
  SELECT (SELECT COUNT(*) FROM information_schema.COLUMNS
          WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'loans') AS degree,
         (SELECT COUNT(*) FROM loans) AS cardinality;
explanation: |
  `loans` has degree 6 and cardinality 52. Degree is a property of the schema (it changes only with
  `ALTER TABLE`); cardinality is a property of the current state. Using `TABLE_ROWS` from
  `information_schema.TABLES` for cardinality would be a mistake: for InnoDB it is only an estimate.
```

```exercise
id: ex2
title: Authors not from the United States — including unknowns
difficulty: medium
concepts: [null, three-valued-logic]
prompt: |
  The library wants a list of authors who are **not known to be from the United States** — that includes
  authors whose country is unknown. Return `author_id` and `name`.
hints:
  - "country <> 'United States' is UNKNOWN when country is NULL, so such rows are filtered out."
  - "Add OR country IS NULL (or use NOT (country <=> 'United States'))."
solution: |
  SELECT author_id, name
  FROM authors
  WHERE country <> 'United States' OR country IS NULL;
explanation: |
  The naive `WHERE country <> 'United States'` misses author 16, whose country is NULL. `<=>` is MySQL's
  NULL-safe equality: `NULL <=> 'United States'` is 0 (false), not NULL, so `NOT (country <=> 'United States')`
  also works — but it is MySQL-specific, while `OR … IS NULL` is portable.
```

```exercise
id: ex3
title: Bag or set?
difficulty: medium
setup: |
  CREATE TABLE reading_list (member_id INT UNSIGNED NOT NULL, book_id INT UNSIGNED NOT NULL);
  INSERT INTO reading_list VALUES (1, 2), (1, 2), (1, 5), (2, 2), (3, 8), (3, 8), (3, 8), (4, 11);
concepts: [relation-properties, bag-semantics]
prompt: |
  `reading_list` was created without a key, and some members added the same book more than once.
  Return one row with: `rows_in_table` (all rows), `distinct_tuples` (number of distinct (member_id, book_id)
  pairs) and `duplicate_rows` (rows that would have to be deleted to make the table a set).
hints:
  - "Count distinct pairs with COUNT(DISTINCT member_id, book_id) — MySQL accepts several expressions — or with a derived table of SELECT DISTINCT."
  - "duplicate_rows = rows_in_table − distinct_tuples."
solution: |
  SELECT COUNT(*) AS rows_in_table,
         COUNT(DISTINCT member_id, book_id) AS distinct_tuples,
         COUNT(*) - COUNT(DISTINCT member_id, book_id) AS duplicate_rows
  FROM reading_list;
explanation: |
  8 rows contain 5 distinct tuples, so 3 rows are redundant. Until they are removed, `ALTER TABLE … ADD PRIMARY
  KEY (member_id, book_id)` fails with error 1062. Multi-argument `COUNT(DISTINCT a, b)` is a MySQL extension;
  the portable form is `SELECT COUNT(*) FROM (SELECT DISTINCT member_id, book_id FROM reading_list) AS s`.
```

```exercise
id: ex4
title: Restrict a domain
difficulty: hard
kind: dml
concepts: [domain]
prompt: |
  The domain of `books.published_year` should be the years **1450 to 2100** inclusive. Add a check constraint
  named **`chk_books_year`** that enforces it. The checker lists the CHECK constraints defined on `books`.
hints:
  - "ALTER TABLE books ADD CONSTRAINT <name> CHECK (<condition>);"
  - "BETWEEN is inclusive on both ends: published_year BETWEEN 1450 AND 2100."
solution: |
  ALTER TABLE books ADD CONSTRAINT chk_books_year CHECK (published_year BETWEEN 1450 AND 2100);
check: |
  SELECT CONSTRAINT_NAME, CONSTRAINT_TYPE, ENFORCED
  FROM information_schema.TABLE_CONSTRAINTS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'books' AND CONSTRAINT_TYPE = 'CHECK';
explanation: |
  MySQL validates existing rows when a CHECK constraint is added, so this works only because every book's year
  is in range. The constraint narrows the domain from "any SMALLINT" to plausible publication years — the closest
  MySQL comes to a standard `CREATE DOMAIN`. An unnamed constraint would get a generated name such as
  `books_chk_1`; naming constraints makes error messages and migrations readable.
```

```exercise
id: ex5
title: Is this spreadsheet a relation?
difficulty: challenge
kind: open
concepts: [relation-properties, domain, null, atomic-values]
prompt: |
  A library volunteer kept this sheet (first row = header):

  | # | Title | Authors | Year | Copies | Notes |
  |---|---|---|---|---|---|
  | 1 | Americanah | Chimamanda Ngozi Adichie | 2013 | 2 | |
  | 2 | High Performance MySQL | Schwartz, Zaitsev, Tkachenko | 2012 | 3 | 3rd ed. |
  | 3 | Americanah | Chimamanda Ngozi Adichie | 2013 | 2 | |
  | 4 | 1Q84 | Haruki Murakami | 2011 | two | sorted by title below this line |
  | 5 | The Dispossessed | Ursula K. Le Guin | n/a | 1 | |

  Identify every way in which the sheet violates the properties of a relation or the idea of a domain, and
  describe a relational design (relation schemas with keys and domains) that fixes each problem.
solution: |
  **Violations**
  - *Duplicate tuples*: rows 1 and 3 state the same fact; the "#" column hides this by making rows unique artificially.
  - *Non-atomic values*: "Authors" in row 2 is a list of three authors.
  - *Domain violations*: "Copies" holds `two` (not an integer); "Year" holds `n/a` (a sentinel string in a numeric domain).
  - *Meaning carried by order*: the note "sorted by title below this line" assumes tuple order matters.
  - *Mixed facts*: "3rd ed." is an edition attribute hidden in free-text notes; "Copies" is a count derived from physical copies.

  **Relational design**
  - `books(book_id PK, isbn UNIQUE NOT NULL, title NOT NULL, edition NULL, published_year SMALLINT NULL CHECK (published_year BETWEEN 1450 AND 2100))` —
    NULL (documented as "unknown") instead of `n/a`; the ISBN as a natural key prevents duplicate books.
  - `authors(author_id PK, name NOT NULL)`.
  - `book_authors(book_id FK, author_id FK, author_order TINYINT NOT NULL, PRIMARY KEY (book_id, author_id))` —
    one row per author, position as an explicit attribute.
  - `copies(copy_id PK, book_id FK, condition ENUM(...))` — the number of copies is then *derived* with
    `COUNT(*)`, so it cannot disagree with reality or contain `two`.
  - No reliance on order: any sorted listing is a query with `ORDER BY title`.

  This is essentially the library dataset's schema.
```

## Knowledge Assessment

```quiz
- id: q1
  question: "A relation has attributes (member_id, name, email, joined_on, membership) and currently holds 20 tuples. What are its degree and cardinality?"
  options: ["Degree 20, cardinality 5", "Degree 5, cardinality 20", "Degree 5, cardinality 5", "Degree 25, cardinality 20"]
  answer: 1
  explanation: "Degree is the number of attributes (5), cardinality the number of tuples (20). They are easy to swap: degree is a schema property, cardinality a state property."
  concepts: [degree, cardinality]
- id: q2
  question: "Which statements are true of a relation in Codd's model? (Select all that apply.)"
  type: multi
  options:
    - "It contains no duplicate tuples"
    - "Its tuples have a defined order"
    - "Each attribute value is atomic (a single domain value)"
    - "It is a subset of the Cartesian product of its attributes' domains"
    - "It may contain the same tuple twice if inserted twice"
  answer: [0, 2, 3]
  explanation: "A relation is a set of tuples drawn from the product of the domains, so there are no duplicates and no order, and values are atomic. SQL tables without keys can contain duplicates — that is a departure from the model, not a property of relations."
  concepts: [relation-properties, relation]
- id: q3
  question: "Author 16 has country NULL. What does `SELECT country <> 'Germany' FROM authors WHERE author_id = 16` return?"
  options: ["1", "0", "NULL", "An error"]
  answer: 2
  explanation: "Any comparison with NULL using = or <> yields UNKNOWN, which MySQL returns as NULL. It is neither true (1) nor false (0), and it is not an error."
  concepts: [null, three-valued-logic]
- id: q4
  question: "Which MySQL construct comes closest to a relational *domain*?"
  options:
    - "A view"
    - "A column's data type combined with NOT NULL and CHECK constraints"
    - "A primary key"
    - "CREATE DOMAIN"
  answer: 1
  explanation: "A domain is the set of allowed values; MySQL approximates it with the type (including ENUM) plus NOT NULL and CHECK. CREATE DOMAIN is standard SQL but not implemented in MySQL; keys and views serve other purposes."
  concepts: [domain]
- id: q5
  question: "A table without a primary key has 8 rows, 5 of them distinct. What happens when you run `ALTER TABLE t ADD PRIMARY KEY (all_columns)`?"
  options:
    - "MySQL silently removes the duplicates"
    - "It fails with a duplicate-entry error until the duplicates are removed"
    - "It succeeds and keeps the duplicates"
    - "It converts the table into a view"
  answer: 1
  explanation: "Adding a key validates existing rows; duplicates cause error 1062. MySQL never deletes data to satisfy a constraint, and a key cannot coexist with duplicates."
  concepts: [relation-properties, bag-semantics]
- id: q6
  question: "`SELECT COUNT(*) FROM authors WHERE country NOT IN ('Germany', NULL)` returns 0. Why?"
  options:
    - "All authors are from Germany"
    - "For every row, country <> NULL is UNKNOWN, so the whole NOT IN condition is never TRUE"
    - "NOT IN is not supported in MySQL"
    - "COUNT(*) ignores NULLs"
  answer: 1
  explanation: "x NOT IN (a, NULL) means x <> a AND x <> NULL; the second part is UNKNOWN, so the conjunction is UNKNOWN or FALSE, never TRUE. COUNT(*) counts rows and ignores nothing; NOT IN is supported."
  concepts: [null, three-valued-logic]
- id: q7
  question: "A query without ORDER BY has always returned rows in primary-key order. After a DBA adds an index, the order changes. What is the correct diagnosis?"
  options:
    - "The index corrupted the table"
    - "Relations are unordered; without ORDER BY no order is guaranteed, and the new plan legitimately returns a different order"
    - "MySQL has a bug in index maintenance"
    - "The primary key was dropped"
  answer: 1
  explanation: "Tuple order is not part of the model or of SQL's contract. Physical order is an implementation detail; a different access path may return rows differently. The fix is an explicit ORDER BY with a unique tie-breaker."
  concepts: [relation-properties]
- id: q8
  question: "Why is storing 'sci-fi,classic' in a single tags column considered non-relational?"
  options:
    - "Because VARCHAR columns cannot contain commas"
    - "Because the value is a list, not an atomic domain value, so the DBMS cannot constrain, index or join on individual tags"
    - "Because MySQL rejects such values in strict mode"
    - "Because tags must be stored as ENUM"
  answer: 1
  explanation: "Atomicity (1NF) requires one domain value per attribute. MySQL happily stores the string — that is the problem: it sees one opaque value. The relational design uses one row per (book, tag)."
  concepts: [relation-properties]
```
