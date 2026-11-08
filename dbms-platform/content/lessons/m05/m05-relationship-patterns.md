---
id: m05-relationship-patterns
dataset: library
summary: The standard relationship patterns and their MySQL implementations — 1:1 (FK + UNIQUE, shared primary key, "at most one current" with a generated column), 1:N, M:N junction tables, recursive relationships with self-referencing foreign keys, and ternary relationships, including the proof that a ternary cannot always be split into binaries.
concepts: [one-to-one, one-to-many, many-to-many, junction-tables, recursive-relationships, ternary-relationships, foreign-key, referential-actions]
related: [m05-relationships-cardinality, m05-weak-entities-eer, m05-er-to-relational, m04-recursive-ctes, m04-cross-self-join, m06-4nf-5nf]
references:
  - title: "MySQL 8.4 Reference Manual — FOREIGN KEY Constraints"
    url: https://dev.mysql.com/doc/refman/8.4/en/create-table-foreign-keys.html
  - title: "MySQL 8.4 Reference Manual — CHECK Constraints (restrictions with foreign keys)"
    url: https://dev.mysql.com/doc/refman/8.4/en/create-table-check-constraints.html
  - title: "MySQL 8.4 Reference Manual — CREATE TABLE and Generated Columns"
    url: https://dev.mysql.com/doc/refman/8.4/en/create-table-generated-columns.html
  - title: "MySQL 8.4 Reference Manual — WITH (Common Table Expressions), Recursive CTEs"
    url: https://dev.mysql.com/doc/refman/8.4/en/with.html
  - title: "MySQL 8.4 Reference Manual — Unique Indexes"
    url: https://dev.mysql.com/doc/refman/8.4/en/create-index.html#create-index-unique
---

## Introduction

Almost every relationship you will ever model is an instance of a handful of patterns. Recognizing the pattern in a requirement — *"each member has one library card"*, *"a book has many copies"*, *"books have several authors and authors write several books"*, *"a staff member reports to another staff member"*, *"a vendor supplies a book to a branch"* — immediately tells you the table structure, the keys, the constraints and the classic mistakes to avoid.

This lesson works through the five patterns on the **library** dataset:

| Pattern | Library example | Core MySQL structure |
|---|---|---|
| **1:1** | Member — library card | FK with `UNIQUE`, or shared primary key |
| **1:N** | Book — copies; member — loans | NOT NULL / NULL foreign key on the N side |
| **M:N** | Books — authors | Junction table `book_authors` with composite PK |
| **Recursive** | Staff — supervisor; book — related book | Self-referencing foreign key, or a junction table to the same table |
| **Ternary** | Vendor supplies book to branch | One table with three foreign keys and a carefully chosen key |

Along the way you will see a few production-grade refinements: the "at most one *current*" pattern (a 1:1 hidden inside a 1:N, like "a copy has at most one open loan"), where to put the foreign key in a 1:1, why junction tables need an index in the reverse direction, and a MySQL-specific restriction between `CHECK` constraints and foreign-key actions that catches many people modelling recursive relationships.

:::note[Dataset]
The library has 12 books, 16 authors linked through `book_authors`, physical `copies` of each book, 20 members and 52 loans (7 still open). We add a few tables of our own (`library_cards`, `member_profiles`, `library_staff`, `branches`, `vendors`, `vendor_supplies`).
:::

## DBMS Theory

:::theory[Binary relationship patterns and their relational mappings]
For a binary relationship R between entity types S and T, the standard mapping (Elmasri & Navathe's ER-to-relational algorithm) is:

- **1:1** — choose one side, say S (preferably the side with **total participation**), and add T's primary key to S as a foreign key, declared **unique**. Alternatives: merge S and T into one relation (if both participate totally), or create a separate relationship relation (the "cross-reference" approach) with a unique key on each FK.
- **1:N** — add the primary key of the "1" side as a foreign key to the "N" side. Relationship attributes go to the N side.
- **M:N** — create a new relation R whose attributes are the primary keys of S and T (as foreign keys) plus R's own attributes. The key of R is the combination of the two foreign keys (unless the relationship can repeat — then a discriminator such as a date is added to the key).

For a **recursive** relationship, S and T are the same entity type in two roles; the same three mappings apply with the foreign key referencing the same table.

For an **n-ary** relationship (n ≥ 3), create a relation with the primary keys of all participants. Its key is the combination of the foreign keys *of the participants on the "many" sides*: if a participant's role has max = 1 given the others, its foreign key is excluded from the key.
:::

### Recursive relationships

A recursive (unary) relationship relates entities of the same type in different **roles**: *supervisor* and *supervisee*, *course* and *prerequisite*, *part* and *component*. Its cardinality follows the same rules:

- **1:N recursive** — a hierarchy (each staff member has at most one supervisor). Mapped with a self-referencing foreign key column. Querying the whole tree requires recursion ([Recursive CTEs](/lessons/m04-recursive-ctes)).
- **M:N recursive** — a graph (a course has many prerequisites and is the prerequisite of many courses; `course_prereqs` in the university dataset). Mapped with a junction table whose two foreign keys reference the same table.

### Ternary relationships

A ternary relationship associates three entities *at once*: "vendor V supplies book B **to** branch Br". The crucial theoretical point is that a ternary relationship is **not**, in general, equivalent to three binary relationships (V–B, B–Br, V–Br). The binaries record which pairs ever co-occurred; they lose which *triples* occurred, and joining them back can produce **spurious tuples**. Formally, the ternary relation can be decomposed losslessly into its binary projections only if it satisfies a corresponding **join dependency** — the subject of 5NF ([4NF and 5NF](/lessons/m06-4nf-5nf)). You will see a concrete spurious tuple below.

## Why It Exists

Patterns exist because the same relationship can be stored in several wrong ways, and each wrong way fails in a characteristic manner:

| Wrong implementation | Failure |
|---|---|
| 1:1 with the FK but no `UNIQUE` | It is silently a 1:N; two members can share one card |
| 1:1 with FKs on *both* sides | Circular dependency; one side must be inserted with NULL then updated; the two FKs can disagree |
| 1:N with the FK on the "1" side | Repeating columns (`copy1_id`, `copy2_id`) or comma lists |
| M:N with a single FK column | Each book can have only one author |
| M:N junction without a key | Duplicate pairs; counts inflated |
| Recursive hierarchy without recursion in queries | "All reports of X" only goes one level deep |
| Ternary split into three binaries | Spurious combinations reported as facts |

Knowing the pattern means getting the structure right the first time — and recognizing these failures instantly in an inherited schema.

## Internal Working

:::internals[What InnoDB maintains for each pattern]
- **Every foreign key needs an index on the child columns** (InnoDB creates one if none exists whose leftmost columns match) and an index on the referenced parent columns (normally the parent's primary key). These indexes are what make the per-row FK checks cheap.
- **1:1 via `UNIQUE`** adds a unique secondary index on the FK column. That same index satisfies the FK's index requirement — no second index is created.
- **1:1 via shared primary key** needs no extra index at all: the child's primary key *is* the foreign key, and the clustered index serves the FK check. Both tables are clustered on the same key values, so joining them is a merge of two identically ordered B+ trees.
- **M:N junction tables** are clustered by their composite primary key, say `(book_id, author_id)`. Navigation from book to authors is a clustered range scan. Navigation from author to books needs the secondary index on `author_id`; InnoDB secondary index entries contain the primary key columns, so the index on `(author_id)` effectively stores `(author_id, book_id)` and is *covering* for pure navigation.
- **Self-referencing foreign keys** work exactly like any other FK. InnoDB checks them immediately for each row, so in a multi-row insert the parent row must be inserted before (or in the same statement ahead of) its children. Cascading actions on self-referencing FKs (`ON DELETE CASCADE` down a tree) are limited by InnoDB to a nesting depth of 15.
- **Referential actions** (`CASCADE`, `SET NULL`) are executed by InnoDB inside the same transaction as the triggering statement. Cascaded changes do **not** fire triggers on the child table.
:::

## MySQL Implementation

### 1:1 — foreign key plus UNIQUE

*"Each member may hold one library card; each card belongs to exactly one member."* The card side participates totally (a card cannot exist without a member), so the foreign key goes there and is `NOT NULL UNIQUE`:

```sql run
CREATE TABLE library_cards (
  card_no     CHAR(10)     NOT NULL PRIMARY KEY,
  member_id   INT UNSIGNED NOT NULL,
  issued_on   DATE         NOT NULL,
  CONSTRAINT uq_cards_member UNIQUE (member_id),
  CONSTRAINT fk_cards_member FOREIGN KEY (member_id) REFERENCES members (member_id)
);
INSERT INTO library_cards VALUES
  ('LC00000001', 1, '2024-01-10'),
  ('LC00000002', 2, '2024-01-12');
SELECT c.card_no, m.member_id, m.name
FROM library_cards AS c JOIN members AS m ON m.member_id = c.member_id
ORDER BY c.card_no;
```

The `UNIQUE` constraint is what makes it 1:1. A second card for member 1 is rejected:

```sql run error
INSERT INTO library_cards VALUES ('LC00000003', 1, '2024-06-01');
```

### 1:1 — shared primary key

When the second entity is an optional *extension* of the first (rarely used or sensitive attributes), the child can use the parent's key as its own primary key. *"Some members fill in an optional profile with reading preferences."*

```sql run
CREATE TABLE member_profiles (
  member_id        INT UNSIGNED NOT NULL PRIMARY KEY,
  favourite_genre  VARCHAR(40)  NULL,
  newsletter_optin BOOLEAN      NOT NULL DEFAULT FALSE,
  CONSTRAINT fk_profiles_member FOREIGN KEY (member_id) REFERENCES members (member_id)
    ON DELETE CASCADE
);
INSERT INTO member_profiles (member_id, favourite_genre, newsletter_optin)
VALUES (1, 'technology', TRUE), (3, 'fiction', FALSE);
SELECT m.member_id, m.name, p.favourite_genre, p.newsletter_optin
FROM members AS m LEFT JOIN member_profiles AS p ON p.member_id = m.member_id
WHERE m.member_id <= 4
ORDER BY m.member_id;
```

The primary key guarantees at most one profile per member; the foreign key guarantees the member exists. No extra index is needed.

### 1:N — the foreign key on the many side

`copies.book_id` is the textbook 1:N: `NOT NULL` (every copy is of some book — total participation), with a foreign key. The default referential action protects the parent:

```sql run error
DELETE FROM books WHERE book_id = 1;
```

Error 1451: book 1 is still referenced. The message names one violated constraint, `fk_book_authors_book` (the M:N junction refers to the book); the book's copies reference it through `fk_copies_book` as well and would equally block the delete — deleting it would orphan them. Whether that should be an error, a cascade, or a `SET NULL` is a business decision — you will compare the options under Alternative Approaches.

### M:N — the junction table

`book_authors` is the M:N relationship between books and authors, with a relationship attribute `author_order` (the position of the author on the cover):

```sql run
SELECT b.title,
       GROUP_CONCAT(a.name ORDER BY ba.author_order SEPARATOR ', ') AS authors,
       COUNT(*) AS author_count
FROM books AS b
JOIN book_authors AS ba ON ba.book_id = b.book_id
JOIN authors AS a       ON a.author_id = ba.author_id
GROUP BY b.book_id, b.title
HAVING author_count > 1
ORDER BY b.book_id;
```

And in the other direction — authors with more than one book, and the author with none:

```sql run
SELECT a.author_id, a.name, COUNT(ba.book_id) AS books
FROM authors AS a
LEFT JOIN book_authors AS ba ON ba.author_id = a.author_id
GROUP BY a.author_id, a.name
HAVING books <> 1
ORDER BY books DESC, a.author_id;
```

## Syntax Breakdown

| Pattern | DDL skeleton | Key decisions |
|---|---|---|
| 1:1 (FK + UNIQUE) | `t.s_id … NOT NULL UNIQUE, FOREIGN KEY (s_id) REFERENCES s (id)` | Put the FK on the side with total participation; `NULL` + `UNIQUE` if optional |
| 1:1 (shared PK) | `t.id … PRIMARY KEY, FOREIGN KEY (id) REFERENCES s (id)` | Child is an optional extension of the parent |
| 1:1 (merged) | All attributes in one table | Both sides total, same lifecycle, same access patterns |
| 1:N | `child.parent_id … [NOT] NULL, FOREIGN KEY (parent_id) REFERENCES parent (id) [ON DELETE …]` | NULL-ability = participation; referential action = lifecycle |
| M:N | `r (a_id, b_id, attrs…, PRIMARY KEY (a_id, b_id), KEY (b_id), FK a, FK b)` | Composite PK; reverse index; repeatable pairs need a third key column |
| Recursive 1:N | `t.parent_id … NULL, FOREIGN KEY (parent_id) REFERENCES t (id)` | Root has NULL; `ON DELETE SET NULL` or `RESTRICT` |
| Recursive M:N | `r (t1_id, t2_id, PRIMARY KEY (t1_id, t2_id), FK → t, FK → t)` | `CHECK (t1_id <> t2_id)` to forbid self-links (only without FK actions on those columns) |
| Ternary | `r (a_id, b_id, c_id, attrs…, PRIMARY KEY (…), FK × 3)` | Key = FKs of the "many" roles; add `UNIQUE` for each functional constraint |

**Referential actions** (`ON DELETE` / `ON UPDATE`):

| Action | Effect on child rows when the parent is deleted/updated | Typical use |
|---|---|---|
| `RESTRICT` / `NO ACTION` (default) | Statement fails (error 1451) | Children are independent facts that must not vanish silently (loans, invoices) |
| `CASCADE` | Children deleted / keys updated | Existence-dependent children (profile, order lines, weak entities) |
| `SET NULL` | FK set to NULL (column must be NULL-able) | Optional relationships (supervisor leaves; staff keep existing) |
| `SET DEFAULT` | Parsed, but rejected by InnoDB | — |

## Practical Examples

**1. The "at most one current" pattern.** A copy has many loans over time (1:N), but at most one loan that is still *open*. That is a 1:1 between Copy and *current* Loan, hidden inside a 1:N. MySQL cannot declare a partial unique index (`UNIQUE … WHERE returned_at IS NULL`, as in PostgreSQL), but a **generated column** that is non-NULL only for open loans, plus a `UNIQUE` index, has exactly the same effect because UNIQUE ignores NULLs:

```sql run
ALTER TABLE loans
  ADD COLUMN open_copy_id INT UNSIGNED AS (IF(returned_at IS NULL, copy_id, NULL)) STORED,
  ADD CONSTRAINT uq_loans_one_open_per_copy UNIQUE (open_copy_id);

SELECT loan_id, copy_id, member_id, loaned_at, returned_at, open_copy_id
FROM loans
WHERE returned_at IS NULL
ORDER BY loan_id;
```

The existing data already satisfies the rule (otherwise the `ALTER` would have failed with a duplicate-key error). Now try to lend an already-lent copy:

```sql run error
INSERT INTO loans (loan_id, copy_id, member_id, loaned_at, due_at)
SELECT 1000, copy_id, 19, '2024-12-01', '2024-12-22'
FROM loans WHERE returned_at IS NULL ORDER BY loan_id LIMIT 1;
```

Returned copies can be lent again freely, because their `open_copy_id` is NULL.

**2. A recursive 1:N relationship — a staff hierarchy.**

```sql run
CREATE TABLE library_staff (
  staff_id      INT UNSIGNED NOT NULL PRIMARY KEY,
  name          VARCHAR(80)  NOT NULL,
  role          VARCHAR(40)  NOT NULL,
  supervisor_id INT UNSIGNED NULL,
  CONSTRAINT fk_staff_supervisor FOREIGN KEY (supervisor_id)
    REFERENCES library_staff (staff_id) ON DELETE SET NULL
);
INSERT INTO library_staff VALUES
  (1, 'Ruth Okafor',   'Head Librarian',      NULL),
  (2, 'Tom Becker',    'Circulation Lead',    1),
  (3, 'Mei Tanaka',    'Cataloguing Lead',    1),
  (4, 'Ivan Petrov',   'Circulation Clerk',   2),
  (5, 'Sara Haddad',   'Circulation Clerk',   2),
  (6, 'Leo Martins',   'Cataloguer',          3);

WITH RECURSIVE chain AS (
  SELECT staff_id, name, supervisor_id, 0 AS depth, CAST(name AS CHAR(200)) AS path
  FROM library_staff WHERE supervisor_id IS NULL
  UNION ALL
  SELECT s.staff_id, s.name, s.supervisor_id, c.depth + 1, CONCAT(c.path, ' > ', s.name)
  FROM library_staff AS s JOIN chain AS c ON s.supervisor_id = c.staff_id
)
SELECT staff_id, depth, path FROM chain ORDER BY path;
```

One self-referencing column represents the whole tree; the recursive CTE walks it to any depth. The root has `supervisor_id` NULL — partial participation in the *supervisee* role.

**3. A MySQL-specific trap: CHECK plus a foreign-key action on the same column.** It is natural to add "nobody supervises themselves" as `CHECK (supervisor_id <> staff_id)`. On this table it fails:

```sql run error
ALTER TABLE library_staff
  ADD CONSTRAINT chk_staff_not_self CHECK (supervisor_id <> staff_id);
```

MySQL prohibits `CHECK` constraints on columns used in a foreign key's referential action (`ON DELETE SET NULL` here) and vice versa (error 3823). The options are: drop the action (use the default `RESTRICT` and reassign subordinates explicitly before deleting a supervisor), or enforce the rule with a trigger. On a recursive **M:N** table whose foreign keys have no referential actions, the CHECK works:

```sql run
CREATE TABLE book_relations (
  book_id         INT UNSIGNED NOT NULL,
  related_book_id INT UNSIGNED NOT NULL,
  relation        ENUM('sequel','same_series','recommended') NOT NULL,
  PRIMARY KEY (book_id, related_book_id),
  KEY idx_book_relations_related (related_book_id),
  CONSTRAINT chk_book_rel_not_self CHECK (book_id <> related_book_id),
  CONSTRAINT fk_book_rel_book    FOREIGN KEY (book_id)         REFERENCES books (book_id),
  CONSTRAINT fk_book_rel_related FOREIGN KEY (related_book_id) REFERENCES books (book_id)
);
INSERT INTO book_relations VALUES (8, 11, 'same_series'), (5, 6, 'recommended'), (10, 12, 'recommended');
SELECT b1.title AS book, r.relation, b2.title AS related_book
FROM book_relations AS r
JOIN books AS b1 ON b1.book_id = r.book_id
JOIN books AS b2 ON b2.book_id = r.related_book_id
ORDER BY r.book_id;
```

```sql run error
INSERT INTO book_relations VALUES (3, 3, 'recommended');
```

**4. A ternary relationship and why it cannot be split.** *"Vendors supply books to branches."* A row means: vendor V supplies book B to branch Br.

```sql run
CREATE TABLE branches (branch_id TINYINT UNSIGNED PRIMARY KEY, name VARCHAR(40) NOT NULL);
CREATE TABLE vendors  (vendor_id TINYINT UNSIGNED PRIMARY KEY, name VARCHAR(40) NOT NULL);
INSERT INTO branches VALUES (1, 'Central'), (2, 'Eastside');
INSERT INTO vendors  VALUES (1, 'BookHub'), (2, 'PaperTrail');

CREATE TABLE vendor_supplies (
  vendor_id  TINYINT UNSIGNED NOT NULL,
  book_id    INT UNSIGNED     NOT NULL,
  branch_id  TINYINT UNSIGNED NOT NULL,
  unit_cost  DECIMAL(8,2)     NOT NULL,
  PRIMARY KEY (vendor_id, book_id, branch_id),
  KEY idx_supplies_book (book_id),
  KEY idx_supplies_branch (branch_id),
  FOREIGN KEY (vendor_id) REFERENCES vendors (vendor_id),
  FOREIGN KEY (book_id)   REFERENCES books (book_id),
  FOREIGN KEY (branch_id) REFERENCES branches (branch_id)
);
INSERT INTO vendor_supplies VALUES
  (1, 1, 2, 41.00),   -- BookHub supplies Database Internals to Eastside
  (1, 2, 1, 38.50),   -- BookHub supplies DDIA to Central
  (2, 1, 1, 43.25);   -- PaperTrail supplies Database Internals to Central
SELECT vendor_id, book_id, branch_id FROM vendor_supplies ORDER BY vendor_id, book_id, branch_id;
```

Now suppose a designer had stored three binary relationships instead — "vendor supplies book", "book is stocked at branch", "vendor serves branch" — the three projections of the ternary — and later reconstructs the triples by joining them:

```sql run
WITH vb  AS (SELECT DISTINCT vendor_id, book_id   FROM vendor_supplies),
     bbr AS (SELECT DISTINCT book_id,   branch_id FROM vendor_supplies),
     vbr AS (SELECT DISTINCT vendor_id, branch_id FROM vendor_supplies)
SELECT vb.vendor_id, vb.book_id, bbr.branch_id,
       EXISTS (SELECT 1 FROM vendor_supplies AS s
               WHERE s.vendor_id = vb.vendor_id AND s.book_id = vb.book_id
                 AND s.branch_id = bbr.branch_id) AS is_real_fact
FROM vb
JOIN bbr ON bbr.book_id = vb.book_id
JOIN vbr ON vbr.vendor_id = vb.vendor_id AND vbr.branch_id = bbr.branch_id
ORDER BY vb.vendor_id, vb.book_id, bbr.branch_id;
```

The join returns a triple that never happened — (BookHub, Database Internals, Central) — because BookHub *does* supply Database Internals (to Eastside), Database Internals *is* stocked at Central (from PaperTrail) and BookHub *does* serve Central (with DDIA). Every pair is true; the triple is false. The binaries lost information. That is why a genuine ternary relationship must be stored as one table.

## Execution Walkthrough

Follow a referential action through a recursive relationship: what InnoDB does when a supervisor leaves.

```walkthrough
title: DELETE a supervisor with ON DELETE SET NULL on a self-referencing FK
steps:
  - title: "1. The tree before the delete"
    explain: Tom Becker (2) supervises Ivan (4) and Sara (5). He reports to Ruth (1).
    sql: SELECT staff_id, name, supervisor_id FROM library_staff ORDER BY staff_id;
  - title: "2. Delete the supervisor"
    explain: >-
      InnoDB deletes row 2, then — inside the same statement and transaction — finds the child rows through the
      index on supervisor_id (created automatically for the foreign key) and sets their supervisor_id to NULL.
    sql: DELETE FROM library_staff WHERE staff_id = 2;
  - title: "3. The orphans are now roots"
    explain: >-
      Ivan and Sara have NULL supervisors. Structurally valid (participation in the supervisee role is partial),
      but the business probably wants them reassigned — SET NULL records "unknown supervisor", it does not choose
      a new one.
    sql: SELECT staff_id, name, supervisor_id FROM library_staff ORDER BY staff_id;
  - title: "4. Reassign explicitly"
    explain: >-
      The business rule "orphaned staff report to the head librarian" is applied by an ordinary UPDATE. In
      production this delete-and-reassign would be one transaction, or the delete would be RESTRICTed until the
      reassignment is done.
    sql: |
      UPDATE library_staff SET supervisor_id = 1 WHERE supervisor_id IS NULL AND staff_id <> 1;
      SELECT staff_id, name, supervisor_id FROM library_staff ORDER BY staff_id;
```

## Output

After the walkthrough the hierarchy is valid again: every staff member except the head librarian has a supervisor, and the tree is two levels deep. Compare the three states — before, after the cascade action, after the reassignment — and note what the database guaranteed (no reference to a deleted staff member ever existed) versus what the business rule required (a sensible new supervisor), which only the explicit `UPDATE` provided.

The ternary example's output is just as important: the reconstruction query returned three real triples plus one spurious triple, flagged by `is_real_fact = 0`. Whenever you see three binary relationships among the same three entities in a design, ask whether the business ever needs the *triple*; if it does, the binaries are not enough.

## Alternative Approaches

**1:1 placement options compared:**

| Option | Pros | Cons | Choose when |
|---|---|---|---|
| FK + UNIQUE on the total side | No NULLs; clear ownership | Extra unique index | Most 1:1s (card → member) |
| FK + UNIQUE, NULL-able | Works when both sides are partial | NULLs; UNIQUE allows many NULLs (fine) | Optional 1:1 (an employee *may* have a parking space) |
| Shared primary key | No extra index; trivially 1:1; aligned clustering | Child cannot exist independently; key must be the same type | Optional extension / vertical partition of one entity |
| Merge into one table | No join | NULL-heavy if one side is optional; mixes lifecycles | Both sides total and always read together |
| Separate relationship table | Neither entity changes; relationship history possible | Two UNIQUE constraints; an extra join | Relationship managed independently, or with history |

**M:N junction keys — composite vs surrogate:**

| Key | Behaviour |
|---|---|
| `PRIMARY KEY (book_id, author_id)` | Enforces "no duplicate pair" by itself; clustered by book; compact |
| Surrogate `id` + `UNIQUE (book_id, author_id)` | Useful when other tables must reference a single relationship instance (e.g. royalties per book–author) or when an ORM insists on single-column keys; costs an extra index |
| Surrogate `id` without the UNIQUE | Wrong: allows duplicate pairs |
| `PRIMARY KEY (book_id, author_id, valid_from)` | The pair can repeat over time (history) |

**Recursive hierarchies** can also be modelled with a *closure table* (all ancestor–descendant pairs), *materialized path* (`'/1/2/4/'`) or *nested sets*. They trade write complexity for faster subtree reads; the adjacency list (self-FK) plus recursive CTEs is the default in MySQL 8 (Module 13, [Anti-Patterns](/lessons/m13-anti-patterns) discusses the trade-offs).

**Referential actions on the 1:N copy–book relationship:** `RESTRICT` (the dataset's choice) forces staff to deal with copies before deleting a book; `CASCADE` would silently delete copies — and then fail on their loans, which also restrict. For records with legal or financial meaning, prefer `RESTRICT` plus soft deletion (an `is_withdrawn` flag) to physical cascades.

## Performance Analysis

The patterns differ mostly in which navigation directions are cheap:

```sql run
EXPLAIN FORMAT=TREE
SELECT book_id FROM book_authors WHERE author_id = 13;
```

The reverse navigation (author → books) is a *covering* lookup on `idx_book_authors_author`: the secondary index holds `author_id` plus the primary key `(book_id, author_id)`, so the clustered index is never touched. Without that index, the query would scan the whole junction table — and, separately, InnoDB would have created an index on `author_id` anyway to support the foreign key.

| Pattern | Cheap direction | Other direction | Watch out for |
|---|---|---|---|
| 1:1 FK + UNIQUE | Both (PK one way, UNIQUE index the other) | — | — |
| 1:1 shared PK | Both via primary keys | — | — |
| 1:N | Child → parent (PK lookup) | Parent → children via FK index | Unindexed FK columns in other DBMSs; InnoDB always indexes them |
| M:N | Leading PK column direction (clustered range) | Reverse via secondary index (covering for IDs) | Missing reverse index on hand-made junction tables without an FK |
| Recursive 1:N | One level via FK index | Whole subtree needs a recursive CTE: one index lookup per node | Deep or wide trees; set `cte_max_recursion_depth` appropriately |
| Ternary | Leading PK column | Each other FK column needs its own index | Choosing the PK column order for the most frequent access path |

The generated-column "one open loan per copy" pattern costs a `STORED` column (4 bytes, or just the NULL bit when NULL) and a unique secondary index. Unlike a PostgreSQL partial index, that index does **not** contain only the open loans: InnoDB stores NULL keys in secondary indexes too, so it has one entry per loan, of which only the open ones are non-NULL. On a loans table with millions of historical rows, that index is as large as any other single-column index.

:::performance[Theory vs measurement]
The plan shapes here are what InnoDB's structures imply. On 52 loans or 17 junction rows every option is instantaneous; measure on realistic volumes with `EXPLAIN ANALYZE` before choosing, for example, a closure table over a recursive CTE.
:::

## Best Practices

:::best-practice[Relationship pattern checklist]
1. **1:1** — put the FK on the side with total participation and declare it `UNIQUE`; use a shared primary key for optional extensions; merge only if both sides are total and always read together.
2. **1:N** — FK on the N side; `NOT NULL` iff total participation; choose the referential action from the business lifecycle, defaulting to `RESTRICT` for records with independent meaning.
3. **M:N** — junction table with a composite primary key and an index leading with the other FK; relationship attributes on the junction table; add a time column to the key if pairs can repeat.
4. **Recursive** — self-FK for trees, junction to the same table for graphs; guard against self-links (CHECK when allowed, otherwise trigger) and design for cycles in graph traversals.
5. **Ternary** — keep it as one table when the triple is the fact; derive the key from the cardinalities; never "simplify" into three binaries without checking the join dependency.
6. **"At most one current"** — generated column + `UNIQUE` in MySQL (the substitute for a partial unique index).
7. **Name constraints after the relationship** (`fk_copies_book`, `uq_cards_member`) for readable errors.
:::

## Common Mistakes

**Mistake 1 — a 1:1 that is really 1:N.** Without `UNIQUE`, nothing stops a second card for the same member:

```sql run
CREATE TABLE library_cards_bad (
  card_no   CHAR(10) PRIMARY KEY,
  member_id INT UNSIGNED NOT NULL,
  FOREIGN KEY (member_id) REFERENCES members (member_id)
);
INSERT INTO library_cards_bad VALUES ('X1', 5), ('X2', 5);
SELECT member_id, COUNT(*) AS cards FROM library_cards_bad GROUP BY member_id;
```

The cardinality you drew (1:1) is not the cardinality you built (1:N). Fix: `UNIQUE (member_id)`.

**Mistake 2 — foreign keys on both sides of a 1:1.** `members.card_no` *and* `library_cards.member_id` create a circular dependency: neither row can be inserted first without a NULL, and the two references can disagree. Store the relationship once.

**Mistake 3 — a junction table without a primary key.** Duplicate pairs inflate every count. Always `PRIMARY KEY (a_id, b_id)`.

**Mistake 4 — splitting a ternary into binaries.** Shown above: the reconstruction invents facts. Conversely, do not *force* a ternary when the business only ever needs pairs — a ternary that is really two independent binaries (e.g. "book has authors" and "book is stocked at branches") causes redundancy instead.

**Mistake 5 — CHECK on a column with an FK action.** Error 3823 surprises people modelling hierarchies; decide between the referential action and the CHECK, or use a trigger.

**Mistake 6 — one-level hierarchy queries.** `SELECT … WHERE supervisor_id = 1` returns direct reports only. "Everyone under Ruth" requires a recursive CTE.

## Industry Application

- **Identity systems** use the shared-primary-key 1:1 to separate hot, small `users` rows from large or sensitive `user_profiles` / `user_credentials` — different access patterns, different privacy controls, same key.
- **E-commerce and content platforms** are built on M:N junction tables: products–categories, posts–tags, users–roles. At scale, the reverse-direction index and the junction PK order are deliberate performance decisions.
- **Organizational charts, comment threads, category trees and bills of materials** are recursive relationships. MySQL 8 teams use adjacency lists with recursive CTEs; earlier MySQL versions (no CTEs) pushed teams toward closure tables and nested sets, which you will still meet in legacy schemas.
- **Pricing, logistics and scheduling** are full of ternaries: supplier–part–warehouse prices, instructor–course–term assignments, doctor–patient–procedure appointments. Mis-modelling them as binaries produces the exact spurious combinations shown above — often as wrong prices.
- **"Exactly one active" rules** — one active subscription per account, one default address per customer, one open shift per employee — are implemented with the generated-column-plus-UNIQUE pattern in MySQL, or with partial unique indexes in PostgreSQL.

## Practice Exercises

```exercise
id: ex1
title: Navigate the M:N relationship
difficulty: easy
concepts: [many-to-many, junction-tables]
prompt: |
  List every author together with the titles of their books, one row per (author, book). Columns: `name`,
  `title`, ordered by `name`, then `title`. Authors without books must not appear.
hints:
  - "Join authors → book_authors → books."
  - "Inner joins drop the author without books, which is what the prompt asks."
solution: |
  SELECT a.name, b.title
  FROM authors AS a
  JOIN book_authors AS ba ON ba.author_id = a.author_id
  JOIN books AS b ON b.book_id = ba.book_id
  ORDER BY a.name, b.title;
orderMatters: true
explanation: |
  An M:N relationship is always navigated through the junction table: two joins, each along a max = 1 role
  from the junction row's point of view, so no row multiplication beyond one row per relationship instance.
```

```exercise
id: ex2
title: Implement a 1:1 with the foreign key on the correct side
difficulty: medium
kind: dml
concepts: [one-to-one, foreign-key]
prompt: |
  Each **locker** in the library can be rented by at most one member, and a member can rent at most one
  locker; many members rent none, and some lockers are free.

  Create table **`lockers`** with `locker_no` (`SMALLINT UNSIGNED`, primary key) and `member_id`
  (`INT UNSIGNED`, NULL-able, references `members`). Enforce the 1:1 cardinality declaratively.

  The checker lists your table's constraint types.
hints:
  - "Both sides are partial, so the FK column must be NULL-able."
  - "UNIQUE (member_id) turns the 1:N foreign key into a 1:1 — and UNIQUE allows many NULLs (free lockers)."
solution: |
  CREATE TABLE lockers (
    locker_no SMALLINT UNSIGNED NOT NULL PRIMARY KEY,
    member_id INT UNSIGNED NULL,
    CONSTRAINT uq_lockers_member UNIQUE (member_id),
    CONSTRAINT fk_lockers_member FOREIGN KEY (member_id) REFERENCES members (member_id)
  );
check: |
  SELECT CONSTRAINT_TYPE
  FROM information_schema.TABLE_CONSTRAINTS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'lockers'
  ORDER BY CONSTRAINT_TYPE;
explanation: |
  The `UNIQUE` index on a NULL-able FK is the standard partial 1:1: free lockers have NULL (any number allowed),
  rented lockers have distinct members. Putting a `locker_no` column on `members` would also work, but adds a
  column that is NULL for most members.
```

```exercise
id: ex3
title: A ternary relationship with a functional constraint
difficulty: hard
kind: dml
dataset: library
setup: |
  CREATE TABLE branches (branch_id TINYINT UNSIGNED PRIMARY KEY, name VARCHAR(40) NOT NULL);
  CREATE TABLE vendors  (vendor_id TINYINT UNSIGNED PRIMARY KEY, name VARCHAR(40) NOT NULL);
  INSERT INTO branches VALUES (1, 'Central'), (2, 'Eastside');
  INSERT INTO vendors  VALUES (1, 'BookHub'), (2, 'PaperTrail');
concepts: [ternary-relationships, composite-key]
prompt: |
  New rule: *for each (book, branch) there is exactly one supplying vendor* — but a vendor may supply many
  books to many branches. Create table **`book_sourcing`** with `book_id` (`INT UNSIGNED`), `branch_id`
  (`TINYINT UNSIGNED`), `vendor_id` (`TINYINT UNSIGNED`), all NOT NULL, with foreign keys to `books`,
  `branches` and `vendors`, and a primary key that enforces the rule.

  The checker lists the columns of your primary key in order.
hints:
  - "If the vendor is determined by (book, branch), the vendor role has max = 1 given the other two."
  - "The key is therefore (book_id, branch_id); vendor_id is a non-key column with a foreign key."
solution: |
  CREATE TABLE book_sourcing (
    book_id   INT UNSIGNED     NOT NULL,
    branch_id TINYINT UNSIGNED NOT NULL,
    vendor_id TINYINT UNSIGNED NOT NULL,
    PRIMARY KEY (book_id, branch_id),
    FOREIGN KEY (book_id)   REFERENCES books (book_id),
    FOREIGN KEY (branch_id) REFERENCES branches (branch_id),
    FOREIGN KEY (vendor_id) REFERENCES vendors (vendor_id)
  );
check: |
  SELECT COLUMN_NAME, ORDINAL_POSITION
  FROM information_schema.KEY_COLUMN_USAGE
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'book_sourcing' AND CONSTRAINT_NAME = 'PRIMARY'
  ORDER BY ORDINAL_POSITION;
orderMatters: true
explanation: |
  In an n-ary relationship, a participant whose role has max = 1 given the others is excluded from the key.
  `PRIMARY KEY (book_id, branch_id, vendor_id)` would allow two vendors for the same book and branch — the
  rule would be lost. `(branch_id, book_id)` would enforce the same rule; the checker expects book first,
  matching the rule's wording, and the order also decides which access path is clustered.
```

```exercise
id: ex4
title: Everyone under the head librarian
difficulty: hard
dataset: library
setup: |
  CREATE TABLE library_staff (
    staff_id INT UNSIGNED PRIMARY KEY, name VARCHAR(80) NOT NULL,
    supervisor_id INT UNSIGNED NULL,
    FOREIGN KEY (supervisor_id) REFERENCES library_staff (staff_id)
  );
  INSERT INTO library_staff VALUES (1,'Ruth',NULL),(2,'Tom',1),(3,'Mei',1),(4,'Ivan',2),(5,'Sara',2),(6,'Leo',3),(7,'Ana',6);
concepts: [recursive-relationships, recursive-cte]
prompt: |
  The table `library_staff (staff_id, name, supervisor_id)` holds a hierarchy. Return every staff member
  **below** Mei (staff_id 3) at any depth, with columns `staff_id`, `name`, `levels_below_mei`
  (1 for direct reports), ordered by `staff_id`.
hints:
  - "Anchor: the direct reports of 3 (depth 1). Recursive step: staff whose supervisor is in the CTE (depth + 1)."
  - "WITH RECURSIVE sub AS (SELECT staff_id, name, 1 AS d FROM library_staff WHERE supervisor_id = 3 UNION ALL …)"
solution: |
  WITH RECURSIVE sub AS (
    SELECT staff_id, name, 1 AS levels_below_mei
    FROM library_staff WHERE supervisor_id = 3
    UNION ALL
    SELECT s.staff_id, s.name, sub.levels_below_mei + 1
    FROM library_staff AS s JOIN sub ON s.supervisor_id = sub.staff_id
  )
  SELECT staff_id, name, levels_below_mei FROM sub ORDER BY staff_id;
orderMatters: true
explanation: |
  A 1:N recursive relationship stores only one level per row; a recursive CTE composes those levels. A plain
  `WHERE supervisor_id = 3` would return Leo only and miss Ana, who reports to Leo.
```

```exercise
id: ex5
title: Ternary or binaries?
difficulty: medium
kind: open
concepts: [ternary-relationships, many-to-many]
prompt: |
  For each scenario decide whether you need one ternary relationship or two/three binary relationships, and
  justify with a sentence about which facts must be recorded: (a) instructors teach courses in terms;
  (b) books have authors, and books are stocked at branches; (c) a doctor prescribes a drug to a patient;
  (d) employees have skills, and employees work on projects (skills are not tied to projects).
solution: |
  (a) Ternary (or an entity *Section* that represents it): the fact is "instructor I teaches course C in
  term T"; knowing that I teaches C (sometime) and teaches in T (some course) does not tell you which course
  in which term.
  (b) Two binaries: author-of and stocked-at are independent facts about a book; a ternary
  (author, book, branch) would repeat every author for every branch — redundancy (a multivalued dependency,
  4NF).
  (c) Ternary: a prescription relates all three at once and has attributes (dose, date); the pairs
  doctor–drug, doctor–patient, patient–drug cannot reconstruct who prescribed what to whom.
  (d) Two binaries: skills and project assignments are independent; storing them together would create the
  employee × skills × projects cross product per employee.
```

```exercise
id: ex6
title: Design review of a relationship-heavy schema
difficulty: challenge
kind: open
concepts: [one-to-one, one-to-many, many-to-many, recursive-relationships, ternary-relationships, referential-actions]
prompt: |
  A teammate proposes this schema for a conference system. Identify every relationship pattern, find the
  defects, and give corrected DDL fragments.

  ~~~sql
  CREATE TABLE attendees (attendee_id INT PRIMARY KEY, name VARCHAR(80), badge_id INT,
                          mentor_id INT, talks_attended VARCHAR(500));
  CREATE TABLE badges (badge_id INT PRIMARY KEY, attendee_id INT,
                       FOREIGN KEY (attendee_id) REFERENCES attendees (attendee_id));
  CREATE TABLE talks (talk_id INT PRIMARY KEY, title VARCHAR(200), speaker_id INT, room_id INT, slot_id INT);
  CREATE TABLE speaker_rooms (speaker_id INT, room_id INT);
  CREATE TABLE speaker_slots (speaker_id INT, slot_id INT);
  ~~~
solution: |
  **Attendee–Badge (1:1)** is stored twice (`attendees.badge_id` and `badges.attendee_id`) — circular and
  able to disagree; neither side is UNIQUE, so it is actually 1:N. Keep one: `badges.attendee_id INT NOT
  NULL UNIQUE` with the FK (the badge cannot exist without an attendee) and drop `attendees.badge_id`.

  **Mentor (recursive 1:N)** — `mentor_id` has no FK. Add `FOREIGN KEY (mentor_id) REFERENCES attendees
  (attendee_id) ON DELETE SET NULL` (and then no CHECK on mentor_id; use a trigger for "not self" or drop
  the action).

  **Attendee–Talk (M:N)** stored as a comma list `talks_attended`. Replace with
  `talk_attendance (attendee_id, talk_id, PRIMARY KEY (attendee_id, talk_id), KEY (talk_id), FK, FK)`.

  **Talk–Speaker / Room / Slot** — `talks.speaker_id`, `room_id`, `slot_id` lack FKs and NOT NULL. If a talk
  can have several speakers, speaker becomes an M:N junction `talk_speakers`.

  **speaker_rooms + speaker_slots** — two binaries that try to record "speaker S is in room R at slot T":
  a ternary split into binaries, which produces spurious combinations. The fact is already recorded by the
  talk (talk → room, slot; talk → speakers). Drop both tables; if the room/slot schedule is a separate
  concept, model it as a ternary/entity `schedule (room_id, slot_id, talk_id, PRIMARY KEY (room_id,
  slot_id), UNIQUE (talk_id))` — one talk per room per slot, one slot per talk.

  General: no table has keys on its junctions, no NOT NULL on mandatory columns, and no naming of
  constraints; add them all.
```

## Knowledge Assessment

```quiz
- id: q1
  question: "For a 1:1 relationship where Card participates totally and Member partially, where should the foreign key go?"
  options:
    - "On library_cards (member_id NOT NULL UNIQUE)"
    - "On members (card_no NULL UNIQUE)"
    - "On both tables"
    - "In a separate table without constraints"
  answer: 0
  explanation: "Placing the FK on the total side avoids NULLs and expresses the existence dependency. On members it would be NULL for every member without a card. FKs on both sides create a circular dependency that can disagree. A separate table without UNIQUE constraints would not be 1:1."
  concepts: [one-to-one]
- id: q2
  question: "What makes `FOREIGN KEY (member_id) REFERENCES members (member_id)` on library_cards a 1:1 instead of a 1:N?"
  options: ["A UNIQUE constraint on member_id", "NOT NULL on member_id", "ON DELETE CASCADE", "An index on card_no"]
  answer: 0
  explanation: "UNIQUE limits each member to one card (max = 1 on both sides). NOT NULL only makes participation total; CASCADE is a lifecycle rule; card_no is already the primary key."
  concepts: [one-to-one, foreign-key]
- id: q3
  question: "Joining the three binary projections of a ternary relation (V,B), (B,Br), (V,Br) produced a triple not in the original. This shows:"
  options:
    - "The ternary relation does not satisfy the join dependency needed for lossless decomposition into those binaries"
    - "The join was written incorrectly"
    - "Ternary relationships are always redundant"
    - "MySQL does not support three-way joins"
  answer: 0
  explanation: "Each pair in the spurious triple is true, but the triple is not — the binaries lost information. Lossless binary decomposition requires a join dependency (5NF topic). The join is correct; ternaries are not always redundant; MySQL handles multi-way joins fine."
  concepts: [ternary-relationships]
- id: q4
  question: "MySQL rejects `CHECK (supervisor_id <> staff_id)` on a table whose FK on supervisor_id has `ON DELETE SET NULL`. Why?"
  options:
    - "MySQL forbids CHECK constraints on columns used in foreign-key referential actions (error 3823)"
    - "CHECK constraints cannot compare two columns"
    - "Self-referencing foreign keys are not supported"
    - "SET NULL is not supported by InnoDB"
  answer: 0
  explanation: "This is a documented MySQL restriction. CHECK can compare columns of the same row; self-referencing FKs and SET NULL are supported. Remove the action or enforce the rule with a trigger."
  concepts: [recursive-relationships, referential-actions]
- id: q5
  type: multi
  question: "Which techniques correctly enforce \"a copy has at most one open loan\" in MySQL 8.4? (Select all that apply.)"
  options:
    - "A generated column IF(returned_at IS NULL, copy_id, NULL) with a UNIQUE index"
    - "A BEFORE INSERT/UPDATE trigger that locks the copy row and checks for an open loan"
    - "CREATE UNIQUE INDEX … ON loans (copy_id) WHERE returned_at IS NULL"
    - "UNIQUE (copy_id) on loans"
  answer: [0, 1]
  explanation: "The generated column makes non-open rows NULL, and UNIQUE ignores NULLs; a properly locking trigger also works. MySQL has no partial (filtered) indexes — that syntax is PostgreSQL. UNIQUE (copy_id) would forbid any second loan ever."
  concepts: [one-to-one, one-to-many]
- id: q6
  question: "In a ternary relationship *vendor supplies book to branch*, the rule \"each (book, branch) has one vendor\" means the primary key should be:"
  options: ["(book_id, branch_id)", "(vendor_id, book_id, branch_id)", "(vendor_id)", "(vendor_id, branch_id)"]
  answer: 0
  explanation: "The vendor is functionally determined by (book, branch), so it is excluded from the key. A key over all three would allow two vendors per (book, branch); the other keys forbid legitimate combinations."
  concepts: [ternary-relationships, composite-key]
- id: q7
  question: "`book_authors` has PRIMARY KEY (book_id, author_id) and an index on author_id. `SELECT book_id FROM book_authors WHERE author_id = 13` uses:"
  options:
    - "A covering lookup on the author_id index, because secondary index entries also contain the primary key columns"
    - "A full table scan"
    - "The clustered index range on book_id"
    - "A temporary table"
  answer: 0
  explanation: "InnoDB secondary indexes store the PK columns, so (author_id) effectively stores (author_id, book_id) — everything the query needs. The clustered index is ordered by book_id first and cannot be range-scanned by author_id."
  concepts: [junction-tables, many-to-many]
- id: q8
  question: "Which query structure is needed to list every staff member below a manager at any depth in an adjacency-list hierarchy?"
  options: ["A recursive CTE", "A single self-join", "GROUP BY supervisor_id", "A UNIQUE index on supervisor_id"]
  answer: 0
  explanation: "Each row stores only its direct supervisor; arbitrary depth requires recursion. A self-join goes one level; GROUP BY counts direct reports; UNIQUE on supervisor_id would forbid a manager from having two reports."
  concepts: [recursive-relationships]
```
