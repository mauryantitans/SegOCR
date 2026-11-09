---
id: m06-4nf-5nf
dataset: empty
summary: Beyond functional dependencies — multivalued dependencies and fourth normal form, join dependencies and fifth normal form. Both are demonstrated on real MySQL tables, with SQL tests for MVDs and JDs, spurious tuples from every binary decomposition of a 5NF example, and an honest look at how often these forms matter in practice.
concepts: [multivalued-dependency, fourth-normal-form, join-dependency, fifth-normal-form, lossless-join, independent-multivalued-facts]
references:
  - title: "MySQL 8.4 Reference Manual — EXCEPT Clause"
    url: https://dev.mysql.com/doc/refman/8.4/en/except.html
  - title: "MySQL 8.4 Reference Manual — CREATE VIEW Statement"
    url: https://dev.mysql.com/doc/refman/8.4/en/create-view.html
  - title: "R. Fagin — Multivalued Dependencies and a New Normal Form for Relational Databases (ACM TODS, 1977)"
    url: https://dl.acm.org/doi/10.1145/320557.320571
  - title: "W. Kent — A Simple Guide to Five Normal Forms in Relational Database Theory (CACM, 1983)"
    url: https://dl.acm.org/doi/10.1145/358024.358054
related: [m06-bcnf, m06-decomposition-properties, m06-1nf, m05-relationship-patterns]
---

## Introduction

Every normal form so far has been defined by **functional dependencies**: "X determines one value of Y". Some redundancy has nothing to do with single values. Consider a course catalogue that records, for each course, the instructors who may teach it and the textbooks it uses — two **independent** lists:

| course_id | instructor | textbook |
|---|---|---|
| CS-157A | Kim | Silberschatz |
| CS-157A | Kim | Elmasri |
| CS-157A | Rossi | Silberschatz |
| CS-157A | Rossi | Elmasri |

There are no non-trivial FDs at all (a course has several instructors and several textbooks), the only key is all three columns, and the table is therefore in **BCNF**. Yet it is obviously redundant: each instructor is listed once per textbook and each textbook once per instructor. Adding a third textbook requires **two** new rows; forgetting one makes the table claim that Rossi does not use the new book — a statement nobody intended.

This redundancy is captured by a **multivalued dependency (MVD)** `course_id ↠ instructor`, and removing it is the job of **fourth normal form (4NF)**. A still more general constraint, the **join dependency (JD)**, defines **fifth normal form (5NF)**: a relation that can be reconstructed from three or more projections but *not* from any two.

This lesson demonstrates both in MySQL — the redundancy and anomalies of a 4NF violation, an SQL test for MVDs, the 4NF decomposition, and a 5NF example where every two-way decomposition produces spurious rows while the three-way join is exact — and then discusses honestly how often you will meet them.

:::note[Dataset]
This lesson uses the **empty** schema: every table is created in your lab, so the data can be chosen to illustrate exactly one dependency at a time. Every output was captured on a real MySQL server.
:::

## DBMS Theory

:::theory[Multivalued dependency and 4NF]
Let X, Y ⊆ R and Z = R − X − Y. The **multivalued dependency X ↠ Y** holds on R if, for every legal instance and every two tuples t₁, t₂ with t₁[X] = t₂[X], there is a tuple t₃ with

> t₃[X] = t₁[X], t₃[Y] = t₁[Y], t₃[Z] = t₂[Z]

— i.e. for each X value, the set of Y values and the set of Z values are **independent**: every Y value appears in combination with every Z value. (Fagin 1977; Silberschatz et al. §7.6; Elmasri & Navathe §14.6.)

An MVD is **trivial** if Y ⊆ X or X ∪ Y = R. R is in **fourth normal form (4NF)** if for every non-trivial MVD X ↠ Y in D⁺, **X is a superkey** of R.
:::

Key properties:

- **Every FD is an MVD:** if X → Y then X ↠ Y (one Y value is trivially "independent" of everything). Hence 4NF ⇒ BCNF.
- **Complementation:** X ↠ Y implies X ↠ R − X − Y. MVDs come in pairs — `course ↠ instructor` and `course ↠ textbook` are the same fact.
- **Fagin's theorem:** R decomposes losslessly into (X ∪ Y) and (X ∪ Z) **if and only if** X ↠ Y holds. This makes MVDs the exact condition for a lossless binary decomposition — the deepest reason they matter.
- **Right sides cannot be split.** Unlike FDs, X ↠ YW does not imply X ↠ Y.

**4NF decomposition:** while some relation has a non-trivial MVD X ↠ Y with X not a superkey, replace it by (X ∪ Y) and (R − Y). Like BCNF decomposition, it is lossless but not always dependency-preserving.

:::theory[Join dependency and 5NF]
A **join dependency** \*(R₁, R₂, …, Rₙ), with R₁ ∪ … ∪ Rₙ = R, holds on R if every legal instance r equals the join of its projections: r = π_R₁(r) ⋈ π_R₂(r) ⋈ … ⋈ π_Rₙ(r). An MVD X ↠ Y is exactly the binary JD \*(XY, XZ).

R is in **fifth normal form (5NF)**, also called **project-join normal form (PJ/NF)**, if every non-trivial JD that holds on R is **implied by the candidate keys** of R. (Fagin 1979; Elmasri & Navathe §14.7.) A JD is trivial if some Rᵢ = R.
:::

The relationship between the forms: **5NF ⇒ 4NF ⇒ BCNF ⇒ 3NF ⇒ 2NF ⇒ 1NF**. Each stronger form considers a more general kind of dependency: FDs (up to BCNF), MVDs (4NF), JDs (5NF).

## Why It Exists

**4NF** exists because the FD-based forms are blind to *independent multi-valued facts*. When a table combines two independent lists about the same entity, BCNF is satisfied — no FD is violated — but:

- **Insertion anomaly:** a new textbook for a course must be inserted once for every instructor (and a new instructor once for every textbook). A course with textbooks but no instructor yet cannot be stored at all without NULLs.
- **Deletion anomaly:** removing the last instructor deletes the textbook list.
- **Update anomaly:** replacing a textbook edition touches one row per instructor.
- **Semantic ambiguity:** if rows are missing, does the table say "Rossi does not use Elmasri" (a dependent, three-way fact) or is the data just incomplete?

**5NF** exists for a rarer but real situation: a three-way relationship that is fully determined by its three two-way projections because of a **cyclic business rule** ("if a supplier supplies a part, and a project uses that part, and the supplier supplies that project at all, then the supplier supplies that part to that project"). Storing the three-way table repeats information implied by the rule; storing the pairs instead removes the redundancy — but only the **three-way** join reconstructs the data. No two-way split can.

## Internal Working

:::internals[Why MVD redundancy grows multiplicatively]
For a course with *i* instructors and *t* textbooks, the combined table must store **i × t** rows; the decomposed tables store **i + t** rows. For 5 instructors and 6 textbooks that is 30 rows versus 11, each with its own clustered-index entry, undo/redo on change, and secondary-index entries. Every insertion of a new textbook is *i* row inserts, executed (and locked) together if the application is correct, or partially if it is not.

The DBMS cannot enforce an MVD: MySQL, like other SQL systems, has no declaration for it, and no index expresses "every Y combines with every Z". An MVD can only be **checked** after the fact (with the queries below) or **made unnecessary** by decomposition. The same applies to join dependencies.
:::

How the engine sees the decomposed design: two narrow tables, each with a composite primary key `(course_id, instructor)` and `(course_id, textbook)`. A query that needs the combinations joins them on `course_id` — the join *recreates* the cross product per course on demand, which is exactly what the MVD says is the meaning of the data.

```mermaid
flowchart LR
  subgraph Before["course_info: i × t rows per course"]
    B1["(CS-157A, Kim, Silberschatz)"]
    B2["(CS-157A, Kim, Elmasri)"]
    B3["(CS-157A, Rossi, Silberschatz)"]
    B4["(CS-157A, Rossi, Elmasri)"]
  end
  subgraph After["4NF: i + t rows per course"]
    I1["course_instructor<br/>(CS-157A, Kim), (CS-157A, Rossi)"]
    T1["course_textbook<br/>(CS-157A, Silberschatz), (CS-157A, Elmasri)"]
  end
  Before -->|"course_id ↠ instructor | textbook"| After
```

## MySQL Implementation

Create the combined table. Its only key is all three columns:

```sql run
CREATE TABLE course_info (
  course_id  VARCHAR(10) NOT NULL,
  instructor VARCHAR(40) NOT NULL,
  textbook   VARCHAR(40) NOT NULL,
  PRIMARY KEY (course_id, instructor, textbook)
);
INSERT INTO course_info VALUES
  ('CS-157A',  'Kim',    'Silberschatz'), ('CS-157A',  'Kim',    'Elmasri'),
  ('CS-157A',  'Rossi',  'Silberschatz'), ('CS-157A',  'Rossi',  'Elmasri'),
  ('DATA-225', 'Haddad', 'Kleppmann'),    ('DATA-225', 'Haddad', 'Petrov'),
  ('DATA-225', 'Haddad', 'Karwin');
SELECT course_id, COUNT(DISTINCT instructor) AS instructors, COUNT(DISTINCT textbook) AS textbooks,
       COUNT(*) AS rows_stored
FROM course_info GROUP BY course_id ORDER BY course_id;
```

### Testing an MVD with SQL

By definition, X ↠ Y holds when, within each X group, every Y value appears with every Z value. Since the primary key forbids duplicate rows, that is equivalent to **rows per group = distinct Y × distinct Z**:

```sql run
SELECT course_id,
       COUNT(*)                                               AS rows_stored,
       COUNT(DISTINCT instructor) * COUNT(DISTINCT textbook)  AS rows_if_independent,
       COUNT(*) = COUNT(DISTINCT instructor) * COUNT(DISTINCT textbook) AS mvd_holds
FROM course_info GROUP BY course_id ORDER BY course_id;
```

The second, more general test uses Fagin's theorem directly: X ↠ Y holds iff R equals the join of its projections on XY and XZ. Compare in both directions with `EXCEPT`:

```sql run
WITH ci AS (SELECT DISTINCT course_id, instructor FROM course_info),
     ct AS (SELECT DISTINCT course_id, textbook   FROM course_info),
     rejoined AS (SELECT ci.course_id, ci.instructor, ct.textbook FROM ci JOIN ct ON ct.course_id = ci.course_id)
SELECT (SELECT COUNT(*) FROM rejoined) AS rejoined_rows,
       (SELECT COUNT(*) FROM (SELECT * FROM rejoined EXCEPT SELECT * FROM course_info) d) AS spurious,
       (SELECT COUNT(*) FROM (SELECT * FROM course_info EXCEPT SELECT * FROM rejoined) d) AS lost;
```

Zero spurious and zero lost: the current instance satisfies the MVD, so the decomposition into `(course_id, instructor)` and `(course_id, textbook)` is lossless for it. As with FDs, the data can refute an MVD but only the business rule ("instructors and textbooks are chosen independently") establishes it.

## Syntax Breakdown

There is no MVD or JD syntax in SQL; everything is done with ordinary queries and DDL.

| Task | SQL pattern | Notes |
|---|---|---|
| Check X ↠ Y per group | `GROUP BY X HAVING COUNT(*) <> COUNT(DISTINCT Y) * COUNT(DISTINCT Z)` | Requires no duplicate rows (a PK); Y and Z may be column lists with `COUNT(DISTINCT a, b)` |
| Check X ↠ Y globally | `π_XY ⋈ π_XZ` compared with R via `EXCEPT` both ways | Works for any instance; the general lossless test |
| Check a JD \*(R₁,…,Rₙ) | Join all projections, `EXCEPT` both ways | The join can never *lose* rows of R, only add spurious ones |
| 4NF decomposition | `CREATE TABLE r1 (X, Y, PRIMARY KEY (X, Y))`, same for (X, Z) | Populate with `INSERT … SELECT DISTINCT` |
| Rebuild the combination | `CREATE VIEW … AS SELECT … FROM r1 JOIN r2 USING (X)` | The view is the "combined table" on demand |
| Enforce an MVD/JD directly | Not possible declaratively | Decompose instead; a trigger could only approximate it |

:::standard[Theory vs SQL]
Normalization theory works with sets; SQL tables are bags unless a key prevents duplicates. Both MVD tests above assume set semantics — which is why every table here has a primary key, and why the projections use `SELECT DISTINCT`.
:::

## Practical Examples

### 1. The anomalies of the 4NF violation

**Insertion.** CS-157A adopts a third textbook, *Date*. The correct change needs one row per instructor; a careless insert adds only one:

```sql run
INSERT INTO course_info VALUES ('CS-157A', 'Kim', 'Date');
SELECT course_id,
       COUNT(*) AS rows_stored,
       COUNT(DISTINCT instructor) * COUNT(DISTINCT textbook) AS rows_if_independent
FROM course_info GROUP BY course_id
HAVING COUNT(*) <> COUNT(DISTINCT instructor) * COUNT(DISTINCT textbook);
```

The table now *says* that Rossi teaches CS-157A without *Date* — a three-way fact nobody stated. Which reading is right is undecidable from the data. The fix requires the missing combination:

```sql run
INSERT INTO course_info VALUES ('CS-157A', 'Rossi', 'Date');
```

**Insertion of an instructor** — a new instructor, Lin, for DATA-225 needs three rows (one per textbook). **Deletion** — if Haddad leaves and is deleted, the three DATA-225 textbooks vanish with him:

```sql run
SELECT COUNT(*) AS data225_textbook_rows_that_would_disappear
FROM course_info WHERE course_id = 'DATA-225' AND instructor = 'Haddad';
```

### 2. Decompose into 4NF

```sql run
CREATE TABLE course_instructor (
  course_id  VARCHAR(10) NOT NULL,
  instructor VARCHAR(40) NOT NULL,
  PRIMARY KEY (course_id, instructor)
);
CREATE TABLE course_textbook (
  course_id VARCHAR(10) NOT NULL,
  textbook  VARCHAR(40) NOT NULL,
  PRIMARY KEY (course_id, textbook)
);
INSERT INTO course_instructor SELECT DISTINCT course_id, instructor FROM course_info;
INSERT INTO course_textbook   SELECT DISTINCT course_id, textbook   FROM course_info;
SELECT (SELECT COUNT(*) FROM course_info)       AS combined_rows,
       (SELECT COUNT(*) FROM course_instructor) AS instructor_rows,
       (SELECT COUNT(*) FROM course_textbook)   AS textbook_rows;
```

Each new table has only trivial MVDs (two attributes, both in the key), so both are in 4NF. The anomalies disappear: a new textbook is one row in `course_textbook`; a new instructor one row in `course_instructor`; deleting an instructor touches nothing about textbooks.

```sql run
INSERT INTO course_instructor VALUES ('DATA-225', 'Lin');
DELETE FROM course_instructor WHERE course_id = 'DATA-225' AND instructor = 'Haddad';
SELECT 'textbooks still known for DATA-225' AS check_, COUNT(*) AS n FROM course_textbook WHERE course_id = 'DATA-225';
```

### 3. When there is NO multivalued dependency

Suppose each instructor chooses *their own* textbooks — Kim uses Silberschatz, Rossi uses Elmasri. Now the three-way rows are **dependent** facts, and decomposition would be wrong:

```sql run
CREATE TABLE teaching_choice (
  course_id  VARCHAR(10) NOT NULL,
  instructor VARCHAR(40) NOT NULL,
  textbook   VARCHAR(40) NOT NULL,
  PRIMARY KEY (course_id, instructor, textbook)
);
INSERT INTO teaching_choice VALUES
  ('CS-157A', 'Kim', 'Silberschatz'), ('CS-157A', 'Rossi', 'Elmasri');
WITH ci AS (SELECT DISTINCT course_id, instructor FROM teaching_choice),
     ct AS (SELECT DISTINCT course_id, textbook   FROM teaching_choice)
SELECT ci.course_id, ci.instructor, ct.textbook,
       EXISTS (SELECT 1 FROM teaching_choice t WHERE t.course_id = ci.course_id
               AND t.instructor = ci.instructor AND t.textbook = ct.textbook) AS in_original
FROM ci JOIN ct ON ct.course_id = ci.course_id
ORDER BY ci.instructor, ct.textbook;
```

The join of the projections invents (Kim, Elmasri) and (Rossi, Silberschatz) — **spurious tuples**. The MVD does not hold, the decomposition would be lossy, and the three-column table is the correct design (it is in 4NF and 5NF: its only JDs are trivial). The same table shape can be a 4NF violation or a perfect design; only the business rule decides.

## Execution Walkthrough

The 5NF example: suppliers, parts and projects with a cyclic rule. The data is the classic SUPPLY relation from Elmasri & Navathe.

```walkthrough
title: A join dependency that no two-way decomposition captures
steps:
  - title: "1. The relation and its rule"
    explain: "SUPPLY(supplier, part, project). Rule: if supplier s supplies part p, project j uses part p, and s supplies project j, then s supplies p to j. There are no FDs and no MVDs; the only key is all three attributes, so the table is in 4NF."
    sql: |
      CREATE TABLE supply (
        supplier VARCHAR(20) NOT NULL, part VARCHAR(20) NOT NULL, project VARCHAR(20) NOT NULL,
        PRIMARY KEY (supplier, part, project));
      INSERT INTO supply VALUES
        ('Smith', 'Bolt', 'ProjX'), ('Smith', 'Nut', 'ProjY'), ('Adamsky', 'Bolt', 'ProjY'),
        ('Walton', 'Nut', 'ProjZ'), ('Adamsky', 'Nail', 'ProjX'), ('Adamsky', 'Bolt', 'ProjX'),
        ('Smith', 'Bolt', 'ProjY');
      SELECT supplier, part, project FROM supply ORDER BY supplier, part, project;
  - title: "2. The three binary projections"
    explain: "R1 = (supplier, part), R2 = (part, project), R3 = (supplier, project). Each projection is a set of pairs; count them."
    sql: |
      SELECT (SELECT COUNT(*) FROM (SELECT DISTINCT supplier, part FROM supply) a)    AS supplier_part,
             (SELECT COUNT(*) FROM (SELECT DISTINCT part, project FROM supply) b)     AS part_project,
             (SELECT COUNT(*) FROM (SELECT DISTINCT supplier, project FROM supply) c) AS supplier_project;
  - title: "3. Every two-way join is lossy"
    explain: "Joining any two projections produces rows that are not in SUPPLY. That proves none of the three possible MVDs holds, so no binary decomposition is lossless."
    sql: |
      WITH sp AS (SELECT DISTINCT supplier, part FROM supply),
           pj AS (SELECT DISTINCT part, project FROM supply),
           sj AS (SELECT DISTINCT supplier, project FROM supply)
      SELECT 'R1 ⋈ R2 (on part)' AS join_, COUNT(*) AS join_rows,
             (SELECT COUNT(*) FROM (SELECT sp.supplier, sp.part, pj.project FROM sp JOIN pj USING (part)
                                    EXCEPT SELECT * FROM supply) d) AS spurious
      FROM sp JOIN pj USING (part)
      UNION ALL
      SELECT 'R1 ⋈ R3 (on supplier)', COUNT(*),
             (SELECT COUNT(*) FROM (SELECT sp.supplier, sp.part, sj.project FROM sp JOIN sj USING (supplier)
                                    EXCEPT SELECT * FROM supply) d)
      FROM sp JOIN sj USING (supplier)
      UNION ALL
      SELECT 'R2 ⋈ R3 (on project)', COUNT(*),
             (SELECT COUNT(*) FROM (SELECT sj.supplier, pj.part, pj.project FROM pj JOIN sj USING (project)
                                    EXCEPT SELECT * FROM supply) d)
      FROM pj JOIN sj USING (project);
  - title: "4. A spurious tuple, inspected"
    explain: "R1 ⋈ R2 pairs every supplier of nuts with every project that uses nuts: (Smith, Nut, ProjZ) and (Walton, Nut, ProjY) are invented. Smith does not supply ProjZ at all, and Walton does not supply ProjY, so the rule does not generate them. The third projection, (supplier, project), is what filters them out."
    sql: |
      WITH sp AS (SELECT DISTINCT supplier, part FROM supply),
           pj AS (SELECT DISTINCT part, project FROM supply)
      SELECT sp.supplier, sp.part, pj.project FROM sp JOIN pj USING (part)
      EXCEPT SELECT * FROM supply
      ORDER BY 1, 2, 3;
  - title: "5. The three-way join is exact"
    explain: "Joining all three projections reproduces SUPPLY precisely: the join dependency *(R1, R2, R3) holds. Because it is not implied by the key (all three attributes), SUPPLY is not in 5NF, and the 5NF design stores the three projections instead."
    sql: |
      WITH sp AS (SELECT DISTINCT supplier, part FROM supply),
           pj AS (SELECT DISTINCT part, project FROM supply),
           sj AS (SELECT DISTINCT supplier, project FROM supply),
           j3 AS (SELECT sp.supplier, sp.part, pj.project
                  FROM sp JOIN pj USING (part)
                  JOIN sj ON sj.supplier = sp.supplier AND sj.project = pj.project)
      SELECT (SELECT COUNT(*) FROM j3) AS three_way_rows,
             (SELECT COUNT(*) FROM (SELECT * FROM j3 EXCEPT SELECT * FROM supply) d) AS spurious,
             (SELECT COUNT(*) FROM (SELECT * FROM supply EXCEPT SELECT * FROM j3) d) AS lost;
```

FD-based analysis cannot see any of this. The analyzer below, given the SUPPLY attributes and no FDs, reports the single key `{supplier, part, project}` and **BCNF** — correct, and a reminder that BCNF is the limit of what FDs can tell you.

```viz
type: fd
attributes: [supplier, part, project]
fds: []
```

## Output

The 5NF design stores the three pair tables and exposes the three-way relation as a view. Because the business rule *is* the join, the view is not a convenience — it is the definition of which supplies exist:

```sql run
CREATE TABLE supplier_part    (supplier VARCHAR(20) NOT NULL, part VARCHAR(20) NOT NULL, PRIMARY KEY (supplier, part));
CREATE TABLE part_project     (part VARCHAR(20) NOT NULL, project VARCHAR(20) NOT NULL, PRIMARY KEY (part, project));
CREATE TABLE supplier_project (supplier VARCHAR(20) NOT NULL, project VARCHAR(20) NOT NULL, PRIMARY KEY (supplier, project));
INSERT INTO supplier_part    SELECT DISTINCT supplier, part FROM supply;
INSERT INTO part_project     SELECT DISTINCT part, project FROM supply;
INSERT INTO supplier_project SELECT DISTINCT supplier, project FROM supply;
CREATE VIEW supply_v AS
SELECT sp.supplier, sp.part, pj.project
FROM supplier_part sp
JOIN part_project pj      ON pj.part = sp.part
JOIN supplier_project sj  ON sj.supplier = sp.supplier AND sj.project = pj.project;
SELECT supplier, part, project FROM supply_v ORDER BY supplier, part, project;
```

The view returns the same seven supplies as the original table. Notice what the 5NF design changes about **insertion**: in the original table, if Walton starts supplying ProjX and ProjX starts using nuts, the rule *requires* the row (Walton, Nut, ProjX), and someone must remember to insert it. In the 5NF design you record only the two new pair facts, and the view derives the supply:

```sql run
INSERT INTO supplier_project VALUES ('Walton', 'ProjX');
INSERT INTO part_project VALUES ('Nut', 'ProjX');
SELECT supplier, part, project FROM supply_v WHERE supplier = 'Walton' ORDER BY part, project;
```

Two independent facts (Walton supplies ProjX; ProjX uses nuts) produced the derived supply (Walton, Nut, ProjX) without anyone writing it — the rule is enforced by the structure.

## Alternative Approaches

| Situation | Recommended design | Why |
|---|---|---|
| Two independent lists per entity (skills and languages of an employee; instructors and textbooks of a course) | Two junction tables (4NF) | Removes i × t redundancy; each list maintained independently |
| A genuine three-way fact (instructor X uses textbook Y in course Z) | One three-column table with a three-column key | No MVD holds; decomposition would be lossy |
| A three-way fact constrained by a cyclic rule (SUPPLY) | Three pair tables + a view (5NF) | Rule enforced structurally; facts entered once |
| Same as above, but the rule is not stable or has exceptions | One three-way table | The JD does not truly hold; 5NF would forbid legitimate exceptions |
| Lists that are not entities and are always read together | JSON array per list (with care) | See [First Normal Form](/lessons/m06-1nf); avoids the cross product without extra tables |

**ER modelling usually prevents 4NF violations.** If each independent relationship (course–instructor, course–textbook) is modelled as its own relationship type, the ER-to-relational mapping ([Mapping ER Diagrams to Tables](/lessons/m05-er-to-relational)) produces separate tables automatically. 4NF violations typically arise when someone merges two relationships into one table for a report or spreadsheet.

**Beyond 5NF.** Two further forms appear in the literature: **domain-key normal form** (Fagin 1981 — every constraint follows from domains and keys; an ideal rather than a practical target) and **sixth normal form** (Date, Darwen & Lorentzos — every relation is irreducible, used mainly for temporal data where each attribute's history is stored separately).

## Performance Analysis

| Aspect | Combined table (MVD) | 4NF tables |
|---|---|---|
| Rows for *i* instructors × *t* textbooks | i × t | i + t |
| Add one textbook | i inserts (atomic in one transaction) | 1 insert |
| List a course's textbooks | `SELECT DISTINCT textbook … WHERE course_id = ?` — reads i × t rows | Index range on `(course_id, textbook)` — reads t rows |
| Produce all combinations | Read rows directly | Join on `course_id` (cross product per course) |

The 4NF design is better on almost every axis; the only query that gets more expensive is the one that explicitly wants the cross product, and that query is rare (and cheap when the lists are short).

For 5NF the trade-off is sharper. The view `supply_v` is a **three-way join** on composite keys. Look at how MySQL executes it for one supplier:

```sql run
ANALYZE TABLE supplier_part, part_project, supplier_project;
EXPLAIN FORMAT=TREE SELECT * FROM supply_v WHERE supplier = 'Smith';
```

(We ran `ANALYZE TABLE` first because the tables were created in this lesson.) The optimizer reads Smith's projects with a covering lookup on the primary key of `supplier_project`, hash-joins them with a scan of `part_project` on `project`, and then probes `supplier_part` with a *single-row* primary-key lookup on `(supplier, part)` for each candidate — the third projection acting as the filter that removes would-be spurious tuples. On large data, a 5NF design trades update simplicity for join cost on every read; it is chosen when the cyclic rule is important to enforce and reads can afford the join — or a summary table maintained from the pairs (see [Normalization vs Denormalization](/lessons/m06-denormalization)).

:::performance[Theory vs measurement]
The row-count formulas are exact; actual speed depends on list sizes and indexes. With realistic list lengths (a handful of instructors and textbooks) the difference is small for reads and large for correctness. Measure the 5NF join with `EXPLAIN ANALYZE` on production-sized data before committing to it.
:::

## Best Practices

:::best-practice[MVDs and JDs in real designs]
1. **Never combine two independent multi-valued facts in one table.** One junction table per independent relationship.
2. **Ask "are these lists independent?" before decomposing a three-column all-key table.** If one value constrains the other (instructor-specific textbooks), the three-way table is correct.
3. **Test candidate MVDs and JDs with the projection-join-EXCEPT pattern** on real data; spurious rows refute the dependency.
4. **Use 5NF only for stable cyclic rules.** Document the rule, store the pair tables, and expose the derived relation as a view.
5. **Model relationships separately in the ER diagram**; correct ER-to-relational mapping avoids most 4NF problems by construction.
6. **Remember that FD tools stop at BCNF.** A schema reported as "BCNF" by an FD analyzer can still contain MVD redundancy.
:::

## Common Mistakes

**Mistake 1 — "it's all-key, so it's fully normalized".** An all-key table has no non-trivial FDs and is always in BCNF, but it can still violate 4NF (`course_info`) or 5NF (`supply`).

**Mistake 2 — decomposing dependent three-way facts.** Practical Example 3 showed the spurious (Kim, Elmasri) and (Rossi, Silberschatz) rows. Before applying 4NF, verify the MVD.

**Mistake 3 — partial inserts into an MVD table.** Adding a textbook for one instructor only silently changes the meaning of the data, as Example 1 showed. If you must keep such a table temporarily, insert the full cross product in one transaction:

```sql run
INSERT INTO course_info (course_id, instructor, textbook)
SELECT DISTINCT course_id, instructor, 'Karwin' FROM course_info WHERE course_id = 'CS-157A';
SELECT instructor, COUNT(*) AS textbooks FROM course_info WHERE course_id = 'CS-157A' GROUP BY instructor ORDER BY instructor;
```

**Mistake 4 — believing two-way decomposition always suffices.** The SUPPLY example has no lossless binary decomposition at all; only the three-way split is lossless.

**Mistake 5 — forcing 5NF on rules with exceptions.** If Smith may supply ProjX with bolts *except* for safety-critical bolts, the JD does not hold, and the 5NF view would derive supplies that are not allowed.

## Industry Application

- **HR and skills platforms** store employee skills, certifications and languages in separate junction tables — a direct application of 4NF; merging them into one "profile facts" table is a classic early-stage mistake.
- **Product catalogues** keep "available colours" and "available sizes" as independent lists (4NF) when every combination is sellable, and switch to an explicit variant table (three-way fact, SKU per colour-size) as soon as some combinations do not exist.
- **Supply chain and procurement** systems occasionally encode cyclic approval rules (approved supplier for a part, part approved for a project, supplier approved for a project) as three pair tables, deriving permitted supplies with a join — a 5NF design in all but name.
- **In practice, 5NF is rare.** Kent (1983) already noted that 4NF and 5NF problems are uncommon compared with 2NF/3NF ones; interviews and exams test them mainly to see whether a candidate understands that BCNF is not the end of normalization.

## Practice Exercises

```exercise
id: ex1
title: Test an MVD per group
difficulty: easy
setup: |
  CREATE TABLE employee_profile (emp_id INT NOT NULL, skill VARCHAR(20) NOT NULL, language VARCHAR(20) NOT NULL,
    PRIMARY KEY (emp_id, skill, language));
  INSERT INTO employee_profile VALUES
    (1, 'SQL', 'English'), (1, 'SQL', 'Spanish'), (1, 'Python', 'English'), (1, 'Python', 'Spanish'),
    (2, 'Java', 'English'), (2, 'Go', 'English'), (2, 'Go', 'Hindi'),
    (3, 'SQL', 'French');
concepts: [multivalued-dependency]
prompt: |
  `employee_profile(emp_id, skill, language)` should satisfy `emp_id ↠ skill` (skills and languages are
  independent). Return the `emp_id` values for which the MVD is **violated** in the data, with columns
  `emp_id`, `rows_stored` and `rows_if_independent`, ordered by `emp_id`.
hints:
  - "Within each emp_id group the MVD requires rows = distinct skills × distinct languages."
  - "Use HAVING COUNT(*) <> COUNT(DISTINCT skill) * COUNT(DISTINCT language)."
solution: |
  SELECT emp_id, COUNT(*) AS rows_stored,
         COUNT(DISTINCT skill) * COUNT(DISTINCT language) AS rows_if_independent
  FROM employee_profile
  GROUP BY emp_id
  HAVING COUNT(*) <> COUNT(DISTINCT skill) * COUNT(DISTINCT language)
  ORDER BY emp_id;
orderMatters: true
explanation: |
  Employee 2 has skills {Java, Go} and languages {English, Hindi} but only 3 of the 4 combinations — (Java,
  Hindi) is missing. Either the data is incomplete or skills and languages are not independent for this
  employee. Employees 1 and 3 satisfy the MVD. The test relies on the primary key preventing duplicate rows.
```

```exercise
id: ex2
title: Find the spurious tuples
difficulty: medium
setup: |
  CREATE TABLE teaching_choice (course_id VARCHAR(10) NOT NULL, instructor VARCHAR(40) NOT NULL, textbook VARCHAR(40) NOT NULL,
    PRIMARY KEY (course_id, instructor, textbook));
  INSERT INTO teaching_choice VALUES
    ('CS-157A', 'Kim', 'Silberschatz'), ('CS-157A', 'Rossi', 'Elmasri'), ('CS-157A', 'Rossi', 'Silberschatz'),
    ('DATA-225', 'Haddad', 'Kleppmann');
concepts: [lossless-join, multivalued-dependency]
prompt: |
  Decomposing `teaching_choice` into `(course_id, instructor)` and `(course_id, textbook)` and joining them back
  on `course_id` would produce spurious rows. Return them (`course_id, instructor, textbook`), ordered by all three columns.
hints:
  - "Build the two projections with SELECT DISTINCT, join them on course_id."
  - "Subtract the original table with EXCEPT."
solution: |
  WITH ci AS (SELECT DISTINCT course_id, instructor FROM teaching_choice),
       ct AS (SELECT DISTINCT course_id, textbook FROM teaching_choice)
  SELECT ci.course_id, ci.instructor, ct.textbook
  FROM ci JOIN ct ON ct.course_id = ci.course_id
  EXCEPT
  SELECT course_id, instructor, textbook FROM teaching_choice
  ORDER BY course_id, instructor, textbook;
orderMatters: true
explanation: |
  The only spurious row is (CS-157A, Kim, Elmasri): Kim does not use Elmasri. Because the join produces a row
  that is not in the table, `course_id ↠ instructor` does not hold, and by Fagin's theorem the decomposition is lossy.
```

```exercise
id: ex3
title: Decompose into 4NF
difficulty: hard
kind: dml
setup: |
  CREATE TABLE employee_profile (emp_id INT NOT NULL, skill VARCHAR(20) NOT NULL, language VARCHAR(20) NOT NULL,
    PRIMARY KEY (emp_id, skill, language));
  INSERT INTO employee_profile VALUES
    (1, 'SQL', 'English'), (1, 'SQL', 'Spanish'), (1, 'Python', 'English'), (1, 'Python', 'Spanish'),
    (2, 'Go', 'English'), (2, 'Go', 'Hindi');
concepts: [fourth-normal-form, multivalued-dependency]
prompt: |
  The MVD `emp_id ↠ skill | language` holds. Decompose into 4NF:

  1. `employee_skill (emp_id INT NOT NULL, skill VARCHAR(20) NOT NULL, PRIMARY KEY (emp_id, skill))`
  2. `employee_language (emp_id INT NOT NULL, language VARCHAR(20) NOT NULL, PRIMARY KEY (emp_id, language))`
  3. Populate both and drop `employee_profile`.

  The checker lists both tables' rows.
hints:
  - "INSERT INTO employee_skill SELECT DISTINCT emp_id, skill FROM employee_profile;"
solution: |
  CREATE TABLE employee_skill (emp_id INT NOT NULL, skill VARCHAR(20) NOT NULL, PRIMARY KEY (emp_id, skill));
  CREATE TABLE employee_language (emp_id INT NOT NULL, language VARCHAR(20) NOT NULL, PRIMARY KEY (emp_id, language));
  INSERT INTO employee_skill SELECT DISTINCT emp_id, skill FROM employee_profile;
  INSERT INTO employee_language SELECT DISTINCT emp_id, language FROM employee_profile;
  DROP TABLE employee_profile;
check: |
  SELECT 'skill' AS kind, emp_id, skill AS value FROM employee_skill
  UNION ALL
  SELECT 'language', emp_id, language FROM employee_language
  ORDER BY kind, emp_id, value;
explanation: |
  Six combined rows become three skill rows and four language rows; with larger lists the saving is
  multiplicative. Without `DISTINCT` the inserts fail on the primary key, because each skill appears once per language.
```

```exercise
id: ex4
title: Prove a join dependency
difficulty: hard
setup: |
  CREATE TABLE agent_sale (agent VARCHAR(10) NOT NULL, company VARCHAR(10) NOT NULL, product VARCHAR(10) NOT NULL,
    PRIMARY KEY (agent, company, product));
  INSERT INTO agent_sale VALUES
    ('Ana', 'Ford', 'car'), ('Ana', 'Ford', 'truck'), ('Ana', 'GM', 'car'),
    ('Bo', 'Ford', 'car'), ('Bo', 'GM', 'bus');
concepts: [join-dependency, fifth-normal-form]
prompt: |
  Test whether the join dependency \*((agent, company), (agent, product), (company, product)) holds in
  `agent_sale`. Return one row with columns `three_way_rows`, `spurious` and `lost` (row counts of the three-way
  join of the projections, of join-minus-table, and of table-minus-join).
hints:
  - "Build three DISTINCT projections and join them on all shared attributes."
  - "Compare with EXCEPT in both directions, inside scalar subqueries."
solution: |
  WITH ac AS (SELECT DISTINCT agent, company FROM agent_sale),
       ap AS (SELECT DISTINCT agent, product FROM agent_sale),
       cp AS (SELECT DISTINCT company, product FROM agent_sale),
       j3 AS (SELECT ac.agent, ac.company, ap.product
              FROM ac JOIN ap ON ap.agent = ac.agent
              JOIN cp ON cp.company = ac.company AND cp.product = ap.product)
  SELECT (SELECT COUNT(*) FROM j3) AS three_way_rows,
         (SELECT COUNT(*) FROM (SELECT * FROM j3 EXCEPT SELECT * FROM agent_sale) d) AS spurious,
         (SELECT COUNT(*) FROM (SELECT * FROM agent_sale EXCEPT SELECT * FROM j3) d) AS lost;
explanation: |
  The three-way join produces one spurious row, (Bo, GM, car): Bo sells for GM, Bo sells cars, and GM makes cars,
  but Bo does not sell GM cars. So the JD does not hold in this instance, and the cyclic rule is not a business
  rule here — the three-way table is the correct design. `lost` is always 0: a join of projections can only add rows.
```

```exercise
id: ex5
title: Which normal form, and should we care?
difficulty: challenge
kind: open
concepts: [fourth-normal-form, fifth-normal-form, independent-multivalued-facts]
prompt: |
  A streaming service stores `title_availability(title_id, country, audio_language)` with key = all three columns.
  Product says: "Every audio track of a title is available in every country where the title is licensed."

  1. Which dependency does this rule express? Is the table in BCNF? In 4NF?
  2. Propose the normalized design and the SQL view that reproduces the original table.
  3. Six months later, licensing deals start restricting some audio languages per country. What happens to your
     design, and what is the lesson about MVDs?
solution: |
  1. The rule is the MVD `title_id ↠ country` (equivalently `title_id ↠ audio_language`): countries and audio
     languages are independent per title. There are no non-trivial FDs and the key is all three attributes, so
     the table is in **BCNF**; it is **not in 4NF**, because `title_id` is not a superkey.
  2. `title_country(title_id, country, PK both)` and `title_audio(title_id, audio_language, PK both)`;
     `CREATE VIEW title_availability_v AS SELECT c.title_id, c.country, a.audio_language FROM title_country c
     JOIN title_audio a ON a.title_id = c.title_id;`
  3. The MVD stops holding: availability becomes a genuine three-way fact. The 4NF design can no longer express
     "German audio is not available in Brazil". Options: go back to a three-column `title_availability` table
     (correct, no longer redundant because combinations are now independent facts), or keep the two tables plus a
     `title_audio_exclusion(title_id, country, audio_language)` table with the view subtracting exclusions — better
     when exceptions are rare. Lesson: MVDs are business rules, and decomposing on them bakes the rule into the
     schema; verify with stakeholders how stable the independence really is.
```

## Knowledge Assessment

```quiz
- id: q1
  question: "A table R(course, instructor, textbook) has key {course, instructor, textbook} and no non-trivial FDs. Which statement is necessarily true?"
  options:
    - "R is in 4NF"
    - "R is in BCNF"
    - "R violates 3NF"
    - "R violates 2NF"
  answer: 1
  explanation: "With no non-trivial FDs, BCNF holds trivially. Whether it is in 4NF depends on whether an MVD such as course ↠ instructor holds — which is a business question."
  concepts: [fourth-normal-form]
- id: q2
  question: "Which statements about MVDs are true? (Select all that apply.)"
  type: multi
  options:
    - "Every FD X → Y is also an MVD X ↠ Y"
    - "X ↠ Y implies X ↠ R − X − Y"
    - "X ↠ YW implies X ↠ Y"
    - "R decomposes losslessly into XY and XZ (Z = R − X − Y) iff X ↠ Y"
  answer: [0, 1, 3]
  explanation: "FDs are special MVDs, MVDs come in complementary pairs, and Fagin's theorem characterises lossless binary decomposition. Unlike FDs, MVD right sides cannot be split."
  concepts: [multivalued-dependency]
- id: q3
  question: "A course has 4 instructors and 5 textbooks, independent of each other. How many rows does the combined table need, and how many rows do the 4NF tables need in total?"
  options: ["20 and 9", "9 and 20", "20 and 20", "5 and 4"]
  answer: 0
  explanation: "The MVD forces every combination: 4 × 5 = 20 rows. The decomposed tables store 4 + 5 = 9 rows."
  concepts: [fourth-normal-form, independent-multivalued-facts]
- id: q4
  question: "For SUPPLY(supplier, part, project), each of the three two-way joins of projections produces spurious rows, but the three-way join reproduces the table. What does this show?"
  options:
    - "SUPPLY is in 5NF"
    - "A join dependency *(SP, PJ, SJ) holds but no MVD holds; SUPPLY is in 4NF but not 5NF"
    - "SUPPLY violates BCNF"
    - "The decomposition is lossy"
  answer: 1
  explanation: "No binary decomposition is lossless, so no MVD holds (4NF is satisfied), but the three-way JD holds and is not implied by the key — a 5NF violation. The three-way decomposition is lossless."
  concepts: [join-dependency, fifth-normal-form]
- id: q5
  question: "The query `GROUP BY course_id HAVING COUNT(*) <> COUNT(DISTINCT instructor) * COUNT(DISTINCT textbook)` returns no rows. What does that mean?"
  options:
    - "The MVD course_id ↠ instructor is proven for the schema"
    - "The current instance satisfies the MVD (assuming no duplicate rows)"
    - "course_id is a key"
    - "There are no textbooks"
  answer: 1
  explanation: "Like FD checks, data can only refute an MVD. The formula relies on set semantics, which the primary key guarantees."
  concepts: [multivalued-dependency]
- id: q6
  question: "Which situation calls for keeping a three-column all-key table instead of decomposing it?"
  options:
    - "Skills and languages of an employee are independent"
    - "Each instructor chooses their own textbooks for a course"
    - "Every colour of a shirt exists in every size"
    - "Countries and audio tracks of a title are independent"
  answer: 1
  explanation: "Instructor-specific textbook choices are a genuine three-way fact; no MVD holds, so decomposition would create spurious tuples. The other three describe independent lists — MVDs that 4NF decomposes."
  concepts: [independent-multivalued-facts, lossless-join]
- id: q7
  question: "Why can't MySQL enforce an MVD with a constraint?"
  options:
    - "MySQL does not support composite primary keys"
    - "SQL has no declaration for MVDs or JDs, and no index can express 'every Y combines with every Z'; decomposition is the remedy"
    - "Because MVDs are only defined for NULL values"
    - "It can, with a CHECK constraint"
  answer: 1
  explanation: "MVDs and JDs are not part of SQL's constraint language. CHECK constraints are row-level and cannot reference other rows. Decomposing makes the constraint unnecessary."
  concepts: [multivalued-dependency, fourth-normal-form]
- id: q8
  question: "An FD-based analyzer reports that a table is in BCNF. What can you conclude?"
  options:
    - "It is fully normalized"
    - "It has no FD-based redundancy, but it may still violate 4NF or 5NF"
    - "It is in 5NF"
    - "It has a single candidate key"
  answer: 1
  explanation: "BCNF is the strongest normal form defined by FDs alone. MVDs and JDs can still cause redundancy. BCNF says nothing about the number of keys."
  concepts: [fourth-normal-form, fifth-normal-form]
```
