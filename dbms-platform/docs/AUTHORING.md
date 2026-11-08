# Content Authoring Guide

This guide is the contract for every lesson, project and assessment on the platform. Read it fully before
writing content. The gold-standard example is
[`content/lessons/m03/m03-aggregate-functions.md`](../content/lessons/m03/m03-aggregate-functions.md) —
match its depth, tone and structure.

## 1. Teaching philosophy (non-negotiable)

Every lesson teaches **one DBMS concept and its MySQL implementation together**, connecting three levels:

1. **Theory** — why the concept exists, what problem it solves, the formal principles and terminology.
2. **Implementation** — how MySQL implements it, the exact syntax, executable examples on realistic data.
3. **Engineering** — how to use it correctly, efficiently and reliably in production.

Always distinguish **general DBMS theory** from **MySQL/InnoDB-specific behaviour**, and **standard SQL** from
**MySQL extensions**. Write for a graduate student (MS in Applied Data Intelligence) who wants professional
competence, not exam tricks. Explain *why*, not just *what*. No placeholder paragraphs, no "in this lesson we
will briefly…" filler, no summaries in place of explanations.

**Target length:** 3,500–6,500 words of genuine explanation per lesson (the capture tool prints a word count).

## 2. File layout

| What | Where |
|---|---|
| Curriculum (titles, levels, prerequisites, topics) | `content/curriculum.yaml` — single source of truth; do not rename ids |
| Lesson | `content/lessons/<module-id>/<lesson-id>.md` |
| Project | `content/projects/<project-id>.md` |
| Module assessment | `content/assessments/<module-id>-assessment.yaml` |
| Datasets | `content/datasets/*.sql`, described in `content/datasets/datasets.yaml` |
| Captured outputs (generated) | `content/generated/captures/<doc-id>.json` — never edit by hand |

A lesson must cover **every topic** listed for it in `curriculum.yaml`.

## 3. Front matter

```yaml
---
id: m03-aggregate-functions          # must equal the curriculum id and the file name
dataset: shop                        # shop | hr | university | bank | library | empty
summary: One or two sentences shown on lesson cards.
concepts: [aggregate-functions, count, null-handling]   # kebab-case tags used for weak-spot tracking
references:                          # official documentation first (MySQL 8.4 manual)
  - title: "MySQL 8.4 Reference Manual — Aggregate Function Descriptions"
    url: https://dev.mysql.com/doc/refman/8.4/en/aggregate-functions.html
---
```

Optional: `noRunnableExamples: true` (only for lessons where *nothing* can run in the sandbox; explain why in
the lesson text) and `related: [lesson-id, ...]`.

## 4. Mandatory sections

Every lesson uses exactly these `##` headings, in this order (the frontend colour-codes them by learning
stage). Use `###` for sub-structure inside a section.

| # | Heading | Stage | What belongs here |
|---|---|---|---|
| 1 | `## Introduction` | Concept | What the concept is; the real questions it answers |
| 2 | `## DBMS Theory` | Concept | Formal principles, terminology, properties (use a `:::theory` callout) |
| 3 | `## Why It Exists` | Concept | The problem it solves; what goes wrong without it |
| 4 | `## Internal Working` | Internals | Algorithms, data structures, InnoDB/server components (`:::internals`) |
| 5 | `## MySQL Implementation` | Implementation | Syntax and first executable examples |
| 6 | `## Syntax Breakdown` | Implementation | Every important keyword/clause/option, ideally as a table |
| 7 | `## Practical Examples` | Implementation | Several realistic, executable examples of increasing sophistication |
| 8 | `## Execution Walkthrough` | Execution | Step-by-step execution with intermediate results (`walkthrough`, EXPLAIN, `concurrency`) |
| 9 | `## Output` | Execution | The final result and how to read it |
| 10 | `## Alternative Approaches` | Optimization | Other ways to solve the same problem; when each fits |
| 11 | `## Performance Analysis` | Optimization | Cost, access paths, plans — theory vs measurement |
| 12 | `## Best Practices` | Optimization | Professional recommendations (`:::best-practice`) |
| 13 | `## Common Mistakes` | Optimization | Wrong code (`sql run error` or misleading results) and the correction |
| 14 | `## Industry Application` | Industry | How professionals use it in production systems |
| 15 | `## Practice Exercises` | Assessment | ≥ 4 `exercise` blocks, easy → challenge |
| 16 | `## Knowledge Assessment` | Assessment | `quiz` block with ≥ 6 questions |

If a section is genuinely not applicable (e.g. "Performance Analysis" for a purely conceptual lesson), keep the
heading and **explain why** it does not apply and what the learner should consider instead. Never force
irrelevant content, never drop the heading.

## 5. Markdown features

- GitHub-flavoured Markdown: tables, lists, `inline code`, **bold**, links.
- **Callouts** (container directives). Allowed names:
  `theory`, `mysql`, `internals`, `example`, `best-practice`, `warning`, `performance`, `industry`, `note`,
  `standard` (standard SQL vs MySQL), `version` (version-specific behaviour, e.g. 8.0 vs 8.4).
  Optional title in brackets:

  ```markdown
  :::warning[NOT IN and NULL]
  Explanation…
  :::
  ```
- **Diagrams**: fenced ```` ```mermaid ```` blocks (`flowchart`, `sequenceDiagram`, `erDiagram`,
  `stateDiagram-v2`, `classDiagram`). Keep labels short; quote labels containing punctuation:
  `A["SELECT … FROM t"]`. Use `<br/>` for line breaks inside labels.
- Cross-references: link to another lesson with `[GROUP BY and HAVING](/lessons/m03-group-by-having)`, to a
  project with `/projects/<id>`, and to the playground with `/playground`. Refer to lessons in other modules by
  title and link even if they are not written yet — every curriculum id resolves.
- No LaTeX. Use Unicode: σ π ⋈ ⟕ ⟖ ⟗ × ∪ ∩ − ÷ γ → ⊆ ∀ ∃ ¬ ∧ ∨ ≤ ≥ ≠.
- **Nested fences:** a ```` ``` ```` line inside a YAML block (for example a code sample in an exercise
  `solution` or `explanation`) closes the outer block early. Use `~~~sql … ~~~` inside, or open the outer block
  with four backticks (```` ````exercise ````) — a fence is only closed by one at least as long.
- **Plans need statistics:** datasets are `ANALYZE`d automatically when a lab loads, but tables you create or bulk
  load in a lesson are not — run `ANALYZE TABLE …` (and explain why) before showing their EXPLAIN output.
- **Structured blocks (below) must be at the top level — never inside a `:::` callout.** Plain display-only
  ```` ```sql ```` inside a callout is fine.

## 6. SQL blocks

| Fence | Behaviour |
|---|---|
| ```` ```sql ```` | Display only (has an "Open in Playground" button). Use for syntax templates, statements the sandbox forbids (GRANT, CREATE USER, SET GLOBAL…), 8.4-only syntax, and server/admin commands. |
| ```` ```sql run ```` | Executed **in document order** in one lab session at capture time; the real result is stored and shown under the block. Must succeed. |
| ```` ```sql run error ```` | Must fail; the real MySQL error is captured and shown (great for Common Mistakes). |
| ```` ```sql run volatile ```` | Must succeed but its output is not shown (non-deterministic: `NOW()`, `RAND()`, `UUID()`, `CONNECTION_ID()`…). |
| ```` ```sql run measured ```` | Output shown with a "measured on the build machine — your timings will differ" notice. Use for `EXPLAIN ANALYZE` and anything with timings. |

Rules for `run` blocks:

- **Never write expected output by hand.** Outputs are captured from a real server. After capturing, read
  `content/generated/captures/<id>.json` and make sure every number/claim in your prose matches the capture.
  Prefer phrasing like "the result shows 20 of 25 customers have a phone" only after checking it.
- Blocks share one session: tables you create stay for later blocks; session variables persist.
- One block may contain at most 60 statements separated by `;` (sandbox limit per run). Stored programs work without `DELIMITER`
  (the splitter understands `BEGIN … END`), but `DELIMITER //` is also supported.
- Make multi-row outputs deterministic (`ORDER BY` with a unique tie-breaker) and keep them readable
  (`LIMIT` ≈ 5–20 rows). The display keeps at most 50 rows.
- Don't destroy dataset tables that later blocks need. Create small illustrative tables with clear names when
  a dataset table would obscure the point (see `sensor_readings` in the exemplar).
- Large data for performance lessons: generate it inside the lab with a recursive CTE, e.g.
  ```sql
  SET SESSION cte_max_recursion_depth = 200000;
  INSERT INTO big (id, grp, val)
  WITH RECURSIVE seq(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM seq WHERE n < 200000)
  SELECT n, n % 100, (n * 7919) % 10007 FROM seq;
  ```
  Keep each statement under ~8 s on modest hardware (limit is 10 s per statement) and total lab size well under
  256 MB. 100k–300k narrow rows is plenty to make plans and costs visible.

### What the sandbox cannot run (use display-only `sql` blocks)

Each lab runs as a MySQL account with privileges **only on its own schema**:
`SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, INDEX, REFERENCES, CREATE TEMPORARY TABLES, LOCK TABLES,
CREATE VIEW, SHOW VIEW, CREATE ROUTINE, ALTER ROUTINE, EXECUTE, TRIGGER, EVENT`.

Forbidden or impossible: `CREATE/DROP DATABASE`, `USE other_db`, `CREATE USER`, `GRANT`, `REVOKE`, roles,
`SET GLOBAL`/`SET PERSIST`, `LOAD DATA`, `SELECT … INTO OUTFILE`, `KILL`, `FLUSH`, `RESET`, `PURGE`,
replication commands, `SHOW ENGINE INNODB STATUS`, `SHOW BINARY LOGS`, reading `mysql.*`, `performance_schema.*`
or `sys.*`. `SHOW VARIABLES`, `SHOW STATUS` (session), `SELECT @@var`, `SET SESSION …`, `EXPLAIN`,
`EXPLAIN ANALYZE`, `ANALYZE TABLE`, `information_schema` (own objects), and the optimizer trace
(`SET SESSION optimizer_trace='enabled=on'` + `information_schema.OPTIMIZER_TRACE`) all work.
The event scheduler is OFF in the sandbox: `CREATE EVENT` works and `SHOW EVENTS` lists events, but they do not
fire — say so in the events lesson.

### MySQL version

The platform targets **MySQL 8.4 LTS**. The local capture server may be an 8.0.x release; captures record the
exact server version and the UI shows it. Therefore:

- Keep `run` blocks to syntax valid in both 8.0.x (≥ 8.0.31) and 8.4.
- Show 8.4-only syntax/behaviour (e.g. `SHOW BINARY LOG STATUS`, `SHOW REPLICA STATUS`, removed
  `mysql_native_password` default, `innodb_change_buffering=none` default, `innodb_adaptive_hash_index=OFF`
  default) as display-only blocks inside or next to a `:::version` callout.
- Mark deprecated/removed features explicitly (query cache — removed in 8.0; `utf8mb3`; integer display width;
  `SQL_CALC_FOUND_ROWS`; `@var :=` in expressions; MASTER/SLAVE terminology removed in 8.4).
- Never invent behaviour, plan output or benchmark numbers. If unsure, check the 8.4 manual
  (`https://dev.mysql.com/doc/refman/8.4/en/…`) or demonstrate it with a `run` block and read the capture.

## 7. Walkthroughs — step-by-step execution with real intermediate results

```walkthrough
title: SELECT … the query being traced
steps:
  - title: "1. FROM — the input rows"
    explain: What happens at this step and why.
    sql: SELECT * FROM t ORDER BY id;        # optional; output is captured and shown
  - title: "2. WHERE — filter"
    explain: A step may have no SQL if it only explains.
```

Use walkthroughs to show input tables, intermediate results (e.g. the rows after WHERE, the groups before
HAVING, the join result before projection) and the final output. Each step's SQL runs in the same lab session.

## 8. Concurrency scenarios — real concurrent sessions

```concurrency
title: Lost update under READ COMMITTED
description: Two clerks update the same account balance.
dataset: bank            # optional; defaults to the lesson dataset
setup: |                 # optional SQL run once after the dataset loads
  CREATE TABLE t (id INT PRIMARY KEY, v INT);
  INSERT INTO t VALUES (1, 10), (2, 20);
steps:
  - session: A
    sql: START TRANSACTION; SELECT v FROM t WHERE id = 1;
    note: Shown under the step. Explain what A sees and why.
  - session: B
    sql: UPDATE t SET v = v + 1 WHERE id = 1;
    expect: blocks        # ok (default) | blocks | error | any
    note: B waits for A's row lock.
  - observe: locks        # snapshot of InnoDB transactions + locks (performance_schema.data_locks) for A/B/C
  - session: A
    sql: COMMIT;
    note: Releases the lock; B's UPDATE completes now.
  - session: B
    sql: SELECT * FROM t;
```

- Each step must finish within ~1.5 s or it is recorded as *blocked*; wrap bulk setup in one transaction or put
  it in `setup`.
- `observe: locks` attributes every lock to the session whose transaction owns it (`ENGINE_TRANSACTION_ID`); when
  `performance_schema` reports it on another session's thread (an implicit lock converted by that session), the
  snapshot also records `reportedByThreadOf`.
- Sessions are `A`, `B`, `C`. Each is its own connection (autocommit ON unless you start a transaction).
- A step with `expect: blocks` must contain **exactly one statement**, and a later step must release the lock.
  The capture records when it completed.
- `expect: error` with `errorCode: 1213` (deadlock), `1205` (lock wait timeout), `3572` (NOWAIT) etc.
  is verified against the real server. If InnoDB picks a different deadlock victim than you expected, fix
  the scenario, not the expectation.
- `observe: locks` captures real lock rows (`LOCK_MODE` such as `X,REC_NOT_GAP`, `X,GAP`, `S`, `IX`,
  `LOCK_STATUS` GRANTED/WAITING, `LOCK_DATA`). Use it to *show* record, gap and next-key locks.
- Use `SET SESSION TRANSACTION ISOLATION LEVEL …` inside the session's first step to demonstrate levels.
- To keep a lock-wait-timeout demo fast: `SET SESSION innodb_lock_wait_timeout = 2;` in that session.

## 9. Visualizations

```viz
type: bptree
order: 4                 # max children per node
keys: [10, 20, 5, 6, 12, 30, 7, 17]
search: 12               # optional: highlight a search path
```

| `type` | Props | Purpose |
|---|---|---|
| `bptree` | `order` (3–7), `keys` (ints, inserted in order), optional `search` | Interactive B+ tree: insert/search/range with node splits animated |
| `join` | `left`/`right`: `{name, columns, rows}`; `leftKey`, `rightKey`; optional `types` subset of `[inner, left, right, full, cross]` | Animated row matching for join types, NULL-extended rows |
| `fd` | `attributes: [A, B, C, D]`, `fds: ["A -> B", "B, C -> D"]` | Attribute closure, candidate keys, prime attributes, highest normal form, BCNF/3NF decomposition |
| `bufferpool` | `capacity` (pages), `accesses` (list of page ids), optional `oldPct` (default 37) | InnoDB midpoint-insertion LRU simulation (young/old sublists) |
| `query-order` | none | Animated logical processing order of SELECT clauses |
| `isolation` | none | Interactive matrix: isolation level × anomaly, with InnoDB-specific notes |

Execution plans need no special block: a `sql run` block containing `EXPLAIN FORMAT=JSON …` is rendered as an
interactive plan tree, and `EXPLAIN FORMAT=TREE` / `EXPLAIN ANALYZE` as a formatted iterator tree.

## 10. Exercises

```exercise
id: ex1                         # unique within the document
title: Customers without a phone number
difficulty: easy                # easy | medium | hard | challenge
kind: query                     # query (default) | dml | open
dataset: shop                   # optional, defaults to the lesson dataset
setup: |                        # optional SQL run before both the learner's and the reference solution
  CREATE TABLE … ;
concepts: [count, null-handling]
prompt: |
  Markdown problem statement. State the exact columns (names and order) expected.
starter: |                      # optional starter code placed in the editor
  SELECT
hints:                          # revealed one at a time; progressively more specific
  - First hint (conceptual).
  - Second hint (more concrete).
solution: |                     # reference SQL — never sent to the browser until unlocked
  SELECT COUNT(*) - COUNT(phone) AS no_phone FROM customers;
orderMatters: false             # true only if the prompt specifies an ordering
check: |                        # dml only: SELECT run afterwards; its result is compared
  SELECT * FROM t ORDER BY id;
explanation: |                  # shown after a correct answer or when the solution unlocks:
  Why the solution works, typical wrong answers, and why it is better than alternatives.
```

Grading compares the **result** of the learner's SQL with the reference solution's result on a freshly loaded
dataset (multiset comparison of rows; column names are not compared; numeric values are normalised;
`orderMatters: true` compares order as well). Consequences for authors:

- The prompt must pin down exactly which columns, in which order, and any rounding.
- The solution must return at least one row (the tool rejects empty results unless `allowEmpty: true`).
- `dml` exercises: the learner's statements run first, then `check`; the same happens with the solution;
  the two `check` results are compared.
- `open` exercises (design/theory): no auto-grading; the learner writes an answer, then sees the model answer
  (`solution`, Markdown). Use for design reviews, trade-off analyses, interview-style questions.
- Provide ≥ 4 exercises per lesson with real difficulty progression; the last one should require combining
  ideas. Include at least one exercise that targets a classic mistake from Common Mistakes.

## 11. Quizzes

```quiz
- id: q1
  type: single                  # single (default) | multi
  question: "Markdown question text"
  options: ["A", "B", "C", "D"]
  answer: 1                     # 0-based index; a list of indexes for multi
  explanation: "Why the answer is right AND why the distractors are wrong."
  concepts: [count]
```

≥ 6 questions per lesson: mix recall, conceptual, scenario-based ("a report shows…"), output prediction and at
least one multi-select. Distractors must be plausible misconceptions. Quote any string containing `: ` or
starting with special characters in YAML.

## 12. Module assessments

`content/assessments/<module-id>-assessment.yaml`:

```yaml
id: m03-assessment
module: m03
title: "Module 3 Assessment — SQL Fundamentals"
passMark: 70
description: What the assessment covers.
questions:          # ≥ 15 quiz questions (same schema as §11), spanning every lesson in the module
  - {id: q1, question: "…", options: [...], answer: 0, explanation: "…", concepts: [...]}
challenges:         # ≥ 3 graded SQL challenges (same schema as §10), scenario-based
  - id: c1
    title: …
    dataset: shop
    difficulty: hard
    prompt: |
      …
    hints: [ … ]
    solution: |
      …
    explanation: |
      …
interview:          # ≥ 5 interview-style questions with model answers (Markdown)
  - id: i1
    question: "Explain the difference between WHERE and HAVING."
    answer: |
      Model answer…
    concepts: [having]
```

## 13. Validate before you finish

```bash
cd backend
python scripts/capture_content.py --doc <lesson-id>     # lint + execute + capture
python scripts/capture_content.py --module m04          # whole module
python scripts/capture_content.py --doc <id> --show     # print captured results while debugging
```

A document is done only when the tool prints ✓, the word count is in range, and you have re-read the captured
outputs against your prose. To explore datasets interactively, the local server also has read-only copies:
`mysql -uroot t_shop`, `t_hr`, `t_university`, `t_bank`, `t_library`.
