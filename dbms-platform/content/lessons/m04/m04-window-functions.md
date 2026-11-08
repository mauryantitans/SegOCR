---
id: m04-window-functions
dataset: hr
summary: How window functions compute values across related rows without collapsing them — OVER, PARTITION BY, ORDER BY, ROWS vs RANGE frames (and the default-frame trap with peers), named windows, aggregate window functions, where windows sit in query evaluation and what they cost.
concepts: [window-functions, over-clause, partition-by, window-frames, rows-vs-range, named-windows, aggregate-window-functions]
related: [m03-group-by-having, m04-ranking-functions, m04-value-functions, m04-analytical-queries]
references:
  - title: "MySQL 8.4 Reference Manual — Window Function Concepts and Syntax"
    url: https://dev.mysql.com/doc/refman/8.4/en/window-functions-usage.html
  - title: "MySQL 8.4 Reference Manual — Window Function Frame Specification"
    url: https://dev.mysql.com/doc/refman/8.4/en/window-functions-frames.html
  - title: "MySQL 8.4 Reference Manual — Named Windows"
    url: https://dev.mysql.com/doc/refman/8.4/en/window-functions-named-windows.html
  - title: "MySQL 8.4 Reference Manual — Window Function Restrictions"
    url: https://dev.mysql.com/doc/refman/8.4/en/window-function-restrictions.html
  - title: "MySQL 8.4 Reference Manual — Window Function Optimization"
    url: https://dev.mysql.com/doc/refman/8.4/en/window-function-optimization.html
  - title: "MySQL 8.4 Reference Manual — Aggregate Function Descriptions"
    url: https://dev.mysql.com/doc/refman/8.4/en/aggregate-functions.html
---

## Introduction

`GROUP BY` answers questions about groups by **collapsing** each group into one row: one row per department with its average salary. But many questions need the group's value **next to every individual row**:

- "Show each employee's salary *and* their department's average, and how far above or below it they are."
- "Show each month's revenue *and* the running total so far this year."
- "Show each day's sales *and* the average of the last seven days."
- "What percentage of the company's payroll does each department represent?"

With only `GROUP BY` you would compute the aggregate in a subquery and join it back to the detail rows — once per question. **Window functions** do it directly. A window function is computed for each row over a set of rows related to it — its **window** — and the row stays in the result:

```sql
SELECT employee_id, department_id, salary,
       AVG(salary) OVER (PARTITION BY department_id) AS dept_avg
FROM employees;
```

Every employee row is returned, and each carries the average of its own department. The `OVER (…)` clause is what turns an ordinary aggregate into a window function.

MySQL has supported window functions since **8.0.2**. This lesson covers the foundations shared by all of them: the `OVER` clause, `PARTITION BY`, `ORDER BY` inside a window, **frames** (`ROWS` vs `RANGE`, and the default frame that surprises almost everyone), named windows, and the aggregate functions used as window functions. The following lessons build on this: [Ranking](/lessons/m04-ranking-functions), [Value Functions](/lessons/m04-value-functions) and [Advanced Analytical Queries](/lessons/m04-analytical-queries).

:::note[Dataset]
The examples use the **hr** dataset: 30 employees in 7 departments (Research has none; employee 124 has no department). Employees **107 and 108** in Engineering (department 20) earn exactly the same salary, 150,000 — those ties matter for frames. Every output shown was captured from a real MySQL server.
:::

## DBMS Theory

:::theory[The window model]
For each row *r* of the input (the rows that survive `FROM`, `WHERE`, `GROUP BY` and `HAVING`), a window function is evaluated over a **frame**, defined in three layers:

1. **Partition** — `PARTITION BY` splits the input into disjoint groups of rows that agree on the partition expressions. The window of *r* never extends beyond *r*'s partition. Without `PARTITION BY`, the whole input is one partition.
2. **Order** — `ORDER BY` inside `OVER` sorts the rows of each partition. Rows with equal sort keys are called **peers**.
3. **Frame** — a sliding subset of the ordered partition, defined *relative to the current row*: in physical rows (`ROWS`) or in ranges of sort-key values (`RANGE`).

The function's result for *r* is computed over *r*'s frame, and *r* is emitted unchanged with the result added as a new column. Hence the defining property: **a window function never changes the number of rows**.
:::

Window functions were added to the SQL standard in SQL:2003 (the "OLAP" extensions) and refined in SQL:2011. They fall into three families:

| Family | Functions in MySQL | Uses the frame? |
|---|---|---|
| **Aggregate** window functions | `SUM`, `AVG`, `COUNT`, `MIN`, `MAX`, `STDDEV_POP`/`STDDEV_SAMP`, `VAR_POP`/`VAR_SAMP`, `BIT_AND`/`BIT_OR`/`BIT_XOR`, `JSON_ARRAYAGG`, `JSON_OBJECTAGG` | yes |
| **Ranking** functions | `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `PERCENT_RANK`, `CUME_DIST`, `NTILE` | no — whole partition |
| **Value** (navigation) functions | `LAG`, `LEAD` (no frame); `FIRST_VALUE`, `LAST_VALUE`, `NTH_VALUE` (frame) | depends |

**Where windows sit in logical evaluation.** Window functions are computed **after** `WHERE`, `GROUP BY` and `HAVING`, and **before** `DISTINCT`, `ORDER BY` and `LIMIT` (see [Logical Query Processing Order](/lessons/m04-logical-query-order)). Two consequences you will use constantly:

- A window function **cannot** appear in `WHERE`, `GROUP BY` or `HAVING` — those clauses run before windows exist. To filter on a window result, compute it in a CTE or derived table and filter in the outer query.
- A window function **can** be applied to the result of a grouped query, including over aggregates: `SUM(SUM(salary)) OVER ()` sums the per-group sums.

:::standard[Standard SQL vs MySQL]
MySQL implements the standard `OVER`, `PARTITION BY`, `ORDER BY`, `ROWS` and `RANGE` frames, and the `WINDOW` clause for named windows. It does **not** implement the SQL:2011 `GROUPS` frame unit, the frame `EXCLUDE` clause, `DISTINCT` inside aggregate window functions (`COUNT(DISTINCT x) OVER …`), the `FILTER` clause, or `GROUP_CONCAT` as a window function — each is rejected with error 1235 (*not yet supported*). `RANGE` with a value offset (e.g. `RANGE INTERVAL 7 DAY PRECEDING`) is supported, but requires exactly one numeric or temporal `ORDER BY` expression.
:::

## Why It Exists

Before window functions (MySQL 5.7 and earlier), each of the questions in the introduction required a workaround:

- **Join back an aggregate.** Compute per-department averages in a derived table, join it to `employees`. One extra join per aggregate and per partitioning — and easy to get wrong when the grouping and joining keys differ.
- **Correlated subqueries.** `(SELECT AVG(salary) FROM employees x WHERE x.department_id = e.department_id)` in the select list, evaluated per row — and for running totals, `(SELECT SUM(…) WHERE x.date <= e.date)`, which is O(n²) work.
- **User variables.** The infamous MySQL idiom `@running := @running + amount` relied on the order in which MySQL happened to evaluate rows — undefined behaviour that produced wrong results when the plan changed. Setting user variables inside expressions is **deprecated** since 8.0 and must not be used for new code.

Window functions replace all three with one declarative construct the optimizer understands. The engine sorts each partition **once** and computes running and moving aggregates incrementally, turning many O(n²) workarounds into a sort plus a linear pass. They also make whole classes of analytical SQL — rankings, running totals, moving averages, period-over-period comparisons, sessionization, gaps-and-islands — expressible at all without procedural code.

## Internal Working

:::internals[How MySQL executes window functions]
After the join, `WHERE`, `GROUP BY` and `HAVING` have produced their rows, the executor runs a **window iterator** for each distinct window specification:

1. **Sort.** Rows are sorted by the window's `PARTITION BY` keys followed by its `ORDER BY` keys (a *filesort*). Windows with compatible specifications can share one sort; windows with different specifications are evaluated one after another, with an intermediate **temporary table** between them.
2. **Scan partitions in order.** The iterator reads the sorted rows. A change in the partition key resets the function state.
3. **Evaluate the frame.** For functions that need only the rows seen so far (`ROW_NUMBER`, running `SUM` with a `ROWS UNBOUNDED PRECEDING` frame), a single pass suffices. Functions that need rows *ahead* of the current row (`LEAD`, frames ending in `FOLLOWING`, `RANGE` frames with peers, whole-partition aggregates) need a **frame buffer**: rows of the current partition are kept in an in-memory temporary table (spilling to disk if large) so the iterator can look forward and backward. `EXPLAIN FORMAT=TREE` labels these iterators *"Window aggregate with buffering"*.
4. **Incremental (inverse) evaluation.** For a moving frame such as `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW`, MySQL does not re-sum the whole frame for every row: it adds the row entering the frame and *removes* the row leaving it. `EXPLAIN FORMAT=JSON` reports this as `"optimized_frame_evaluation": true`. For floating-point `SUM`/`AVG`, removal can accumulate rounding error, so by default (`windowing_use_high_precision = ON`) MySQL avoids inverse evaluation where it would lose precision. `MIN`/`MAX` cannot be "un-added" and are recomputed over the frame.
:::

```mermaid
flowchart LR
  F["FROM / WHERE /<br/>GROUP BY / HAVING"] --> S["Sort by<br/>PARTITION BY + ORDER BY"]
  S --> W["Window iterator<br/>per partition: reset state,<br/>maintain frame (buffer)"]
  W --> W2["Next window spec<br/>(temp table + re-sort)"]
  W2 --> P["SELECT list, DISTINCT"]
  P --> O["ORDER BY / LIMIT"]
```

The practical consequence: **each distinct window specification may cost a sort**, and frames that look ahead cost a buffer. Window functions are not free, but one sort is far cheaper than the O(n²) correlated subqueries they replace.

## MySQL Implementation

```sql
function_name([args]) OVER (
    [PARTITION BY expr [, expr] ...]
    [ORDER BY expr [ASC | DESC] [, ...]]
    [frame_clause]
)
-- or:  function_name([args]) OVER window_name
--      ... WINDOW window_name AS (window_spec) [, ...]
```

**Empty window — the whole result.** `OVER ()` treats every row as one partition. Compare each salary with the company average and the company payroll:

```sql run
SELECT employee_id, first_name, salary,
       ROUND(AVG(salary) OVER (), 2)                AS company_avg,
       ROUND(100 * salary / SUM(salary) OVER (), 2) AS pct_of_payroll
FROM employees
ORDER BY salary DESC
LIMIT 6;
```

**PARTITION BY — one window per group.** Each employee next to their department's figures. Note that the result still has one row per employee:

```sql run
SELECT department_id, employee_id, salary,
       COUNT(*)    OVER (PARTITION BY department_id) AS dept_headcount,
       ROUND(AVG(salary) OVER (PARTITION BY department_id), 2) AS dept_avg,
       salary - MAX(salary) OVER (PARTITION BY department_id) AS below_dept_max
FROM employees
WHERE department_id IN (30, 50)
ORDER BY department_id, salary DESC, employee_id;
```

**ORDER BY inside the window — running aggregates.** Adding `ORDER BY` to the window turns a whole-partition aggregate into a **cumulative** one: payroll accumulated in hiring order.

```sql run
SELECT employee_id, hire_date, salary,
       SUM(salary) OVER (ORDER BY hire_date, employee_id) AS cumulative_payroll,
       COUNT(*)    OVER (ORDER BY hire_date, employee_id) AS headcount_so_far
FROM employees
ORDER BY hire_date, employee_id
LIMIT 8;
```

The `ORDER BY` inside `OVER` defines the order for the *computation*; the final `ORDER BY` defines the order of the *output*. They are independent — always write both when the output order matters.

## Syntax Breakdown

| Element | Meaning and rules |
|---|---|
| `OVER ()` | Window = the whole result (one partition, no order, frame = all rows). |
| `PARTITION BY expr, …` | Splits rows into independent partitions. Any expressions, not just columns. |
| `ORDER BY expr [ASC\|DESC], …` | Orders rows within each partition; defines peers (equal keys). Required by ranking and `LAG`/`LEAD` to be meaningful. |
| Frame unit `ROWS` | Frame bounds counted in **physical rows** relative to the current row. |
| Frame unit `RANGE` | Frame bounds defined by **values** of the `ORDER BY` key; all peers of the current row are always included together. |
| `UNBOUNDED PRECEDING` | Frame starts at the first row of the partition. |
| `n PRECEDING` / `n FOLLOWING` | `ROWS`: n rows before/after. `RANGE`: key value − n / + n (numeric) or `INTERVAL n unit` (temporal). |
| `CURRENT ROW` | `ROWS`: the current row. `RANGE`: the current row **and all its peers**. |
| `UNBOUNDED FOLLOWING` | Frame ends at the last row of the partition. |
| `BETWEEN start AND end` | Full form; `ROWS n PRECEDING` alone means `BETWEEN n PRECEDING AND CURRENT ROW`. |
| Default frame | With `ORDER BY`: **`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`**. Without `ORDER BY`: the whole partition. |
| `WINDOW w AS (…)` | Named window, defined after `HAVING` and before `ORDER BY`; referenced as `OVER w` or refined as `OVER (w ORDER BY …)`. |

:::warning[The default frame is RANGE, not ROWS]
`SUM(x) OVER (ORDER BY k)` means `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. With `RANGE`, "current row" includes **all peers** — every row with the same `k`. If `k` has duplicates, all tied rows get the *same* running total: the total *through the last peer*. That is rarely what people expect from a "running total". Make the order key unique or write `ROWS` explicitly.
:::

## Practical Examples

**1. The default-frame trap, shown on real ties.** In Engineering (department 20), employees 107 and 108 both earn 150,000. Compare a running total with the default frame, with explicit `ROWS`, and with explicit `RANGE`:

```sql run
SELECT employee_id, salary,
       SUM(salary) OVER (ORDER BY salary)                                    AS default_frame,
       SUM(salary) OVER (ORDER BY salary
                         RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)  AS range_frame,
       SUM(salary) OVER (ORDER BY salary, employee_id
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)   AS rows_frame
FROM employees
WHERE department_id = 20
ORDER BY salary, employee_id;
```

Look at the rows for 107 and 108. `default_frame` and `range_frame` are identical (the default *is* `RANGE`), and both tied rows show the same total, which already includes **both** 150,000 salaries. Only `rows_frame` grows one row at a time. Note the tie-breaker `employee_id` in the `ROWS` window: without it, the order of 107 and 108 would be arbitrary and so would their running totals — a non-deterministic report.

**2. ROWS vs RANGE with value offsets — moving windows over dates.** Create a small table of daily sales with **missing days** (no sales on 3, 4 and 7 March):

```sql run
CREATE TABLE daily_sales (sale_date DATE PRIMARY KEY, amount DECIMAL(10,2) NOT NULL);
INSERT INTO daily_sales VALUES
  ('2024-03-01', 100), ('2024-03-02', 200), ('2024-03-05', 400),
  ('2024-03-06', 300), ('2024-03-08', 500), ('2024-03-09', 100);

SELECT sale_date, amount,
       SUM(amount) OVER (ORDER BY sale_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS last_3_rows,
       SUM(amount) OVER (ORDER BY sale_date
                         RANGE BETWEEN INTERVAL 2 DAY PRECEDING AND CURRENT ROW)   AS last_3_days
FROM daily_sales
ORDER BY sale_date;
```

`ROWS 2 PRECEDING` sums the current and the two previous **rows**, however many days apart they are — on 5 March it adds 1 and 2 March. `RANGE INTERVAL 2 DAY PRECEDING` sums rows whose date lies within the last **two days**: on 5 March, the window 3–5 March contains only 5 March itself. For calendar-based metrics ("last 7 days") `RANGE` with an interval is correct; `ROWS` is correct only when there is exactly one row per period (for example after filling gaps with a generated calendar).

**3. Centered moving average and range-based neighbourhoods.** A 3-row centered average smooths a series; a numeric `RANGE` finds "salaries within ±10,000 of mine":

```sql run
SELECT employee_id, salary,
       ROUND(AVG(salary) OVER (ORDER BY salary, employee_id
                               ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING), 2) AS centered_avg_3,
       COUNT(*) OVER (ORDER BY salary
                      RANGE BETWEEN 10000 PRECEDING AND 10000 FOLLOWING)     AS within_10k
FROM employees
WHERE department_id = 20
ORDER BY salary, employee_id;
```

The first and last rows have only two rows in their centered frame (there is no row before the first), so their "3-row" average is really a 2-row average — a common edge effect to handle or document in reports.

**4. Windows over grouped results.** Window functions run after `GROUP BY`, so they can aggregate the aggregates — each department's share of total payroll in a single query:

```sql run
SELECT department_id,
       SUM(salary)                                       AS dept_payroll,
       SUM(SUM(salary)) OVER ()                          AS company_payroll,
       ROUND(100 * SUM(salary) / SUM(SUM(salary)) OVER (), 1) AS pct
FROM employees
GROUP BY department_id
ORDER BY dept_payroll DESC;
```

`SUM(salary)` is the ordinary group aggregate; `SUM(SUM(salary)) OVER ()` is a window function over the grouped rows. Employee 124's NULL department forms its own group, as `GROUP BY` always does with NULLs.

**5. Named windows.** When several functions share a window, name it once in the `WINDOW` clause. A named window can also be *refined* — here `w_dept` is reused with an added `ORDER BY`:

```sql run
SELECT employee_id, department_id, hire_date, salary,
       ROUND(AVG(salary) OVER w_dept, 2)                 AS dept_avg,
       MAX(salary)       OVER w_dept                     AS dept_max,
       SUM(salary)       OVER (w_dept ORDER BY hire_date, employee_id
                               ROWS UNBOUNDED PRECEDING) AS dept_payroll_by_seniority
FROM employees
WHERE department_id IN (50, 60)
WINDOW w_dept AS (PARTITION BY department_id)
ORDER BY department_id, hire_date, employee_id;
```

A refinement may add `ORDER BY` and a frame but may not change the partitioning of the window it builds on.

**6. Salary history — the year-over-year comparison partner of aggregates.** Each employee's salary history with their personal maximum so far, using a partition per employee:

```sql run
SELECT employee_id, effective_date, salary,
       MAX(salary) OVER (PARTITION BY employee_id ORDER BY effective_date
                         ROWS UNBOUNDED PRECEDING) AS max_so_far,
       COUNT(*)    OVER (PARTITION BY employee_id) AS history_rows
FROM salary_history
WHERE employee_id IN (107, 108)
ORDER BY employee_id, effective_date;
```

## Execution Walkthrough

Trace `SUM(salary) OVER (PARTITION BY department_id ORDER BY salary)` for two small departments, step by step. Watch how partitions reset the computation and how the default `RANGE` frame treats peers.

```walkthrough
title: SELECT department_id, employee_id, salary, SUM(salary) OVER (PARTITION BY department_id ORDER BY salary) FROM employees WHERE department_id IN (40, 60)
steps:
  - title: "1. FROM + WHERE — input rows"
    explain: Window functions see only rows that survive WHERE. Here that is Sales (40) and Finance (60).
    sql: SELECT department_id, employee_id, salary FROM employees WHERE department_id IN (40, 60) ORDER BY employee_id;
  - title: "2. Partition and sort"
    explain: The window iterator sorts by department_id (partition key) then salary (window order key). Each department is now a contiguous block of rows in salary order. ROW_NUMBER is added only to make the position visible.
    sql: |
      SELECT department_id, employee_id, salary,
             ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary, employee_id) AS pos
      FROM employees WHERE department_id IN (40, 60)
      ORDER BY department_id, salary, employee_id;
  - title: "3. Determine each row's frame"
    explain: >-
      With ORDER BY and no frame clause, the frame is RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW: every row
      from the start of the partition up to the current row's last peer. COUNT(*) over that frame shows the frame
      size; it restarts at 1 when the partition changes.
    sql: |
      SELECT department_id, employee_id, salary,
             COUNT(*) OVER (PARTITION BY department_id ORDER BY salary) AS frame_rows
      FROM employees WHERE department_id IN (40, 60)
      ORDER BY department_id, salary, employee_id;
  - title: "4. Aggregate over the frame"
    explain: The SUM over each row's frame is the running payroll within the department. At the partition boundary the sum restarts.
    sql: |
      SELECT department_id, employee_id, salary,
             SUM(salary) OVER (PARTITION BY department_id ORDER BY salary) AS running_payroll
      FROM employees WHERE department_id IN (40, 60)
      ORDER BY department_id, salary, employee_id;
  - title: "5. The physical plan"
    explain: The plan shows the Window aggregate iterator on top of a Sort by the partition and order keys, on top of the filtered scan — the steps above, in iterator form.
    sql: |
      EXPLAIN FORMAT=TREE
      SELECT department_id, employee_id, salary,
             SUM(salary) OVER (PARTITION BY department_id ORDER BY salary) AS running_payroll
      FROM employees WHERE department_id IN (40, 60);
```

## Output

The final output of the walkthrough has exactly as many rows as its input — window functions add columns, never remove rows. In each department the running payroll starts at the lowest salary and ends, on the department's highest-paid employee, at the department's total payroll. You can verify that last claim directly:

```sql run
SELECT department_id, SUM(salary) AS dept_payroll
FROM employees
WHERE department_id IN (40, 60)
GROUP BY department_id
ORDER BY department_id;
```

When reading any window output, ask three questions: *what is the partition* (where does the computation restart), *what is the order* (what does "before" mean), and *what is the frame* (exactly which rows were aggregated for this row — especially when there are ties).

## Alternative Approaches

"Each employee's salary compared with the department average" can be written three ways. All return the same rows; they differ in readability and plan.

```sql run
-- A: window function
SELECT employee_id, department_id, salary,
       ROUND(salary - AVG(salary) OVER (PARTITION BY department_id), 2) AS diff_from_avg
FROM employees
WHERE department_id = 30
ORDER BY employee_id;
```

```sql run
-- B: join to a grouped derived table
SELECT e.employee_id, e.department_id, e.salary,
       ROUND(e.salary - d.avg_salary, 2) AS diff_from_avg
FROM employees e
JOIN (SELECT department_id, AVG(salary) AS avg_salary
      FROM employees GROUP BY department_id) d
  ON d.department_id = e.department_id
WHERE e.department_id = 30
ORDER BY e.employee_id;
```

```sql run
-- C: correlated scalar subquery
SELECT e.employee_id, e.department_id, e.salary,
       ROUND(e.salary - (SELECT AVG(x.salary) FROM employees x
                         WHERE x.department_id = e.department_id), 2) AS diff_from_avg
FROM employees e
WHERE e.department_id = 30
ORDER BY e.employee_id;
```

| Approach | Strengths | Weaknesses |
|---|---|---|
| A — window | Shortest; one pass + sort; easy to add more metrics over the same window | Needs MySQL 8.0+; one sort per distinct window |
| B — join to grouped derived table | Works in any SQL version; the grouped table can use a covering index | Extra join per aggregate level; easy to mismatch keys (NULL departments silently drop out of an inner join) |
| C — correlated subquery | Reads naturally for one value | Evaluated per outer row unless the optimizer decorrelates it; running totals this way are O(n²) (see [Correlated Subqueries](/lessons/m04-correlated-subqueries)) |

Note a subtle difference: in B, employee 124 (NULL department) would vanish from an unfiltered query because `NULL = NULL` is not true in the join, whereas `PARTITION BY department_id` puts NULLs in their own partition. Running totals are where the gap is largest: the correlated version `(SELECT SUM(salary) FROM employees x WHERE x.hire_date <= e.hire_date)` re-reads the qualifying rows for every row, while the window version reads them once after a sort. [Advanced Analytical Queries](/lessons/m04-analytical-queries) compares these strategies with `EXPLAIN ANALYZE`.

## Performance Analysis

**Plans.** Each window specification shows up as an iterator over a sort:

```sql run
EXPLAIN FORMAT=TREE
SELECT employee_id,
       SUM(salary) OVER (ORDER BY hire_date) AS by_hire,
       AVG(salary) OVER (PARTITION BY department_id) AS dept_avg
FROM employees;
```

Two windows with different specifications produce two window iterators, two sorts and a temporary table between them. Read the tree bottom-up: scan, sort by `hire_date`, first window aggregate, temporary table, sort by `department_id`, second window aggregate. Wherever you can, **share a window**: functions with the same `PARTITION BY`/`ORDER BY` use one sort, which is another reason to use named windows.

**Frame buffer and optimized evaluation.** The JSON plan exposes the details of a moving frame:

```sql run
EXPLAIN FORMAT=JSON
SELECT employee_id,
       SUM(salary) OVER (PARTITION BY department_id ORDER BY hire_date
                         ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_sum
FROM employees;
```

Look under `windowing`: `using_filesort` with the sort key, and `frame_buffer` with `optimized_frame_evaluation: true` — MySQL maintains the moving sum incrementally instead of re-adding three rows for every row.

**Does an index remove the sort?** It is natural to expect that an index on `(department_id, salary)` lets MySQL skip sorting for `PARTITION BY department_id ORDER BY salary`. Test it on a 50,000-row table:

```sql run
SET SESSION cte_max_recursion_depth = 60000;
CREATE TABLE win_big (id INT PRIMARY KEY, grp INT NOT NULL, val INT NOT NULL, KEY idx_grp_val (grp, val));
INSERT INTO win_big
WITH RECURSIVE seq (n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM seq WHERE n < 50000)
SELECT n, n % 100, (n * 7919) % 10007 FROM seq;
ANALYZE TABLE win_big;
SET SESSION cte_max_recursion_depth = DEFAULT;
```

```sql run
EXPLAIN FORMAT=TREE
SELECT grp, val, SUM(val) OVER (PARTITION BY grp ORDER BY val ROWS UNBOUNDED PRECEDING) AS running
FROM win_big;
```

In the captured plan the optimizer reads `win_big` through the index `idx_grp_val` — and **still** sorts by `grp, val` before the window iterator. Do not assume an index eliminates window sorts; check the plan on your server version. What an index *can* do is reduce the rows reaching the sort, when a selective `WHERE` uses it.

```sql run measured
EXPLAIN ANALYZE
SELECT grp, val, SUM(val) OVER (PARTITION BY grp ORDER BY val ROWS UNBOUNDED PRECEDING) AS running
FROM win_big;
```

:::performance[Theory vs measurement]
Expected cost of a window query ≈ cost of producing the input rows + one sort per distinct window specification (O(n log n)) + one linear pass per window, plus a frame buffer for look-ahead frames. Sorts spill to disk when they exceed `sort_buffer_size`; frame buffers and inter-window temporary tables spill when they exceed TempTable's memory limits. The measured `EXPLAIN ANALYZE` above shows where time actually goes on the build machine (look at the `Sort` line versus the `Window aggregate` line); measure your own data before tuning.
:::

## Best Practices

:::best-practice[Using window functions well]
1. **Always make window ordering deterministic** for `ROWS` frames and running totals: add a unique tie-breaker (`ORDER BY hire_date, employee_id`).
2. **Write the frame explicitly** whenever you use `ORDER BY` in an aggregate window. `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` for running totals; `RANGE … INTERVAL …` for calendar windows. Never rely on the default unless you *want* peer grouping.
3. **Use `RANGE` with intervals for time-based windows** over gappy data, or fill the gaps with a generated calendar before using `ROWS`.
4. **Name shared windows** with `WINDOW w AS (…)` — fewer typos, and identical specifications share one sort.
5. **Filter in an outer query.** Compute window results in a CTE, filter them outside; never try to put a window function in `WHERE`.
6. **Keep the final `ORDER BY`.** The window's ordering does not guarantee output order.
7. **Mind edge effects** of moving windows (the first rows have smaller frames) and decide whether to show, hide or flag them.
8. **Retire user-variable tricks** (`@x := @x + …`): deprecated and unreliable; window functions are the supported replacement.
:::

## Common Mistakes

**Mistake 1 — a window function in `WHERE`.** "Employees earning more than their department average":

```sql run error
SELECT employee_id, salary
FROM employees
WHERE salary > AVG(salary) OVER (PARTITION BY department_id);
```

Error 3593: windows are computed after `WHERE`. Compute first, filter outside:

```sql run
WITH w AS (
  SELECT employee_id, department_id, salary,
         AVG(salary) OVER (PARTITION BY department_id) AS dept_avg
  FROM employees
)
SELECT employee_id, department_id, salary, ROUND(dept_avg, 2) AS dept_avg
FROM w
WHERE salary > dept_avg
ORDER BY department_id, employee_id;
```

**Mistake 2 — trusting the default frame with ties.** Shown in Practical Example 1: with duplicate order keys, a "running total" written as `SUM(x) OVER (ORDER BY k)` jumps over all peers at once. Write `ROWS` and add a tie-breaker.

**Mistake 3 — unsupported constructs.** `COUNT(DISTINCT …)` cannot be used as a window function in MySQL:

```sql run error
SELECT employee_id,
       COUNT(DISTINCT job_id) OVER (PARTITION BY department_id) AS distinct_jobs
FROM employees;
```

Error 1235. Workarounds: compute the distinct count with `GROUP BY` and join it back, or use `MAX(DENSE_RANK() …)` in a two-step query (the ranking lesson shows the trick).

**Mistake 4 — a `RANGE` offset with a composite or non-numeric order key.**

```sql run error
SELECT employee_id,
       SUM(salary) OVER (ORDER BY salary, employee_id
                         RANGE BETWEEN 1000 PRECEDING AND CURRENT ROW) AS s
FROM employees;
```

Error 3587: a `RANGE` frame with `n PRECEDING/FOLLOWING` needs exactly one numeric or temporal `ORDER BY` expression, because "1000 less than the current key" must be computable.

**Mistake 5 — redefining the partition of a referenced window.**

```sql run error
SELECT employee_id, SUM(salary) OVER (w PARTITION BY job_id) AS s
FROM employees
WINDOW w AS (ORDER BY hire_date);
```

Error 3581: a window that builds on another may add ordering or a frame, but cannot define partitioning.

**Mistake 6 — confusing the window's `ORDER BY` with the output order.** A query with `SUM(x) OVER (ORDER BY d)` but no final `ORDER BY` may return rows in any order. Always add the outer `ORDER BY`.

## Industry Application

- **Financial reporting.** Running balances, year-to-date revenue and month-over-month change are window aggregates over a time-ordered partition per account or business unit. Ledgers order by `(posted_at, entry_id)` precisely to make running balances deterministic.
- **Product analytics dashboards.** 7-day and 28-day moving averages of active users or revenue smooth out weekday effects; they are `RANGE … INTERVAL` windows over a calendar-filled daily table, often precomputed into summary tables by a nightly job.
- **Compensation and HR analytics.** Comparing each salary with the band, department median or peer average — exactly the `PARTITION BY department_id` pattern — feeds pay-equity reviews.
- **Operations and monitoring.** Rolling error rates and latency percentiles per service, over the last N minutes, detect anomalies; window functions compute them over event tables or rollups.
- **Feature engineering for ML.** Data scientists derive "customer's average order value so far", "days since previous purchase" and "rank of this purchase" with window functions, taking care that frames only include *past* rows to avoid leaking future information into training data.
- **Data quality.** Windows detect duplicates and out-of-order events within a key (`COUNT(*) OVER (PARTITION BY natural_key) > 1`).

## Practice Exercises

```exercise
id: ex1
title: Salary versus department average
difficulty: easy
concepts: [window-functions, partition-by]
prompt: |
  For every employee **with a department**, return `employee_id`, `department_id`, `salary` and
  `dept_avg` — the average salary of the employee's department, rounded to 2 decimals.
hints:
  - "AVG(salary) OVER (PARTITION BY department_id) computes the department average on every row."
  - "Filter WHERE department_id IS NOT NULL before the window is computed."
solution: |
  SELECT employee_id, department_id, salary,
         ROUND(AVG(salary) OVER (PARTITION BY department_id), 2) AS dept_avg
  FROM employees
  WHERE department_id IS NOT NULL;
explanation: |
  The window function keeps every row and adds the partition's average. Because WHERE runs first, employee 124
  is removed before partitioning. A GROUP BY solution would return one row per department instead of one row
  per employee.
```

```exercise
id: ex2
title: Department share of payroll
difficulty: medium
concepts: [aggregate-window-functions, window-functions]
prompt: |
  Return one row per department (ignore employees without a department) with `department_id`,
  `payroll` (sum of salaries) and `pct_of_total` — the department's percentage of the payroll of all
  employees **with** a department, rounded to 1 decimal. Use a window function over the grouped result.
hints:
  - "GROUP BY department_id gives the per-department SUM(salary)."
  - "SUM(SUM(salary)) OVER () is the total of the grouped sums."
solution: |
  SELECT department_id,
         SUM(salary) AS payroll,
         ROUND(100 * SUM(salary) / SUM(SUM(salary)) OVER (), 1) AS pct_of_total
  FROM employees
  WHERE department_id IS NOT NULL
  GROUP BY department_id;
explanation: |
  The inner `SUM(salary)` is the ordinary group aggregate; the outer `SUM(…) OVER ()` is a window over the
  grouped rows (windows are computed after GROUP BY). A scalar subquery for the total also works but scans the
  table a second time.
```

```exercise
id: ex3
title: A correct running total despite ties
difficulty: medium
concepts: [window-frames, rows-vs-range]
prompt: |
  For Engineering (`department_id = 20`), return `employee_id`, `salary` and `running_total`: the sum of
  salaries from the lowest salary up to and including the current row, **one row at a time**, ordering by
  salary and breaking ties by `employee_id`. Employees 107 and 108 must have *different* running totals.
hints:
  - "The default frame with ORDER BY is RANGE, which lumps peers (equal salaries) together."
  - "Use ORDER BY salary, employee_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW."
solution: |
  SELECT employee_id, salary,
         SUM(salary) OVER (ORDER BY salary, employee_id
                           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
  FROM employees
  WHERE department_id = 20;
explanation: |
  `SUM(salary) OVER (ORDER BY salary)` gives 107 and 108 the same total because RANGE includes all peers of the
  current row. Adding `employee_id` alone to the ORDER BY would also break the tie (no peers remain), but
  writing `ROWS` explicitly documents the intent and protects against future duplicate keys.
```

```exercise
id: ex4
title: Three-day rolling revenue over gaps
difficulty: hard
concepts: [rows-vs-range, window-frames]
dataset: empty
setup: |
  CREATE TABLE daily_sales (sale_date DATE PRIMARY KEY, amount DECIMAL(10,2) NOT NULL);
  INSERT INTO daily_sales VALUES
    ('2024-03-01', 100), ('2024-03-02', 200), ('2024-03-05', 400),
    ('2024-03-06', 300), ('2024-03-08', 500), ('2024-03-09', 100);
prompt: |
  For each row of `daily_sales`, return `sale_date`, `amount` and `rolling_3d`: the total sales of the
  **calendar days** from two days before `sale_date` up to and including `sale_date`. Days without a row count
  as zero.
hints:
  - "ROWS 2 PRECEDING counts rows, not days — wrong when days are missing."
  - "RANGE BETWEEN INTERVAL 2 DAY PRECEDING AND CURRENT ROW works on the date values."
solution: |
  SELECT sale_date, amount,
         SUM(amount) OVER (ORDER BY sale_date
                           RANGE BETWEEN INTERVAL 2 DAY PRECEDING AND CURRENT ROW) AS rolling_3d
  FROM daily_sales;
explanation: |
  On 2024-03-05 the correct window (3–5 March) contains only that day, so the result is 400; a ROWS frame would
  wrongly add 1 and 2 March (700). RANGE with a temporal offset requires exactly one DATE/DATETIME/numeric
  ORDER BY expression.
```

```exercise
id: ex5
title: Above-average earners per department
difficulty: challenge
concepts: [window-functions, partition-by, named-windows]
prompt: |
  List employees who earn **more than their department's average** salary. Return `employee_id`,
  `department_id`, `salary`, `dept_avg` (rounded to 2 decimals) and `rank_in_dept`, where `rank_in_dept` is the
  number of employees in the same department with a salary **greater than or equal to** the employee's own
  salary (so the top earner has 1, and tied employees share the larger count). Ignore employees without a
  department. Define the department window once with a `WINDOW` clause.
hints:
  - "Compute the window values in a CTE, then filter salary > dept_avg in the outer query."
  - "COUNT(*) over a window ordered by salary DESC with the default RANGE frame counts all rows with salary ≥ the current one — peers included."
  - "WINDOW d AS (PARTITION BY department_id); then AVG(salary) OVER d and COUNT(*) OVER (d ORDER BY salary DESC)."
solution: |
  WITH w AS (
    SELECT employee_id, department_id, salary,
           AVG(salary) OVER d                      AS dept_avg,
           COUNT(*)    OVER (d ORDER BY salary DESC) AS rank_in_dept
    FROM employees
    WHERE department_id IS NOT NULL
    WINDOW d AS (PARTITION BY department_id)
  )
  SELECT employee_id, department_id, salary, ROUND(dept_avg, 2) AS dept_avg, rank_in_dept
  FROM w
  WHERE salary > dept_avg;
explanation: |
  This exercise uses the default RANGE frame *deliberately*: with `ORDER BY salary DESC`, the frame runs from
  the top salary to the current row's last peer, so `COUNT(*)` is "how many earn at least as much as me". The
  filter must be applied outside the CTE because window functions cannot appear in WHERE. (The next lesson's
  RANK() is a related but different measure: it counts only strictly higher salaries, plus one.)
```

## Knowledge Assessment

```quiz
- id: q1
  question: "What is the defining difference between `SUM(salary) OVER (PARTITION BY department_id)` and `SUM(salary) … GROUP BY department_id`?"
  options:
    - "The window version is always faster"
    - "The window version keeps one row per input row and attaches the partition's sum; GROUP BY collapses each group into one row"
    - "GROUP BY ignores NULL departments; PARTITION BY does not"
    - "There is no difference"
  answer: 1
  explanation: "Window functions never change the number of rows. Both treat NULL department as its own group/partition; speed depends on the plan."
  concepts: [window-functions]
- id: q2
  question: "With no frame clause, what frame does `SUM(x) OVER (ORDER BY k)` use?"
  options:
    - "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
    - "RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
    - "The whole partition"
    - "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW"
  answer: 1
  explanation: "With ORDER BY, the default is RANGE from the partition start to the current row including all its peers. Without ORDER BY, the default is the whole partition."
  concepts: [window-frames, rows-vs-range]
- id: q3
  question: "Salaries ordered ascending are 125500, 146500, 150000, 150000, 153500. What does `SUM(salary) OVER (ORDER BY salary)` return for the first 150000 row?"
  options: ["422000", "572000", "150000", "Undefined"]
  answer: 1
  explanation: "The default RANGE frame includes both peers with salary 150000: 125500 + 146500 + 150000 + 150000 = 572000. 422000 would be the ROWS result for the first of the two peers."
  concepts: [rows-vs-range]
- id: q4
  question: "Daily sales exist for 1, 2 and 5 March. On 5 March, what does `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW` include, compared with `RANGE BETWEEN INTERVAL 2 DAY PRECEDING AND CURRENT ROW`?"
  options:
    - "Both include only 5 March"
    - "ROWS includes 1, 2 and 5 March; RANGE includes only 5 March"
    - "ROWS includes only 5 March; RANGE includes 1, 2 and 5 March"
    - "Both include 1, 2 and 5 March"
  answer: 1
  explanation: "ROWS counts physical rows regardless of date gaps; RANGE uses the date values (3–5 March), and only 5 March has a row in that range."
  concepts: [rows-vs-range]
- id: q5
  question: "Which of the following are rejected by MySQL 8.4? (Select all that apply.)"
  type: multi
  options:
    - "COUNT(DISTINCT job_id) OVER (PARTITION BY department_id)"
    - "SUM(SUM(salary)) OVER () in a query with GROUP BY"
    - "A GROUPS frame unit"
    - "WHERE ROW_NUMBER() OVER (ORDER BY salary) = 1"
    - "AVG(salary) OVER w … WINDOW w AS (PARTITION BY department_id)"
  answer: [0, 2, 3]
  explanation: "DISTINCT in window aggregates and the GROUPS unit are not supported (error 1235); window functions are not allowed in WHERE (error 3593). Windows over grouped aggregates and named windows are fine."
  concepts: [window-functions, named-windows]
- id: q6
  question: "A query computes `SUM(x) OVER (ORDER BY d)` and `AVG(x) OVER (PARTITION BY g)`. What does EXPLAIN FORMAT=TREE typically show?"
  options:
    - "One sort shared by both windows"
    - "Two window iterators with two sorts and a temporary table between them"
    - "No sort, because window functions read rows in index order"
    - "A nested-loop join between the two windows"
  answer: 1
  explanation: "Windows with different specifications are evaluated one after another, each preceded by its own sort, with intermediate results in a temporary table. Identical specifications can share a sort."
  concepts: [window-functions]
- id: q7
  question: "Why is the old idiom `SELECT @run := @run + amount …` a bad way to compute running totals in MySQL 8.x?"
  options:
    - "It is too slow"
    - "Its result depends on the undefined order in which MySQL evaluates rows and expressions, and setting variables in expressions is deprecated"
    - "User variables cannot hold DECIMAL values"
    - "It only works with MyISAM"
  answer: 1
  explanation: "The idiom relied on unspecified evaluation order and could give wrong answers when the plan changed; MySQL 8.0 deprecated assigning user variables inside expressions. Window functions are the supported replacement."
  concepts: [aggregate-window-functions]
- id: q8
  question: "In which logical phase are window functions evaluated?"
  options:
    - "Before WHERE"
    - "Between WHERE and GROUP BY"
    - "After GROUP BY/HAVING and before DISTINCT/ORDER BY/LIMIT"
    - "After LIMIT"
  answer: 2
  explanation: "Windows operate on the rows remaining after filtering and grouping, which is why they can aggregate group results but cannot be used in WHERE or HAVING; ORDER BY and LIMIT then apply to the windowed rows."
  concepts: [window-functions]
```
