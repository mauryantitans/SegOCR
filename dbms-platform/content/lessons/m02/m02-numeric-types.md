---
id: m02-numeric-types
dataset: shop
summary: MySQL's integer, fixed-point, floating-point, BIT and BOOLEAN types — their ranges and storage sizes, how strict mode rejects out-of-range values, why DECIMAL is exact and FLOAT/DOUBLE are not, what the deprecated display width and ZEROFILL really did, and how to choose the right type.
concepts: [integer-types, unsigned, decimal-precision, floating-point, bit-boolean, strict-mode, choosing-types]
references:
  - title: "MySQL 8.4 Reference Manual — Numeric Data Types"
    url: https://dev.mysql.com/doc/refman/8.4/en/numeric-types.html
  - title: "MySQL 8.4 Reference Manual — Integer Types (Exact Value)"
    url: https://dev.mysql.com/doc/refman/8.4/en/integer-types.html
  - title: "MySQL 8.4 Reference Manual — Fixed-Point Types (Exact Value) - DECIMAL, NUMERIC"
    url: https://dev.mysql.com/doc/refman/8.4/en/fixed-point-types.html
  - title: "MySQL 8.4 Reference Manual — Floating-Point Types (Approximate Value)"
    url: https://dev.mysql.com/doc/refman/8.4/en/floating-point-types.html
  - title: "MySQL 8.4 Reference Manual — Out-of-Range and Overflow Handling"
    url: https://dev.mysql.com/doc/refman/8.4/en/out-of-range-and-overflow.html
  - title: "MySQL 8.4 Reference Manual — Numeric Type Attributes"
    url: https://dev.mysql.com/doc/refman/8.4/en/numeric-type-attributes.html
  - title: "MySQL 8.4 Reference Manual — Data Type Storage Requirements"
    url: https://dev.mysql.com/doc/refman/8.4/en/storage-requirements.html
---

## Introduction

Choosing a numeric type looks trivial — "it's a number, use `INT`" — until a primary key overflows at 2,147,483,647 and the application stops accepting orders, an invoice total is off by a cent because it was summed as `FLOAT`, or a percentage column silently rounds 12.345 to 12.35. Numeric types are the most common place where a column's **domain** (from relational theory) meets **physical storage** (bytes on a page) and **arithmetic semantics** (exact versus approximate).

MySQL offers four families:

| Family | Types | Values are |
|---|---|---|
| Integer | `TINYINT`, `SMALLINT`, `MEDIUMINT`, `INT`, `BIGINT` (optionally `UNSIGNED`) | Exact whole numbers |
| Fixed-point | `DECIMAL(M,D)` (synonyms `NUMERIC`, `DEC`, `FIXED`) | Exact decimal numbers with a fixed number of digits |
| Floating-point | `FLOAT`, `DOUBLE` (`REAL` = `DOUBLE` by default) | Approximate binary fractions |
| Bit / boolean | `BIT(M)`; `BOOLEAN`/`BOOL` (alias of `TINYINT(1)`) | Bit fields; truth values stored as small integers |

This lesson shows, with real outputs, each type's range and storage size, what MySQL does with values that do not fit (errors in strict mode, clamping with warnings otherwise), how `DECIMAL` rounding works, why `FLOAT` comparisons fail, what the **deprecated** display width (`INT(11)`), `ZEROFILL` and `FLOAT(M,D)` syntax really meant, and how to choose types professionally.

:::note[Dataset]
The examples use small demonstration tables plus the **shop** dataset, whose money columns (`price`, `cost`, `amount`, `unit_price`) are `DECIMAL(10,2)` and whose ids are `INT UNSIGNED`.
:::

## DBMS Theory

:::theory[Domains, exactness and representation]
In the relational model every attribute draws its values from a **domain** — a set of permissible values ([The Relational Model](/lessons/m01-relational-model)). A data type is the DBMS's implementation of a domain: it defines (1) the **set of values** (e.g. integers from −2³¹ to 2³¹−1), (2) the **operations** and their semantics (is division exact? how is rounding done?), and (3) the **physical representation** (how many bytes, which encoding).

The SQL standard divides numeric types into **exact numeric** types — `SMALLINT`, `INTEGER`, `BIGINT`, `NUMERIC(p,s)`, `DECIMAL(p,s)` — whose values are represented exactly, and **approximate numeric** types — `FLOAT(p)`, `REAL`, `DOUBLE PRECISION` — represented as binary floating-point with a mantissa and an exponent, where most decimal fractions can only be approximated.
:::

Two consequences of this theory guide every choice in this lesson:

1. **Exact types are for counting and money; approximate types are for measurement.** A count of items, a balance in cents or a price must be exact — `0.1 + 0.2` must equal `0.3`. A sensor reading, a coordinate or a model weight is already an approximation of the physical world; a tiny relative error is harmless, and the huge range and speed of floating-point are worth it.
2. **A narrower domain is an integrity constraint.** A `TINYINT UNSIGNED` rating column cannot hold 300 or −1; the type itself rejects nonsense. Choosing the tightest type that comfortably fits the domain is a form of data validation (complemented by `CHECK` constraints — [Constraints in MySQL](/lessons/m02-constraints-mysql)).

**Two's complement and unsigned.** Integers are stored in binary. A signed n-bit integer (two's complement) covers −2ⁿ⁻¹ … 2ⁿ⁻¹−1; an unsigned one covers 0 … 2ⁿ−1. `UNSIGNED` doesn't add storage — it shifts the range to non-negative values, doubling the positive maximum.

**IEEE 754.** `FLOAT` is a 32-bit IEEE 754 single-precision number (24-bit significand ≈ 7 significant decimal digits) and `DOUBLE` a 64-bit double-precision number (53-bit significand ≈ 15–17 digits). Integers above 2²⁴ = 16,777,216 cannot all be represented in `FLOAT`; decimal fractions such as 0.1 cannot be represented exactly in either.

## Why It Exists

Why not one "number" type for everything?

- **Storage and performance.** A `BIGINT` takes 8 bytes, a `TINYINT` 1. In InnoDB every secondary index entry also stores the primary key, so a wide primary key makes *every* index larger. Over hundreds of millions of rows and several indexes, the difference between `INT` (4 bytes) and `BIGINT` (8 bytes) — or between `INT` and a `DECIMAL(20,0)` used as an id — is gigabytes of disk and buffer pool.
- **Correct arithmetic.** Financial systems need exact decimal arithmetic with defined rounding; scientific data needs range (10⁻³⁰⁸ … 10³⁰⁸) more than exactness. No single representation provides both efficiently.
- **Validation.** Types enforce domains. Without them, invalid values would have to be caught in every application.

What goes wrong with poor choices:

- **Key exhaustion.** A signed `INT AUTO_INCREMENT` key tops out at 2,147,483,647. High-volume tables (events, logs, messages) reach that, and then every insert fails. Changing the type of a primary key on a huge table is a COPY operation ([Creating, Altering and Dropping Tables](/lessons/m02-tables-ddl)) — a major migration.
- **Money in FLOAT.** Sums drift by fractions of a cent; reconciliations fail; equality checks (`WHERE amount = 19.99`) miss rows.
- **Silent data change.** In non-strict SQL modes, MySQL clamps out-of-range values to the nearest limit and only warns — data is silently corrupted.

## Internal Working

:::internals[How InnoDB stores numeric values]
**Integers** are stored in fixed width (1, 2, 3, 4 or 8 bytes) in **big-endian** byte order with the **sign bit inverted**. This encoding makes the raw bytes of keys sort in the same order as the numeric values, so InnoDB can compare B+ tree keys with a simple byte comparison (`memcmp`). Signed and unsigned integers of the same width use the same space.

**DECIMAL(M,D)** uses a packed **binary-coded decimal-like format**: the integer part and the fractional part are stored separately, each in groups of **9 decimal digits packed into 4 bytes**, with leftover digits taking 1–4 bytes:

| Leftover digits | 0 | 1–2 | 3–4 | 5–6 | 7–9 |
|---|---|---|---|---|---|
| Bytes | 0 | 1 | 2 | 3 | 4 |

So `DECIMAL(10,2)` = 8 integer digits (4 bytes: 8 leftover digits) + 2 fractional digits (1 byte) = **5 bytes**; `DECIMAL(18,9)` = 9 + 9 digits = 4 + 4 = 8 bytes; `DECIMAL(65,30)` is the maximum precision. Arithmetic on DECIMAL uses MySQL's own precision-math library, with up to 65 digits, and is exact.

**FLOAT/DOUBLE** are stored as raw IEEE 754 values: 4 or 8 bytes. Arithmetic uses the CPU's floating-point unit — fast, but approximate. When MySQL *displays* a `FLOAT`, it prints it with about 6 significant digits, so the displayed value can hide the representation error that comparisons expose.

**BIT(M)** occupies (M + 7) / 8 bytes (rounded down), i.e. 1 byte for BIT(1)…BIT(8). **BOOLEAN** is just `TINYINT(1)`: 1 byte, any value −128…127.

**NULL** costs no space in the value area: InnoDB's `DYNAMIC`/`COMPACT` row formats keep a NULL bitmap with one bit per nullable column in the row header.
:::

| Type | Bytes | Signed range | Unsigned range |
|---|---|---|---|
| `TINYINT` | 1 | −128 … 127 | 0 … 255 |
| `SMALLINT` | 2 | −32,768 … 32,767 | 0 … 65,535 |
| `MEDIUMINT` | 3 | −8,388,608 … 8,388,607 | 0 … 16,777,215 |
| `INT` / `INTEGER` | 4 | −2,147,483,648 … 2,147,483,647 | 0 … 4,294,967,295 |
| `BIGINT` | 8 | −2⁶³ … 2⁶³−1 (≈ ±9.22 × 10¹⁸) | 0 … 2⁶⁴−1 (≈ 1.84 × 10¹⁹) |
| `DECIMAL(M,D)` | packed, 4 bytes per 9 digits | M ≤ 65 digits, D ≤ 30, D ≤ M | (UNSIGNED deprecated) |
| `FLOAT` | 4 | ≈ ±3.4 × 10³⁸, ~7 significant digits | (UNSIGNED deprecated) |
| `DOUBLE` | 8 | ≈ ±1.8 × 10³⁰⁸, ~15–17 significant digits | (UNSIGNED deprecated) |
| `BIT(M)` | ⌊(M+7)/8⌋ | M = 1 … 64 bits | — |

## MySQL Implementation

Create a table with one column of each family and look at how MySQL records the types:

```sql run
CREATE TABLE numeric_demo (
  tiny_signed    TINYINT,
  tiny_unsigned  TINYINT UNSIGNED,
  price          DECIMAL(5,2),
  approx_single  FLOAT,
  approx_double  DOUBLE,
  flags          BIT(8),
  is_enabled     BOOLEAN
);
SELECT COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, NUMERIC_PRECISION, NUMERIC_SCALE
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'numeric_demo'
ORDER BY ORDINAL_POSITION;
```

`BOOLEAN` was stored as `tinyint(1)`. `NUMERIC_PRECISION` is the number of decimal digits (3 for TINYINT, whose max 255 has 3 digits; 5 for DECIMAL(5,2)), or for FLOAT/DOUBLE the number of digits MySQL reports for display purposes; for BIT it is the number of bits.

**Strict mode rejects out-of-range values.** The default SQL mode includes `STRICT_TRANS_TABLES`, so an insert that does not fit fails with error 1264 and nothing is stored:

```sql run error
INSERT INTO numeric_demo (tiny_signed) VALUES (128);
```

```sql run error
INSERT INTO numeric_demo (tiny_unsigned) VALUES (-1);
```

**DECIMAL rounds the fraction, rejects the integer part.** A value with too many *fractional* digits is rounded (with a Note, not an error); a value with too many *integer* digits is out of range:

```sql run
INSERT INTO numeric_demo (price) VALUES (1.005);
SELECT price FROM numeric_demo WHERE price IS NOT NULL;
```

`1.005` was stored as `1.01` — rounded half away from zero — and MySQL raised Note 1265 *"Data truncated for column 'price'"*. `DECIMAL(5,2)` has 3 integer digits, so its range is −999.99 … 999.99:

```sql run error
INSERT INTO numeric_demo (price) VALUES (1234.5);
```

Even `999.995` fails, because rounding it to two decimals gives 1000.00, which no longer fits.

```sql run error
INSERT INTO numeric_demo (price) VALUES (999.995);
```

## Syntax Breakdown

```sql
{TINYINT | SMALLINT | MEDIUMINT | INT | INTEGER | BIGINT} [UNSIGNED]
DECIMAL[(M[,D])] | NUMERIC[(M[,D])]          -- default M = 10, D = 0
FLOAT | DOUBLE [PRECISION] | REAL
FLOAT(p)                                     -- p 0..24 → FLOAT, 25..53 → DOUBLE (standard syntax, still valid)
BIT[(M)]                                     -- default M = 1
BOOL | BOOLEAN                               -- = TINYINT(1)
SERIAL                                       -- = BIGINT UNSIGNED NOT NULL AUTO_INCREMENT UNIQUE
```

| Element | Meaning | Standard / MySQL | Status |
|---|---|---|---|
| `SMALLINT`, `INTEGER`/`INT`, `BIGINT` | Exact integers | Standard | Current |
| `TINYINT`, `MEDIUMINT` | 1- and 3-byte integers | MySQL extension | Current |
| `UNSIGNED` on integers | Non-negative range, same storage | MySQL extension | Current |
| `UNSIGNED` on `DECIMAL`/`FLOAT`/`DOUBLE` | Disallows negatives (no range gain) | MySQL extension | **Deprecated since 8.0.17** — use a `CHECK (col >= 0)` constraint |
| `DECIMAL(M,D)` / `NUMERIC(M,D)` | M total digits, D after the point | Standard (MySQL implements NUMERIC as DECIMAL) | Current |
| `FLOAT`, `DOUBLE`, `REAL`, `FLOAT(p)` | IEEE 754 single/double | Standard | Current |
| `FLOAT(M,D)`, `DOUBLE(M,D)` | Display rounding to D decimals | MySQL extension | **Deprecated since 8.0.17** |
| `INT(N)` — display width | Formerly a *display hint* used with ZEROFILL; never limited the range | MySQL extension | **Deprecated since 8.0.17**; not shown in `SHOW CREATE TABLE` (except `tinyint(1)`) |
| `ZEROFILL` | Left-pads displayed value with zeros to the display width; implies UNSIGNED | MySQL extension | **Deprecated since 8.0.17** — use `LPAD()` or a `CHAR` column |
| `BIT(M)` | Bit field, 1–64 bits | MySQL (standard BIT was removed from SQL:2003) | Current |
| `BOOLEAN` | Alias of `TINYINT(1)`; `TRUE` = 1, `FALSE` = 0 | Standard name, non-standard implementation (no real boolean type) | Current |
| `SERIAL` | Shorthand for an auto-increment `BIGINT UNSIGNED` | MySQL/PostgreSQL-style alias | Current |

:::version[Deprecated numeric syntax — what changed and when]
Since **MySQL 8.0.17** the following produce deprecation warning 1681 and are slated for removal in a future version (still accepted in 8.4):

- integer display width (`INT(11)`, `BIGINT(20)`) — except `TINYINT(1)`, kept because connectors use it to recognise booleans;
- `ZEROFILL`;
- `FLOAT(M,D)` / `DOUBLE(M,D)`;
- `UNSIGNED` on `DECIMAL`, `FLOAT` and `DOUBLE`.

Since **8.0.19**, `SHOW CREATE TABLE` and `information_schema.COLUMNS.COLUMN_TYPE` no longer display integer widths unless `ZEROFILL` is used. Tools that compare schemas across 5.7 and 8.x see `int(11)` vs `int` — a cosmetic difference, not a type change.
:::

**Arithmetic rules worth knowing:**

| Expression | Result type / behaviour |
|---|---|
| integer `+ - *` integer | `BIGINT`; overflow → error 1690 (no silent wrap-around) |
| any operand `UNSIGNED` | Result is `BIGINT UNSIGNED`; a negative result → error 1690 (unless `NO_UNSIGNED_SUBTRACTION` mode) |
| `a / b` with exact operands | `DECIMAL`, scale = scale(a) + `div_precision_increment` (default 4) |
| `a DIV b` | Integer division (truncates) |
| `/ 0`, `% 0` | `NULL` with warning 1365 (an *error* only when inserting/updating in strict mode with `ERROR_FOR_DIVISION_BY_ZERO`) |
| literal `0.1` | Exact `DECIMAL` literal |
| literal `0.1e0` / `1e-1` | `DOUBLE` literal |
| `ROUND(x)` on DECIMAL | Round half away from zero |
| `ROUND(x)` on DOUBLE | Uses the C library — typically round half to even ("banker's rounding") |

## Practical Examples

**1. Exact vs approximate in one line.** Decimal literals are exact in MySQL; `e`-notation literals are `DOUBLE`:

```sql run
SELECT 0.1 + 0.2 = 0.3           AS decimal_equal,
       0.1e0 + 0.2e0 = 0.3e0     AS double_equal,
       0.1e0 + 0.2e0             AS double_sum;
```

The decimal comparison is true; the double comparison is false because the binary sum is 0.30000000000000004.

**2. The FLOAT equality trap.** Store 0.1 in a `FLOAT` and a `DOUBLE` column, then look it up:

```sql run
INSERT INTO numeric_demo (approx_single, approx_double) VALUES (0.1, 0.1);
SELECT approx_single,
       approx_double,
       approx_single = 0.1             AS float_equals_0_1,
       approx_double = 0.1             AS double_equals_0_1,
       CAST(approx_single AS DOUBLE)   AS float_actual_value
FROM numeric_demo
WHERE approx_single IS NOT NULL;
```

Both columns *display* `0.1`, but the `FLOAT` column fails the equality test: what is stored is the nearest single-precision value, 0.10000000149011612, which differs from the double-precision value of 0.1 used in the comparison. The `DOUBLE` column happens to match because the literal is converted to the same double. **Never use `=` on floating-point columns**; compare with a tolerance (`ABS(x - 0.1) < 1e-6`) or use `DECIMAL`.

**3. FLOAT's 24-bit significand.** Integers above 2²⁴ cannot all be represented in `FLOAT`:

```sql run
CREATE TABLE float_precision (f FLOAT, d DOUBLE);
INSERT INTO float_precision VALUES (16777217, 16777217), (123456789, 123456789);
SELECT f, f + 0 AS float_stored, d FROM float_precision ORDER BY d;
```

16,777,217 became 16,777,216 and 123,456,789 became 123,456,792 in the `FLOAT` column (`f + 0` reveals the stored value; plain `f` is displayed with only ~6 significant digits). The `DOUBLE` column stores both exactly. This is why ids, counters and anything that must be exact never belong in floating-point columns.

**4. Rounding: DECIMAL vs DOUBLE.**

```sql run
SELECT ROUND(2.5)    AS dec_2_5,
       ROUND(3.5)    AS dec_3_5,
       ROUND(-2.5)   AS dec_neg_2_5,
       ROUND(2.5e0)  AS dbl_2_5,
       ROUND(3.5e0)  AS dbl_3_5;
```

Exact values round half **away from zero** (2.5 → 3, −2.5 → −3). Approximate values use the C library's rounding, which on this server rounds half to **even** (2.5 → 2, 3.5 → 4). Two "identical" reports can therefore disagree if one computes on `DOUBLE` and the other on `DECIMAL`.

**5. Division and integer division.**

```sql run
SELECT 7 / 2    AS slash,
       7 DIV 2  AS div_operator,
       7 % 2    AS modulo,
       1 / 3    AS one_third,
       1 / 0    AS by_zero;
```

`/` on integers produces a `DECIMAL` with 4 extra decimal places (`div_precision_increment`), not an integer — unlike Java, C or PostgreSQL integer division. `DIV` truncates. Division by zero returns `NULL` with warning 1365 in a `SELECT`.

**6. Unsigned arithmetic surprises.** If any operand is unsigned, the result is unsigned — and a negative result is an error, not a negative number:

```sql run error
SELECT CAST(1 AS UNSIGNED) - 2;
```

This bites in queries such as `stock_qty_unsigned - ordered_qty`. Cast to signed (`CAST(a AS SIGNED) - b`) when the difference may be negative. Signed overflow is an error too — MySQL never silently wraps around:

```sql run error
SELECT 9223372036854775807 + 1;
```

**7. BIT and BOOLEAN.**

```sql run
INSERT INTO numeric_demo (flags, is_enabled) VALUES (b'1010', 2);
SELECT flags + 0      AS flags_as_number,
       BIN(flags)     AS flags_binary,
       is_enabled,
       is_enabled = TRUE AS equals_true,
       is_enabled IS TRUE AS is_true
FROM numeric_demo
WHERE flags IS NOT NULL;
```

`BIT` values are binary strings; use `col + 0` or `BIN(col)` to read them (a raw `BIT` value is sent to clients as bytes). `BOOLEAN` accepted the value 2 — it is a `TINYINT`. Note the trap: `is_enabled = TRUE` compares with the integer 1 and is **false** for 2, while `is_enabled IS TRUE` tests "non-zero" and is true. Constrain boolean columns with `CHECK (is_enabled IN (0, 1))` if other code might write other values.

**8. Display width and ZEROFILL — deprecated.** These clauses still parse, with warnings:

```sql run
CREATE TABLE legacy_widths (
  a  INT(11),
  z  INT(5) ZEROFILL,
  fl FLOAT(7,2),
  du DECIMAL(5,2) UNSIGNED
);
SHOW CREATE TABLE legacy_widths;
```

Five deprecation warnings (code 1681). In the stored definition, `INT(11)` became plain `int` — the width never meant anything for storage or range. `ZEROFILL` kept its width and implicitly added `UNSIGNED`. The width only affects formatting:

```sql run
INSERT INTO legacy_widths (a, z) VALUES (2147483647, 42);
SELECT a, z, CAST(z AS CHAR) AS z_as_text, LPAD(42, 5, '0') AS modern_alternative FROM legacy_widths;
```

`INT(11)` stored 2,147,483,647 — its full 4-byte range — regardless of the "11". `ZEROFILL` padded 42 to `00042` only when converted to text (the numeric value sent to clients is 42). Use `LPAD()` in queries or a `CHAR(5)` column if leading zeros are part of the data (ZIP codes, account numbers — which are identifiers, not numbers).

## Execution Walkthrough

What happens when MySQL inserts the value `'12.345'` into a `DECIMAL(4,2)` column, under strict and non-strict modes? The steps below trace the conversion with real statements.

```walkthrough
title: INSERT INTO dec_target VALUES ('12.345') — conversion, rounding, range check
steps:
  - title: "1. The target column"
    explain: >-
      DECIMAL(4,2) has 2 integer digits and 2 fractional digits: range -99.99 … 99.99. Storage: 1 byte for the
      2 integer digits + 1 byte for the 2 fractional digits.
    sql: |
      CREATE TABLE dec_target (v DECIMAL(4,2));
      SELECT COLUMN_TYPE, NUMERIC_PRECISION, NUMERIC_SCALE FROM information_schema.COLUMNS
      WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'dec_target';
  - title: "2. Convert the string, round the fraction"
    explain: >-
      The string '12.345' is converted to an exact decimal, then rounded to scale 2 (half away from zero) →
      12.35. Losing fractional digits is only a Note (1265), even in strict mode.
    sql: |
      INSERT INTO dec_target VALUES ('12.345');
      SELECT v FROM dec_target;
  - title: "3. Integer part too large — strict mode"
    explain: >-
      123.4 needs 3 integer digits. Under STRICT_TRANS_TABLES the statement fails with error 1264 and no row
      is inserted.
    sql: INSERT INTO dec_target VALUES (123.4);
    expect: error
  - title: "4. The same value without strict mode"
    explain: >-
      With an empty sql_mode, MySQL clamps the value to the column's maximum (99.99) and only warns (1264).
      The row is stored with a value nobody entered.
    sql: |
      SET SESSION sql_mode = '';
      INSERT INTO dec_target VALUES (123.4);
      SET SESSION sql_mode = DEFAULT;
      SELECT v FROM dec_target ORDER BY v;
```

## Output

The final table contains `12.35` (the rounded string) and `99.99` (the clamped value inserted with strict mode off). Read the outputs as a lesson in **what the client is told**:

- Rounding the fraction produced a *Note* — many client libraries never surface notes or warnings to the application.
- The out-of-range value produced an *error* in strict mode (the safe outcome: the application learns something is wrong) and only a *warning* in non-strict mode (the dangerous outcome: wrong data is stored).

This is why strict mode is the default since MySQL 5.7 and why you should never disable it. If your application deliberately relies on rounding, round explicitly in SQL (`ROUND(x, 2)`) so the intent is visible.

## Alternative Approaches

**Money.** Options, in order of preference for most systems:

| Representation | Example | Pros | Cons |
|---|---|---|---|
| `DECIMAL(p,s)` | `DECIMAL(12,2)`, or `DECIMAL(19,4)` for currencies with sub-cent pricing | Exact, readable, SQL arithmetic is exact | Slightly slower than integers; choose scale carefully |
| Integer minor units | `BIGINT` amount in cents | Exact, fastest, no rounding inside the DB | Scale lives in application code; currencies with 0 or 3 decimals need care |
| `DOUBLE` | — | — | **Wrong for money**: inexact sums and comparisons |

**Booleans.** `BOOLEAN` (`TINYINT(1)`) is the idiomatic MySQL choice; `BIT(1)` saves nothing per row (both 1 byte) and is awkward for clients; `ENUM('N','Y')` or `CHAR(1)` are legacy patterns. Several flags can be packed into one `BIT(M)` or `SET` column, at the cost of indexability and readability.

**Identifiers.** `INT UNSIGNED` (4.29 billion values) suffices for most entity tables; `BIGINT UNSIGNED` for events, logs and anything that could ever grow beyond that; UUIDs as `BINARY(16)` when ids must be generated outside the database ([JSON and Binary Data Types](/lessons/m02-json-binary-types)).

**Percentages and ratios.** Store `DECIMAL(5,2)` (e.g. 12.34 %) or a fraction `DECIMAL(6,5)`; use `DOUBLE` only for statistical outputs where approximation is acceptable.

**Scientific measurements.** `DOUBLE` (or `FLOAT` when precision of ~7 digits is enough and storage matters). Never compare with `=`.

## Performance Analysis

Numeric type choice affects performance through **size**, not arithmetic speed:

- **Row and index size.** A primary key is stored in the clustered index and repeated in every secondary index entry. Changing a key from `INT` (4 bytes) to `BIGINT` (8 bytes) adds 4 bytes per row to the clustered index and to each secondary index. For a table with 1 billion rows and 4 secondary indexes, that is about 4 bytes × 1 billion × 5 ≈ 20 GB more data — more pages to cache in the buffer pool and to read from disk. Conversely, choosing `INT` and running out of values is far more expensive than the extra bytes, so size keys for the table's *lifetime* growth.
- **Comparison cost.** Integer comparison is a fixed-width byte comparison; `DECIMAL` comparison is also byte-comparable in its packed format; floating-point comparison is fast but semantically fragile. In practice, CPU differences between numeric types are negligible next to I/O.
- **Type mismatches in predicates** can prevent index use. Comparing an integer column with an integer literal is ideal; comparing it with a string or a float forces conversions. A classic case is comparing a `VARCHAR` column with a number, which converts every row's string to a number and cannot use the index ([Sargable Queries](/lessons/m07-sargable-queries)).
- **Aggregation precision.** `SUM` over `DECIMAL` returns `DECIMAL` (exact); over `FLOAT`/`DOUBLE` returns `DOUBLE`, and the result can even depend on the order rows are processed, because floating-point addition is not associative.

:::performance[Theory vs measurement]
The storage figures above come from MySQL's documented storage requirements, not a benchmark. On your own server you can measure the effect of a type change by comparing `DATA_LENGTH` and `INDEX_LENGTH` in `information_schema.TABLES` (after `ANALYZE TABLE`) for two copies of a large table that differ only in key type.
:::

## Best Practices

:::best-practice[Choosing numeric types]
1. **Money → `DECIMAL`** with explicit precision and scale (e.g. `DECIMAL(12,2)`), or integer minor units. Never `FLOAT`/`DOUBLE`.
2. **Surrogate keys → `INT UNSIGNED` or `BIGINT UNSIGNED`.** Estimate lifetime growth; when in doubt for high-volume tables, choose `BIGINT`. Use the same type for the foreign keys that reference it.
3. **Small domains → small types** (`TINYINT UNSIGNED` for ratings, `SMALLINT UNSIGNED` for years of service), plus a `CHECK` constraint for the exact domain (`CHECK (rating BETWEEN 1 AND 5)`).
4. **Measurements → `DOUBLE`**; compare with tolerances, never with `=`.
5. **Do not use deprecated syntax** in new DDL: no `INT(11)`, no `ZEROFILL`, no `FLOAT(M,D)`, no `UNSIGNED` on `DECIMAL`/`FLOAT`/`DOUBLE`.
6. **Keep strict mode on** so out-of-range values are errors, not silent clamping.
7. **Store identifiers that look numeric but are not quantities as strings**: ZIP codes, phone numbers, account numbers with leading zeros, credit-card numbers. You never add two ZIP codes.
8. **Round only at presentation** (`ROUND(x, 2)` in the final SELECT or in the application), and use `DECIMAL` intermediate results so rounding happens once.
:::

## Common Mistakes

**Mistake 1 — thinking `INT(11)` means "11 digits".** As shown above, it is a deprecated display hint; `INT(1)` and `INT(11)` both store −2,147,483,648 … 2,147,483,647.

**Mistake 2 — running out of `AUTO_INCREMENT` values.** When the counter reaches the type's maximum, the next generated value collides with the last one:

```sql run
CREATE TABLE tiny_keys (id TINYINT AUTO_INCREMENT PRIMARY KEY, note VARCHAR(10));
INSERT INTO tiny_keys VALUES (127, 'last one');
```

```sql run error
INSERT INTO tiny_keys (note) VALUES ('one more');
```

The insert fails with duplicate-key error 1062 for value 127 — the counter cannot advance beyond `TINYINT`'s maximum. On an `INT` key the same thing happens at 2,147,483,647 and takes down production writes. Monitor key headroom: `SELECT AUTO_INCREMENT FROM information_schema.TABLES WHERE …` compared with the type maximum.

**Mistake 3 — money in `FLOAT`.** Totals drift and equality lookups fail, as in Practical Example 2. Migrate to `DECIMAL` (a COPY operation) and fix the reconciliation.

**Mistake 4 — unsigned subtraction.** `SELECT stock - reserved` where both are `UNSIGNED` fails with error 1690 as soon as `reserved > stock`. Cast to `SIGNED` or design the columns as signed with a `CHECK (stock >= 0)`.

**Mistake 5 — disabling strict mode to "fix" errors.** Out-of-range errors are the database protecting your data. With strict mode off, values are clamped (127 for TINYINT, 99.99 for DECIMAL(4,2)) and the application never knows.

**Mistake 6 — `WHERE flag = TRUE` on a TINYINT that contains other values.** Use `WHERE flag` / `flag IS TRUE` (non-zero), or constrain the column to 0/1.

## Industry Application

- **Payments and accounting.** Ledgers use `DECIMAL(19,4)` or `BIGINT` minor units, with currency in a separate `CHAR(3)` column; rounding rules (half-up vs half-even) are specified by finance and applied explicitly at defined steps, not left to floating-point behaviour.
- **Key-exhaustion incidents.** Several well-known production outages were caused by 32-bit integer ids overflowing. Mature teams alert at, for example, 70 % of the key type's range and plan `BIGINT` migrations with online schema-change tools well in advance.
- **Analytics and ML features.** Feature stores and telemetry tables use `DOUBLE`/`FLOAT` for metrics and embeddings (MySQL 9.0 added a dedicated `VECTOR` type for embeddings), and `DECIMAL` for monetary features. When data is exported to Python, `DECIMAL` arrives as `decimal.Decimal` and `DOUBLE` as `float` — the same exactness distinction continues in pandas/NumPy.
- **Storage optimisation at scale.** Large platforms shrink hot tables by narrowing types (`BIGINT` status codes → `TINYINT`, `DATETIME` → appropriate types), because smaller rows mean more rows per 16 KB page and a higher buffer-pool hit rate.
- **Data quality.** Validation pipelines check that numeric columns are within domain (`MIN`, `MAX`, `COUNT(*) WHERE amount < 0`) — but well-chosen types and `CHECK` constraints prevent many violations at write time.

## Practice Exercises

```exercise
id: ex1
title: Design a sensor table with correct types
difficulty: easy
kind: dml
dataset: empty
concepts: [integer-types, decimal-precision, floating-point]
prompt: |
  Create a table `sensor_sample` with these columns, choosing types as described:

  1. `sample_id` — `BIGINT UNSIGNED`, not null, auto-increment primary key
  2. `sensor_no` — the smallest **unsigned** integer type that can hold values 0–60,000
  3. `temperature_c` — an approximate 8-byte floating-point type
  4. `calibration_fee` — exact money with 8 digits in total and 2 after the decimal point

  The checker reads `COLUMN_NAME` and `COLUMN_TYPE` from `information_schema.COLUMNS`.
hints:
  - "TINYINT UNSIGNED holds up to 255, SMALLINT UNSIGNED up to 65,535."
  - "8-byte floating point is DOUBLE; exact money is DECIMAL(M,D)."
solution: |
  CREATE TABLE sensor_sample (
    sample_id        BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    sensor_no        SMALLINT UNSIGNED NOT NULL,
    temperature_c    DOUBLE,
    calibration_fee  DECIMAL(8,2)
  );
check: |
  SELECT COLUMN_NAME, COLUMN_TYPE
  FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'sensor_sample'
  ORDER BY ORDINAL_POSITION;
explanation: |
  `SMALLINT UNSIGNED` (2 bytes, 0–65,535) is the tightest fit for 0–60,000; signed SMALLINT stops at 32,767
  and MEDIUMINT wastes a byte. `DOUBLE` is right for a physical measurement. `DECIMAL(8,2)` stores up to
  999,999.99 exactly in 4 bytes (6 integer digits → 3 bytes, 2 fractional digits → 1 byte).
```

```exercise
id: ex2
title: Find the out-of-range rows before migrating
difficulty: medium
dataset: shop
concepts: [integer-types, choosing-types]
prompt: |
  The team wants to change `products.stock_qty` (currently `INT`) to `SMALLINT UNSIGNED`. Before running the
  migration, list every product whose `stock_qty` would **not** fit in `SMALLINT UNSIGNED`, plus a summary.

  Return **one row** with columns: `min_stock`, `max_stock`, `rows_out_of_range` (count of products whose
  stock is below 0 or above 65,535).
hints:
  - "SMALLINT UNSIGNED covers 0 … 65,535."
  - "Use MIN, MAX and a conditional count: SUM(stock_qty < 0 OR stock_qty > 65535)."
solution: |
  SELECT MIN(stock_qty) AS min_stock,
         MAX(stock_qty) AS max_stock,
         SUM(stock_qty < 0 OR stock_qty > 65535) AS rows_out_of_range
  FROM products;
explanation: |
  Checking the actual data range before narrowing a type is standard practice: in strict mode the ALTER (a
  COPY operation) would fail at the first out-of-range row with error 1264 after copying part of the table;
  in non-strict mode it would silently clamp values. Here every stock value fits, so the migration is safe —
  but remember to also consider *future* values (can stock ever exceed 65,535 units?).
```

```exercise
id: ex3
title: Exact revenue versus floating-point revenue
difficulty: hard
dataset: shop
concepts: [decimal-precision, floating-point]
prompt: |
  Compute the total revenue of all order items (Σ quantity × unit_price) in two ways and return one row:

  1. `exact_revenue` — using the `DECIMAL` columns as they are
  2. `double_revenue` — casting `unit_price` to `DOUBLE` before multiplying
  3. `difference` — `double_revenue - exact_revenue`
hints:
  - "SUM(quantity * unit_price) is exact because unit_price is DECIMAL(10,2)."
  - "CAST(unit_price AS DOUBLE) switches the computation to floating point."
solution: |
  SELECT SUM(quantity * unit_price)                    AS exact_revenue,
         SUM(quantity * CAST(unit_price AS DOUBLE))    AS double_revenue,
         SUM(quantity * CAST(unit_price AS DOUBLE)) - SUM(quantity * unit_price) AS difference
  FROM order_items;
explanation: |
  The DECIMAL sum is exact to the cent. The DOUBLE sum may differ by a tiny amount (or happen to round to the
  same printed value) — the point is that it is not *guaranteed* to be exact, and its value can depend on the
  order in which rows are added. Mixing a DOUBLE and a DECIMAL in the subtraction converts to DOUBLE, so the
  difference shows the representation error directly. Financial systems keep the whole pipeline in DECIMAL
  (or integer cents) for this reason.
```

```exercise
id: ex4
title: Boolean pitfalls
difficulty: medium
kind: dml
dataset: empty
concepts: [bit-boolean]
setup: |
  CREATE TABLE feature_flag (name VARCHAR(20) PRIMARY KEY, enabled BOOLEAN NOT NULL);
  INSERT INTO feature_flag VALUES ('dark_mode', 1), ('beta_search', 2), ('old_ui', 0), ('chat', -1);
prompt: |
  Legacy code wrote values other than 0 and 1 into `feature_flag.enabled`. Normalise the column so that
  every non-zero value becomes 1 (and 0 stays 0), then add a CHECK constraint named `chk_enabled_bool` that
  allows only 0 and 1.

  The checker reads all rows and the constraint name.
hints:
  - "UPDATE … SET enabled = 1 WHERE enabled <> 0 (or WHERE enabled IS TRUE)."
  - "ALTER TABLE feature_flag ADD CONSTRAINT chk_enabled_bool CHECK (enabled IN (0, 1));"
solution: |
  UPDATE feature_flag SET enabled = 1 WHERE enabled <> 0;
  ALTER TABLE feature_flag ADD CONSTRAINT chk_enabled_bool CHECK (enabled IN (0, 1));
check: |
  SELECT name AS item, CAST(enabled AS CHAR) AS v FROM feature_flag
  UNION ALL
  SELECT CONSTRAINT_NAME, CONSTRAINT_TYPE FROM information_schema.TABLE_CONSTRAINTS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'feature_flag' AND CONSTRAINT_TYPE = 'CHECK';
explanation: |
  `UPDATE … WHERE enabled = TRUE` would be wrong: it matches only rows equal to 1, leaving 2 and −1 untouched.
  The order matters too — adding the CHECK constraint first fails, because MySQL validates existing rows when
  a CHECK constraint is added (error 3819). After this, `WHERE enabled = TRUE` and `WHERE enabled` agree.
```

```exercise
id: ex5
title: Type review for a payments schema
difficulty: challenge
kind: open
concepts: [choosing-types, integer-types, decimal-precision, unsigned]
prompt: |
  Review this proposed table and rewrite it with correct, modern MySQL 8.4 types. Justify each change.

  ~~~text
  CREATE TABLE payment (
    id            INT(11) AUTO_INCREMENT PRIMARY KEY,   -- ~5 million payments per day expected
    order_id      INT(11),
    amount        FLOAT(10,2),
    fee_pct       DOUBLE,
    currency      INT,                                  -- ISO 4217 numeric code, e.g. 840
    refunded      INT(1),
    account_no    BIGINT ZEROFILL                       -- may start with zeros
  );
  ~~~
solution: |
  ~~~sql
  CREATE TABLE payment (
    payment_id   BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    order_id     BIGINT UNSIGNED NOT NULL,          -- same type as orders' primary key
    amount       DECIMAL(15,2)   NOT NULL,
    fee_pct      DECIMAL(6,4)    NOT NULL,          -- e.g. 2.9000 %
    currency     CHAR(3)         NOT NULL,          -- ISO 4217 alphabetic code 'USD'
    refunded     BOOLEAN         NOT NULL DEFAULT FALSE,
    account_no   VARCHAR(34)     NOT NULL,          -- identifier, not a number (IBAN max 34 chars)
    CONSTRAINT chk_payment_amount CHECK (amount >= 0),
    CONSTRAINT chk_payment_refunded CHECK (refunded IN (0, 1))
  );
  ~~~

  - **id**: 5 million/day ≈ 1.8 billion/year — a signed INT (2.1 billion) would overflow within about a
    year. `BIGINT UNSIGNED`. `INT(11)` display width is deprecated and meaningless.
  - **order_id**: must match the referenced key's type exactly (foreign keys require compatible types); NOT NULL.
  - **amount**: money must be exact → `DECIMAL`; `FLOAT(M,D)` is approximate *and* deprecated syntax.
  - **fee_pct**: a rate used in money calculations should be exact too → `DECIMAL(6,4)`; `DOUBLE` would make
    fee computations inexact.
  - **currency**: the numeric code is an identifier; `CHAR(3)` alphabetic codes are clearer (or SMALLINT
    UNSIGNED if the numeric code is required).
  - **refunded**: a boolean → `BOOLEAN` with a CHECK for 0/1; `INT(1)` wastes 3 bytes and the width means nothing.
  - **account_no**: leading zeros are part of the value, it is never used in arithmetic, and `ZEROFILL` is
    deprecated → a string type.
  - Make NOT NULL explicit everywhere it applies; `UNSIGNED` on DECIMAL is deprecated, so use CHECK constraints
    for non-negativity.
```

## Knowledge Assessment

```quiz
- id: q1
  question: "What is the maximum value of a signed `INT` column, and of `INT(11)`?"
  options: ["2,147,483,647 for both", "2,147,483,647 and 99,999,999,999", "4,294,967,295 for both", "11 digits for INT(11), 10 for INT"]
  answer: 0
  explanation: "INT is always 4 bytes: −2,147,483,648 … 2,147,483,647 signed. The (11) is a deprecated display width that never affected range or storage. 4,294,967,295 is the INT UNSIGNED maximum."
  concepts: [integer-types]
- id: q2
  question: "Under the default SQL mode, `INSERT INTO t (d) VALUES (1.005)` where d is DECIMAL(5,2). What is stored?"
  options: ["1.00, silently", "1.01, with a Note", "An error 1264 is raised", "1.005"]
  answer: 1
  explanation: "Excess fractional digits are rounded half away from zero (1.005 → 1.01) and MySQL issues Note 1265. Only excess integer digits raise an out-of-range error. DECIMAL(5,2) cannot hold three decimals."
  concepts: [decimal-precision, strict-mode]
- id: q3
  question: "A FLOAT column contains 0.1 (inserted as the literal 0.1). Why does `WHERE f = 0.1` return no rows?"
  options:
    - "Because MySQL stores FLOAT as text"
    - "Because the stored value is the nearest single-precision number (≈0.10000000149), which differs from the double-precision 0.1 it is compared with"
    - "Because 0.1 is out of FLOAT's range"
    - "Because FLOAT columns cannot be compared in WHERE clauses"
  answer: 1
  explanation: "Binary floating point cannot represent 0.1 exactly; single and double precision approximate it differently. The display rounds to ~6 digits and hides this. Compare with a tolerance or use DECIMAL."
  concepts: [floating-point]
- id: q4
  question: "Which features are deprecated in MySQL 8.0.17+ / 8.4? (Select all that apply.)"
  type: multi
  options: ["Integer display width such as BIGINT(20)", "ZEROFILL", "DECIMAL(10,2)", "FLOAT(7,2)", "UNSIGNED on INT"]
  answer: [0, 1, 3]
  explanation: "Display width (except TINYINT(1)), ZEROFILL, FLOAT(M,D)/DOUBLE(M,D) and UNSIGNED on DECIMAL/FLOAT/DOUBLE are deprecated. DECIMAL(M,D) is the standard exact type, and UNSIGNED on integer types is still fully supported."
  concepts: [integer-types, floating-point]
- id: q5
  question: "What does `SELECT CAST(5 AS UNSIGNED) - 10;` return in the default SQL mode?"
  options: ["-5", "0", "18446744073709551611", "Error 1690: BIGINT UNSIGNED value is out of range"]
  answer: 3
  explanation: "If an operand is unsigned, the result is BIGINT UNSIGNED; a negative result raises error 1690. (Only with the NO_UNSIGNED_SUBTRACTION SQL mode would the result be signed −5.) MySQL does not silently wrap around."
  concepts: [unsigned]
- id: q6
  question: "How many bytes does InnoDB use to store a DECIMAL(10,2) value?"
  options: ["4", "5", "8", "10"]
  answer: 1
  explanation: "8 integer digits take 4 bytes (7–9 leftover digits → 4 bytes) and 2 fractional digits take 1 byte: 5 bytes in total. DECIMAL is not stored as one byte per digit."
  concepts: [decimal-precision]
- id: q7
  question: "A `BOOLEAN` column contains the value 2. Which predicates are true for that row? (Select all that apply.)"
  type: multi
  options: ["col = TRUE", "col IS TRUE", "col <> 0", "col = 1"]
  answer: [1, 2]
  explanation: "BOOLEAN is TINYINT(1). TRUE is the integer 1, so col = TRUE and col = 1 are false for 2. IS TRUE and <> 0 test for non-zero and are true."
  concepts: [bit-boolean]
- id: q8
  question: "A messaging table will receive about 20 million new rows per day for many years. Which primary key type should you choose?"
  options: ["INT", "INT UNSIGNED", "BIGINT UNSIGNED", "DECIMAL(20,0)"]
  answer: 2
  explanation: "20 million/day exceeds INT UNSIGNED's 4.29 billion in under 7 months. BIGINT UNSIGNED (8 bytes) lasts effectively forever. DECIMAL(20,0) works but is larger (9 bytes) and slower to compare than a native integer."
  concepts: [choosing-types, integer-types]
```
