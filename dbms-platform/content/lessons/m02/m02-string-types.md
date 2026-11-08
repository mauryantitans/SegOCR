---
id: m02-string-types
dataset: shop
summary: CHAR, VARCHAR, the TEXT family, ENUM, SET, BINARY and VARBINARY — how MySQL measures length in characters but stores bytes, how utf8mb4 changes sizes, how trailing spaces are padded, stripped and compared, and which row-size and index-size limits shape table design.
concepts: [char-varchar, text-types, enum-set, binary-strings, character-length, row-size-limits, trailing-spaces]
references:
  - title: "MySQL 8.4 Reference Manual — String Data Types"
    url: https://dev.mysql.com/doc/refman/8.4/en/string-types.html
  - title: "MySQL 8.4 Reference Manual — The CHAR and VARCHAR Types"
    url: https://dev.mysql.com/doc/refman/8.4/en/char.html
  - title: "MySQL 8.4 Reference Manual — The BLOB and TEXT Types"
    url: https://dev.mysql.com/doc/refman/8.4/en/blob.html
  - title: "MySQL 8.4 Reference Manual — The ENUM Type"
    url: https://dev.mysql.com/doc/refman/8.4/en/enum.html
  - title: "MySQL 8.4 Reference Manual — The SET Type"
    url: https://dev.mysql.com/doc/refman/8.4/en/set.html
  - title: "MySQL 8.4 Reference Manual — Limits on Table Column Count and Row Size"
    url: https://dev.mysql.com/doc/refman/8.4/en/column-count-limit.html
  - title: "MySQL 8.4 Reference Manual — InnoDB Row Formats"
    url: https://dev.mysql.com/doc/refman/8.4/en/innodb-row-format.html
---

## Introduction

Names, e-mail addresses, product descriptions, status codes, country codes, JSON blobs stored as text, hashes, tokens — most columns in a typical schema are strings. MySQL offers several string types, and they differ in ways that matter far beyond "how long can it be":

- **Fixed vs variable length**: `CHAR(n)` vs `VARCHAR(n)` — and what each does with trailing spaces.
- **Characters vs bytes**: `VARCHAR(50)` means 50 *characters*, but with `utf8mb4` a character takes 1 to 4 *bytes* — and every storage limit (row size, index size, TEXT capacity) is in bytes.
- **Text vs binary**: `VARCHAR` stores characters interpreted through a **character set** and compared with a **collation**; `VARBINARY` stores raw bytes compared byte by byte.
- **In-row vs off-page**: `TEXT` and `BLOB` values can live outside the row, which changes defaults, indexing and performance.
- **Constrained domains**: `ENUM` and `SET` restrict values to a list — useful, but with sharp edges.

This lesson demonstrates each behaviour with real outputs. Character sets and collations in depth — case and accent sensitivity, `utf8mb4` vs `utf8mb3`, collation mismatches — are the subject of [Character Sets and Collations](/lessons/m02-charsets-collations); here you will learn just enough to understand string storage and comparison.

:::note[Dataset]
The examples use small demonstration tables and the **shop** dataset, whose `customers` table uses `VARCHAR` for names and e-mails, `CHAR(2)` for US state codes and `ENUM` for the loyalty tier.
:::

## DBMS Theory

:::theory[Character strings, binary strings and their domains]
The SQL standard distinguishes **character string types** — `CHARACTER(n)` (fixed length) and `CHARACTER VARYING(n)` (variable length), plus `CHARACTER LARGE OBJECT` (CLOB) — from **binary string types** — `BINARY(n)`, `BINARY VARYING(n)` and `BINARY LARGE OBJECT` (BLOB). A character string is a sequence of characters from a **character repertoire**, encoded by a **character set**, and ordered/compared by a **collation**. A binary string is a sequence of octets with no character interpretation.

Fixed-length character strings are conceptually padded with spaces to their declared length. The standard's comparison rule for such strings, **PAD SPACE**, compares two strings as if the shorter were padded with spaces — so `'ab'` and `'ab  '` are equal. Newer collations may instead use **NO PAD**, in which trailing spaces are significant.
:::

From the relational point of view a string type is a domain: `CHAR(2)` for a US state code says "exactly two characters"; `ENUM('bronze','silver','gold')` says "one of three values". A well-chosen string type documents meaning and rejects invalid data.

Three distinctions recur through the lesson:

| Distinction | Character types | Binary types |
|---|---|---|
| Unit of length | characters (`VARCHAR(10)` = 10 characters) | bytes (`VARBINARY(10)` = 10 bytes) |
| Comparison | by collation (case-/accent-insensitive by default in MySQL 8) | byte by byte |
| Typical use | human-readable text | hashes, UUIDs, encrypted data, file contents |

**Length in characters vs bytes.** Unicode defines over a hundred thousand characters; the UTF-8 encoding stores ASCII characters in 1 byte, most accented Latin, Greek, Cyrillic, Hebrew and Arabic letters in 2 bytes, most CJK characters in 3 bytes, and emoji and other supplementary characters in 4 bytes. MySQL's `utf8mb4` is full UTF-8 (up to 4 bytes); the deprecated `utf8mb3` (still aliased by `utf8` in 8.4) stores at most 3 bytes and therefore **cannot store emoji**.

## Why It Exists

Why so many string types?

- **`CHAR` vs `VARCHAR`** trade space for simplicity. Values of a truly fixed length (ISO country codes, hashes in hex) fit `CHAR`; most text varies in length and wastes no space in `VARCHAR`.
- **`TEXT`/`BLOB`** exist because rows have size limits. A product description or an article body can be many kilobytes; storing it inline would make rows huge and pages hold few rows. Large-object types can be stored off-page.
- **`BINARY`/`VARBINARY`/`BLOB`** exist because not all strings are text. A SHA-256 hash is 32 bytes; storing it as a 64-character hex `CHAR` doubles its size and subjects it to case-insensitive comparison rules.
- **`ENUM`/`SET`** exist for compact storage of small fixed vocabularies — a status stored in 1 byte instead of a `VARCHAR`.

What goes wrong without understanding them:

- **Emoji break inserts** in `utf8mb3` columns (error 1366), or legacy `latin1` tables garble non-Latin names.
- **Indexes fail to create** because a `VARCHAR(1000)` in `utf8mb4` can be 4,000 bytes, exceeding the 3,072-byte index key limit.
- **Lookups miss rows** because `CHAR` strips trailing spaces while `VARCHAR` keeps them, and the default collation treats them as significant.
- **`CREATE TABLE` fails** with "row size too large" because many wide columns exceed InnoDB's row limits.

## Internal Working

:::internals[How InnoDB stores strings]
**VARCHAR(n)** stores a **length prefix** followed by the actual bytes. The prefix is **1 byte** if the column's maximum byte length (n × max bytes per character) is ≤ 255, otherwise **2 bytes**. `VARCHAR(60)` in utf8mb4 can need 240 bytes → 1-byte prefix; `VARCHAR(64)` can need 256 bytes → 2-byte prefix. (This boundary also decides whether widening a VARCHAR can be done in place — see [Creating, Altering and Dropping Tables](/lessons/m02-tables-ddl).)

**CHAR(n)** is fixed-length for single-byte character sets: n bytes, padded with spaces. For variable-width character sets such as utf8mb4, InnoDB's `COMPACT`/`DYNAMIC` row formats store `CHAR(n)` in **at least n bytes** and up to n × 4, trimming trailing spaces beyond n bytes — a space-saving optimisation. On retrieval, **trailing spaces are removed** from `CHAR` values (unless the `PAD_CHAR_TO_FULL_LENGTH` SQL mode, deprecated, is enabled).

**BINARY(n)** is n bytes, padded on the right with `0x00` bytes (not spaces), and the padding is **not** removed on retrieval. **VARBINARY(n)** stores a length prefix plus the bytes.

**TEXT and BLOB** values are stored with a 1–4 byte length prefix. In the `DYNAMIC` row format (the default), a long variable-length value that does not fit in the row is stored **entirely off-page** in overflow pages, and the row keeps a **20-byte pointer**. Short values may stay in the row.

**ENUM** stores the **1-based index** of the value in the list: 1 byte for up to 255 members, 2 bytes for up to 65,535. **SET** stores a **bitmap**: 1, 2, 3, 4 or 8 bytes for up to 8, 16, 24, 32 or 64 members.

**Row-size limits.** Two limits apply:

1. **The server's 65,535-byte row limit** — the sum of the *maximum* byte lengths of all columns (plus length prefixes and NULL bits), where each TEXT/BLOB column counts only 9–12 bytes because its body is stored separately. Exceeding it at `CREATE TABLE` gives error 1118.
2. **InnoDB's page limit** — a row's in-page data must fit in about half a page (8,126 bytes for 16 KB pages) so that each page holds at least two rows. Variable-length columns that are too long are moved off-page (DYNAMIC), but fixed-length data cannot be, so tables with many wide `CHAR` columns can fail with "Row size too large (> 8126)".

**Index key limit.** With `DYNAMIC`/`COMPRESSED` row formats, an index key prefix can be at most **3,072 bytes** (767 bytes for `COMPACT`/`REDUNDANT`). In utf8mb4 that is 768 characters for a full-column index.
:::

| Type | Maximum length | Storage | Notes |
|---|---|---|---|
| `CHAR(n)` | 255 characters | n × charset width (InnoDB: ≥ n bytes for utf8mb4) | Trailing spaces stripped on read |
| `VARCHAR(n)` | 65,535 **bytes** shared by the row (16,383 utf8mb4 characters max) | 1–2 byte prefix + data | Trailing spaces kept |
| `TINYTEXT` | 255 bytes | L + 1 | |
| `TEXT` | 65,535 bytes (64 KB) | L + 2 | `TEXT(n)` picks the smallest type for n characters |
| `MEDIUMTEXT` | 16,777,215 bytes (16 MB) | L + 3 | |
| `LONGTEXT` | 4,294,967,295 bytes (4 GB) | L + 4 | Practically limited by `max_allowed_packet` |
| `BINARY(n)` / `VARBINARY(n)` | 255 / 65,535 bytes | n / L + 1–2 | Byte strings; BINARY pads with `0x00` |
| `TINYBLOB` … `LONGBLOB` | same as TEXT family, in bytes | L + 1 … L + 4 | Binary large objects |
| `ENUM(…)` | 65,535 members | 1–2 bytes | Stores the index |
| `SET(…)` | 64 members | 1, 2, 3, 4 or 8 bytes | Stores a bitmap |

## MySQL Implementation

A table with each major string type:

```sql run
CREATE TABLE string_demo (
  fixed_code   CHAR(5),
  var_text     VARCHAR(5),
  fixed_bytes  BINARY(5),
  var_bytes    VARBINARY(5),
  short_note   TINYTEXT
);
INSERT INTO string_demo VALUES ('ab  ', 'ab  ', 'ab', 'ab', 'ab  ');
SELECT CONCAT('[', fixed_code, ']') AS char_value,
       CONCAT('[', var_text, ']')   AS varchar_value,
       HEX(fixed_bytes)             AS binary_hex,
       HEX(var_bytes)               AS varbinary_hex,
       LENGTH(fixed_code)           AS char_len,
       LENGTH(var_text)             AS varchar_len,
       LENGTH(fixed_bytes)          AS binary_len
FROM string_demo;
```

The same value `'ab  '` (with two trailing spaces) comes back differently:

- `CHAR(5)` returned `[ab]` — trailing spaces are **stripped on retrieval**, so you cannot tell whether they were there.
- `VARCHAR(5)` returned `[ab  ]` — trailing spaces are **preserved**.
- `BINARY(5)` padded `'ab'` with zero bytes to `6162000000` (hex), length 5 — and keeps the padding.
- `VARBINARY(5)` stored just the two bytes `6162`.

**Characters vs bytes.** `CHAR_LENGTH()` counts characters, `LENGTH()` counts bytes:

```sql run
SELECT CHAR_LENGTH('héllo') AS chars,  LENGTH('héllo') AS bytes,
       CHAR_LENGTH('東京')   AS cjk_chars, LENGTH('東京') AS cjk_bytes,
       CHAR_LENGTH('😀')     AS emoji_chars, LENGTH('😀') AS emoji_bytes,
       HEX('é')             AS e_acute_utf8;
```

`é` is 2 bytes (`C3A9`), each CJK character 3 bytes, the emoji 4 bytes. A `VARCHAR(5)` accepts five characters regardless of their byte size:

```sql run
INSERT INTO string_demo (var_text) VALUES ('東京大阪市');
SELECT var_text, CHAR_LENGTH(var_text) AS chars, LENGTH(var_text) AS bytes
FROM string_demo WHERE var_text LIKE '東%';
```

Five characters, fifteen bytes — within the declared length of 5 characters. Byte-based limits (row size, index keys, TEXT capacity) are where the difference bites.

**Strict mode and too-long values.** Inserting six characters into `VARCHAR(5)` fails:

```sql run error
INSERT INTO string_demo (var_text) VALUES ('abcdef');
```

But excess **trailing spaces** are silently trimmed with only a Note:

```sql run
INSERT INTO string_demo (var_text) VALUES ('abcde   ');
SELECT CONCAT('[', var_text, ']') AS stored_value FROM string_demo WHERE var_text LIKE 'abcde%';
```

## Syntax Breakdown

```sql
CHAR[(n)]            [CHARACTER SET cs] [COLLATE coll]     -- n = 0..255, default 1
VARCHAR(n)           [CHARACTER SET cs] [COLLATE coll]     -- n required
{TINYTEXT | TEXT[(n)] | MEDIUMTEXT | LONGTEXT} [CHARACTER SET cs] [COLLATE coll]
BINARY[(n)] | VARBINARY(n)
{TINYBLOB | BLOB[(n)] | MEDIUMBLOB | LONGBLOB}
ENUM('v1', 'v2', ...) [CHARACTER SET cs] [COLLATE coll]
SET('v1', 'v2', ...)  [CHARACTER SET cs] [COLLATE coll]
```

| Element | Meaning | Standard / MySQL | Notes |
|---|---|---|---|
| `CHAR(n)`, `VARCHAR(n)` | Fixed / variable character strings, n in **characters** | Standard (`CHARACTER`, `CHARACTER VARYING`) | `NATIONAL CHAR`/`NCHAR` = utf8mb3 in MySQL — avoid |
| `TEXT` family | Large character objects | MySQL names (standard: `CLOB`) | Cannot have a literal `DEFAULT`; expression default `DEFAULT ('…')` allowed since 8.0.13 |
| `BINARY`, `VARBINARY` | Fixed / variable byte strings | Standard (`BINARY`, `BINARY VARYING`) | No character set; byte comparison |
| `BLOB` family | Large binary objects | Standard name `BLOB` | Same storage rules as TEXT |
| `ENUM` | One value from a list | MySQL extension | Stored as index; sorts by index |
| `SET` | Zero or more values from a list | MySQL extension | Stored as bitmap; violates 1NF if used for lists |
| `CHARACTER SET`, `COLLATE` | Per-column encoding and comparison rules | Standard concepts, MySQL syntax | Default inherited from table → database → server |
| `utf8` / `utf8mb3` | 3-byte UTF-8 subset | MySQL | **Deprecated**; `utf8` is an alias of `utf8mb3` in 8.4 and produces warning 3719 |

:::version[utf8mb3 deprecation]
`utf8mb3` (and its alias `utf8`) has been deprecated since MySQL 8.0 and will be removed in a future release; in 8.4 `utf8` still means `utf8mb3` and produces warning 3719 recommending `utf8mb4`. The plan is for `utf8` eventually to become an alias of `utf8mb4`. Always write `utf8mb4` explicitly.
:::

## Practical Examples

**1. Trailing spaces and the default collation.** MySQL 8's default collation `utf8mb4_0900_ai_ci` is a **NO PAD** collation: trailing spaces are significant in comparisons. Older collations such as `utf8mb4_general_ci` and `utf8mb4_bin` are **PAD SPACE**:

```sql run
SELECT COLLATION_NAME, PAD_ATTRIBUTE
FROM information_schema.COLLATIONS
WHERE COLLATION_NAME IN ('utf8mb4_0900_ai_ci', 'utf8mb4_0900_bin', 'utf8mb4_general_ci', 'utf8mb4_bin')
ORDER BY COLLATION_NAME;
```

Combine this with how `CHAR` and `VARCHAR` store values:

```sql run
SELECT fixed_code = 'ab'    AS char_equals_ab,
       var_text   = 'ab'    AS varchar_equals_ab,
       var_text   = 'ab  '  AS varchar_equals_ab_spaces,
       fixed_code = 'AB'    AS char_equals_upper_ab
FROM string_demo
WHERE fixed_code IS NOT NULL;
```

The `CHAR` column matches `'ab'` because its trailing spaces were stripped. The `VARCHAR` column holds `'ab  '`, which under the NO PAD collation is **not** equal to `'ab'`. And `'ab' = 'AB'` is true because the collation is case-insensitive (`_ci`). Data imported with stray trailing spaces into `VARCHAR` columns is a classic source of "the row is there but WHERE can't find it" bugs — trim on input.

**2. ENUM: a compact, constrained domain.** The shop's `customers.loyalty_tier` is `ENUM('bronze','silver','gold')`. Create a similar column:

```sql run
CREATE TABLE tshirt (
  sku   VARCHAR(10) PRIMARY KEY,
  size  ENUM('S', 'M', 'L', 'XL') NOT NULL,
  tags  SET('cotton', 'organic', 'recycled') NOT NULL DEFAULT ''
);
INSERT INTO tshirt VALUES ('T1', 'M', 'recycled,cotton'), ('T2', 'XL', ''), ('T3', 'S', 'organic,cotton');
SELECT sku, size, size + 0 AS size_index, tags, tags + 0 AS tags_bitmap
FROM tshirt ORDER BY sku;
```

`size + 0` reveals the stored index (S = 1, M = 2, L = 3, XL = 4); `tags + 0` reveals the bitmap (cotton = 1, organic = 2, recycled = 4, so `'cotton,recycled'` = 5). MySQL normalised the SET value to definition order. In strict mode, a value outside the list is rejected:

```sql run error
INSERT INTO tshirt VALUES ('T4', 'XXL', '');
```

**3. ENUM sorts by index, not alphabetically.** This is usually what you want for sizes and statuses, but surprises people:

```sql run
SELECT size FROM tshirt ORDER BY size;
```

```sql run
SELECT size FROM tshirt ORDER BY CAST(size AS CHAR);
```

The first query sorts S, M, XL (definition order); casting to a string sorts alphabetically (M, S, XL).

**4. SET membership.** Query a SET with `FIND_IN_SET` (or bit operations):

```sql run
SELECT sku, tags FROM tshirt WHERE FIND_IN_SET('cotton', tags) > 0 ORDER BY sku;
```

This works, but it cannot use an index efficiently, and a SET column stores a *list* in one attribute — a violation of first normal form. A junction table (`product_tag(product_id, tag)`) is the relational design ([Relationship Patterns](/lessons/m05-relationship-patterns)).

**5. Binary strings compare bytes.** Case-insensitive comparison applies only to character strings:

```sql run
SELECT 'abc' = 'ABC'                              AS text_compare,
       CAST('abc' AS BINARY) = CAST('ABC' AS BINARY) AS binary_compare,
       HEX(UNHEX(SHA2('hello', 256)))             AS sha256_hex,
       LENGTH(UNHEX(SHA2('hello', 256)))          AS sha256_bytes,
       LENGTH(SHA2('hello', 256))                 AS sha256_hex_chars;
```

A SHA-256 digest is 32 bytes as binary (`BINARY(32)`) but 64 characters as hex text (`CHAR(64)`, and up to 256 bytes of row space reserved in utf8mb4). Store hashes, tokens and UUIDs as binary.

**6. Row-size limit.** Three `VARCHAR(10000)` utf8mb4 columns could need 3 × 40,000 bytes — far beyond the 65,535-byte row limit:

```sql run error
CREATE TABLE too_wide (a VARCHAR(10000), b VARCHAR(10000), c VARCHAR(10000));
```

Error 1118 suggests changing some columns to TEXT or BLOB, whose bodies are stored separately and count only a few bytes toward the limit:

```sql run
CREATE TABLE wide_ok (a VARCHAR(5000), b VARCHAR(5000), c TEXT, d TEXT);
SELECT COLUMN_NAME, COLUMN_TYPE, CHARACTER_MAXIMUM_LENGTH, CHARACTER_OCTET_LENGTH
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'wide_ok'
ORDER BY ORDINAL_POSITION;
```

`CHARACTER_OCTET_LENGTH` shows the maximum **bytes**: 20,000 for `VARCHAR(5000)` in utf8mb4, 65,535 for `TEXT` — note that a `TEXT` column holds 65,535 *bytes*, i.e. as few as 16,383 four-byte characters.

**7. Index key limit.** A full index on `VARCHAR(1000)` utf8mb4 would need up to 4,000 bytes per key:

```sql run error
CREATE TABLE long_key (url VARCHAR(1000), INDEX idx_url (url));
```

Solutions: a shorter column, a **prefix index** (`INDEX (url(255))` — indexes the first 255 characters; [Covering, Prefix and Functional Indexes](/lessons/m07-covering-prefix-functional)), or an index on a hash of the value stored in a `BINARY(32)` generated column.

## Execution Walkthrough

How does MySQL store, retrieve and compare the value `'ab '` (one trailing space) in a `CHAR(4)` and a `VARCHAR(4)` column?

```walkthrough
title: The life of 'ab ' in CHAR(4) and VARCHAR(4)
steps:
  - title: "1. Store"
    explain: >-
      CHAR(4) pads the value to 4 characters ('ab  '); VARCHAR(4) stores the three characters as given with a
      1-byte length prefix (max 16 bytes in utf8mb4 ≤ 255).
    sql: |
      CREATE TABLE pad_trace (c CHAR(4), v VARCHAR(4));
      INSERT INTO pad_trace VALUES ('ab ', 'ab ');
  - title: "2. Retrieve"
    explain: >-
      On retrieval CHAR strips all trailing spaces, VARCHAR returns exactly what was stored. The lengths make
      the difference visible.
    sql: |
      SELECT CONCAT('[', c, ']') AS c_value, CHAR_LENGTH(c) AS c_chars,
             CONCAT('[', v, ']') AS v_value, CHAR_LENGTH(v) AS v_chars
      FROM pad_trace;
  - title: "3. Compare under the default NO PAD collation"
    explain: >-
      With utf8mb4_0900_ai_ci trailing spaces count. The stripped CHAR value equals 'ab'; the VARCHAR value
      'ab ' does not, but equals 'ab '.
    sql: SELECT c = 'ab' AS c_eq, v = 'ab' AS v_eq, v = 'ab ' AS v_eq_space FROM pad_trace;
  - title: "4. Compare under a PAD SPACE collation"
    explain: >-
      Forcing utf8mb4_general_ci (PAD SPACE) makes trailing spaces insignificant: now the VARCHAR value also
      equals 'ab'. The comparison result depends on the collation, not just the data.
    sql: SELECT v = 'ab' COLLATE utf8mb4_general_ci AS v_eq_pad_space FROM pad_trace;
  - title: "5. LIKE never pads"
    explain: >-
      LIKE compares character by character with no padding in any collation, so 'ab ' LIKE 'ab' is false.
    sql: SELECT v LIKE 'ab' AS v_like_ab, v LIKE 'ab%' AS v_like_ab_prefix FROM pad_trace;
```

## Output

The walkthrough shows four facts in its outputs: `CHAR` returned `[ab]` with 2 characters while `VARCHAR` returned `[ab ]` with 3; under the default NO PAD collation the `VARCHAR` value equals `'ab '` but not `'ab'`; under the PAD SPACE `utf8mb4_general_ci` it equals `'ab'`; and `LIKE` ignores padding rules entirely.

Practical reading: if two systems disagree on whether a value "exists", check (1) the column type (CHAR strips), (2) the collation's pad attribute, and (3) whether one side uses `LIKE`. The cleanest fix is upstream — normalise (trim) values before they are stored.

## Alternative Approaches

| Need | Options | Recommendation |
|---|---|---|
| Fixed-length codes (ISO country, currency) | `CHAR(2)`, `CHAR(3)`, `VARCHAR`, lookup table | `CHAR(n)` with a binary or `_bin` collation if case matters, or a lookup table with a foreign key |
| Free text up to a few hundred characters | `VARCHAR(n)` | `VARCHAR` with a realistic n (it also documents intent and limits abuse) |
| Long text (descriptions, articles) | `TEXT`, `MEDIUMTEXT`, external storage | `TEXT`/`MEDIUMTEXT`; very large documents in object storage with a reference |
| Small fixed vocabulary (status) | `ENUM`, `VARCHAR` + `CHECK`, lookup table + FK | Lookup table if values change or carry attributes; `ENUM` or `VARCHAR` + `CHECK` for stable small sets |
| Multi-valued attribute (tags) | `SET`, JSON array, junction table | Junction table (normalised, indexable); JSON if semi-structured |
| Hashes, tokens, UUIDs | `CHAR(64)` hex, `BINARY(32)` | `BINARY(n)` — half the size, byte comparison |
| Case-sensitive identifiers | `VARCHAR … COLLATE utf8mb4_bin` or `utf8mb4_0900_bin`, `VARBINARY` | A `_bin` collation keeps character semantics; `VARBINARY` loses them |

**ENUM vs lookup table.** Adding a value to the *end* of an ENUM list is an instant metadata change; inserting in the middle or removing values requires a table rebuild and remaps indexes. A lookup table changes with a plain `INSERT` and can carry extra columns (label, sort order, active flag). Many teams use `VARCHAR` with a `CHECK (status IN (…))` constraint as a middle ground: readable values, validated domain, easy to change.

## Performance Analysis

- **Row width drives I/O.** Narrower rows mean more rows per 16 KB page, so fewer pages to read for scans and a higher buffer-pool hit rate. Declaring `VARCHAR(255)` everywhere does not waste disk (only actual bytes are stored), but the declared maximum *does* matter: in-memory temporary tables and sort buffers may allocate space based on the declared maximum, and indexes are limited by maximum byte length.
- **Off-page TEXT/BLOB.** Large values stored off-page cost an extra page read when the column is selected — and nothing when it is not. Avoid `SELECT *` on tables with big TEXT columns; select only what you need. Since 8.0.13 the TempTable engine handles BLOB/TEXT in internal temporary tables, but such operations still use more memory.
- **Index size.** Secondary index entries store the full key plus the primary key. A `VARCHAR(255)` utf8mb4 key can be up to 1,022 bytes (255 × 4 + 2); a `BINARY(16)` key is 16. Prefix and hash indexes keep long strings indexable at reasonable cost.
- **Collation cost.** Comparing strings under a linguistic collation such as `utf8mb4_0900_ai_ci` is more work than a byte comparison (`_bin` or binary types). This rarely dominates a query, but for very hot equality lookups on identifiers, binary comparison is both faster and semantically correct.
- **ENUM is compact** (1–2 bytes) and fast to compare; `VARCHAR` status columns are larger but more flexible.

:::performance[Theory vs measurement]
The byte counts here (length prefixes, 20-byte off-page pointers, 3,072-byte key limit) are documented storage rules; the performance effects are expected consequences. Measure on your own data by comparing `AVG_ROW_LENGTH`, `DATA_LENGTH` and `INDEX_LENGTH` in `information_schema.TABLES` (after `ANALYZE TABLE`) for alternative designs, and with `EXPLAIN ANALYZE` for queries that touch TEXT columns.
:::

## Best Practices

:::best-practice[String columns done right]
1. **Use `utf8mb4` everywhere** (the 8.x default) and write it explicitly in DDL; never create new `utf8`/`utf8mb3` columns.
2. **Size `VARCHAR` realistically** — `VARCHAR(255)` is not free (index limits, memory for temporary tables) and a meaningful limit documents the domain. E-mail addresses: `VARCHAR(254)`/`(255)`; person names: `VARCHAR(100)`; URLs: `VARCHAR(2048)` with a hash or prefix index.
3. **Use `CHAR` only for truly fixed-length values**, and remember it strips trailing spaces.
4. **Store binary data in binary types**: hashes as `BINARY(32)`, UUIDs as `BINARY(16)`, encrypted values as `VARBINARY`/`BLOB`.
5. **Trim and normalise input** before storing; under the default NO PAD collation, trailing spaces change equality.
6. **Prefer lookup tables or `VARCHAR` + `CHECK` over `SET`**, and use `ENUM` only for small, stable vocabularies whose order is meaningful.
7. **Keep large text out of hot tables** or at least out of `SELECT *`; consider a separate table for rarely read large columns.
8. **Know the byte limits**: 65,535 bytes per row (excluding TEXT/BLOB bodies), about 8 KB of in-page data per InnoDB row, 3,072 bytes per index key.
:::

## Common Mistakes

**Mistake 1 — `utf8` that cannot store emoji.**

```sql run
CREATE TABLE legacy_comments (body VARCHAR(100) CHARACTER SET utf8);
SHOW CREATE TABLE legacy_comments;
```

The warning (3719) says `utf8` is currently `utf8mb3`, and `SHOW CREATE TABLE` confirms it. Now a customer writes a comment with an emoji:

```sql run error
INSERT INTO legacy_comments VALUES ('Great product 👍');
```

Error 1366: the 4-byte character cannot be encoded in utf8mb3. The fix is `ALTER TABLE legacy_comments CONVERT TO CHARACTER SET utf8mb4` (a table copy) — or never creating such columns.

**Mistake 2 — giving a TEXT column a literal default.**

```sql run error
CREATE TABLE notes_bad (body TEXT NOT NULL DEFAULT '');
```

`TEXT`/`BLOB`/`JSON` columns cannot have literal defaults (error 1101). Since 8.0.13 an expression default works:

```sql run
CREATE TABLE notes_ok (body TEXT NOT NULL DEFAULT (''));
INSERT INTO notes_ok () VALUES ();
SELECT CONCAT('[', body, ']') AS body FROM notes_ok;
```

**Mistake 3 — counting bytes when you mean characters.** `LENGTH()` returns bytes. A validation such as `WHERE LENGTH(name) > 50` flags a 30-character Japanese name (90 bytes). Use `CHAR_LENGTH()` for character counts.

**Mistake 4 — numbers in string columns.** Comparing a `VARCHAR` column with a number converts every value to a number (`'007' = 7` is true; `'abc' = 0` is true with a warning) and prevents index use. Compare strings with strings.

**Mistake 5 — appending to or reordering an ENUM carelessly.** Values are stored as indexes; inserting a new member in the middle of the list (`ENUM('S','M','L')` → `ENUM('S','M','ML','L')`) forces a rebuild and changes the sort order. Append at the end, or use a lookup table.

**Mistake 6 — trailing spaces in imported data.** A CSV import that leaves `'SJC '` in a `VARCHAR` column makes `WHERE code = 'SJC'` miss the row under the default collation. `UPDATE t SET code = TRIM(code)` and validate on input.

## Industry Application

- **Internationalisation.** Global products must store names and addresses in every script, plus emoji in user-generated content; `utf8mb4` became the MySQL default in 8.0 for exactly this reason. Migrations of legacy `latin1`/`utf8mb3` schemas to `utf8mb4` are common projects — they require table copies and re-checking index-length limits (a `VARCHAR(255)` index that fitted in 767 bytes under utf8mb3 may not under utf8mb4 with older row formats).
- **Identity and security data.** Password hashes (bcrypt/argon2 strings) are stored as `VARCHAR`/`CHAR` in their textual encoding; API tokens and session ids are often stored as hashes in `BINARY(32)` so that a database leak does not expose usable tokens.
- **Content systems.** CMSs and e-commerce platforms keep long descriptions in `TEXT`/`MEDIUMTEXT`, often in a separate "details" table, and store media files in object storage (S3, GCS) with the URL or key in a `VARCHAR`.
- **Search.** Text columns that must be searched by words use `FULLTEXT` indexes ([Full-Text Search](/lessons/m13-full-text-search)) or an external search engine; `LIKE '%word%'` cannot use a B+ tree index.
- **Data pipelines.** When exporting to analytics systems, `CHAR` padding, collation-based equality and `ENUM` ordering are frequent sources of mismatches between MySQL results and Spark/pandas results, where string comparison is usually binary and case-sensitive.

## Practice Exercises

```exercise
id: ex1
title: Choose string types for a user table
difficulty: easy
kind: dml
dataset: empty
concepts: [char-varchar, binary-strings]
prompt: |
  Create a table `app_user` with these columns, in order:

  1. `user_id` — `INT UNSIGNED`, not null, primary key
  2. `email` — variable-length text up to 255 characters, not null
  3. `country_code` — exactly 2 characters, not null
  4. `password_hash` — a 32-byte binary value (SHA-256), not null
  5. `bio` — long free text up to 64 KB, nullable

  The checker reads `COLUMN_NAME`, `COLUMN_TYPE` and `IS_NULLABLE`.
hints:
  - "Text: VARCHAR(255), CHAR(2), TEXT. Binary: BINARY(32)."
  - "Remember NOT NULL where required; columns are nullable by default."
solution: |
  CREATE TABLE app_user (
    user_id       INT UNSIGNED NOT NULL PRIMARY KEY,
    email         VARCHAR(255) NOT NULL,
    country_code  CHAR(2)      NOT NULL,
    password_hash BINARY(32)   NOT NULL,
    bio           TEXT         NULL
  );
check: |
  SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE
  FROM information_schema.COLUMNS
  WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'app_user'
  ORDER BY ORDINAL_POSITION;
explanation: |
  `BINARY(32)` stores a raw SHA-256 digest in 32 bytes; `CHAR(64)` hex would take at least 64 bytes and
  compare case-insensitively. (In real systems, password hashes use slow algorithms such as bcrypt or argon2,
  whose output is a text string — but tokens and fingerprints are often SHA-256.) `TEXT` holds up to 65,535
  bytes — about 16K characters if every character needs 4 bytes.
```

```exercise
id: ex2
title: Bytes versus characters in customer names
difficulty: medium
concepts: [character-length]
setup: |
  CREATE TABLE intl_names (id INT PRIMARY KEY, full_name VARCHAR(50) NOT NULL);
  INSERT INTO intl_names VALUES
    (1, 'Ana Souza'), (2, 'José Núñez'), (3, '山田太郎'), (4, 'Zoë 😀'), (5, 'Ольга Петрова');
dataset: empty
prompt: |
  For every row of `intl_names`, return `id`, `chars` (number of characters in `full_name`), `bytes`
  (number of bytes) and `multibyte` (1 if the name contains at least one character that needs more than one
  byte, else 0). Order by `id`.
hints:
  - "CHAR_LENGTH counts characters; LENGTH counts bytes."
  - "A name contains a multi-byte character exactly when bytes > chars."
solution: |
  SELECT id,
         CHAR_LENGTH(full_name) AS chars,
         LENGTH(full_name)      AS bytes,
         LENGTH(full_name) > CHAR_LENGTH(full_name) AS multibyte
  FROM intl_names
  ORDER BY id;
orderMatters: true
explanation: |
  Only the pure-ASCII name has equal character and byte counts. Accented Latin and Cyrillic letters need 2
  bytes each, the kanji 3 and the emoji 4. Validation rules ("name at most 50 characters") must use
  CHAR_LENGTH; storage planning (row size, index limits) must use the maximum byte length.
```

```exercise
id: ex3
title: Clean up trailing spaces
difficulty: medium
kind: dml
dataset: empty
setup: |
  CREATE TABLE airport (code VARCHAR(5) PRIMARY KEY, city VARCHAR(40) NOT NULL);
  INSERT INTO airport VALUES ('SJC  ', 'San Jose'), ('SFO', 'San Francisco'), (' OAK', 'Oakland');
prompt: |
  An import left leading and trailing spaces in `airport.code`, so `WHERE code = 'SJC'` finds nothing.
  Fix the data so that every code has no leading or trailing spaces.

  The checker returns the codes that can now be found with exact equality for 'SJC', 'SFO' and 'OAK'.
hints:
  - "TRIM() removes leading and trailing spaces."
  - "UPDATE airport SET code = TRIM(code);"
solution: |
  UPDATE airport SET code = TRIM(code);
check: |
  SELECT code, city FROM airport WHERE code IN ('SJC', 'SFO', 'OAK') ORDER BY code;
explanation: |
  The default collation utf8mb4_0900_ai_ci is NO PAD, so 'SJC  ' ≠ 'SJC'; and leading spaces are significant
  in every collation. TRIM fixes both. Because `code` is the primary key, the UPDATE could fail if trimming
  made two keys equal (duplicate-key error 1062) — in real clean-ups, check for collisions first with
  `SELECT TRIM(code), COUNT(*) … GROUP BY TRIM(code) HAVING COUNT(*) > 1`.
```

```exercise
id: ex4
title: Replace a SET with a junction table
difficulty: hard
kind: dml
dataset: empty
setup: |
  CREATE TABLE article (article_id INT PRIMARY KEY, title VARCHAR(60) NOT NULL,
                        topics SET('sql', 'python', 'cloud') NOT NULL DEFAULT '');
  INSERT INTO article VALUES (1, 'Window functions', 'sql'),
                             (2, 'Pandas meets MySQL', 'sql,python'),
                             (3, 'Serverless ETL', 'python,cloud'),
                             (4, 'Release notes', '');
prompt: |
  Normalise the multi-valued `topics` column: create a table `article_topic` with columns
  `article_id INT NOT NULL` and `topic VARCHAR(20) NOT NULL` and primary key `(article_id, topic)`, and fill
  it with one row per (article, topic) pair taken from `article.topics`. Leave `article` unchanged.

  The checker reads all rows of `article_topic`.
hints:
  - "FIND_IN_SET('sql', topics) > 0 tells you whether an article has a topic."
  - "Insert one INSERT … SELECT per topic, or UNION ALL three SELECTs in one INSERT."
solution: |
  CREATE TABLE article_topic (
    article_id INT NOT NULL,
    topic      VARCHAR(20) NOT NULL,
    PRIMARY KEY (article_id, topic)
  );
  INSERT INTO article_topic (article_id, topic)
  SELECT article_id, 'sql'    FROM article WHERE FIND_IN_SET('sql', topics) > 0
  UNION ALL
  SELECT article_id, 'python' FROM article WHERE FIND_IN_SET('python', topics) > 0
  UNION ALL
  SELECT article_id, 'cloud'  FROM article WHERE FIND_IN_SET('cloud', topics) > 0;
check: |
  SELECT article_id, topic FROM article_topic ORDER BY article_id, topic;
explanation: |
  The junction table is in first normal form: one topic per row, indexable by `(article_id, topic)` (and by
  `topic` with a secondary index for "articles about python"), extensible without ALTER TABLE, and joinable to
  a `topic` lookup table with a foreign key. Article 4 correctly gets no rows. Using `topics LIKE '%sql%'`
  would be fragile (a topic named 'nosql' would match).
```

```exercise
id: ex5
title: Diagnose a failing schema
difficulty: challenge
kind: open
concepts: [row-size-limits, char-varchar, text-types]
prompt: |
  This table definition fails on MySQL 8.4 with utf8mb4 defaults. Identify **every** problem (there are at
  least four), explain the limit involved, and rewrite it.

  ~~~text
  CREATE TABLE product_page (
    product_id   INT PRIMARY KEY,
    slug         VARCHAR(1000) NOT NULL UNIQUE,
    title        VARCHAR(5000) NOT NULL,
    summary      VARCHAR(5000),
    body         VARCHAR(20000),
    lang         CHAR(5) CHARACTER SET utf8 DEFAULT 'en_US',
    keywords     SET('sale','new','eco','kids','men','women'),
    notes        TEXT DEFAULT ''
  );
  ~~~
solution: |
  1. **`slug VARCHAR(1000) UNIQUE`** — a unique index on up to 4,000 bytes exceeds the 3,072-byte InnoDB key
     limit (error 1071). Use a realistic length (`VARCHAR(200)` is plenty for slugs) or a prefix/hash index.
  2. **`body VARCHAR(20000)`** — exceeds the VARCHAR maximum of 16,383 utf8mb4 characters (error 1074);
     and even smaller widths make the row's maximum size exceed 65,535 bytes: title + summary + body could
     need (5,000 + 5,000 + 20,000) × 4 bytes. Use `TEXT`/`MEDIUMTEXT` for long text (bodies stored off-page,
     counting only a few bytes toward the row limit).
  3. **`notes TEXT DEFAULT ''`** — TEXT cannot have a literal default (error 1101); use `DEFAULT ('')`
     (8.0.13+) or make it nullable without a default.
  4. **`CHARACTER SET utf8`** — deprecated alias of utf8mb3 (warning 3719); use utf8mb4 (or `VARCHAR(10)`
     with an ASCII/`utf8mb4_bin` collation for locale codes such as `en_US`, `zh-Hant-TW` which can exceed 5).
  5. **`SET(...)` keywords** — multi-valued, not first normal form, hard to index; use a junction table.

  ~~~sql
  CREATE TABLE product_page (
    product_id INT UNSIGNED NOT NULL PRIMARY KEY,
    slug       VARCHAR(200)  NOT NULL,
    title      VARCHAR(300)  NOT NULL,
    summary    VARCHAR(1000) NULL,
    body       MEDIUMTEXT    NULL,
    lang       VARCHAR(16)   NOT NULL DEFAULT 'en-US',
    notes      TEXT          NULL,
    UNIQUE KEY uq_product_page_slug (slug)
  ) DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci;

  CREATE TABLE product_page_keyword (
    product_id INT UNSIGNED NOT NULL,
    keyword    VARCHAR(30)  NOT NULL,
    PRIMARY KEY (product_id, keyword),
    CONSTRAINT fk_ppk_page FOREIGN KEY (product_id) REFERENCES product_page (product_id)
  );
  ~~~
```

## Knowledge Assessment

```quiz
- id: q1
  question: "A CHAR(5) column and a VARCHAR(5) column both receive the value 'ab  ' (two trailing spaces). What does SELECT return for each?"
  options: ["'ab  ' and 'ab  '", "'ab' and 'ab  '", "'ab' and 'ab'", "'ab   ' (padded to 5) and 'ab  '"]
  answer: 1
  explanation: "CHAR values are padded for storage and trailing spaces are stripped on retrieval; VARCHAR preserves trailing spaces. CHAR never returns its padding (unless the deprecated PAD_CHAR_TO_FULL_LENGTH mode is set)."
  concepts: [char-varchar, trailing-spaces]
- id: q2
  question: "`SELECT LENGTH('東京'), CHAR_LENGTH('東京');` with utf8mb4 returns:"
  options: ["2, 2", "6, 2", "2, 6", "4, 2"]
  answer: 1
  explanation: "LENGTH counts bytes and CHAR_LENGTH counts characters. Each of these CJK characters takes 3 bytes in UTF-8, so 6 bytes and 2 characters."
  concepts: [character-length]
- id: q3
  question: "Why can't a utf8mb3 (`utf8`) column store '👍'?"
  options:
    - "Because emoji are not Unicode"
    - "Because utf8mb3 stores at most 3 bytes per character and the emoji needs 4"
    - "Because VARCHAR cannot store symbols"
    - "Because strict mode forbids emoji"
  answer: 1
  explanation: "Emoji are supplementary Unicode characters that need 4 bytes in UTF-8. utf8mb3 supports only up to 3 bytes (the Basic Multilingual Plane), so the insert fails with error 1366 in strict mode. utf8mb4 is full UTF-8."
  concepts: [character-length]
- id: q4
  question: "Under MySQL 8's default collation utf8mb4_0900_ai_ci, a VARCHAR column holds 'SJC ' (one trailing space). Which predicates are true? (Select all that apply.)"
  type: multi
  options: ["code = 'SJC'", "code = 'sjc '", "code LIKE 'SJC%'", "code = 'SJC' COLLATE utf8mb4_general_ci"]
  answer: [1, 2, 3]
  explanation: "utf8mb4_0900_ai_ci is NO PAD (trailing spaces significant) and case-insensitive, so 'SJC ' equals 'sjc ' but not 'SJC'. LIKE 'SJC%' matches the prefix. utf8mb4_general_ci is PAD SPACE, so trailing spaces are ignored and the comparison is true."
  concepts: [trailing-spaces]
- id: q5
  question: "Why does `CREATE TABLE t (url VARCHAR(1000), INDEX (url))` fail with utf8mb4?"
  options:
    - "VARCHAR cannot exceed 255 characters"
    - "The index key could need 4,000 bytes, more than the 3,072-byte limit"
    - "URLs must be stored as TEXT"
    - "Indexes on VARCHAR require a prefix length in all cases"
  answer: 1
  explanation: "Index key length is limited in bytes (3,072 for DYNAMIC rows). 1,000 utf8mb4 characters can take 4,000 bytes. VARCHAR can be much longer than 255, and full-column indexes are fine when they fit."
  concepts: [row-size-limits]
- id: q6
  question: "An ENUM('low','medium','high') column is sorted with ORDER BY. In what order do the values appear?"
  options: ["high, low, medium (alphabetical)", "low, medium, high (definition order)", "Random", "By insertion time"]
  answer: 1
  explanation: "ENUM values are stored as their 1-based index in the definition and sort by that index. Cast to CHAR to sort alphabetically."
  concepts: [enum-set]
- id: q7
  question: "Which statements about TEXT and BLOB columns in MySQL 8.4 are true? (Select all that apply.)"
  type: multi
  options:
    - "A TEXT column holds up to 65,535 bytes, which may be fewer than 65,535 characters"
    - "They can have literal defaults such as DEFAULT ''"
    - "Their bodies can be stored off-page, counting only a few bytes toward the 65,535-byte row limit"
    - "BLOB values are compared using the column's collation"
  answer: [0, 2]
  explanation: "TEXT capacity is in bytes; DYNAMIC rows store long values off-page with a 20-byte pointer, and TEXT/BLOB count 9–12 bytes toward the row limit. Literal defaults are not allowed (expression defaults are, since 8.0.13). BLOBs are binary and compared byte by byte."
  concepts: [text-types, row-size-limits]
- id: q8
  question: "You need to store SHA-256 hashes for fast exact lookups. Which column type is best?"
  options: ["VARCHAR(255)", "CHAR(64) with utf8mb4_0900_ai_ci", "BINARY(32)", "TEXT"]
  answer: 2
  explanation: "BINARY(32) stores the raw 32-byte digest, compares bytes exactly and makes a small index. CHAR(64) hex doubles the size and compares case-insensitively; VARCHAR(255) and TEXT waste space and TEXT needs a prefix index."
  concepts: [binary-strings]
```
