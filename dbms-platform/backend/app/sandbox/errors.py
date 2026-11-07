"""Learner-friendly explanations for common MySQL error codes.

Codes and their meanings follow the MySQL 8.4 Server Error Message Reference:
https://dev.mysql.com/doc/mysql-errors/8.4/en/server-error-reference.html
"""
from __future__ import annotations

_HINTS: dict[int, str] = {
    1046: "No database selected. In the sandbox your lab schema is selected automatically — reset the lab if you see this.",
    1048: "A NOT NULL column received NULL. Supply a value or give the column a DEFAULT.",
    1050: "A table with that name already exists. Use CREATE TABLE IF NOT EXISTS, DROP it first, or reset the lab.",
    1051: "The table you tried to drop does not exist. DROP TABLE IF EXISTS avoids the error.",
    1052: "A column name is ambiguous because it exists in more than one joined table. Qualify it with the table name or alias, e.g. o.customer_id.",
    1054: "Unknown column. Check spelling, the table alias, and remember that SELECT-list aliases are not visible in WHERE (WHERE is evaluated before SELECT).",
    1055: "ONLY_FULL_GROUP_BY: every selected column must be grouped, aggregated, or functionally dependent on the GROUP BY columns.",
    1060: "Duplicate column name in the result or table definition. Give expressions distinct aliases.",
    1061: "An index with that name already exists on the table.",
    1062: "Duplicate entry: the value violates a PRIMARY KEY or UNIQUE constraint. Use a different key, or INSERT ... ON DUPLICATE KEY UPDATE if an upsert is intended.",
    1064: "Syntax error. MySQL shows the text right after the point where parsing failed — look just before that fragment for a missing comma, quote, parenthesis or keyword.",
    1091: "The column or index you tried to drop does not exist.",
    1093: "MySQL does not allow updating/deleting a table while selecting from the same table in a subquery. Wrap the subquery in a derived table or use a JOIN.",
    1111: "Invalid use of a group function: aggregates cannot appear in WHERE. Filter on aggregates with HAVING.",
    1136: "The number of values does not match the number of columns. List the target columns explicitly in INSERT.",
    1140: "Mixing aggregate and non-aggregate columns without GROUP BY. Add a GROUP BY clause.",
    1146: "The table does not exist in your lab. If the example depends on earlier steps, run them first (or use 'Replay from start').",
    1205: "Lock wait timeout: another transaction held a conflicting lock for longer than innodb_lock_wait_timeout. Only the statement was rolled back, not the transaction (unless innodb_rollback_on_timeout is ON).",
    1213: "Deadlock: InnoDB detected a cycle of transactions waiting for each other and rolled this transaction back. The application should retry the whole transaction.",
    1216: "Foreign key violation: the referenced parent row does not exist.",
    1217: "Foreign key violation: child rows still reference this parent row.",
    1227: "Access denied: this statement needs a privilege the sandbox account does not have (for safety, server administration is disabled).",
    1242: "A scalar subquery returned more than one row. Use IN/EXISTS, add a condition, or aggregate inside the subquery.",
    1264: "Out of range value for the column's data type (strict SQL mode rejects it instead of clamping).",
    1265: "Data truncated: the value does not fit the column (for example an invalid ENUM value or too-long string).",
    1292: "Incorrect value for the data type, often an invalid date/time literal or a string compared/converted to a number.",
    1317: "The statement was interrupted. In the sandbox this usually means it exceeded the time limit.",
    1364: "A NOT NULL column without a DEFAULT was not given a value.",
    1366: "Incorrect value for the column type (for example text inserted into an integer column).",
    1406: "The string is longer than the column allows.",
    1416: "Cannot get geometry object from data you send to the GEOMETRY field.",
    1442: "A trigger cannot modify the table that fired it.",
    1451: "Foreign key violation: cannot delete or update a parent row because child rows reference it. Delete the children first, or define ON DELETE CASCADE/SET NULL deliberately.",
    1452: "Foreign key violation: cannot add or update a child row because the referenced parent row does not exist.",
    1690: "Value out of range during arithmetic, often subtraction on an UNSIGNED column going below zero.",
    3024: "The query exceeded max_execution_time and was aborted by the server.",
    3572: "Statement aborted because NOWAIT was requested and the row is locked by another transaction.",
    3819: "CHECK constraint violated: the new row does not satisfy the constraint expression.",
    1644: "A user-defined error raised with SIGNAL.",
    4031: "The connection was closed by the server (idle timeout). Run again to reconnect.",
}


def explain_error(code: int | None, message: str, timeout_ms: int) -> str | None:
    if code in (1317, 3024):
        return f"The statement exceeded the sandbox time limit of {timeout_ms / 1000:.0f} s and was stopped. " \
               "Check for a missing join condition (accidental Cartesian product), an unbounded recursive CTE, or add an index."
    if code == 1044 or code == 1142 or code == 1227 or code == 1045:
        return "Access denied: the sandbox account can only work inside its own lab schema, and server administration is disabled."
    if code == 1226:
        return "Your lab exceeded its hourly resource quota (queries per hour). Wait a little and try again."
    return _HINTS.get(code)
