#!/bin/bash
# Creates the sandbox admin account: it can create/drop lab schemas and per-lab accounts
# (privileges limited to `lab\_%` schemas), kill runaway queries, and read lock metadata.
set -euo pipefail
mysql -uroot -p"${MYSQL_ROOT_PASSWORD}" <<SQL
CREATE USER IF NOT EXISTS 'sandbox_admin'@'%' IDENTIFIED BY '${SANDBOX_ADMIN_PASSWORD}';
GRANT CREATE USER, CONNECTION_ADMIN, PROCESS ON *.* TO 'sandbox_admin'@'%';
GRANT ALL PRIVILEGES ON \`lab\\_%\`.* TO 'sandbox_admin'@'%' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON \`sandbox_meta\`.* TO 'sandbox_admin'@'%';
GRANT SELECT ON performance_schema.* TO 'sandbox_admin'@'%';
SQL
