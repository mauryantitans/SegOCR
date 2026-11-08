-- One-time setup for a LOCAL MySQL server used as the sandbox during development
-- (run as root: mysql -uroot < docker/sandbox-local-setup.sql). Use a dedicated server in production.
CREATE USER IF NOT EXISTS 'sandbox_admin'@'localhost' IDENTIFIED BY 'sandbox_admin_pw';
GRANT CREATE USER, CONNECTION_ADMIN, PROCESS ON *.* TO 'sandbox_admin'@'localhost';
GRANT ALL PRIVILEGES ON `lab\_%`.* TO 'sandbox_admin'@'localhost' WITH GRANT OPTION;
GRANT ALL PRIVILEGES ON `sandbox_meta`.* TO 'sandbox_admin'@'localhost';
GRANT SELECT ON performance_schema.* TO 'sandbox_admin'@'localhost';
-- With binary logging enabled (the default), creating stored functions/triggers needs this:
SET GLOBAL log_bin_trust_function_creators = 1;
SET GLOBAL event_scheduler = OFF;
