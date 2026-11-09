# DB Academy — Integrated DBMS & MySQL Learning Platform

An interactive course that teaches **database systems theory and MySQL together**: every lesson goes
*concept → DBMS theory → MySQL implementation → internal working → execution walkthrough → optimization →
industry application → graded practice*, against a **real MySQL server**.

- **Curriculum:** 13 modules, 154 lessons, 13 module assessments, 7 projects and 3 cumulative exams
  (`content/curriculum.yaml`), from DBMS fundamentals to indexing, transactions, internals, security, production
  engineering and data warehousing — about 700,000 words, 920+ graded exercises and 1,500+ quiz/assessment
  questions. Every lesson and project is written and validated; see [docs/STATUS.md](docs/STATUS.md).
- **Real SQL sandbox:** each learner gets private, resettable MySQL schemas (playground, per lesson, per exercise,
  concurrency labs), isolated by per-lab MySQL accounts and resource limits.
- **Verified content:** every example output in a lesson was captured by running it on MySQL
  (`backend/scripts/capture_content.py`); nothing is hand-written or simulated.
- **Interactive tools:** Monaco SQL editor, schema browser, visual EXPLAIN, two-session concurrency lab with live
  InnoDB lock inspection, B+ tree, join, ER-builder (→ DDL), normalization analyzer, buffer-pool LRU, isolation
  matrix and query-order visualizations.
- **Assessment:** exercises graded by comparing your result with the reference solution on fresh data, progressive
  hints, solutions unlocked only after real attempts, quizzes, module assessments with SQL challenges and interview
  questions, cumulative exams, and weak-concept tracking across everything you answer.

## Run locally (development)

Prerequisites: Python 3.12+, Node 20+, a local MySQL 8.x server used as the sandbox.

```bash
mysql -uroot < docker/sandbox-local-setup.sql        # creates the sandbox admin account
```

```bash
cd backend
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000             # app data in SQLite by default (dbms_app.db)
```

```bash
cd frontend
npm install
npm run dev                                           # http://localhost:5173 (proxies /api to :8000)
```

Configuration is environment-based (`DBMS_*` variables, see `backend/app/config.py`). To use MySQL for application
data, set `DBMS_APP_DATABASE_URL=mysql+pymysql://user:pass@host/dbms_app` and run `alembic upgrade head`.

## Run with Docker (production-style)

```bash
cp .env.example .env      # then set real secrets
docker compose up --build # http://localhost:8080
```

This starts two MySQL 8.4 servers — `app-db` for application data and a hardened `sandbox-db` for learner SQL —
plus the FastAPI backend and an nginx-served frontend.

## Tests and content validation

```bash
cd backend && python -m pytest -q                       # API, sandbox isolation, grading, splitter, guard
cd frontend && npm test && npm run build                # algorithm tests + type-check + production build
cd backend && python scripts/capture_content.py --all   # lint + execute every lesson against MySQL
cd backend && python scripts/status_report.py           # regenerate docs/STATUS.md
```

## Repository layout

```
content/      curriculum.yaml · lessons/<module>/<lesson>.md · assessments/ · projects/ · datasets/ · generated/captures/
backend/      FastAPI app (app/), sandbox engine (app/sandbox/), grading, Alembic migrations, tests, scripts
frontend/     React + TypeScript SPA (pages/, components/lesson/, viz/)
docs/         ARCHITECTURE.md · AUTHORING.md (content contract) · STATUS.md (generated progress record)
docker/       sandbox server initialisation
```

## Re-capturing outputs on MySQL 8.4

Lesson outputs were captured on the MySQL server available while the course was built (8.0.46; see the
limitations below). To refresh every captured output on the Docker MySQL 8.4 sandbox:

```bash
docker compose run --rm -v ./content:/srv/content backend python scripts/capture_content.py --all
```

Any document whose behaviour differs on 8.4 is reported with the failing block, so prose can be updated.

## Known limitations

- **Captured on MySQL 8.0.46, targeting 8.4 LTS.** The build environment could not run an 8.4 server, so outputs,
  plans and error messages were captured on 8.0.46. Run blocks use syntax valid on both; 8.4-only behaviour is
  shown display-only in `version` callouts and has not been executed. Re-capture on 8.4 (above) before relying on
  exact plan or error text.
- **Docker deployment not exercised here.** `docker compose` could not run in the build environment; the backend,
  frontend and sandbox were tested against a locally installed MySQL instead.
- **Timings are indicative.** `measured` blocks were captured on a shared build machine while other captures ran;
  lessons argue from rows examined and plan shape rather than milliseconds.
- **Server-administration output is illustrative.** The sandbox deliberately cannot run GRANT/CREATE USER,
  SHOW ENGINE INNODB STATUS, mysqldump/mysqlbinlog, replication or performance_schema queries. Where lessons show
  such output it is labelled as an illustrative example, not a capture.
- **One backend worker per instance.** Concurrent lab sessions (A/B/C) live in the backend process; scale out with
  sticky sessions.
- **Projects reveal earlier answers.** Later project stages open with "checkpoint" blocks containing the reference
  solution of the previous stage, so that each stage can build on a known schema.
- **Synthetic data.** Datasets and generated tables are synthetic; for example the analytics project's retention
  rates are higher than real products typically see.

## MySQL version

The course targets **MySQL 8.4 LTS** (the Docker images). Each lesson shows which server version its outputs were
captured on; differences between 8.0 and 8.4 are called out in `version` callouts.
