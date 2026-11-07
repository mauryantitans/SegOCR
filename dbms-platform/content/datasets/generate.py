"""Deterministic generator for the sample datasets used by lessons and labs.

Run:  python content/datasets/generate.py
Writes one <name>.sql file per dataset next to this script. The generated SQL
is committed so the platform never depends on running this script, but the
script is the source of truth: edit it, re-run it, then re-run content capture.

Design rules for every dataset:
  * Plain DDL + INSERTs only (no CREATE DATABASE / USE / DELIMITER / routines):
    files are loaded into a learner's private lab schema by the sandbox user.
  * Small enough to read results by eye, realistic enough to be meaningful.
  * Deliberate edge cases: NULLs, rows with no matches (for outer/anti joins),
    ties (for ranking), hierarchies (for self joins / recursive CTEs).
"""
from __future__ import annotations

import datetime as dt
import random
from decimal import Decimal
from pathlib import Path

OUT = Path(__file__).resolve().parent


def lit(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, Decimal)):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, dt.datetime):
        return "'" + v.strftime("%Y-%m-%d %H:%M:%S") + "'"
    if isinstance(v, dt.date):
        return "'" + v.isoformat() + "'"
    return "'" + str(v).replace("\\", "\\\\").replace("'", "''") + "'"


def inserts(table: str, cols: list[str], rows: list[tuple]) -> str:
    out = [f"INSERT INTO {table} ({', '.join(cols)}) VALUES"]
    out.append(",\n".join("  (" + ", ".join(lit(v) for v in r) + ")" for r in rows) + ";")
    return "\n".join(out)


def money(x: float) -> Decimal:
    return Decimal(f"{x:.2f}")


FIRST = ["Aarav", "Olivia", "Mateo", "Priya", "Liam", "Sofia", "Chen", "Amara", "Noah", "Fatima",
         "Lucas", "Mei", "Ethan", "Zara", "Diego", "Hana", "Omar", "Grace", "Ravi", "Isabella",
         "Kofi", "Elena", "Jonas", "Aisha", "Daniel", "Yuki", "Samuel", "Leila", "Marcus", "Nina",
         "Arjun", "Chloe", "Tomas", "Ingrid", "Kwame", "Lucia", "Hiro", "Sara", "Victor", "Anika"]
LAST = ["Sharma", "Johnson", "Garcia", "Nguyen", "Smith", "Rossi", "Wang", "Okafor", "Brown", "Khan",
        "Silva", "Lin", "Miller", "Ahmed", "Lopez", "Sato", "Haddad", "Kim", "Patel", "Martin",
        "Mensah", "Petrova", "Berg", "Rahman", "Wilson", "Tanaka", "Adeyemi", "Moradi", "Clark", "Novak"]
US_CITIES = [("San Jose", "CA"), ("San Francisco", "CA"), ("Los Angeles", "CA"), ("Seattle", "WA"),
             ("Austin", "TX"), ("New York", "NY"), ("Boston", "MA"), ("Chicago", "IL"), ("Denver", "CO")]


# ---------------------------------------------------------------------------
# shop — e-commerce
# ---------------------------------------------------------------------------
def shop() -> str:
    r = random.Random(42)
    ddl = """-- Dataset: shop (e-commerce). Tables: customers, categories, products, orders,
-- order_items, payments, reviews.
CREATE TABLE customers (
  customer_id   INT UNSIGNED NOT NULL AUTO_INCREMENT,
  first_name    VARCHAR(50)  NOT NULL,
  last_name     VARCHAR(50)  NOT NULL,
  email         VARCHAR(255) NOT NULL,
  phone         VARCHAR(20)  NULL,
  city          VARCHAR(60)  NOT NULL,
  state         CHAR(2)      NOT NULL,
  loyalty_tier  ENUM('bronze','silver','gold') NULL,
  signup_date   DATE         NOT NULL,
  PRIMARY KEY (customer_id),
  UNIQUE KEY uq_customers_email (email)
) ENGINE=InnoDB;

CREATE TABLE categories (
  category_id  INT UNSIGNED NOT NULL AUTO_INCREMENT,
  name         VARCHAR(60)  NOT NULL,
  parent_id    INT UNSIGNED NULL,
  PRIMARY KEY (category_id),
  UNIQUE KEY uq_categories_name (name),
  CONSTRAINT fk_categories_parent FOREIGN KEY (parent_id) REFERENCES categories (category_id)
) ENGINE=InnoDB;

CREATE TABLE products (
  product_id   INT UNSIGNED NOT NULL AUTO_INCREMENT,
  sku          VARCHAR(20)   NOT NULL,
  name         VARCHAR(120)  NOT NULL,
  category_id  INT UNSIGNED  NULL,
  price        DECIMAL(10,2) NOT NULL,
  cost         DECIMAL(10,2) NOT NULL,
  stock_qty    INT           NOT NULL DEFAULT 0,
  is_active    BOOLEAN       NOT NULL DEFAULT TRUE,
  created_at   DATETIME      NOT NULL,
  PRIMARY KEY (product_id),
  UNIQUE KEY uq_products_sku (sku),
  KEY idx_products_category (category_id),
  CONSTRAINT fk_products_category FOREIGN KEY (category_id) REFERENCES categories (category_id),
  CONSTRAINT chk_products_price CHECK (price >= 0)
) ENGINE=InnoDB;

CREATE TABLE orders (
  order_id       INT UNSIGNED NOT NULL AUTO_INCREMENT,
  customer_id    INT UNSIGNED NOT NULL,
  order_date     DATETIME     NOT NULL,
  status         ENUM('pending','paid','shipped','delivered','cancelled','refunded') NOT NULL,
  shipped_at     DATETIME     NULL,
  discount_code  VARCHAR(20)  NULL,
  PRIMARY KEY (order_id),
  KEY idx_orders_customer (customer_id),
  KEY idx_orders_date (order_date),
  CONSTRAINT fk_orders_customer FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
) ENGINE=InnoDB;

CREATE TABLE order_items (
  order_id    INT UNSIGNED  NOT NULL,
  product_id  INT UNSIGNED  NOT NULL,
  quantity    INT UNSIGNED  NOT NULL,
  unit_price  DECIMAL(10,2) NOT NULL,
  PRIMARY KEY (order_id, product_id),
  KEY idx_order_items_product (product_id),
  CONSTRAINT fk_items_order   FOREIGN KEY (order_id)   REFERENCES orders (order_id),
  CONSTRAINT fk_items_product FOREIGN KEY (product_id) REFERENCES products (product_id),
  CONSTRAINT chk_items_qty CHECK (quantity > 0)
) ENGINE=InnoDB;

CREATE TABLE payments (
  payment_id  INT UNSIGNED  NOT NULL AUTO_INCREMENT,
  order_id    INT UNSIGNED  NOT NULL,
  amount      DECIMAL(10,2) NOT NULL,
  method      ENUM('card','paypal','bank_transfer','gift_card') NOT NULL,
  status      ENUM('succeeded','failed','refunded') NOT NULL,
  paid_at     DATETIME      NOT NULL,
  PRIMARY KEY (payment_id),
  KEY idx_payments_order (order_id),
  CONSTRAINT fk_payments_order FOREIGN KEY (order_id) REFERENCES orders (order_id)
) ENGINE=InnoDB;

CREATE TABLE reviews (
  review_id    INT UNSIGNED NOT NULL AUTO_INCREMENT,
  product_id   INT UNSIGNED NOT NULL,
  customer_id  INT UNSIGNED NOT NULL,
  rating       TINYINT UNSIGNED NOT NULL,
  comment      VARCHAR(500) NULL,
  created_at   DATETIME     NOT NULL,
  PRIMARY KEY (review_id),
  UNIQUE KEY uq_reviews_product_customer (product_id, customer_id),
  KEY idx_reviews_customer (customer_id),
  CONSTRAINT fk_reviews_product  FOREIGN KEY (product_id)  REFERENCES products (product_id),
  CONSTRAINT fk_reviews_customer FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
  CONSTRAINT chk_reviews_rating CHECK (rating BETWEEN 1 AND 5)
) ENGINE=InnoDB;
"""
    parts = [ddl]

    # customers: 25; customers 23-25 never order (outer/anti join practice)
    customers = []
    used = set()
    for cid in range(1, 26):
        while True:
            fn, ln = r.choice(FIRST), r.choice(LAST)
            if (fn, ln) not in used:
                used.add((fn, ln))
                break
        city, state = r.choice(US_CITIES)
        phone = None if cid % 5 == 0 else f"+1-{r.randint(200, 989)}-555-{r.randint(1000, 9999)}"
        tier = None if cid % 7 == 3 else r.choice(["bronze", "bronze", "silver", "gold"])
        signup = dt.date(2023, 1, 1) + dt.timedelta(days=r.randint(0, 540))
        customers.append((cid, fn, ln, f"{fn.lower()}.{ln.lower()}{cid}@example.com", phone, city, state, tier, signup))
    parts.append(inserts("customers", ["customer_id", "first_name", "last_name", "email", "phone", "city", "state",
                                       "loyalty_tier", "signup_date"], customers))

    categories = [
        (1, "Electronics", None), (2, "Computers", 1), (3, "Laptops", 2), (4, "Accessories", 2),
        (5, "Audio", 1), (6, "Home & Kitchen", None), (7, "Coffee", 6), (8, "Books", None),
        (9, "Databases", 8), (10, "Fitness", None),
    ]
    parts.append(inserts("categories", ["category_id", "name", "parent_id"], categories))

    catalog = [
        ("Ultrabook 14 Pro", 3, 1299.00), ("Ultrabook 13 Air", 3, 999.00), ("Gaming Laptop 16", 3, 1799.00),
        ("Chromebook 11", 3, 249.00), ("Wireless Mouse", 4, 29.99), ("Mechanical Keyboard", 4, 89.99),
        ("USB-C Hub 7-in-1", 4, 49.99), ("27\" 4K Monitor", 2, 379.00), ("Laptop Stand", 4, 39.99),
        ("Noise-Cancelling Headphones", 5, 249.99), ("Bluetooth Speaker", 5, 79.99), ("Wireless Earbuds", 5, 129.99),
        ("Studio Microphone", 5, 149.00), ("Espresso Machine", 7, 499.00), ("Burr Coffee Grinder", 7, 129.00),
        ("Pour-Over Kettle", 7, 59.00), ("Single-Origin Beans 1kg", 7, 34.50), ("Chef's Knife", 6, 89.00),
        ("Cast Iron Skillet", 6, 44.99), ("Database Internals (Book)", 9, 54.99),
        ("Designing Data-Intensive Applications", 9, 49.99), ("High Performance MySQL", 9, 59.99),
        ("SQL Antipatterns", 9, 39.99), ("Fiction Bestseller", 8, 18.99), ("Yoga Mat", 10, 35.00),
        ("Adjustable Dumbbells", 10, 299.00), ("Resistance Bands Set", 10, 24.99), ("Fitness Tracker", 10, 99.00),
        ("Gift Card $50", None, 50.00), ("Webcam 1080p", 4, 69.99),
    ]
    products = []
    for pid, (name, cat, price) in enumerate(catalog, start=1):
        cost = money(price * r.uniform(0.45, 0.75))
        stock = 0 if pid in (4, 13) else r.randint(3, 250)
        active = pid != 4
        created = dt.datetime(2022, 6, 1) + dt.timedelta(days=r.randint(0, 600), minutes=r.randint(0, 1439))
        products.append((pid, f"SKU-{1000 + pid}", name, cat, money(price), cost, stock, active, created))
    parts.append(inserts("products", ["product_id", "sku", "name", "category_id", "price", "cost", "stock_qty",
                                      "is_active", "created_at"], products))
    price_of = {p[0]: p[4] for p in products}

    orders, items, payments = [], [], []
    statuses = ["delivered"] * 6 + ["shipped"] * 2 + ["paid", "pending", "cancelled", "refunded"]
    pay_id = 1
    for oid in range(1, 81):
        cid = r.randint(1, 22)
        odate = dt.datetime(2024, 1, 1, 8) + dt.timedelta(days=r.randint(0, 364), minutes=r.randint(0, 720))
        status = r.choice(statuses)
        shipped = odate + dt.timedelta(days=r.randint(1, 4), hours=r.randint(0, 12)) if status in ("shipped", "delivered", "refunded") else None
        code = r.choice([None, None, None, "WELCOME10", "SPRING15", "VIP20"])
        orders.append((oid, cid, odate, status, shipped, code))
        n = r.choice([1, 1, 2, 2, 3, 4])
        pids = r.sample([p for p in range(1, 31) if p != 4], n)
        total = Decimal("0")
        for pid in pids:
            q = r.choice([1, 1, 1, 2, 3])
            up = price_of[pid]
            if r.random() < 0.15:  # historical price differs from current list price
                up = money(float(up) * 0.9)
            items.append((oid, pid, q, up))
            total += up * q
        if status != "pending":
            method = r.choice(["card", "card", "card", "paypal", "bank_transfer", "gift_card"])
            pstat = {"cancelled": "failed", "refunded": "refunded"}.get(status, "succeeded")
            if status == "cancelled" and r.random() < 0.5:
                continue  # some cancelled orders never reached payment
            payments.append((pay_id, oid, total, method, pstat, odate + dt.timedelta(minutes=r.randint(1, 50))))
            pay_id += 1
    parts.append(inserts("orders", ["order_id", "customer_id", "order_date", "status", "shipped_at", "discount_code"], orders))
    parts.append(inserts("order_items", ["order_id", "product_id", "quantity", "unit_price"], items))
    parts.append(inserts("payments", ["payment_id", "order_id", "amount", "method", "status", "paid_at"], payments))

    comments = ["Excellent quality, would buy again.", "Does what it says.", "Arrived late but works fine.",
                "Not worth the price.", "Five stars!", None, None, "Great for daily use.", "Packaging was damaged."]
    reviews, seen = [], set()
    bought = [(o[1], it[1], o[2]) for o in orders for it in items if it[0] == o[0] and o[3] == "delivered"]
    r.shuffle(bought)
    rid = 1
    for cid, pid, odate in bought:
        if (pid, cid) in seen or len(reviews) >= 45:
            continue
        seen.add((pid, cid))
        rating = r.choice([5, 5, 4, 4, 4, 3, 2, 1])
        reviews.append((rid, pid, cid, rating, r.choice(comments), odate + dt.timedelta(days=r.randint(5, 30))))
        rid += 1
    parts.append(inserts("reviews", ["review_id", "product_id", "customer_id", "rating", "comment", "created_at"], reviews))
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# hr — employees with a management hierarchy
# ---------------------------------------------------------------------------
def hr() -> str:
    r = random.Random(7)
    ddl = """-- Dataset: hr (human resources). Tables: locations, departments, jobs, employees,
-- salary_history. employees.manager_id is a self-referencing hierarchy.
CREATE TABLE locations (
  location_id  INT UNSIGNED NOT NULL,
  city         VARCHAR(60)  NOT NULL,
  country      VARCHAR(60)  NOT NULL,
  PRIMARY KEY (location_id)
) ENGINE=InnoDB;

CREATE TABLE departments (
  department_id  INT UNSIGNED  NOT NULL,
  name           VARCHAR(60)   NOT NULL,
  location_id    INT UNSIGNED  NOT NULL,
  budget         DECIMAL(12,2) NOT NULL,
  PRIMARY KEY (department_id),
  UNIQUE KEY uq_departments_name (name),
  CONSTRAINT fk_departments_location FOREIGN KEY (location_id) REFERENCES locations (location_id)
) ENGINE=InnoDB;

CREATE TABLE jobs (
  job_id      VARCHAR(10)   NOT NULL,
  title       VARCHAR(60)   NOT NULL,
  min_salary  DECIMAL(10,2) NOT NULL,
  max_salary  DECIMAL(10,2) NOT NULL,
  PRIMARY KEY (job_id),
  CONSTRAINT chk_jobs_range CHECK (min_salary <= max_salary)
) ENGINE=InnoDB;

CREATE TABLE employees (
  employee_id     INT UNSIGNED  NOT NULL,
  first_name      VARCHAR(50)   NOT NULL,
  last_name       VARCHAR(50)   NOT NULL,
  email           VARCHAR(255)  NOT NULL,
  hire_date       DATE          NOT NULL,
  job_id          VARCHAR(10)   NOT NULL,
  department_id   INT UNSIGNED  NULL,
  manager_id      INT UNSIGNED  NULL,
  salary          DECIMAL(10,2) NOT NULL,
  commission_pct  DECIMAL(4,2)  NULL,
  PRIMARY KEY (employee_id),
  UNIQUE KEY uq_employees_email (email),
  KEY idx_employees_department (department_id),
  KEY idx_employees_manager (manager_id),
  CONSTRAINT fk_employees_job        FOREIGN KEY (job_id)        REFERENCES jobs (job_id),
  CONSTRAINT fk_employees_department FOREIGN KEY (department_id) REFERENCES departments (department_id),
  CONSTRAINT fk_employees_manager    FOREIGN KEY (manager_id)    REFERENCES employees (employee_id)
) ENGINE=InnoDB;

CREATE TABLE salary_history (
  employee_id     INT UNSIGNED  NOT NULL,
  effective_date  DATE          NOT NULL,
  salary          DECIMAL(10,2) NOT NULL,
  PRIMARY KEY (employee_id, effective_date),
  CONSTRAINT fk_salary_history_employee FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
) ENGINE=InnoDB;
"""
    parts = [ddl]
    locations = [(1, "San Jose", "United States"), (2, "New York", "United States"), (3, "London", "United Kingdom"),
                 (4, "Bengaluru", "India"), (5, "Toronto", "Canada")]
    departments = [(10, "Executive", 1, Decimal("900000.00")), (20, "Engineering", 1, Decimal("4200000.00")),
                   (30, "Data Platform", 4, Decimal("1800000.00")), (40, "Sales", 2, Decimal("2100000.00")),
                   (50, "Marketing", 3, Decimal("950000.00")), (60, "Finance", 2, Decimal("700000.00")),
                   (70, "Research", 5, Decimal("500000.00"))]  # Research has no employees
    jobs = [("CEO", "Chief Executive Officer", 250000, 400000), ("VP", "Vice President", 180000, 280000),
            ("ENG_MGR", "Engineering Manager", 160000, 230000), ("SWE", "Software Engineer", 110000, 190000),
            ("DBA", "Database Administrator", 115000, 185000), ("DATA_ENG", "Data Engineer", 115000, 190000),
            ("SALES_MGR", "Sales Manager", 120000, 180000), ("SALES_REP", "Sales Representative", 60000, 110000),
            ("MKT_SPEC", "Marketing Specialist", 65000, 115000), ("ACCOUNT", "Accountant", 70000, 120000)]
    jobs = [(j, t, Decimal(f"{a}.00"), Decimal(f"{b}.00")) for j, t, a, b in jobs]
    # (id, job, dept, manager)
    org = [(100, "CEO", 10, None), (101, "VP", 20, 100), (102, "VP", 40, 100), (103, "VP", 60, 100),
           (104, "ENG_MGR", 20, 101), (105, "ENG_MGR", 30, 101), (106, "SWE", 20, 104), (107, "SWE", 20, 104),
           (108, "SWE", 20, 104), (109, "DBA", 20, 104), (110, "DATA_ENG", 30, 105), (111, "DATA_ENG", 30, 105),
           (112, "DBA", 30, 105), (113, "SWE", 30, 105), (114, "SALES_MGR", 40, 102), (115, "SALES_REP", 40, 114),
           (116, "SALES_REP", 40, 114), (117, "SALES_REP", 40, 114), (118, "SALES_REP", 40, 114),
           (119, "MKT_SPEC", 50, 102), (120, "MKT_SPEC", 50, 119), (121, "ACCOUNT", 60, 103),
           (122, "ACCOUNT", 60, 103), (123, "SWE", 20, 104), (124, "SWE", None, 104),  # 124 not yet assigned
           (125, "DATA_ENG", 30, 105), (126, "SALES_REP", 40, 114), (127, "SWE", 20, 104),
           (128, "MKT_SPEC", 50, 119), (129, "DBA", 20, 101)]
    jr = {j[0]: j for j in jobs}
    emps, hist, used = [], [], set()
    for eid, job, dept, mgr in org:
        while True:
            fn, ln = r.choice(FIRST), r.choice(LAST)
            if (fn, ln) not in used:
                used.add((fn, ln))
                break
        lo, hi = float(jr[job][2]), float(jr[job][3])
        sal = Decimal(f"{round(r.uniform(lo, hi) / 500) * 500:.2f}")
        if eid in (107, 108):  # deliberate salary tie for ranking lessons
            sal = Decimal("150000.00")
        hire = dt.date(2015, 1, 1) + dt.timedelta(days=r.randint(0, 3400))
        if eid == 100:
            hire = dt.date(2014, 3, 3)
        comm = Decimal(f"{r.choice([0.05, 0.10, 0.15]):.2f}") if job in ("SALES_REP", "SALES_MGR") else None
        emps.append((eid, fn, ln, f"{fn.lower()}.{ln.lower()}@corp.example", hire, job, dept, mgr, sal, comm))
        # salary history: hire salary + raises
        s = float(sal) * r.uniform(0.70, 0.85)
        d = hire
        while d < dt.date(2024, 6, 1):
            hist.append((eid, d, Decimal(f"{round(s / 100) * 100:.2f}")))
            d = dt.date(d.year + 1, d.month, min(d.day, 28))
            s = min(s * r.uniform(1.03, 1.09), float(sal))
        hist[-1] = (eid, hist[-1][1], sal)
    parts.append(inserts("locations", ["location_id", "city", "country"], locations))
    parts.append(inserts("departments", ["department_id", "name", "location_id", "budget"], departments))
    parts.append(inserts("jobs", ["job_id", "title", "min_salary", "max_salary"], jobs))
    parts.append(inserts("employees", ["employee_id", "first_name", "last_name", "email", "hire_date", "job_id",
                                       "department_id", "manager_id", "salary", "commission_pct"], emps))
    parts.append(inserts("salary_history", ["employee_id", "effective_date", "salary"], hist))
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# university
# ---------------------------------------------------------------------------
def university() -> str:
    r = random.Random(2024)
    ddl = """-- Dataset: university. Tables: departments, instructors, students, courses,
-- course_prereqs, sections, enrollments, advisors.
CREATE TABLE departments (
  dept_id   SMALLINT UNSIGNED NOT NULL,
  name      VARCHAR(60)  NOT NULL,
  building  VARCHAR(40)  NOT NULL,
  budget    DECIMAL(12,2) NOT NULL,
  PRIMARY KEY (dept_id),
  UNIQUE KEY uq_departments_name (name)
) ENGINE=InnoDB;

CREATE TABLE instructors (
  instructor_id  INT UNSIGNED NOT NULL,
  name           VARCHAR(80)  NOT NULL,
  dept_id        SMALLINT UNSIGNED NOT NULL,
  salary         DECIMAL(10,2) NOT NULL,
  hired_on       DATE NOT NULL,
  PRIMARY KEY (instructor_id),
  CONSTRAINT fk_instructors_dept FOREIGN KEY (dept_id) REFERENCES departments (dept_id)
) ENGINE=InnoDB;

CREATE TABLE students (
  student_id       INT UNSIGNED NOT NULL,
  name             VARCHAR(80)  NOT NULL,
  email            VARCHAR(255) NOT NULL,
  dept_id          SMALLINT UNSIGNED NULL,
  enrollment_year  YEAR NOT NULL,
  total_credits    SMALLINT UNSIGNED NOT NULL DEFAULT 0,
  PRIMARY KEY (student_id),
  UNIQUE KEY uq_students_email (email),
  CONSTRAINT fk_students_dept FOREIGN KEY (dept_id) REFERENCES departments (dept_id)
) ENGINE=InnoDB;

CREATE TABLE courses (
  course_id  VARCHAR(10) NOT NULL,
  title      VARCHAR(100) NOT NULL,
  dept_id    SMALLINT UNSIGNED NOT NULL,
  credits    TINYINT UNSIGNED NOT NULL,
  PRIMARY KEY (course_id),
  CONSTRAINT fk_courses_dept FOREIGN KEY (dept_id) REFERENCES departments (dept_id),
  CONSTRAINT chk_courses_credits CHECK (credits BETWEEN 1 AND 6)
) ENGINE=InnoDB;

CREATE TABLE course_prereqs (
  course_id  VARCHAR(10) NOT NULL,
  prereq_id  VARCHAR(10) NOT NULL,
  PRIMARY KEY (course_id, prereq_id),
  CONSTRAINT fk_prereqs_course FOREIGN KEY (course_id) REFERENCES courses (course_id),
  CONSTRAINT fk_prereqs_prereq FOREIGN KEY (prereq_id) REFERENCES courses (course_id)
) ENGINE=InnoDB;

CREATE TABLE sections (
  section_id     INT UNSIGNED NOT NULL,
  course_id      VARCHAR(10)  NOT NULL,
  term           ENUM('Spring','Summer','Fall') NOT NULL,
  year           YEAR NOT NULL,
  instructor_id  INT UNSIGNED NULL,
  room           VARCHAR(20)  NULL,
  capacity       SMALLINT UNSIGNED NOT NULL,
  PRIMARY KEY (section_id),
  UNIQUE KEY uq_sections_offering (course_id, term, year, room),
  CONSTRAINT fk_sections_course     FOREIGN KEY (course_id)     REFERENCES courses (course_id),
  CONSTRAINT fk_sections_instructor FOREIGN KEY (instructor_id) REFERENCES instructors (instructor_id)
) ENGINE=InnoDB;

CREATE TABLE enrollments (
  student_id  INT UNSIGNED NOT NULL,
  section_id  INT UNSIGNED NOT NULL,
  grade       CHAR(2) NULL,
  PRIMARY KEY (student_id, section_id),
  KEY idx_enrollments_section (section_id),
  CONSTRAINT fk_enrollments_student FOREIGN KEY (student_id) REFERENCES students (student_id),
  CONSTRAINT fk_enrollments_section FOREIGN KEY (section_id) REFERENCES sections (section_id)
) ENGINE=InnoDB;

CREATE TABLE advisors (
  student_id     INT UNSIGNED NOT NULL,
  instructor_id  INT UNSIGNED NOT NULL,
  PRIMARY KEY (student_id),
  CONSTRAINT fk_advisors_student    FOREIGN KEY (student_id)    REFERENCES students (student_id),
  CONSTRAINT fk_advisors_instructor FOREIGN KEY (instructor_id) REFERENCES instructors (instructor_id)
) ENGINE=InnoDB;
"""
    parts = [ddl]
    depts = [(1, "Computer Science", "Duncan Hall", Decimal("1500000.00")),
             (2, "Data Science", "MacQuarrie Hall", Decimal("1100000.00")),
             (3, "Mathematics", "MacQuarrie Hall", Decimal("800000.00")),
             (4, "Business", "Boccardo Hall", Decimal("950000.00")),
             (5, "Physics", "Science Building", Decimal("700000.00"))]
    instr = []
    used = set()
    for i in range(1, 13):
        while True:
            n = f"{r.choice(FIRST)} {r.choice(LAST)}"
            if n not in used:
                used.add(n)
                break
        d = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 1][i - 1]
        instr.append((1000 + i, n, d, Decimal(f"{r.randint(95, 190) * 1000}.00"),
                      dt.date(2005, 1, 1) + dt.timedelta(days=r.randint(0, 6500))))
    courses = [("CS-046", "Introduction to Programming", 1, 4), ("CS-146", "Data Structures and Algorithms", 1, 3),
               ("CS-157A", "Introduction to Database Management Systems", 1, 3),
               ("CS-157B", "Database Management Systems II", 1, 3), ("CS-149", "Operating Systems", 1, 3),
               ("DATA-200", "Computational Programming for Analytics", 2, 3), ("DATA-225", "Database Systems for Analytics", 2, 3),
               ("DATA-226", "Data Warehousing", 2, 3), ("DATA-245", "Machine Learning", 2, 3),
               ("MATH-042", "Discrete Mathematics", 3, 3), ("MATH-161A", "Applied Probability and Statistics", 3, 3),
               ("BUS-170", "Business Analytics", 4, 3), ("BUS-190", "Operations Management", 4, 3),
               ("PHYS-050", "General Physics", 5, 4), ("CS-160", "Software Engineering", 1, 3)]
    prereqs = [("CS-146", "CS-046"), ("CS-146", "MATH-042"), ("CS-157A", "CS-146"), ("CS-157B", "CS-157A"),
               ("CS-149", "CS-146"), ("CS-160", "CS-146"), ("DATA-225", "DATA-200"), ("DATA-226", "DATA-225"),
               ("DATA-245", "DATA-200"), ("DATA-245", "MATH-161A")]
    students = []
    for s in range(1, 41):
        while True:
            fn, ln = r.choice(FIRST), r.choice(LAST)
            n = f"{fn} {ln}"
            if n not in used:
                used.add(n)
                break
        d = None if s in (13, 37) else r.choice([1, 1, 2, 2, 2, 3, 4, 5])
        students.append([20230000 + s, n, f"{fn.lower()}.{ln.lower()}@student.example.edu", d,
                         r.choice([2021, 2022, 2023, 2024]), 0])
    sections = []
    sid = 1
    for year in (2023, 2024):
        for term in ("Spring", "Fall"):
            for c in r.sample(courses, 6):
                dept_instr = [i[0] for i in instr if i[2] == c[2]]
                ins = None if sid == 7 else r.choice(dept_instr)  # one section still unassigned
                room = None if sid == 11 else f"{r.choice(['DH', 'MH', 'BH', 'SCI'])}-{r.randint(100, 450)}"
                sections.append((sid, c[0], term, year, ins, room, r.choice([30, 40, 60, 120])))
                sid += 1
    grades = ["A", "A", "A-", "B+", "B", "B", "B-", "C+", "C", "D", "F"]
    credits_of = {c[0]: c[3] for c in courses}
    enroll, seen = [], set()
    for st in students:
        for sec in r.sample(sections, r.randint(2, 6)):
            if (st[0], sec[1]) in seen:
                continue
            seen.add((st[0], sec[1]))
            g = None if (sec[3] == 2024 and sec[2] == "Fall" and r.random() < 0.6) else r.choice(grades)
            enroll.append((st[0], sec[0], g))
            if g not in (None, "F"):
                st[5] += credits_of[sec[1]]
    students[39][5] = 0
    enroll = [e for e in enroll if e[0] != students[39][0]]  # newest student has no enrollments yet
    advisors = [(st[0], r.choice([i[0] for i in instr if i[2] == st[3]] or [instr[0][0]]))
                for st in students if st[0] % 4 != 0]
    parts.append(inserts("departments", ["dept_id", "name", "building", "budget"], depts))
    parts.append(inserts("instructors", ["instructor_id", "name", "dept_id", "salary", "hired_on"], instr))
    parts.append(inserts("students", ["student_id", "name", "email", "dept_id", "enrollment_year", "total_credits"],
                         [tuple(s) for s in students]))
    parts.append(inserts("courses", ["course_id", "title", "dept_id", "credits"], courses))
    parts.append(inserts("course_prereqs", ["course_id", "prereq_id"], prereqs))
    parts.append(inserts("sections", ["section_id", "course_id", "term", "year", "instructor_id", "room", "capacity"], sections))
    parts.append(inserts("enrollments", ["student_id", "section_id", "grade"], enroll))
    parts.append(inserts("advisors", ["student_id", "instructor_id"], advisors))
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# bank
# ---------------------------------------------------------------------------
def bank() -> str:
    r = random.Random(99)
    ddl = """-- Dataset: bank. Tables: branches, customers, accounts, transactions.
-- accounts.balance always equals the balance_after of the account's latest transaction.
CREATE TABLE branches (
  branch_id  SMALLINT UNSIGNED NOT NULL,
  name       VARCHAR(60) NOT NULL,
  city       VARCHAR(60) NOT NULL,
  PRIMARY KEY (branch_id)
) ENGINE=InnoDB;

CREATE TABLE customers (
  customer_id  INT UNSIGNED NOT NULL,
  full_name    VARCHAR(100) NOT NULL,
  email        VARCHAR(255) NOT NULL,
  date_of_birth DATE NOT NULL,
  kyc_verified BOOLEAN NOT NULL DEFAULT FALSE,
  PRIMARY KEY (customer_id),
  UNIQUE KEY uq_customers_email (email)
) ENGINE=InnoDB;

CREATE TABLE accounts (
  account_id   INT UNSIGNED NOT NULL,
  customer_id  INT UNSIGNED NOT NULL,
  branch_id    SMALLINT UNSIGNED NOT NULL,
  account_type ENUM('checking','savings') NOT NULL,
  balance      DECIMAL(14,2) NOT NULL DEFAULT 0.00,
  status       ENUM('active','frozen','closed') NOT NULL DEFAULT 'active',
  opened_at    DATETIME NOT NULL,
  version      INT UNSIGNED NOT NULL DEFAULT 0,
  PRIMARY KEY (account_id),
  KEY idx_accounts_customer (customer_id),
  CONSTRAINT fk_accounts_customer FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
  CONSTRAINT fk_accounts_branch   FOREIGN KEY (branch_id)   REFERENCES branches (branch_id),
  CONSTRAINT chk_accounts_balance CHECK (balance >= 0)
) ENGINE=InnoDB;

CREATE TABLE transactions (
  txn_id         BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  account_id     INT UNSIGNED NOT NULL,
  txn_type       ENUM('deposit','withdrawal','transfer_in','transfer_out','fee','interest') NOT NULL,
  amount         DECIMAL(14,2) NOT NULL,
  balance_after  DECIMAL(14,2) NOT NULL,
  reference      VARCHAR(40) NULL,
  created_at     DATETIME NOT NULL,
  PRIMARY KEY (txn_id),
  KEY idx_transactions_account_time (account_id, created_at),
  CONSTRAINT fk_transactions_account FOREIGN KEY (account_id) REFERENCES accounts (account_id),
  CONSTRAINT chk_transactions_amount CHECK (amount > 0)
) ENGINE=InnoDB;
"""
    parts = [ddl]
    branches = [(1, "Downtown", "San Jose"), (2, "Mission", "San Francisco"), (3, "Capitol Hill", "Seattle")]
    customers, used = [], set()
    for c in range(1, 16):
        while True:
            n = f"{r.choice(FIRST)} {r.choice(LAST)}"
            if n not in used:
                used.add(n)
                break
        customers.append((c, n, n.lower().replace(" ", ".") + "@mail.example", dt.date(1960, 1, 1) + dt.timedelta(days=r.randint(0, 15000)),
                          c != 9))
    accounts, txns = [], []
    aid, tid = 1, 1
    events = []
    for c in customers:
        for kind in (["checking", "savings"] if c[0] % 3 else ["checking"]):
            opened = dt.datetime(2022, 1, 1, 9) + dt.timedelta(days=r.randint(0, 400), minutes=r.randint(0, 480))
            status = "frozen" if aid == 7 else ("closed" if aid == 12 else "active")
            accounts.append([aid, c[0], r.randint(1, 3), kind, Decimal("0.00"), status, opened, 0])
            events.append((opened, aid, "deposit", money(r.uniform(500, 8000)), "OPENING"))
            aid += 1
    for _ in range(130):
        a = r.choice(accounts)
        if a[5] == "closed":
            continue
        when = a[6] + dt.timedelta(days=r.randint(1, 700), minutes=r.randint(0, 600))
        typ = r.choice(["deposit", "withdrawal", "withdrawal", "fee", "interest"] if a[3] == "checking"
                       else ["deposit", "interest", "interest", "withdrawal"])
        amt = {"deposit": r.uniform(50, 3000), "withdrawal": r.uniform(20, 900), "fee": r.choice([5, 12, 25]),
               "interest": r.uniform(1, 40)}[typ]
        events.append((when, a[0], typ, money(amt), None))
    for i in range(12):  # transfers
        src, dst = r.sample([a for a in accounts if a[5] == "active"], 2)
        when = max(src[6], dst[6]) + dt.timedelta(days=r.randint(10, 600), minutes=r.randint(0, 600))
        amt = money(r.uniform(50, 1500))
        ref = f"TRF-{2024000 + i}"
        events.append((when, src[0], "transfer_out", amt, ref))
        events.append((when, dst[0], "transfer_in", amt, ref))
    events.sort(key=lambda e: (e[0], e[1]))
    bal = {a[0]: Decimal("0.00") for a in accounts}
    for when, acc, typ, amt, ref in events:
        sign = 1 if typ in ("deposit", "transfer_in", "interest") else -1
        if sign < 0 and bal[acc] - amt < 0:
            continue
        bal[acc] += sign * amt
        txns.append((tid, acc, typ, amt, bal[acc], ref, when))
        tid += 1
    # Drop transfer halves whose partner was skipped, keeping the data consistent.
    refs = {}
    for t in txns:
        if t[5] and t[5].startswith("TRF"):
            refs.setdefault(t[5], []).append(t)
    broken = {k for k, v in refs.items() if len(v) != 2}
    if broken:
        txns = [t for t in txns if t[5] not in broken]
        bal = {a[0]: Decimal("0.00") for a in accounts}
        fixed = []
        for i, t in enumerate(sorted(txns, key=lambda t: (t[6], t[1])), start=1):
            sign = 1 if t[2] in ("deposit", "transfer_in", "interest") else -1
            bal[t[1]] += sign * t[3]
            fixed.append((i, t[1], t[2], t[3], bal[t[1]], t[5], t[6]))
        txns = fixed
    for a in accounts:
        a[4] = bal[a[0]]
        if a[5] == "closed":
            a[4] = Decimal("0.00")
    txns = [t for t in txns if not (accounts[t[1] - 1][5] == "closed" and t[2] != "deposit")]
    # closed account: opening deposit then full withdrawal
    closed = accounts[11]
    first = next(t for t in txns if t[1] == closed[0])
    txns = [t for t in txns if t[1] != closed[0]] + [first]
    txns.append((0, closed[0], "withdrawal", first[3], Decimal("0.00"), "ACCOUNT-CLOSE", first[6] + dt.timedelta(days=200)))
    txns.sort(key=lambda t: (t[6], t[1]))
    txns = [(i,) + t[1:] for i, t in enumerate(txns, start=1)]
    parts.append(inserts("branches", ["branch_id", "name", "city"], branches))
    parts.append(inserts("customers", ["customer_id", "full_name", "email", "date_of_birth", "kyc_verified"], customers))
    parts.append(inserts("accounts", ["account_id", "customer_id", "branch_id", "account_type", "balance", "status",
                                      "opened_at", "version"], [tuple(a) for a in accounts]))
    parts.append(inserts("transactions", ["txn_id", "account_id", "txn_type", "amount", "balance_after", "reference",
                                          "created_at"], txns))
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# library
# ---------------------------------------------------------------------------
def library() -> str:
    r = random.Random(11)
    ddl = """-- Dataset: library. Tables: authors, books, book_authors (M:N), members, copies, loans.
CREATE TABLE authors (
  author_id   INT UNSIGNED NOT NULL,
  name        VARCHAR(100) NOT NULL,
  birth_year  SMALLINT NULL,
  country     VARCHAR(60) NULL,
  PRIMARY KEY (author_id)
) ENGINE=InnoDB;

CREATE TABLE books (
  book_id         INT UNSIGNED NOT NULL,
  isbn            CHAR(13) NOT NULL,
  title           VARCHAR(200) NOT NULL,
  genre           VARCHAR(40) NOT NULL,
  published_year  SMALLINT NOT NULL,
  publisher       VARCHAR(80) NOT NULL,
  PRIMARY KEY (book_id),
  UNIQUE KEY uq_books_isbn (isbn)
) ENGINE=InnoDB;

CREATE TABLE book_authors (
  book_id       INT UNSIGNED NOT NULL,
  author_id     INT UNSIGNED NOT NULL,
  author_order  TINYINT UNSIGNED NOT NULL DEFAULT 1,
  PRIMARY KEY (book_id, author_id),
  KEY idx_book_authors_author (author_id),
  CONSTRAINT fk_book_authors_book   FOREIGN KEY (book_id)   REFERENCES books (book_id),
  CONSTRAINT fk_book_authors_author FOREIGN KEY (author_id) REFERENCES authors (author_id)
) ENGINE=InnoDB;

CREATE TABLE members (
  member_id    INT UNSIGNED NOT NULL,
  name         VARCHAR(100) NOT NULL,
  email        VARCHAR(255) NOT NULL,
  joined_on    DATE NOT NULL,
  membership   ENUM('standard','student','premium') NOT NULL,
  PRIMARY KEY (member_id),
  UNIQUE KEY uq_members_email (email)
) ENGINE=InnoDB;

CREATE TABLE copies (
  copy_id      INT UNSIGNED NOT NULL,
  book_id      INT UNSIGNED NOT NULL,
  acquired_on  DATE NOT NULL,
  `condition`  ENUM('new','good','worn','damaged') NOT NULL,
  PRIMARY KEY (copy_id),
  KEY idx_copies_book (book_id),
  CONSTRAINT fk_copies_book FOREIGN KEY (book_id) REFERENCES books (book_id)
) ENGINE=InnoDB;

CREATE TABLE loans (
  loan_id      INT UNSIGNED NOT NULL,
  copy_id      INT UNSIGNED NOT NULL,
  member_id    INT UNSIGNED NOT NULL,
  loaned_at    DATE NOT NULL,
  due_at       DATE NOT NULL,
  returned_at  DATE NULL,
  PRIMARY KEY (loan_id),
  KEY idx_loans_copy (copy_id),
  KEY idx_loans_member (member_id),
  CONSTRAINT fk_loans_copy   FOREIGN KEY (copy_id)   REFERENCES copies (copy_id),
  CONSTRAINT fk_loans_member FOREIGN KEY (member_id) REFERENCES members (member_id),
  CONSTRAINT chk_loans_due CHECK (due_at >= loaned_at)
) ENGINE=InnoDB;
"""
    parts = [ddl]
    authors = [(1, "Alex Petrov", None, "Germany"), (2, "Martin Kleppmann", 1985, "Germany"),
               (3, "Baron Schwartz", None, "United States"), (4, "Peter Zaitsev", None, "Russia"),
               (5, "Vadim Tkachenko", None, "Russia"), (6, "Bill Karwin", None, "United States"),
               (7, "Abraham Silberschatz", 1952, "Israel"), (8, "Henry F. Korth", None, "United States"),
               (9, "S. Sudarshan", None, "India"), (10, "Ramez Elmasri", None, "Egypt"),
               (11, "Shamkant B. Navathe", None, "India"), (12, "C. J. Date", 1941, "United Kingdom"),
               (13, "Ursula K. Le Guin", 1929, "United States"), (14, "Chimamanda Ngozi Adichie", 1977, "Nigeria"),
               (15, "Haruki Murakami", 1949, "Japan"), (16, "Unpublished Author", None, None)]
    books = [(1, "9781492040347", "Database Internals", "Technology", 2019, "O'Reilly"),
             (2, "9781449373320", "Designing Data-Intensive Applications", "Technology", 2017, "O'Reilly"),
             (3, "9781449314286", "High Performance MySQL, 3rd Edition", "Technology", 2012, "O'Reilly"),
             (4, "9781680508987", "SQL Antipatterns, Volume 1", "Technology", 2022, "Pragmatic Bookshelf"),
             (5, "9780078022159", "Database System Concepts", "Textbook", 2019, "McGraw-Hill"),
             (6, "9780133970777", "Fundamentals of Database Systems", "Textbook", 2015, "Pearson"),
             (7, "9781491941171", "SQL and Relational Theory", "Technology", 2015, "O'Reilly"),
             (8, "9780441478125", "The Left Hand of Darkness", "Fiction", 1969, "Ace Books"),
             (9, "9780307455925", "Americanah", "Fiction", 2013, "Knopf"),
             (10, "9780307593313", "1Q84", "Fiction", 2011, "Knopf"),
             (11, "9780553382563", "The Dispossessed", "Fiction", 1974, "Harper & Row"),
             (12, "9781400095100", "Kafka on the Shore", "Fiction", 2005, "Vintage")]
    book_authors = [(1, 1, 1), (2, 2, 1), (3, 3, 1), (3, 4, 2), (3, 5, 3), (4, 6, 1), (5, 7, 1), (5, 8, 2), (5, 9, 3),
                    (6, 10, 1), (6, 11, 2), (7, 12, 1), (8, 13, 1), (9, 14, 1), (10, 15, 1), (11, 13, 1), (12, 15, 1)]
    members, used = [], set()
    for m in range(1, 21):
        while True:
            n = f"{r.choice(FIRST)} {r.choice(LAST)}"
            if n not in used:
                used.add(n)
                break
        members.append((m, n, n.lower().replace(" ", "_") + "@reader.example", dt.date(2021, 1, 1) + dt.timedelta(days=r.randint(0, 1000)),
                        r.choice(["standard", "standard", "student", "premium"])))
    copies = []
    cid = 1
    for b in books:
        for _ in range(r.choice([1, 2, 2, 3]) if b[0] != 7 else 1):
            copies.append((cid, b[0], dt.date(2020, 1, 1) + dt.timedelta(days=r.randint(0, 900)),
                           r.choice(["new", "good", "good", "worn", "damaged"])))
            cid += 1
    loans = []
    lid = 1
    open_copies = set()
    for _ in range(70):
        c = r.choice(copies)
        m = r.choice(members[:18])  # members 19-20 have never borrowed
        start = max(m[3], c[2]) + dt.timedelta(days=r.randint(1, 900))
        if start > dt.date(2024, 12, 15):
            start = dt.date(2024, 12, 1) - dt.timedelta(days=r.randint(0, 200))
        due = start + dt.timedelta(days=21)
        if c[0] in open_copies:
            continue
        roll = r.random()
        if roll < 0.12:
            returned = None
            open_copies.add(c[0])
        elif roll < 0.30:
            returned = due + dt.timedelta(days=r.randint(1, 20))  # late
        else:
            returned = start + dt.timedelta(days=r.randint(2, 21))
        loans.append((lid, c[0], m[0], start, due, returned))
        lid += 1
    parts.append(inserts("authors", ["author_id", "name", "birth_year", "country"], authors))
    parts.append(inserts("books", ["book_id", "isbn", "title", "genre", "published_year", "publisher"], books))
    parts.append(inserts("book_authors", ["book_id", "author_id", "author_order"], book_authors))
    parts.append(inserts("members", ["member_id", "name", "email", "joined_on", "membership"], members))
    parts.append(inserts("copies", ["copy_id", "book_id", "acquired_on", "`condition`"], copies))
    parts.append(inserts("loans", ["loan_id", "copy_id", "member_id", "loaned_at", "due_at", "returned_at"], loans))
    return "\n\n".join(parts) + "\n"


DATASETS = {"shop": shop, "hr": hr, "university": university, "bank": bank, "library": library}

if __name__ == "__main__":
    for name, fn in DATASETS.items():
        (OUT / f"{name}.sql").write_text(fn())
        print("wrote", name)
    (OUT / "empty.sql").write_text("-- Dataset: empty. A blank schema for lessons that build their own tables.\n")
