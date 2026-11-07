-- Dataset: library. Tables: authors, books, book_authors (M:N), members, copies, loans.
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


INSERT INTO authors (author_id, name, birth_year, country) VALUES
  (1, 'Alex Petrov', NULL, 'Germany'),
  (2, 'Martin Kleppmann', 1985, 'Germany'),
  (3, 'Baron Schwartz', NULL, 'United States'),
  (4, 'Peter Zaitsev', NULL, 'Russia'),
  (5, 'Vadim Tkachenko', NULL, 'Russia'),
  (6, 'Bill Karwin', NULL, 'United States'),
  (7, 'Abraham Silberschatz', 1952, 'Israel'),
  (8, 'Henry F. Korth', NULL, 'United States'),
  (9, 'S. Sudarshan', NULL, 'India'),
  (10, 'Ramez Elmasri', NULL, 'Egypt'),
  (11, 'Shamkant B. Navathe', NULL, 'India'),
  (12, 'C. J. Date', 1941, 'United Kingdom'),
  (13, 'Ursula K. Le Guin', 1929, 'United States'),
  (14, 'Chimamanda Ngozi Adichie', 1977, 'Nigeria'),
  (15, 'Haruki Murakami', 1949, 'Japan'),
  (16, 'Unpublished Author', NULL, NULL);

INSERT INTO books (book_id, isbn, title, genre, published_year, publisher) VALUES
  (1, '9781492040347', 'Database Internals', 'Technology', 2019, 'O''Reilly'),
  (2, '9781449373320', 'Designing Data-Intensive Applications', 'Technology', 2017, 'O''Reilly'),
  (3, '9781449314286', 'High Performance MySQL, 3rd Edition', 'Technology', 2012, 'O''Reilly'),
  (4, '9781680508987', 'SQL Antipatterns, Volume 1', 'Technology', 2022, 'Pragmatic Bookshelf'),
  (5, '9780078022159', 'Database System Concepts', 'Textbook', 2019, 'McGraw-Hill'),
  (6, '9780133970777', 'Fundamentals of Database Systems', 'Textbook', 2015, 'Pearson'),
  (7, '9781491941171', 'SQL and Relational Theory', 'Technology', 2015, 'O''Reilly'),
  (8, '9780441478125', 'The Left Hand of Darkness', 'Fiction', 1969, 'Ace Books'),
  (9, '9780307455925', 'Americanah', 'Fiction', 2013, 'Knopf'),
  (10, '9780307593313', '1Q84', 'Fiction', 2011, 'Knopf'),
  (11, '9780553382563', 'The Dispossessed', 'Fiction', 1974, 'Harper & Row'),
  (12, '9781400095100', 'Kafka on the Shore', 'Fiction', 2005, 'Vintage');

INSERT INTO book_authors (book_id, author_id, author_order) VALUES
  (1, 1, 1),
  (2, 2, 1),
  (3, 3, 1),
  (3, 4, 2),
  (3, 5, 3),
  (4, 6, 1),
  (5, 7, 1),
  (5, 8, 2),
  (5, 9, 3),
  (6, 10, 1),
  (6, 11, 2),
  (7, 12, 1),
  (8, 13, 1),
  (9, 14, 1),
  (10, 15, 1),
  (11, 13, 1),
  (12, 15, 1);

INSERT INTO members (member_id, name, email, joined_on, membership) VALUES
  (1, 'Marcus Moradi', 'marcus_moradi@reader.example', '2022-07-28', 'premium'),
  (2, 'Marcus Haddad', 'marcus_haddad@reader.example', '2023-05-26', 'standard'),
  (3, 'Mei Tanaka', 'mei_tanaka@reader.example', '2022-06-09', 'premium'),
  (4, 'Anika Tanaka', 'anika_tanaka@reader.example', '2021-07-10', 'standard'),
  (5, 'Marcus Khan', 'marcus_khan@reader.example', '2021-05-26', 'standard'),
  (6, 'Kwame Tanaka', 'kwame_tanaka@reader.example', '2023-07-01', 'standard'),
  (7, 'Victor Miller', 'victor_miller@reader.example', '2023-09-15', 'premium'),
  (8, 'Anika Mensah', 'anika_mensah@reader.example', '2021-06-11', 'standard'),
  (9, 'Ingrid Garcia', 'ingrid_garcia@reader.example', '2021-03-02', 'standard'),
  (10, 'Ethan Clark', 'ethan_clark@reader.example', '2021-09-05', 'standard'),
  (11, 'Nina Silva', 'nina_silva@reader.example', '2022-03-28', 'standard'),
  (12, 'Ingrid Okafor', 'ingrid_okafor@reader.example', '2022-10-18', 'student'),
  (13, 'Chloe Sharma', 'chloe_sharma@reader.example', '2022-11-10', 'standard'),
  (14, 'Nina Mensah', 'nina_mensah@reader.example', '2021-10-12', 'premium'),
  (15, 'Lucia Novak', 'lucia_novak@reader.example', '2023-05-11', 'standard'),
  (16, 'Omar Silva', 'omar_silva@reader.example', '2023-02-16', 'standard'),
  (17, 'Tomas Khan', 'tomas_khan@reader.example', '2021-01-31', 'standard'),
  (18, 'Hiro Wilson', 'hiro_wilson@reader.example', '2021-04-21', 'premium'),
  (19, 'Chen Moradi', 'chen_moradi@reader.example', '2021-10-25', 'premium'),
  (20, 'Liam Sharma', 'liam_sharma@reader.example', '2023-05-18', 'standard');

INSERT INTO copies (copy_id, book_id, acquired_on, `condition`) VALUES
  (1, 1, '2020-08-02', 'new'),
  (2, 1, '2021-04-26', 'worn'),
  (3, 2, '2021-03-05', 'new'),
  (4, 2, '2021-08-02', 'good'),
  (5, 2, '2022-03-08', 'good'),
  (6, 3, '2020-03-30', 'good'),
  (7, 3, '2020-12-06', 'new'),
  (8, 4, '2022-02-15', 'new'),
  (9, 4, '2020-05-17', 'good'),
  (10, 4, '2021-12-24', 'new'),
  (11, 5, '2020-03-02', 'worn'),
  (12, 6, '2020-06-30', 'damaged'),
  (13, 6, '2020-07-11', 'worn'),
  (14, 6, '2021-06-05', 'good'),
  (15, 7, '2022-01-19', 'good'),
  (16, 8, '2021-10-21', 'worn'),
  (17, 8, '2020-04-29', 'worn'),
  (18, 8, '2021-03-06', 'good'),
  (19, 9, '2020-10-03', 'damaged'),
  (20, 10, '2020-01-21', 'good'),
  (21, 10, '2020-07-10', 'worn'),
  (22, 11, '2020-02-13', 'good'),
  (23, 12, '2021-03-28', 'good'),
  (24, 12, '2020-01-10', 'damaged');

INSERT INTO loans (loan_id, copy_id, member_id, loaned_at, due_at, returned_at) VALUES
  (1, 11, 10, '2022-10-06', '2022-10-27', NULL),
  (2, 3, 7, '2024-06-22', '2024-07-13', '2024-08-02'),
  (3, 12, 12, '2024-07-17', '2024-08-07', '2024-08-06'),
  (4, 16, 5, '2024-03-27', '2024-04-17', '2024-04-02'),
  (5, 10, 8, '2024-04-10', '2024-05-01', '2024-04-18'),
  (6, 6, 18, '2021-11-09', '2021-11-30', '2021-11-23'),
  (7, 16, 3, '2023-08-15', '2023-09-05', NULL),
  (8, 4, 2, '2024-11-01', '2024-11-22', '2024-11-10'),
  (9, 24, 13, '2023-08-01', '2023-08-22', '2023-08-22'),
  (10, 6, 3, '2022-10-17', '2022-11-07', '2022-11-25'),
  (11, 21, 3, '2023-03-23', '2023-04-13', '2023-04-20'),
  (12, 24, 1, '2022-10-07', '2022-10-28', '2022-11-12'),
  (13, 8, 2, '2023-07-13', '2023-08-03', '2023-08-15'),
  (14, 17, 5, '2021-08-29', '2021-09-19', '2021-09-14'),
  (15, 5, 2, '2023-06-14', '2023-07-05', '2023-06-27'),
  (16, 23, 10, '2021-10-10', '2021-10-31', NULL),
  (17, 21, 3, '2023-10-16', '2023-11-06', NULL),
  (18, 10, 11, '2022-08-15', '2022-09-05', '2022-08-19'),
  (19, 15, 18, '2023-01-31', '2023-02-21', '2023-02-06'),
  (20, 22, 16, '2023-05-07', '2023-05-28', '2023-05-22'),
  (21, 1, 16, '2024-09-25', '2024-10-16', NULL),
  (22, 22, 13, '2023-12-04', '2023-12-25', '2023-12-25'),
  (23, 3, 3, '2022-09-10', '2022-10-01', '2022-09-20'),
  (24, 14, 11, '2023-04-30', '2023-05-21', '2023-05-20'),
  (25, 15, 15, '2024-08-27', '2024-09-17', '2024-08-31'),
  (26, 17, 17, '2021-03-03', '2021-03-24', '2021-03-07'),
  (27, 20, 16, '2023-11-05', '2023-11-26', '2023-11-18'),
  (28, 10, 5, '2023-11-19', '2023-12-10', '2023-12-07'),
  (29, 6, 11, '2024-02-02', '2024-02-23', '2024-02-19'),
  (30, 8, 11, '2023-05-17', '2023-06-07', '2023-05-25'),
  (31, 7, 7, '2024-10-13', '2024-11-03', '2024-11-14'),
  (32, 7, 5, '2021-10-11', '2021-11-01', '2021-10-14'),
  (33, 6, 4, '2022-10-15', '2022-11-05', '2022-10-23'),
  (34, 14, 13, '2024-08-12', '2024-09-02', '2024-08-24'),
  (35, 3, 2, '2024-03-06', '2024-03-27', '2024-03-09'),
  (36, 22, 9, '2022-10-08', '2022-10-29', '2022-10-28'),
  (37, 15, 7, '2023-10-11', '2023-11-01', '2023-10-21'),
  (38, 8, 5, '2024-05-11', '2024-06-01', NULL),
  (39, 4, 18, '2023-06-04', '2023-06-25', '2023-06-17'),
  (40, 3, 7, '2024-04-07', '2024-04-28', '2024-04-17'),
  (41, 6, 1, '2024-09-09', '2024-09-30', '2024-09-12'),
  (42, 6, 8, '2022-03-17', '2022-04-07', '2022-04-05'),
  (43, 20, 6, '2024-08-07', '2024-08-28', '2024-08-16'),
  (44, 3, 14, '2023-10-22', '2023-11-12', '2023-11-07'),
  (45, 15, 7, '2024-11-30', '2024-12-21', '2024-12-20'),
  (46, 24, 4, '2022-02-14', '2022-03-07', '2022-03-20'),
  (47, 3, 10, '2023-03-09', '2023-03-30', '2023-03-21'),
  (48, 9, 1, '2023-07-20', '2023-08-10', '2023-07-23'),
  (49, 15, 11, '2023-10-13', '2023-11-03', '2023-10-23'),
  (50, 3, 14, '2024-01-10', '2024-01-31', NULL),
  (51, 18, 11, '2024-03-01', '2024-03-22', '2024-03-07'),
  (52, 24, 17, '2023-06-11', '2023-07-02', '2023-06-27');
