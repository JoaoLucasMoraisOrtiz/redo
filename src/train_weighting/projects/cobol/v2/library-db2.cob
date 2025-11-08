       IDENTIFICATION DIVISION.
       PROGRAM-ID. LIBRARY-DB2.
       ENVIRONMENT DIVISION.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       EXEC SQL INCLUDE SQLCA END-EXEC.
       01  WS-ISBN        PIC X(13).
       01  WS-TITLE       PIC X(50).
       01  WS-AUTHOR      PIC X(30).
       01  WS-AVAILABLE   PIC X(01).
       PROCEDURE DIVISION.
           DISPLAY "COBOL + DB2 CRUD V2".
           PERFORM DB-CONNECT.
           PERFORM SAMPLE-OPERATIONS.
           PERFORM DB-DISCONNECT.
           STOP RUN.

       DB-CONNECT.
           EXEC SQL
               CONNECT TO LEGACYDB
           END-EXEC.
           IF SQLCODE NOT = 0 THEN
               DISPLAY "CONNECT FAILED" SQLCODE
               STOP RUN
           END-IF.

       SAMPLE-OPERATIONS.
           MOVE '978-0-2000000' TO WS-ISBN.
           MOVE 'Mainframe Patterns' TO WS-TITLE.
           MOVE 'IBM Press' TO WS-AUTHOR.
           MOVE 'Y' TO WS-AVAILABLE.
           EXEC SQL
               INSERT INTO LIB_BOOK (ISBN, TITLE, AUTHOR, AVAILABLE)
               VALUES (:WS-ISBN, :WS-TITLE, :WS-AUTHOR, :WS-AVAILABLE)
           END-EXEC.
           DISPLAY "Inserted " WS-ISBN.

           EXEC SQL
               SELECT TITLE, AUTHOR, AVAILABLE
               INTO :WS-TITLE, :WS-AUTHOR, :WS-AVAILABLE
               FROM LIB_BOOK WHERE ISBN = :WS-ISBN
           END-EXEC.
           IF SQLCODE = 0 THEN
               DISPLAY "Read " WS-TITLE
           END-IF.

           MOVE 'N' TO WS-AVAILABLE.
           EXEC SQL
               UPDATE LIB_BOOK
               SET AVAILABLE = :WS-AVAILABLE
               WHERE ISBN = :WS-ISBN
           END-EXEC.
           DISPLAY "Updated availability".

           EXEC SQL
               DELETE FROM LIB_BOOK WHERE ISBN = :WS-ISBN
           END-EXEC.
           DISPLAY "Deleted " WS-ISBN.

       DB-DISCONNECT.
           EXEC SQL COMMIT END-EXEC.
           EXEC SQL DISCONNECT CURRENT END-EXEC.
