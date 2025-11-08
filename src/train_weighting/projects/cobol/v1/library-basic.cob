       IDENTIFICATION DIVISION.
       PROGRAM-ID. LIBRARY-BASIC.
       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT BOOK-FILE ASSIGN TO "books.dat"
              ORGANIZATION IS LINE SEQUENTIAL.
       DATA DIVISION.
       FILE SECTION.
       FD  BOOK-FILE.
       01  BOOK-RECORD.
           05 ISBN        PIC X(13).
           05 TITLE       PIC X(40).
           05 AUTHOR      PIC X(30).
           05 AVAILABLE   PIC X(01).
       WORKING-STORAGE SECTION.
       01  WS-OPTION      PIC X VALUE SPACE.
       01  WS-END         PIC X VALUE 'N'.
       01  WS-ISBN        PIC X(13).
       01  WS-TITLE       PIC X(40).
       01  WS-AUTHOR      PIC X(30).
       PROCEDURE DIVISION.
       MAIN-LOOP.
           DISPLAY "LIBRARY CRUD - COBOL V1".
           PERFORM UNTIL WS-END = 'Y'
               DISPLAY "C)reate R)ead U)pdate D)elete L)ist E)xit"
               ACCEPT WS-OPTION
               EVALUATE WS-OPTION
                   WHEN 'C' PERFORM ADD-BOOK
                   WHEN 'R' PERFORM READ-BOOK
                   WHEN 'U' PERFORM UPDATE-BOOK
                   WHEN 'D' PERFORM DELETE-BOOK
                   WHEN 'L' PERFORM LIST-BOOKS
                   WHEN 'E' MOVE 'Y' TO WS-END
                   WHEN OTHER DISPLAY "INVALID OPTION"
               END-EVALUATE
           END-PERFORM
           STOP RUN.

       ADD-BOOK.
           DISPLAY "ISBN: "
           ACCEPT WS-ISBN
           DISPLAY "TITLE: "
           ACCEPT WS-TITLE
           DISPLAY "AUTHOR: "
           ACCEPT WS-AUTHOR
           MOVE WS-ISBN TO ISBN
           MOVE WS-TITLE TO TITLE
           MOVE WS-AUTHOR TO AUTHOR
           MOVE 'Y' TO AVAILABLE
           OPEN EXTEND BOOK-FILE
           WRITE BOOK-RECORD
           CLOSE BOOK-FILE.

       READ-BOOK.
           DISPLAY "ISBN TO SEARCH: "
           ACCEPT WS-ISBN
           OPEN INPUT BOOK-FILE
           PERFORM UNTIL EOF
               READ BOOK-FILE
                   AT END EXIT PERFORM
               END-READ
               IF ISBN = WS-ISBN THEN
                   DISPLAY "FOUND: " TITLE " BY " AUTHOR
                   DISPLAY "AVAILABLE: " AVAILABLE
                   EXIT PERFORM
               END-IF
           END-PERFORM
           CLOSE BOOK-FILE.

       UPDATE-BOOK.
           DISPLAY "ISBN TO UPDATE: "
           ACCEPT WS-ISBN
           DISPLAY "NEW TITLE: "
           ACCEPT WS-TITLE
           OPEN I-O BOOK-FILE
           PERFORM UNTIL EOF
               READ BOOK-FILE
                   AT END EXIT PERFORM
               END-READ
               IF ISBN = WS-ISBN THEN
                   MOVE WS-TITLE TO TITLE
                   REWRITE BOOK-RECORD
                   DISPLAY "UPDATED"
                   EXIT PERFORM
               END-IF
           END-PERFORM
           CLOSE BOOK-FILE.

       DELETE-BOOK.
           DISPLAY "ISBN TO DELETE: "
           ACCEPT WS-ISBN
           OPEN I-O BOOK-FILE
           PERFORM UNTIL EOF
               READ BOOK-FILE
                   AT END EXIT PERFORM
               END-READ
               IF ISBN = WS-ISBN THEN
                   DELETE BOOK-FILE RECORD
                   DISPLAY "DELETED"
                   EXIT PERFORM
               END-IF
           END-PERFORM
           CLOSE BOOK-FILE.

       LIST-BOOKS.
           OPEN INPUT BOOK-FILE
           PERFORM UNTIL EOF
               READ BOOK-FILE
                   AT END EXIT PERFORM
               END-READ
               DISPLAY ISBN " - " TITLE " (" AUTHOR ")"
           END-PERFORM
           CLOSE BOOK-FILE.
