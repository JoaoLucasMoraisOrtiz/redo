Legacy COBOL program using embedded SQL for IBM DB2.
Requires DB2 precompiler (DSNHPC) and catalog table LIB_BOOK with columns:
  ISBN CHAR(13) PRIMARY KEY
  TITLE VARCHAR(50)
  AUTHOR VARCHAR(30)
  AVAILABLE CHAR(1)
Compile with Enterprise COBOL 4.x or compatible, enabling embedded SQL.
