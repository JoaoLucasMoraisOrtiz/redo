object dmLibrary: TdmLibrary
  OldCreateOrder = False
  Height = 240
  Width = 320
  object Connection: TADOConnection
    LoginPrompt = False
    Left = 24
    Top = 24
  end
  object TableBooks: TADOTable
    Connection = Connection
    CursorType = ctStatic
    TableName = 'LIB_BOOK'
    Left = 120
    Top = 24
  end
end
