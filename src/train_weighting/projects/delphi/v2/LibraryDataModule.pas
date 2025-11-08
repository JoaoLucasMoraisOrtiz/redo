unit LibraryDataModule;

interface

uses
  SysUtils, Classes, DB, ADODB;

type
  TdmLibrary = class(TDataModule)
    Connection: TADOConnection;
    TableBooks: TADOTable;
  public
    procedure SetupConnection;
    procedure CreateBook(const AIsbn, ATitle, AAuthor: string);
    function FindBook(const AIsbn: string): Boolean;
    procedure UpdateAvailability(const AIsbn: string; AAvailable: Boolean);
    procedure DeleteBook(const AIsbn: string);
  end;

var
  dmLibrary: TdmLibrary;

implementation

{$R *.dfm}

procedure TdmLibrary.SetupConnection;
begin
  Connection.ConnectionString := 'Provider=Microsoft.Jet.OLEDB.4.0;Data Source=library.mdb;Persist Security Info=False;';
  Connection.LoginPrompt := False;
  Connection.Open;
  TableBooks.Connection := Connection;
  TableBooks.TableName := 'LIB_BOOK';
  TableBooks.Open;
end;

procedure TdmLibrary.CreateBook(const AIsbn, ATitle, AAuthor: string);
begin
  TableBooks.Insert;
  TableBooks.FieldByName('ISBN').AsString := AIsbn;
  TableBooks.FieldByName('TITLE').AsString := ATitle;
  TableBooks.FieldByName('AUTHOR').AsString := AAuthor;
  TableBooks.FieldByName('AVAILABLE').AsBoolean := True;
  TableBooks.Post;
end;

function TdmLibrary.FindBook(const AIsbn: string): Boolean;
begin
  Result := TableBooks.Locate('ISBN', AIsbn, []);
end;

procedure TdmLibrary.UpdateAvailability(const AIsbn: string; AAvailable: Boolean);
begin
  if FindBook(AIsbn) then
  begin
    TableBooks.Edit;
    TableBooks.FieldByName('AVAILABLE').AsBoolean := AAvailable;
    TableBooks.Post;
  end;
end;

procedure TdmLibrary.DeleteBook(const AIsbn: string);
begin
  if FindBook(AIsbn) then
    TableBooks.Delete;
end;

end.
