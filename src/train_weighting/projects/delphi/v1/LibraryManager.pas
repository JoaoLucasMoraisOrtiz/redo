unit LibraryManager;

interface

uses
  Classes, SysUtils;

type
  TBook = class
  private
    FIsbn: string;
    FTitle: string;
    FAuthor: string;
    FAvailable: Boolean;
  public
    constructor Create(const AIsbn, ATitle, AAuthor: string);
    property Isbn: string read FIsbn;
    property Title: string read FTitle write FTitle;
    property Author: string read FAuthor write FAuthor;
    property Available: Boolean read FAvailable write FAvailable;
  end;

  TLibraryManager = class
  private
    FBooks: TList;
    function FindBook(const AIsbn: string): TBook;
  public
    constructor Create;
    destructor Destroy; override;
    procedure AddBook(const AIsbn, ATitle, AAuthor: string);
    function GetBook(const AIsbn: string): TBook;
    procedure UpdateTitle(const AIsbn, ATitle: string);
    procedure RemoveBook(const AIsbn: string);
    procedure ListBooks(AStrings: TStrings);
  end;

implementation

{ TBook }

constructor TBook.Create(const AIsbn, ATitle, AAuthor: string);
begin
  inherited Create;
  FIsbn := AIsbn;
  FTitle := ATitle;
  FAuthor := AAuthor;
  FAvailable := True;
end;

{ TLibraryManager }

constructor TLibraryManager.Create;
begin
  inherited Create;
  FBooks := TList.Create;
end;

destructor TLibraryManager.Destroy;
var
  I: Integer;
begin
  for I := 0 to FBooks.Count - 1 do
    TObject(FBooks[I]).Free;
  FBooks.Free;
  inherited Destroy;
end;

function TLibraryManager.FindBook(const AIsbn: string): TBook;
var
  I: Integer;
  Book: TBook;
begin
  Result := nil;
  for I := 0 to FBooks.Count - 1 do
  begin
    Book := TBook(FBooks[I]);
    if SameText(Book.Isbn, AIsbn) then
    begin
      Result := Book;
      Exit;
    end;
  end;
end;

procedure TLibraryManager.AddBook(const AIsbn, ATitle, AAuthor: string);
begin
  FBooks.Add(TBook.Create(AIsbn, ATitle, AAuthor));
end;

function TLibraryManager.GetBook(const AIsbn: string): TBook;
begin
  Result := FindBook(AIsbn);
end;

procedure TLibraryManager.UpdateTitle(const AIsbn, ATitle: string);
var
  Book: TBook;
begin
  Book := FindBook(AIsbn);
  if Assigned(Book) then
    Book.Title := ATitle;
end;

procedure TLibraryManager.RemoveBook(const AIsbn: string);
var
  Book: TBook;
begin
  Book := FindBook(AIsbn);
  if Assigned(Book) then
  begin
    FBooks.Remove(Book);
    Book.Free;
  end;
end;

procedure TLibraryManager.ListBooks(AStrings: TStrings);
var
  I: Integer;
  Book: TBook;
begin
  AStrings.Clear;
  for I := 0 to FBooks.Count - 1 do
  begin
    Book := TBook(FBooks[I]);
    AStrings.Add(Format('%s - %s (%s)', [Book.Isbn, Book.Title, Book.Author]));
  end;
end;

end.
