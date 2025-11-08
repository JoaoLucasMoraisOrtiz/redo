import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Basic CRUD for a legacy Java 6 console application managing books in a library.
 */
public class LibraryManagement {
    private final List<Book> catalog = new ArrayList<Book>();

    public void createBook(String isbn, String title, String author) {
        Book book = new Book(isbn, title, author);
        catalog.add(book);
        System.out.println("Created book: " + book);
    }

    public Book readBook(String isbn) {
        for (Book book : catalog) {
            if (book.getIsbn().equals(isbn)) {
                return book;
            }
        }
        return null;
    }

    public void updateBookTitle(String isbn, String newTitle) {
        Book book = readBook(isbn);
        if (book != null) {
            book.setTitle(newTitle);
            System.out.println("Updated book: " + book);
        } else {
            System.out.println("Book not found: " + isbn);
        }
    }

    public void deleteBook(String isbn) {
        for (Iterator<Book> iterator = catalog.iterator(); iterator.hasNext();) {
            Book book = iterator.next();
            if (book.getIsbn().equals(isbn)) {
                iterator.remove();
                System.out.println("Deleted book: " + isbn);
                return;
            }
        }
        System.out.println("Book not found: " + isbn);
    }

    public void listBooks() {
        System.out.println("Catalog:");
        for (Book book : catalog) {
            System.out.println(" - " + book);
        }
    }

    public static void main(String[] args) {
        LibraryManagement manager = new LibraryManagement();
        manager.createBook("978-0-00", "Legacy Java 6", "Sun Microsystems");
        manager.createBook("978-0-01", "Patterns", "GoF");
        manager.listBooks();
        manager.updateBookTitle("978-0-01", "Design Patterns");
        manager.listBooks();
        manager.deleteBook("978-0-00");
        manager.listBooks();
    }
}

class Book {
    private String isbn;
    private String title;
    private String author;

    public Book(String isbn, String title, String author) {
        this.isbn = isbn;
        this.title = title;
        this.author = author;
    }

    public String getIsbn() {
        return isbn;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getAuthor() {
        return author;
    }

    public String toString() {
        return isbn + " - " + title + " (" + author + ")";
    }
}
