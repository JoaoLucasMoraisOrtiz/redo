import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class LibraryManager {
    private final List<Book> catalog = new ArrayList<Book>();

    public void create(Book book) {
        catalog.add(book);
    }

    public Optional<Book> read(String isbn) {
        return catalog.stream().filter(book -> book.getIsbn().equals(isbn)).findFirst();
    }

    public void updateAvailability(String isbn, boolean available) {
        read(isbn).ifPresent(book -> {
            book.setAvailable(available);
            book.setLastUpdated(LocalDate.now());
        });
    }

    public void delete(String isbn) {
        catalog.removeIf(book -> book.getIsbn().equals(isbn));
    }

    public List<Book> listAvailable() {
        return catalog.stream().filter(Book::isAvailable).collect(Collectors.toList());
    }

    public static void main(String[] args) {
        LibraryManager manager = new LibraryManager();
        manager.create(new Book("978-0-100", "Java 8 in Action", "Raoul-Gabriel", true));
        manager.create(new Book("978-0-101", "Refactoring", "Martin Fowler", true));
        System.out.println("Available books: " + manager.listAvailable().size());
        manager.updateAvailability("978-0-101", false);
        System.out.println("After checkout: " + manager.listAvailable().size());
        manager.delete("978-0-100");
        System.out.println("Remaining: " + manager.catalog.size());
    }
}

class Book {
    private final String isbn;
    private final String title;
    private final String author;
    private boolean available;
    private LocalDate lastUpdated;

    public Book(String isbn, String title, String author, boolean available) {
        this.isbn = isbn;
        this.title = title;
        this.author = author;
        this.available = available;
        this.lastUpdated = LocalDate.now();
    }

    public String getIsbn() {
        return isbn;
    }

    public String getTitle() {
        return title;
    }

    public String getAuthor() {
        return author;
    }

    public boolean isAvailable() {
        return available;
    }

    public void setAvailable(boolean available) {
        this.available = available;
    }

    public LocalDate getLastUpdated() {
        return lastUpdated;
    }

    public void setLastUpdated(LocalDate lastUpdated) {
        this.lastUpdated = lastUpdated;
    }

    public String toString() {
        return isbn + " - " + title + " (" + author + ")";
    }
}
