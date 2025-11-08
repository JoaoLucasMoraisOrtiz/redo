import java.io.Serializable;
import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;

/**
 * Hibernate 3 / JPA 1 entity for legacy Java 6 application.
 */
@Entity
@Table(name = "LIB_BOOK")
public class LibraryEntity implements Serializable {
    private static final long serialVersionUID = 1L;

    @Id
    @Column(name = "ISBN", nullable = false, length = 20)
    private String isbn;

    @Column(name = "TITLE", nullable = false, length = 150)
    private String title;

    @Column(name = "AUTHOR", nullable = false, length = 100)
    private String author;

    public LibraryEntity() {
    }

    public LibraryEntity(String isbn, String title, String author) {
        this.isbn = isbn;
        this.title = title;
        this.author = author;
    }

    public String getIsbn() {
        return isbn;
    }

    public void setIsbn(String isbn) {
        this.isbn = isbn;
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

    public void setAuthor(String author) {
        this.author = author;
    }
}
