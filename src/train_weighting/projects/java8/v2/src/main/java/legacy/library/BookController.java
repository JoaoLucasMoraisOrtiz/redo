package legacy.library;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/books")
public class BookController {
    private final BookRepository repository;

    public BookController(BookRepository repository) {
        this.repository = repository;
    }

    @PostMapping
    public Book create(@RequestBody Book book) {
        return repository.save(book);
    }

    @GetMapping("/{isbn}")
    public ResponseEntity<Book> read(@PathVariable String isbn) {
        Book book = repository.findByIsbn(isbn);
        return book != null ? ResponseEntity.ok(book) : ResponseEntity.notFound().build();
    }

    @GetMapping
    public List<Book> list() {
        return repository.findByAvailable(true);
    }

    @PutMapping("/{isbn}/checkout")
    public ResponseEntity<Book> checkout(@PathVariable String isbn) {
        Book book = repository.findByIsbn(isbn);
        if (book == null) {
            return ResponseEntity.notFound().build();
        }
        book.setAvailable(false);
        return ResponseEntity.ok(repository.save(book));
    }

    @DeleteMapping("/{isbn}")
    public ResponseEntity<Void> delete(@PathVariable String isbn) {
        Book book = repository.findByIsbn(isbn);
        if (book == null) {
            return ResponseEntity.notFound().build();
        }
        repository.delete(book);
        return ResponseEntity.noContent().build();
    }
}
