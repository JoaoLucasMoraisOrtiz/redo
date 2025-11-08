import java.util.List;

import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.Transaction;
import org.hibernate.cfg.Configuration;

/**
 * Legacy DAO using Hibernate 3 for CRUD operations.
 */
public class LibraryDao {
    private static final SessionFactory SESSION_FACTORY;

    static {
        Configuration configuration = new Configuration().configure("hibernate.cfg.xml");
        SESSION_FACTORY = configuration.buildSessionFactory();
    }

    public void create(LibraryEntity entity) {
        Session session = SESSION_FACTORY.openSession();
        Transaction tx = session.beginTransaction();
        session.save(entity);
        tx.commit();
        session.close();
    }

    public LibraryEntity read(String isbn) {
        Session session = SESSION_FACTORY.openSession();
        LibraryEntity entity = (LibraryEntity) session.get(LibraryEntity.class, isbn);
        session.close();
        return entity;
    }

    @SuppressWarnings("unchecked")
    public List<LibraryEntity> list() {
        Session session = SESSION_FACTORY.openSession();
        List<LibraryEntity> result = session.createQuery("from LibraryEntity").list();
        session.close();
        return result;
    }

    public void updateTitle(String isbn, String newTitle) {
        Session session = SESSION_FACTORY.openSession();
        Transaction tx = session.beginTransaction();
        LibraryEntity entity = (LibraryEntity) session.get(LibraryEntity.class, isbn);
        if (entity != null) {
            entity.setTitle(newTitle);
            session.update(entity);
        }
        tx.commit();
        session.close();
    }

    public void delete(String isbn) {
        Session session = SESSION_FACTORY.openSession();
        Transaction tx = session.beginTransaction();
        LibraryEntity entity = (LibraryEntity) session.get(LibraryEntity.class, isbn);
        if (entity != null) {
            session.delete(entity);
        }
        tx.commit();
        session.close();
    }
}
