package Server;
import java.util.concurrent.ThreadFactory;

/**
 * This class implements a thread factory for worker threads.
 * 
 * @author research-aera
 *
 */
public class WorkerThreadFactory implements ThreadFactory {

    String name;
     /**
     * Constructs a WorkerThreadFactory with the given thread name prefix.
     *
     * @param name the name to assign to threads created by this factory
     */
    public WorkerThreadFactory (String name) {
        
        this.name = name;
    }
    /** 
     * Creates a new {@link Thread} for the given {@link Runnable} task.
     *
     * @param runnable the task to be executed by the new thread
     * @return a new Thread configured to run the provided runnable
     */
    //@Override
    public Thread newThread(Runnable runnable) {

        Thread worker = new Thread(runnable);

        return worker;
    }

}



