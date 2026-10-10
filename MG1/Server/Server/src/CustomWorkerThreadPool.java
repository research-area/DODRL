package Server;

/**
 * CustomWorkerThreadPool extends ThreadPoolExecutor to hook into the lifecycle
 * of each task executed by the M/G/1 server simulation.
 *
 * <p>It overrides {@code afterExecute()} to capture the system state (queue length)
 * immediately after a job departs, updating the shared state histogram stored in
 * {@link Server#hmap} and tracking the highest state ever observed in
 * {@link Server#highest_state}. These statistics are used to compute the
 * steady-state probabilities of the queuing system.</p>
 *
 * <p>It also overrides {@code beforeExecute()} as a hook point for any logic that
 * needs to run just before a worker thread picks up a task (currently delegates
 * to the parent implementation).</p>
 *
 * @author research-aera
 */

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.RejectedExecutionHandler;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;



public class CustomWorkerThreadPool extends ThreadPoolExecutor {


	public CustomWorkerThreadPool(int corePoolSize, int maximumPoolSize, long keepAliveTime, TimeUnit unit, BlockingQueue<Runnable> workQueue,
			ThreadFactory threadFactory, RejectedExecutionHandler handler) {
		super(corePoolSize, maximumPoolSize, keepAliveTime, unit, workQueue, threadFactory, handler);

	}
/**
 * Called by the thread pool immediately after a task finishes execution
 * (i.e., a job has departed the system).
 *
 * If the task threw an exception, the error is logged to stdout.
 * Otherwise, the method:
 * 
 *   Reads the current queue length as the observable system state (number
 *       of jobs still waiting or in service).
 *   Updates {@link Server#highest_state} if this is a new maximum.
 *   Increments the counter for this state in the histogram map
 *       {@link Server#hmap}, provided the state is within the supported
 *       range [0, 1000]. States above 1000 are treated as overflow and
 *       reported to stdout.
 *
 * @param r the {@link Runnable} that has completed
 * @param t the exception that caused termination, or {@code null} if
 *          execution completed normally
 */
	@Override
	protected void afterExecute(Runnable r, Throwable t) {
		// Invoke the parent's afterExecute() to preserve standard pool bookkeeping.
		super.afterExecute(r, t);
		if (t != null) {
			// A runtime exception escaped the task — log it for diagnostics.
			System.out.println("Task encountered the following exception(CustomWorkerThread): " + t.getMessage());
		} else {

			// Perform afterExecute() logic
			int state = WorkerThreadPool.queue.size();
			// Record's the highest state reached.
			if (state >= Server.highest_state) {
				Server.highest_state = state;
			}
			if(state<=1000){
				// Increment the frequency counter for this state in the histogram.
				// The lambda (k, v) -> v + 1 atomically adds 1 to the existing value.
				Server.hmap.compute(state, (k, v) -> v + 1);
			}else
				// State exceeds the pre-allocated histogram range — signal overflow.
				System.out.println("OverFlow");
			
		}
	}
	
/**
 * Called by the thread pool just before a worker thread executes a task.
 * Currently delegates entirely to the parent implementation.
 * Can be extended to record pre-execution timestamps or inject
 * per-task context when needed.
 *
 * @param t the thread that will run task {@code r}
 * @param r the {@link Runnable} about to be executed
 */
	protected void beforeExecute(Thread t,Runnable r) {
		super.beforeExecute(t,r);
	}

}


