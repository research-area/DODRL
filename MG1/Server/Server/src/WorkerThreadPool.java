package Server;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.concurrent.*;

import net.openhft.affinity.*;
import org.apache.commons.collections4.bag.SynchronizedSortedBag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.plaf.synth.SynthSpinnerUI;

import static Server.Job.logger;

//import net.openhft.affinity.AffinityStrategies;
//import net.openhft.affinity.AffinityThreadFactory;

/**
 * This class implements a Worker Thread Pool.
 *
 * @author research-aera
 *
 */
public class WorkerThreadPool {

	public static ArrayBlockingQueue<Runnable> queue = new ArrayBlockingQueue<Runnable>(1000);

	public static ThreadFactory workerThreadFactory;

	public static CustomWorkerThreadPool executorPool;

	/**
	 * Initializes and starts the worker thread pool with a single core/max thread,
	 * a rejection handler, and CPU affinity pinning to different cores.
	 */
	public void runWorkerThreadPool() {
		RejectedExecutionHandlerImpl rejectionHandler = new RejectedExecutionHandlerImpl();

		workerThreadFactory = new AffinityThreadFactory("Worker",AffinityStrategies.DIFFERENT_CORE );
		executorPool = new CustomWorkerThreadPool(1, 1, 10, TimeUnit.SECONDS, queue, workerThreadFactory, rejectionHandler);

	}

}




