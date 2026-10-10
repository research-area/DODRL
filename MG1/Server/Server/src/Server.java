package Server;
import net.openhft.affinity.AffinityLock;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.concurrent.LinkedBlockingQueue;


/**
 * The Server with the main class for starting it.
 * 
 * @author research-aera
 *
 */
public class Server {

	static int RUN = 1;
	static LinkedBlockingQueue<JobData> jobDataQueue;
	static long counter;
	public static int highest_state = 0;
	public static HashMap<Integer, Integer> hmap = new HashMap<Integer, Integer>();
	private Thread serverThread, resetServer;
	static WorkerThreadPool workerThreadPool;
	

	public static double lambda;
	public static float duration;

	private static PrintWriter telemetryWriter = null;

	/**
	 * Optional live telemetry: when the MG1_TELEMETRY_FILE environment
	 * variable is set, every completed job appends one CSV line
	 * (wall_ms,response_ns,calc_ns,wait_ns,cpu_ns,repeat) to that file so an
	 * external controller can compute per-window response times while the
	 * single-server queue simulation is running.
	 */
	public static void telemetry(long responseNs, long calcNs, long waitNs, long cpuNs, long repeat) {
		if (telemetryWriter == null) {
			return;
		}
		telemetryWriter.printf("%d,%d,%d,%d,%d,%d%n",
				System.currentTimeMillis(), responseNs, calcNs, waitNs, cpuNs, repeat);
	}

	private static void openTelemetry() {
		String telemetryFile = System.getenv("MG1_TELEMETRY_FILE");
		if (telemetryFile == null) {
			return;
		}
		try {
			telemetryWriter = new PrintWriter(new FileWriter(telemetryFile, true), true);
			System.out.println("MG1 telemetry enabled -> " + telemetryFile);
		} catch (IOException e) {
			System.err.println("Could not open telemetry file: " + telemetryFile);
		}
	}


	/** Initialization of the server running on port and its threads */
	public Server(int port) throws UnknownHostException, IOException {
		
		jobDataQueue = new LinkedBlockingQueue<JobData>();
		serverThread = new Thread(new ServerThread(port));
		resetServer = new Thread(new ResetListener());
		workerThreadPool = new WorkerThreadPool();

		serverThread.start();
		resetServer.start();
		workerThreadPool.runWorkerThreadPool();
	}
/**
 * Entry point. Reads the arrival rate (lambda) and duration from args, logs CPU availability
 * and affinity info, then starts the server on port 9999.
 *
 * @param args args[0] - the arrival rate lambda (double)
 * 			   args[1] - the duration of the test (float)
 * @throws UnknownHostException if the local host cannot be resolved
 * @throws IOException if an I/O error occurs while starting the server
 */
	public static void main(String[] args) throws UnknownHostException, IOException {


		lambda = Double.parseDouble(args[0]);
		duration = Float.parseFloat(args[1]);
		openTelemetry();

		// Check cpu availability
		int availableProcessors = Runtime.getRuntime().availableProcessors();
		// Get the number of CPU cores
		System.out.println("Available processors: " + availableProcessors);
		
		// Log CPU affinity lock state if available
		try { System.out.println(AffinityLock.dumpLocks()); } catch (Throwable t) { System.out.println("Affinity info unavailable: " + t.getMessage()); }
		
		// Start the server on the default port
		new Server(9999);
	}
}
