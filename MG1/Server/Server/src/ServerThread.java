package Server;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;

/**
 * ServerThread is the main listener thread for incoming UDP job requests.
 *
 * It continuously listens on a UDP socket for job packets sent by clients.
 * For each received packet.
 * This class is a key component of the M/G/1 queueing model simulation,
 * responsible for job arrival handling in the server-side implementation.
 *
 * @author research-aera
 */
public class ServerThread implements Runnable {

	private int state;
	
	private int state_before_entering;

	private DatagramSocket datagramSocket;

	private byte[] buffer;

	private DatagramPacket packet;
	private static final Logger log = LoggerFactory.getLogger(ServerThread.class);

/**
 * Constructs a ServerThread that listens for UDP packets on the given port.
 *
 * @param port the UDP port number to bind and listen on
 */
	public ServerThread(int port) {

		try {
			datagramSocket = new DatagramSocket(port);
		} catch (SocketException e) {
			e.printStackTrace();
		}

		this.state = Server.RUN;

	}
/**
 * Main loop of the server thread.
 *
 * Continuously receives UDP packets, extracts job parameters,
 * records the queue depth at arrival time, and dispatches jobs
 * to the worker thread pool for execution.
 */
	// @Override
	public void run() {
		System.out.println("Server running...");
		while (this.state == Server.RUN) {

			buffer = new byte[128];
			packet = new DatagramPacket(buffer, buffer.length);
			/* waiting for incoming jobs */
			try {
				datagramSocket.receive(packet);

			} catch (IOException e) {
				e.printStackTrace();
			}
			int packetLen = packet.getLength();
			int offSet = packet.getOffset();
			byte[] packetContents = new byte[packetLen];
			System.arraycopy(buffer, offSet, packetContents, 0, packetLen);

			String content = new String(packetContents);
			int n = Integer.parseInt(content);
			System.out.println("Packet Received: " + content);
			Job currentJob = new Job(System.nanoTime(), n);
			
			// Capture the number of jobs currently in the system (active workers + queued jobs)
			// This represents the queue length seen by the arriving job (used for state statistics)
			state_before_entering = WorkerThreadPool.executorPool.getActiveCount() + WorkerThreadPool.executorPool.getQueue().size();


			if (state_before_entering >= Server.highest_state) {
				Server.highest_state = state_before_entering;
			}
			if(state_before_entering<=1000){
				Server.hmap.putIfAbsent(state_before_entering,0);
				Server.hmap.compute(state_before_entering, (k, v) -> v + 1);

			}else
				System.out.println("OverFlow");
			
			// Submit the job to the thread pool for asynchronous execution
			WorkerThreadPool.executorPool.execute(currentJob);

			Server.counter++;

			// Log which CPU core this thread is currently running on (for CPU affinity diagnostics)
			int cpuCore = CpuCoreID.CLibrary.INSTANCE.sched_getcpu();
			log.info("Thread CPU Core: " + cpuCore);
		}
	}
}


