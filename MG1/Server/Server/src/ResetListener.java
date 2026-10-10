package Server;
/**
 * Listens for Reset packets. Packages each packet into Resetjob class and add's to the executorPool's queue.
 * Sets the log flag to true or false based on the time of it's arrival.
 * @author research-aera
 */

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.*;

public class ResetListener implements Runnable {

	private static final Logger log = LoggerFactory.getLogger(ResetListener.class);
	private DatagramSocket resetSocket;
	private int resetPort = 9950;
	private byte[] resetBuffer;
	private DatagramPacket resetPacket;
	private int state;
	private String ipaddr = "10.0.0.2";
	public static long arrivalTime=0;
	/**
	* Binds a UDP socket to {@code resetPort} on the configured server IP,
	* then mirrors the server's run-state flag.
	*/
	public ResetListener() {
		try {
			resetSocket = new DatagramSocket(resetPort, InetAddress.getByName(ipaddr));
		} catch (SocketException e) {
			e.printStackTrace();
		} catch (UnknownHostException e) {
            throw new RuntimeException("on server address IP "+ e);
        }
        this.state = Server.RUN;
	}
	/**
	 * Spins until {@link WorkerThreadPool#executorPool} is initialised,
	 * sleeping 50 ms between checks. Returns {@code null} if the thread
	 * is interrupted while waiting.
	 *
	 * @return the ready executor pool, or {@code null} on interruption
	 */
	private CustomWorkerThreadPool awaitExecutor() {
        CustomWorkerThreadPool exec;
        while ((exec = WorkerThreadPool.executorPool) == null) {
            try { Thread.sleep(50); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); return null; }
        }
        return exec;
    }
	/**
	 * Main loop: blocks on UDP receive, decodes each packet, waits for the
	 * executor pool to be ready, then dispatches a {@link ResetJob}.
	 * 
	 * {@code RESET1} – high-priority reset, inserted at the head of the queue ({@code front=true})
	 * anything else – normal reset, appended at the tail of the queue ({@code front=false})
	 * 
	 */
	public void run() {
		System.out.println("ResetListener running...");

		while (this.state == Server.RUN) {
			System.out.println(ipaddr);
			resetBuffer = new byte[128];
			resetPacket = new DatagramPacket(resetBuffer, resetBuffer.length);

			try {
				resetSocket.receive(resetPacket);
			} catch (IOException e) {
				e.printStackTrace();
			}
			int packetLen = resetPacket.getLength();
			int offSet = resetPacket.getOffset();
			byte[] packetContents = new byte[packetLen];
			System.arraycopy(resetBuffer, offSet, packetContents, 0, packetLen);
			String content = new String(packetContents);
			
			CustomWorkerThreadPool exec = awaitExecutor();
            if (exec == null) return;

			if (content.equals("RESET1")) {
				System.out.println("RESET JOB 1 ADDED in the start of the queue");

				exec.execute(new ResetJob(System.nanoTime(), 0, true));

			} else {

				System.out.println("RESET JOB 2 ADDED in the end of the queue");
				exec.execute(new ResetJob(System.nanoTime(), 0, false));				
				int i=0;
				while(WorkerThreadPool.executorPool == null && i<10){
					log.info("WorkerThreadPool is null");
					WorkerThreadPool.executorPool.execute(new ResetJob(System.nanoTime(), 0, false));
					i++;
				}
			}

		}

		}



}

