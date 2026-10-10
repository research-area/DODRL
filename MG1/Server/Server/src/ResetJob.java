package Server;

/**
 * This class represents a ResetJob. ResetJob's are received before and after the arrival of Job packets.
 *  
 * @author research-aera
 */

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.LocalTime;

public class ResetJob extends Job {

	private boolean log = false;
	Process p;
	private final Logger logger = LoggerFactory.getLogger(ResetJob.class);

	public ResetJob(long timeStamp, int repeat, boolean log) {
		super(timeStamp, repeat);
		this.log = log;
	}

	// @Override
	public void run() {
		/*
		 * This if block represents first ResetJob. First initialize all variables to
		 * null. 
		 */
		if (this.log) {
			System.out.println("***************************************************************");
			LocalTime time01 = LocalTime.now();
			System.out.println("Time at start of Reset job number 1 : " + time01);

			System.out.println("Initializing HashMap in ResetJob");
			for (int i = 0; i <= 1000; i++) {
				Server.hmap.put(i, 0);
			}

			//logger.info(""+Server.hmap);
			LocalTime time011 = LocalTime.now();
			System.out.println("Time at end of Reset job number 1 : " + time011);
			System.out.println("***************************************************************");

			/*
			 * Wait for first Job to arrive after ResetJob.
			 */
			try {
				Thread.sleep(1100);
			} catch (InterruptedException ioe) {
				ioe.printStackTrace();
			}
			try {
				LocalTime time02 = LocalTime.now();
				System.out.println("Logging Actually Started : " + time02);
			} catch (Throwable t) {
				t.printStackTrace();
			}

		}
		/*
		 * This else block represents last ResetJob. Calculate
		 * Metrics and initialize variables to null.
		 */
		else {
			LocalTime time2 = LocalTime.now();
			System.out.println("Logging Finished : " + time2);
			System.out.println("***************************************************************");
			LocalTime time01 = LocalTime.now();
			System.out.println("Time at start of Reset job number 2 : " + time01);

			Metrics.calculate();// Do calculation

			// Initialize variables to null for next run of the system.
			Server.counter = 0;
			Server.jobDataQueue.clear();

			Metrics.MeanRspTime = 0;
			Metrics.MeanPacketLength = 0;
			Metrics.MeanSrvcTime = 0;
			Metrics.RspTime = 0;
			Metrics.PacketLength = 0;
			Metrics.SrvcTime = 0;
			
			Server.highest_state = 0;
			System.out.println("Initializing HashMap in ResetJob");
			for (int i = 0; i <= 1000; i++) {
				Server.hmap.put(i, 0);
			}

			LocalTime time011 = LocalTime.now();
			System.out.println("Time at end of Reset job number 2: " + time011);
			System.out.println("***************************************************************");
			WorkerThreadPool.executorPool.shutdown();
			Server.RUN = 0;
			/**
			 * ensure the full system exiting when jobs are served
			 */
			//System.out.println("after while ");
			System.exit(0);
		}

	}
}
