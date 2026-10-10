package Server;

import com.sun.management.ThreadMXBean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.management.ManagementFactory;

import java.security.NoSuchAlgorithmException;

/** This class represents a job. To each job a time stamp and a repeat value is assigned. The
 * time stamp stands for the arrival of the job at the server. The repeat value is an integer in
 * [repeat, 1.33 x repeat]; this number decides the workload of each job. The calc() implements the processing
 * logic of the job.
 * @author research-aera
 */

public class Job implements Runnable {
	public static final Logger logger = LoggerFactory.getLogger(Job.class);
	public long timeStamp;
	public int repeat;
	public static long jobscounter=0;
	private JobData data = new JobData();
	private long endTimeCurrentRequest;
	private long startTimeStampServiceTime;
	private long endTimeStampServiceTime;
	private long calcTime;
	private long packetLength;
	private long responseTime;
	private long startTimeCpuTime;
	private long endTimeCpuTime;

	private long cpuTime;
	
	private long interArrivalTime = 0;
	private long currentTime = 0;
	private long waitingTime=0;
	public Job(long timeStamp, int repeat) {
		this.timeStamp = timeStamp;
		this.repeat = repeat;
	}

	// @Override
	public void run() {
		try {
			calc(this.repeat);
		} catch (NoSuchAlgorithmException e) {
			e.printStackTrace();
		}

		endTimeCurrentRequest = System.nanoTime();
		responseTime = endTimeCurrentRequest - this.timeStamp;//millisecond
		interArrivalTime=timeStamp-ResetListener.arrivalTime;
		ResetListener.arrivalTime=timeStamp;
		data.setResponseTime(responseTime);
		data.setInterArrivaleTime(interArrivalTime);
		Server.jobDataQueue.offer(data);
		Server.telemetry(responseTime, calcTime, waitingTime, cpuTime, this.repeat);

	}

	private void calc(int repeat) throws NoSuchAlgorithmException {

		//check the cpu id used to process the job
		int cpuCore = CpuCoreID.CLibrary.INSTANCE.sched_getcpu();
		logger.info("Job CPU Core: " + cpuCore);


		/*******start computing cpu time ************/
		ThreadMXBean threadMXBean = (ThreadMXBean) ManagementFactory.getThreadMXBean();
		if (threadMXBean.isThreadCpuTimeSupported() && threadMXBean.isThreadCpuTimeEnabled()) {

		//computing execution time
		startTimeStampServiceTime = System.nanoTime();

			// Start cpuTime in nanoseconds
			 startTimeCpuTime = threadMXBean.getCurrentThreadCpuTime();

			/** job */

		 for (long i = 1; i <= repeat; i++) {
			Math.pow(Math.sin(i)*Math.cos(i)/Math.pow(i,i),i); //repeat scaled by a factor in [1, 1.33],

	  	 }
			// Get end CPU time
			endTimeCpuTime = threadMXBean.getCurrentThreadCpuTime(); // In nanoseconds

		//end computing execution time
		endTimeStampServiceTime = System.nanoTime();
		/**this else is if the system couldn't compte the cpu time */
		} else {
			System.out.println("Thread CPU time measurement is not supported on this JVM.");
		}

		/*******stop computing cpu time ************/

		//execution time
		calcTime = (endTimeStampServiceTime - startTimeStampServiceTime); //nanoseconds

		// Calculate and display CPU time
		cpuTime = endTimeCpuTime - startTimeCpuTime;

		//compute waiting time using time stamp as the time the job added to the queue and startTimeStampServiceTime as startExecusion time
		waitingTime= startTimeStampServiceTime-this.timeStamp;
		data.setWaitingTime(waitingTime);

		data.setCalcTime(calcTime);
		data.setCpuTime(cpuTime);
		packetLength = repeat;
		data.setPacketLength(packetLength);

	}

}




