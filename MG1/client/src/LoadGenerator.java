package client;
import java.io.File;
import java.io.FileWriter;
import java.time.LocalTime;
import java.util.LinkedList;
import java.util.List;
import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;

/**
 * Sends packets to a server with rate lambda and packet size repeat.
 * 
 * @author research-aera
 */
public class LoadGenerator implements Runnable {

   private DatagramSocket resetBroadcastSocket;
   private DatagramSocket loadGeneratorSocket;
   private double lambda = (double)0.0F;
   private static int duration = 0;
   private static int repeat = 0;
   private double rate = (double)0.0F;
   private Distribution dist;
   private InetAddress serverIPAddress;
   private static final int jobListenerPort = 9999;
   private static final int resetListenerPort = 9950;
   private static final String serverIP = "10.0.0.2";
   private static final int STOP = 0;
   private static final int RUN = 1;
   private int counter = 0;
   private long starttime = 0L;
   private long endtime = 0L;
   private int state = 0;

    // constructor 
    public LoadGenerator(double lambda) {

        this.lambda = lambda;

        try {
            this.serverIPAddress = InetAddress.getByName(serverIP);
        } catch (UnknownHostException e1) {
            
            e1.printStackTrace();
        }

        dist = new Distribution();

        try {
            this.loadGeneratorSocket = new DatagramSocket();
        } catch (SocketException e) {
            
            e.printStackTrace();
        }

        try {
            this.resetBroadcastSocket = new DatagramSocket();
        } catch (SocketException e) {
            
            e.printStackTrace();
        }
    
        this.loadGeneratorSocket.connect(serverIPAddress, jobListenerPort);
        
    }

    //@Override
    /** This method sends packets to the server with rate lambda. The time 
     * between two packet is exponentially distributed  */
    public void run() {
    	List<Long> repeatTimes = new LinkedList<Long>();
        while (this.state == RUN) {
        //packet with a bounded repeat factor in [1, 1.33] is sent to the server,
        //so the job size spans [1M, 1.33M] for repeat = 1M (near-deterministic service)
            double nextExpNumber=this.dist.nextExponentialRepeat();
            long service_repeat_time = Math.round(nextExpNumber * this.repeat);
            
			byte[] buffer = Long.toString(service_repeat_time).getBytes();

			try {
                this.loadGeneratorSocket.send(new DatagramPacket(buffer,
                        buffer.length, serverIPAddress, jobListenerPort));
            } catch (IOException e) {
                e.printStackTrace();
            }

            // The number of packets already sent : number of jobs
            counter++;
            repeatTimes.add(service_repeat_time);
            
            // calculation of the exponentially distributed waiting time before the next packet is sent
		 	rate = dist.nextExponential(1.0 / (double) lambda);
            //ensure that repeat not null
		 	if (rate == 0)
			 	rate = 1;
			
            try {
			Thread.sleep(Math.round(1000 * this.rate));
		  	} catch (InterruptedException e) {
          	  e.printStackTrace();
		 	}
        }
		
	// new text file
		FileWriter writer;
		File file;
		//file = new File("SendTimes.csv");
		file = new File("Repeat.csv");
		// write values in text file
		try {
			writer = new FileWriter(file, true);
			//writer.write("No. of Job;Time Stamp;" + System.lineSeparator());
			int i = 1;
			for (Long l : repeatTimes) {
				writer.write(Integer.toString(i) + ";" + Long.toString(l) + System.lineSeparator());
				i++;
			}
			writer.flush();
			writer.close();
		} catch (IOException e1) {
			e1.printStackTrace();
		}
    }

    /**
     * Receives the values for lambda, duration, and repeat, and starts the 
     * Load Generator with these values.
     * 
     * @param args
     *          lambda, duration, repeat
     * @throws UnknownHostException
     * @throws IOException
     * @throws InterruptedException
     */
    public static void main(String[] args) throws UnknownHostException,IOException, InterruptedException {
        
        double lambda = Double.parseDouble(args[0]);
        duration = Integer.parseInt(args[1]);
        repeat = Integer.parseInt(args[2]);
        
            
        System.out.println("lambda: " + lambda);
        System.out.println("duration: " + duration);
        System.out.println("repeat: " + repeat);

        LoadGenerator loadGen = new LoadGenerator(lambda);

        // Reset: All measurement values at the server are reset

        String resetText1 = "RESET1";
        byte[] sendData1 = new byte[256];
        sendData1 = resetText1.getBytes();  

        DatagramPacket resetPacket1 = new DatagramPacket(sendData1,
                sendData1.length, loadGen.serverIPAddress, resetListenerPort);
        try {
            loadGen.resetBroadcastSocket.send(resetPacket1);

        } catch (IOException e1) {
            
            e1.printStackTrace();
        }

        Thread.sleep(7000);//important for sent and received jobs to match
        
        // RUN
        loadGen.state = LoadGenerator.RUN;
        LocalTime starttime = LocalTime.now();
        loadGen.starttime = System.currentTimeMillis();
        
        new Thread(loadGen).start();
        System.out.println("stared");

        // Packets are sent for duration ms before the sending is stopped
        Thread.sleep(duration);

        // STOP
        loadGen.state = LoadGenerator.STOP;
        loadGen.endtime = System.currentTimeMillis();
        LocalTime endtime = LocalTime.now();
        System.out.println("#Requests: " + loadGen.counter);
        System.out.println("#Requests / s: " + loadGen.counter/((loadGen.endtime - loadGen.starttime) / 1000.0));
        
        System.out.println("StartTime " + starttime);
        System.out.println("EndTime " + endtime);
		System.out.println("client counter " + loadGen.counter);

        
        loadGen.counter = 0;
                
        String resetText2 = "RESET2";
        byte[] sendData2 = new byte[256];
        sendData2 = resetText2.getBytes();
        DatagramPacket resetPacket3 = new DatagramPacket(sendData2, sendData2.length,
        		loadGen.serverIPAddress, resetListenerPort);

        try {
            loadGen.resetBroadcastSocket.send(resetPacket3);
        } catch (IOException e1) {
            
            e1.printStackTrace();
        }
    }
}


