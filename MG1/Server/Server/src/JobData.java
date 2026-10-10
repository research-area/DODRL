package Server;

/**
 * Stores the measurement values for a Job.
 * get and set methods for the following values:
 * - calcTime
 * - packetLength
 * - responseTime
 * - cpuTime
 * - interArrivaleTime
 * - waitingTime
 * @author research-aera
 *
 */
public class JobData {

    private long calcTime;
    private long packetLength;
    private long responseTime;
    private long cpuTime;
    private long interArrivaleTime=0;
    private long waintingTime=0;
    
    public void setCalcTime (long calcTime) { this.calcTime = calcTime;
    }
    public void setInterArrivaleTime (long interArrivaleTime1) { this.interArrivaleTime = interArrivaleTime1;
    }
    public void setWaitingTime (long waitingTime1) { this.waintingTime = waitingTime1;
    }

    public void setPacketLength (long packetLength) {
        this.packetLength = packetLength;
    }
    
    public void setResponseTime (long responseTime) {
        this.responseTime = responseTime;
    }
    public void setCpuTime(long cpuTime){this.cpuTime=cpuTime;}
    
    public long getCalcTime () {
        return calcTime;
    }
    
    public long getPacketLength () {
        return packetLength;
    }

    public long getResponseTime () {
        return responseTime;
    }
    public long getCpuTime(){return cpuTime;}
    public long getInterArrivaleTime () { return interArrivaleTime;}
    public long getWaintingTime () { return waintingTime;}
}


