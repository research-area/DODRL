package Server;
/**
*This class simplify communication with cpu
*@author research-aera
*
*/
import com.sun.jna.Library;

import com.sun.jna.Native;

public class CpuCoreID {
    public CpuCoreID() {
       }
    public interface CLibrary extends Library {
        CLibrary INSTANCE= (CLibrary) Native.loadLibrary("c",CLibrary.class);
        int sched_getcpu();
    }
}
