
import edu.utn.gisiq.ml4j.metrics.pairwise.LongestCommonSubsequence;
import edu.utn.gisiq.ml4j.metrics.pairwise.LongestCommonSubsequence;
import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author ezequiel
 */
public class TestLCSS {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        new TestLCSS(0);
//        new TestKMedoidsLCSS(1);
    }

    public TestLCSS(int test) {
        if(test==0)
            test0();
    }
    
    private void test0(){            
        INDArray a = Nd4j.randn(200, 30);
        INDArray b = Nd4j.randn(100, 30);
       
        int delta = (int) (((a.rows()+b.rows())/2)*0.25); //set delta as 25% of the average length
        
        LongestCommonSubsequence lcss = new LongestCommonSubsequence(delta, 0.5, 0, false);
       
        long start = System.currentTimeMillis();
        double dist = lcss.distance(a, b);
        long finish = System.currentTimeMillis();
        System.out.println("D1: "+dist+" in "+(finish-start)/1000+" ms");
    }    
}
