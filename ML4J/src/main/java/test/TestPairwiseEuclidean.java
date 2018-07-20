package test;


import edu.utn.gisiq.ml4j.metrics.pairwise.EuclideanDistance;
import edu.utn.gisiq.ml4j.metrics.pairwise.Pairwise;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author ezequiel
 */
public class TestPairwiseEuclidean {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        new TestPairwiseEuclidean(0);
//        new TestKMedoidsEuclidean(1);
    }

    public TestPairwiseEuclidean(int test) {
        if(test==0)
            test0();
        if(test==1)
            test1();
    }
    
    private void test0(){
        INDArray dataset = Nd4j.randn(2, 2);
        INDArray dataset2 = Nd4j.randn(2, 2);
        
        System.out.println("Dataset generated");
        INDArray dist1 = Pairwise.getDistance(dataset, new EuclideanDistance(), true);
        INDArray dist2 = Pairwise.getDistanceIncremental(Nd4j.vstack(dataset, dataset2), dist1, new EuclideanDistance(), true);
        
        System.out.println("Distance Calculated");
        System.out.println(dist1);
        System.out.println(dist2);
    }
    
    private void test1(){
        
    }
    
    
}
