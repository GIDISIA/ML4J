package test;


import edu.utn.gisiq.ml4j.clustering.KMedoids;
import edu.utn.gisiq.ml4j.metrics.pairwise.EuclideanDistance;
import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author ezequiel
 */
public class TestKMedoidsEuclidean {
    private List<INDArray> dataset;
    private KMedoids kmediods;
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        new TestKMedoidsEuclidean();
    }

    public TestKMedoidsEuclidean() {        
        dataset = new ArrayList<>();
        test0();        
        test1();
    }
    
    private void test0(){
        for(int i=0;i<50;i++){
            dataset.add(Nd4j.randn(1, 2));
            dataset.add(Nd4j.randn(1, 2).add(-1));
            dataset.add(Nd4j.randn(1, 2).add(2));
            dataset.add(Nd4j.randn(1, 2).add(1));
        }
        System.out.println("Dataset generated");
        kmediods = new KMedoids(4, 50, new EuclideanDistance());
        System.out.println("KMedoids instanciated");
        kmediods.fit(dataset, true);
        System.out.println("KMedoids Fitted");
        System.out.println(kmediods.getMedoids());
        
        
    }
    
    private void test1(){
        for(int i=0;i<50;i++){            
            dataset.add(Nd4j.randn(1, 2).add(-3));
            dataset.add(Nd4j.randn(1, 2).add(-5));
            dataset.add(Nd4j.randn(1, 2).add(5));
        }
        System.out.println("Dataset: points added");       
        kmediods.fit(dataset, true);
        System.out.println("KMedoids incremental Fitted");
        System.out.println(kmediods.getMedoids());
    }
    
    
}
