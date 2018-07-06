
import edu.utn.gisiq.ml4j.clustering.KMedoids;
import edu.utn.gisiq.ml4j.metrics.pairwise.EuclideanDistance;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author ezequiel
 */
public class TestKMeansEuclidean {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        new TestKMeansEuclidean();
    }

    public TestKMeansEuclidean() {
        INDArray dataset = Nd4j.rand(50, 10);
        System.out.println("Dataset generated");
        KMedoids kmediods = new KMedoids(5, 50, new EuclideanDistance());
        System.out.println("KMedoids instanciated");
        kmediods.fit(dataset);
        System.out.println("KMedoids Fitted");
        System.out.println(kmediods.getMedoids());
    }
    
    
}
