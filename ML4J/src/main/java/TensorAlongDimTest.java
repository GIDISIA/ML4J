
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author ezequiel
 */
public class TensorAlongDimTest {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        INDArray dataset = Nd4j.create(new double[]{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}, new int[]{3,2,3});
        System.out.println(dataset);
        System.out.println(dataset.getRow(0));
    }
    
}
