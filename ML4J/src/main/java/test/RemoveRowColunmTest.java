package test;


import edu.utn.gisiq.ml4j.utils.NDArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author ezequiel
 */
public class RemoveRowColunmTest {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        INDArray dataset = Nd4j.create(new double[]{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, new int[]{4,4});
        System.out.println(dataset);
        System.out.println("------------------------------------");
        
        //dataset without row and colunm1
        System.out.println(NDArrayUtils.remove(NDArrayUtils.remove(dataset, 1, 0), 1, 1));
    }
    
    
    
}
