
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


/**
 *
 * @author ezequiel
 */
public class NewMain {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        new NewMain();
    }

    public NewMain() {
        INDArray r = Nd4j.rand(3, 4);
        INDArray[] rta = Nd4j.sortWithIndices(r, 0, true);
        System.out.println("EXIT");
    }
    
    
    
}
