
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;


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
        
        INDArray r = Nd4j.create(new double[]{1,2,3,4,5,6,7,8,9,10});
        INDArray r2 = Nd4j.create(new double[]{1,1,3,3,3,4,2,1,2,3});
        
        System.out.println(r);
        INDArray rta = r2.condi(Conditions.equals(3));
        System.out.println(rta);
        INDArray rta2 = r2.cond(Conditions.equals(3));
        System.out.println(rta2);
//        INDArray rta3 = r.get(rta);
//        System.out.println(rta2);
        
    }
    
    
    
}
