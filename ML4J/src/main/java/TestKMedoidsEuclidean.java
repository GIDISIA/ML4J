
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

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        new TestKMedoidsEuclidean(0);
//        new TestKMedoidsEuclidean(1);
    }

    public TestKMedoidsEuclidean(int test) {
        if(test==0)
            test0();
        if(test==1)
            test1();
    }
    
    private void test0(){
        List<INDArray> dataset = new ArrayList<>();
        for(int i=0;i<50;i++){
            dataset.add(Nd4j.randn(1, 2));
            dataset.add(Nd4j.randn(1, 2).add(-1));
            dataset.add(Nd4j.randn(1, 2).add(2));
            dataset.add(Nd4j.randn(1, 2).add(1));
        }
        System.out.println("Dataset generated");
        KMedoids kmediods = new KMedoids(4, 50, new EuclideanDistance());
        System.out.println("KMedoids instanciated");
        kmediods.fit(dataset, true);
        System.out.println("KMedoids Fitted");
        System.out.println(kmediods.getMedoids());
        
        
//        double[] x = dataset.getColumn(0).toDoubleVector();
//        double[] y = dataset.getColumn(1).toDoubleVector();
//        
//        // create your PlotPanel (you can use it as a JPanel)
//        Plot2DPanel plot = new Plot2DPanel();
//
//        // add a line plot to the PlotPanel
//        plot.addScatterPlot(
//                "Dataset", 
//                Color.BLUE, 
//                x, 
//                y);
//        plot.addScatterPlot(
//                "Medoids", 
//                Color.RED, 
//                kmediods.getMedoids().getColumn(0).toDoubleVector(),
//                kmediods.getMedoids().getColumn(1).toDoubleVector());
//
//        // put the PlotPanel in a JFrame, as a JPanel
//        JFrame frame = new JFrame("a plot panel");
//        frame.setSize(300, 300);
//        frame.setContentPane(plot);
//        frame.setVisible(true);
    }
    
    private void test1(){
        List<INDArray> dataset = new ArrayList<>();
        for(int i=0;i<250;i++){
            dataset.add(Nd4j.randn(1, 2));
            dataset.add(Nd4j.randn(1, 2).add(-1));
            dataset.add(Nd4j.randn(1, 2).add(2));
            dataset.add(Nd4j.randn(1, 2).add(1));
        }
        System.out.println("Dataset generated");
        KMedoids kmediods = new KMedoids(8, 50, new EuclideanDistance());
        System.out.println("KMedoids instanciated");
        kmediods.fit(dataset, true);
        System.out.println("KMedoids Fitted");
        System.out.println(kmediods.getMedoids());
    }
    
    
}
