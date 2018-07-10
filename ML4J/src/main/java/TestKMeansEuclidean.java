
import edu.utn.gisiq.ml4j.clustering.KMedoids;
import edu.utn.gisiq.ml4j.metrics.pairwise.EuclideanDistance;
import java.awt.Color;
import javax.swing.JFrame;
import org.math.plot.Plot2DPanel;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

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
        INDArray dataset = Nd4j.randn(5, 2);
        INDArray dataset1 = Nd4j.randn(5, 2).add(2);
        dataset = Nd4j.vstack(dataset,dataset1);
        INDArray dataset2 = Nd4j.randn(5, 2).add(4);
        dataset = Nd4j.vstack(dataset,dataset2);
        INDArray dataset3 = Nd4j.randn(5, 2).add(6);
        dataset = Nd4j.vstack(dataset,dataset3);
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
    
    
}
