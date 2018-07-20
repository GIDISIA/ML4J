package test;


import edu.utn.gisiq.ml4j.clustering.KMedoids;
import edu.utn.gisiq.ml4j.metrics.pairwise.LongestCommonSubsequence;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author ezequiel
 */
public class TestKMedoidsLCSS {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        new TestKMedoidsLCSS(1);
//        new TestKMedoidsLCSS(1);
    }

    public TestKMedoidsLCSS(int test) throws IOException {
        if (test == 0) {
            test0();
        }
        if (test == 1) {
            test1();
        }
    }

    private void test0() {
        List<INDArray> dataset = new ArrayList<>();
        for (int i = 0; i < 1; i++) {
            dataset.add(Nd4j.randn(50, 2));
            dataset.add(Nd4j.randn(20, 2));
            dataset.add(Nd4j.randn(100, 2));
            dataset.add(Nd4j.randn(10, 2));
        }
        int delta = (int) (5 * (50 + 20 + 100 + 10) / dataset.size() * 0.25);
        System.out.println("Dataset generated");
        KMedoids kmediods = new KMedoids(2, 50, new LongestCommonSubsequence(delta, 0.1, 0.1, false));
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

    private void test1() throws FileNotFoundException, IOException {
        // create a file that is really a directory
        File aDirectory = new File("/media/ezequiel/Datos/datasets/tctodd/");
        List<File> tsFiles = new ArrayList();
        // get a listing of all files in the subdirectories
        File[] filesInDir = aDirectory.listFiles();
        for (File f : filesInDir) {
            if (f.isDirectory()) {
                File[] subfiles = f.listFiles(new FilenameFilter() {
                    @Override
                    public boolean accept(File dir, String name) {
                        return name.toLowerCase().endsWith(".tsd");
                    }
                });
                for (File sf : subfiles) {
                    tsFiles.add(sf);
                }
            }
        }
        List<INDArray> tss = new ArrayList<>();
        for(File f : tsFiles){
            tss.add(Nd4j.readNumpy(f.getAbsolutePath(), "\t"));
        }

        //calc delta value
        int sum = 0;
        for(INDArray ts : tss){
            sum += ts.rows();
        }
        double delta = sum/tss.size()*0.25;
        
        KMedoids kmediods = new KMedoids(5, 50, new LongestCommonSubsequence(delta, 0.2, 0.1, false));
        
        long start = System.currentTimeMillis();
        kmediods.fit(tss.subList(0, 50), true);
        long finish = System.currentTimeMillis();
        System.out.println(kmediods.getMedoids());
        System.out.println("KMedoids Fitted in "+(finish-start)/1000+" ms");
    }

}
