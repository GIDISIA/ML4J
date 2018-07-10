
import javax.swing.JFrame;
import org.apache.commons.math3.random.MersenneTwister;
import org.math.plot.Plot2DPanel;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author ezequiel
 */
public class JMathPlotTest {
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        MersenneTwister mt = new MersenneTwister();
        double[] x = new double[50];
        for(int i=0;i<x.length;i++)
            x[i]=mt.nextDouble();
        double[] y = new double[50];
        for(int i=0;i<x.length;i++)
            y[i]=mt.nextDouble();
        // create your PlotPanel (you can use it as a JPanel)
        Plot2DPanel plot = new Plot2DPanel();

        // add a line plot to the PlotPanel
        plot.addScatterPlot("my plot", x, y);

        // put the PlotPanel in a JFrame, as a JPanel
        JFrame frame = new JFrame("a plot panel");
        frame.setSize(300, 300);
        frame.setContentPane(plot);
        frame.setVisible(true);
    }
    
}
