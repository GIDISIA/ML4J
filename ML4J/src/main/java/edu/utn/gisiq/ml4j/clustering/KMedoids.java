package edu.utn.gisiq.ml4j.clustering;

import edu.utn.gisiq.ml4j.metrics.pairwise.DistanceMetric;
import edu.utn.gisiq.ml4j.metrics.pairwise.Pairwise;
import edu.utn.gisiq.ml4j.random.MersenneTwisterFast;
import java.awt.Color;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.swing.JFrame;
import org.math.plot.Plot2DPanel;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 *
 * @author Ezequiel Beccaria
 * @date 27/06/2018
 */
public class KMedoids {

    /* Distance measure to measure the distance between instances */
    private DistanceMetric dm;

    /* Number of clusters to generate */
    private int numberOfClusters;

    /* Random generator for selection of candidate medoids */
    private MersenneTwisterFast rg;

    /* The maximum number of iterations the algorithm is allowed to run. */
    private int maxIterations;
    //This are the fitted medoids idxs
    private INDArray medoidsIdx;
    private INDArray dataset;
    private boolean trained;
    

    /**
     * Creates a new instance of the k-medoids algorithm with the specified
     * parameters.
     *
     * @param numberOfClusters the number of clusters to generate
     * @param maxIterations the maximum number of iteration the algorithm is
     * allowed to run
     * @param DistanceMeasure dm the distance metric to use for measuring the
     * distance between instances
     *
     */
    public KMedoids(int numberOfClusters, int maxIterations, DistanceMetric dm) {
        super();
        this.numberOfClusters = numberOfClusters;
        this.maxIterations = maxIterations;
        this.dm = dm;
        rg = new MersenneTwisterFast(System.currentTimeMillis());
        trained = false;
    }

    public void fit(INDArray data, boolean debug) {
        dataset = data.dup();
        int n_samples = data.rows();
        int iteration = 1;
        double prev_distortion = Double.MAX_VALUE;
        
        /**
         * Step 1: (Select initial medoids) 1-1. Calculate the distance between
         * every pair of all objects based on the chosen dissimilarity measure.
         */
        INDArray distMatrix = Pairwise.getDistance(data, dm, false);
        /**
         * 1-2. Calculate vj for object j as follows:
         * v_j=\sum_{i=1}^{n}\frac{d_{ij}}{\sum_{l=1}^{n}d_{il}}
         */
        INDArray vj = this.vj(distMatrix, n_samples);
        /**
         * 1-3. Sort vj’s in ascending order. Select k objects having the first
         * k smallest values as initial medoids.
         */
        //TODO wrong
        INDArray idx = Nd4j.sortWithIndices(vj.dup(), 1, true)[0];
        medoidsIdx = idx.get(NDArrayIndex.interval(0, numberOfClusters));
        /**
         * 1-4. Obtain the initial cluster result by assigning each object to
         * the nearest medoid.
         */
        INDArray assignments = assign(distMatrix);
        
        /**
         * 1-5. Calculate the sum of distances from all objects to their
         * medoids.
         */
        double distortion = calcDistortion(assignments, distMatrix);
        
        if(debug)
            this.printCurrentState("Iteration "+iteration);

        while (distortion < prev_distortion && iteration<maxIterations) {
            prev_distortion = distortion;
            /**
             * Step 2: (Update medoids) Find a new medoid of each cluster, which
             * is the object minimizing the total distance to other objects in
             * its cluster. Update the current medoid in each cluster by
             * replacing with the new medoid.
             */
            recalculateMedoids(assignments, distMatrix);
            /**
             * Step 3: (Assign objects to medoids) 3-1. Assign each object to
             * the nearest medoid and obtain the cluster result. 
             */
            assignments = assign(distMatrix);
            /**
             * 3-2. Calculate the sum of distance from all objects to their 
             * medoids. If the sum is equal to the previous one, then stop the 
             * algorithm. Otherwise, go back to the Step 2.
             */           
            distortion = calcDistortion(assignments, distMatrix);
            iteration++;
            
            if(debug){
                System.out.println("Iteration "+iteration+" finish");
                this.printCurrentState("Iteration "+iteration);
            }    
        }
        trained = true;
        System.out.println("done");
    }

    private INDArray vj(INDArray dist, int n) {
        INDArray out = Nd4j.zeros(n);
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                double sum_dil = 0d;
                for (int l = 0; l < n; l++) {
                    sum_dil += dist.getDouble(i, l);
                }
                out.putScalar(j, out.getDouble(j) + dist.getDouble(i, j) / sum_dil);
            }
        }
        return out;
    }

    /**
     * Assign all instances from the data set to the medoids.
     *
     * @param medoids candidate medoids
     * @param data the data to assign to the medoids
     * @return best cluster indices for each instance in the data set
     */
    private INDArray assign(INDArray distMatrix) {
        INDArray out = Nd4j.zeros(distMatrix.rows());
        for (int i = 0; i < distMatrix.rows(); i++) {
            double bestDistance = Double.MAX_VALUE;
            for (int j = 0; j < medoidsIdx.length(); j++) {
                double tmpDistance = distMatrix.getDouble(i, medoidsIdx.getInt(j));
                if (tmpDistance < bestDistance) {
                    bestDistance = tmpDistance;
                    out.putScalar(i, medoidsIdx.getInt(j));                    
                }
            }
        }
        return out;

    }

    /**
     * Return a array with on each position the clusterIndex to which the
     * Instance on that position in the dataset belongs.
     * @param assignments
     * @param distMatrix
     * @return 
     */
    private void recalculateMedoids(INDArray assignments, INDArray distMatrix) {
        for (int i = 0; i < numberOfClusters; i++) {
            List<Integer> points_idx = new ArrayList<>();
            for (int j = 0; j < assignments.length(); j++) {
                if (assignments.getInt(j) == medoidsIdx.getInt(i)) {
                    points_idx.add(j);
                }
            }
            Map<Integer, Double> clusterDistance = new ConcurrentHashMap<>(points_idx.size());         
            // Sumarization of distances for each point of the cluster to the others
            for(int j=0;j<points_idx.size();j++){
                double distance = 0;
                for(int l=0;l<points_idx.size(); l++){
                    if(points_idx.get(j)!=points_idx.get(l))
                        distance += Math.abs(distMatrix.getDouble(points_idx.get(j), points_idx.get(l)));
                }
                clusterDistance.put(points_idx.get(j), distance);
            }           
            // Check witch clustes is the new medoid
            double minTotalDistance = Double.MAX_VALUE;
            for(Integer key : clusterDistance.keySet()){
                Double dist = clusterDistance.get(key);
                if(minTotalDistance > dist){
                    minTotalDistance = dist;
                    medoidsIdx.putScalar(i, key);
                }
            }
        }
    }
    
    private double calcDistortion(INDArray assignments, INDArray distMatrix){
        double distortion = 0;
        for(int i=0;i<assignments.length();i++){
            distortion += distMatrix.getDouble(i, assignments.getInt(i));
        }
        return distortion;
    }
    
    public INDArray getMedoids(){
        if(trained)
            return dataset.getRows(medoidsIdx.toIntVector());
        return null;
    }
    
    public void printCurrentState(String title){
        double[] x = dataset.getColumn(0).toDoubleVector();
        double[] y = dataset.getColumn(1).toDoubleVector();
        // create your PlotPanel (you can use it as a JPanel)
        Plot2DPanel plot = new Plot2DPanel();

        // add a line plot to the PlotPanel
        plot.addScatterPlot(
                "Dataset", 
                Color.BLUE, 
                x, 
                y);
        plot.addScatterPlot(
                "Medoids", 
                Color.RED, 
                dataset.getRows(medoidsIdx.toIntVector()).getColumn(0).toDoubleVector(),
                dataset.getRows(medoidsIdx.toIntVector()).getColumn(1).toDoubleVector());

        // put the PlotPanel in a JFrame, as a JPanel
        JFrame frame = new JFrame(title);
        frame.setSize(300, 300);
        frame.setContentPane(plot);
        frame.setVisible(true);
    }

}
