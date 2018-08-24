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
import org.apache.commons.lang3.ArrayUtils;
import org.math.plot.Plot2DPanel;
import org.math.plot.Plot3DPanel;
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
    private int[] medoidsIdx;
    private List<INDArray> dataset;
    private boolean trained;
    private INDArray distMatrix; // Distance matrix
    private INDArray assignments;    

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
        distMatrix = null;
    }
    
    public boolean isFitted(){
        return trained;
    }
    
    public void fit(List<INDArray> data, Integer numberOfClusters, boolean debug){
        if((numberOfClusters != null) && (numberOfClusters != this.numberOfClusters)){
            this.numberOfClusters = numberOfClusters;
            this.medoidsIdx = null;
        }
        this.fit(data, debug);
    }

    public void fit(List<INDArray> data, boolean debug) {
        long startTime = System.currentTimeMillis();
        
        if(debug)
            System.out.println("K-Medoids fitting initiation");
        
        dataset = data;
        int n_samples = data.size();
        int iteration = 1;
        double prev_distortion = Double.MAX_VALUE;
        
        /**
         * Step 1: (Select initial medoids) 1-1. Calculate the distance between
         * every pair of all objects based on the chosen dissimilarity measure.
         */
        if(distMatrix == null){
            distMatrix = Pairwise.getDistance(data, dm, false);
        }else{
            // if k-medoids was trained before, use incremetal distance matrix            
            distMatrix = Pairwise.getDistanceIncremental(data, distMatrix, dm, false);
        }
        
        if(debug){
            long distMatrixTime = System.currentTimeMillis();
            System.out.println("K-Medoids distance matrix calculated in: "+(startTime-distMatrixTime)/1000000+" ms");            
        }
        
        if(medoidsIdx == null){
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
            medoidsIdx = idx.get(NDArrayIndex.interval(0, numberOfClusters)).toIntVector();
        }
        
        /**
         * 1-4. Obtain the initial cluster result by assigning each object to
         * the nearest medoid.
         */
        assignments = assign(distMatrix);        
        
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
                long iterationTime = System.currentTimeMillis();
                System.out.println("K-Medoids Iteration "+iteration+" finish in: "+(iterationTime-startTime)/1000000+" ms");      
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
            for (int j = 0; j < medoidsIdx.length; j++) {
                double tmpDistance = distMatrix.getDouble(i, medoidsIdx[j]);
                if (tmpDistance < bestDistance) {
                    bestDistance = tmpDistance;
                    out.putScalar(i, medoidsIdx[j]);                    
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
                if (assignments.getInt(j) == medoidsIdx[i]) {
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
                    medoidsIdx[i] = key;
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
    
    public List<INDArray> getMedoids(){
        if(trained){
            List<INDArray> out = new ArrayList<>();
            for(int i=0;i<medoidsIdx.length;i++)
                out.add(dataset.get(medoidsIdx[i]));
            return out;
        }
        return null;
    }
    
    public int[] getMedoidsIdx(){
        if(trained)
            return medoidsIdx;
        return null;
    }
    
    public int[] getAssignationsToMedoid(int medoidIdx){
        List<Integer> assign = new ArrayList<>();
        for(int i=0;i<assignments.length();i++){
            if(assignments.getInt(i)==medoidIdx)
                assign.add(i);
        }
        return assign.stream().mapToInt(i->i).toArray();
    }
    
    public void printCurrentState(String title){        
        // create your PlotPanel (you can use it as a JPanel)        
        if(dataset.get(0).isRowVectorOrScalar()){
            Plot2DPanel plot = new Plot2DPanel();
            double[][] x = new double[dataset.size()][dataset.get(0).length()];          
            double[][] medoids = new double[medoidsIdx.length][dataset.get(0).getRow(0).length()];      
            // fill x to for plotting
            for(int i=0;i<dataset.size();i++){
                double[] p = dataset.get(i).toDoubleVector();
                for(int j=0;j<p.length;j++){
                    x[i][j] = p[j];         
                    if(j==1) 
                        break;
                }    
            }
            // fill medoids for plotting
            for(int i=0;i<medoidsIdx.length;i++){
                medoids[i] = x[medoidsIdx[i]];
            }            

            // add a scatter plot to the PlotPanel
            plot.addScatterPlot(
                    "Dataset", 
                    Color.BLUE, 
                    x);
            plot.addScatterPlot(
                    "Medoids", 
                    Color.RED, 
                    medoids);
            // put the PlotPanel in a JFrame, as a JPanel
            JFrame frame = new JFrame(title);
            frame.setSize(300, 300);
            frame.setContentPane(plot);
            frame.setVisible(true);
        }else{
            Plot3DPanel plot = new Plot3DPanel();
            for(int i=0;i<dataset.size();i++){
                double[][] x = dataset.get(i).getColumns(new int[]{0,1,2}).toDoubleMatrix();
                if(ArrayUtils.contains(medoidsIdx, i)){
                    plot.addLinePlot("t"+i, x);
                }else{
                    plot.addLinePlot("t"+i, Color.BLUE, x);
                }                
            }
            // put the PlotPanel in a JFrame, as a JPanel
            JFrame frame = new JFrame(title);
            frame.setSize(300, 300);
            frame.setContentPane(plot);
            frame.setVisible(true);
        }
        
        
    }

}
