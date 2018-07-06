package edu.utn.gisiq.ml4j.clustering;

import edu.utn.gisiq.ml4j.metrics.pairwise.DistanceMetric;
import edu.utn.gisiq.ml4j.metrics.pairwise.Pairwise;
import edu.utn.gisiq.ml4j.random.MersenneTwisterFast;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;
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

    public void fit(INDArray data) {
        dataset = data.dup();
        int n_samples = data.rows();
        boolean changed = true;
        int iteration = 1;
        
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
        INDArray[] idx = Nd4j.sortWithIndices(vj, 0, true);
        medoidsIdx = idx[0].get(NDArrayIndex.interval(0, numberOfClusters));
        /**
         * 1-4. Obtain the initial cluster result by assigning each object to
         * the nearest medoid.
         */
        INDArray assignments = assign(distMatrix);
        
        /**
         * 1-5. Calculate the sum of distances from all objects to their
         * medoids.
         */

        while (changed && iteration<maxIterations) {
            /**
             * Step 2: (Update medoids) Find a new medoid of each cluster, which
             * is the object minimizing the total distance to other objects in
             * its cluster. Update the current medoid in each cluster by
             * replacing with the new medoid.
             */
            changed = recalculateMedoids(assignments, distMatrix);
            /**
             * Step 3: (Assign objects to medoids) 3-1. Assign each object to
             * the nearest medoid and obtain the cluster result. 3-2. Calculate
             * the sum of distance from all objects to their medoids. If the sum
             * is equal to the previous one, then stop the algorithm. Otherwise,
             * go back to the Step 2.
             */
            assignments = assign(distMatrix);
            System.out.println("Iteration "+iteration+" finish");
            iteration++;
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
            for (int j = 1; j < medoidsIdx.length(); j++) {
                double tmpDistance = distMatrix.getDouble(i, medoidsIdx.getInt(j));
                if (tmpDistance < bestDistance) {
                    bestDistance = tmpDistance;
                    out.putScalar(i, j);
                    if (bestDistance == 0D) {
                        break;
                    }
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
    private boolean recalculateMedoids(INDArray assignments, INDArray distMatrix) {
        boolean changed = false;
        INDArray oldMedoidsIdx = medoidsIdx.dup(); //Copy of the ndarray
        for (int i = 0; i < numberOfClusters; i++) {
            List<Integer> points_idx = new ArrayList<>();
            for (int j = 0; j < assignments.length(); j++) {
                if (assignments.getInt(j) == i) {
                    points_idx.add(j);
                }
            }
            if (points_idx.isEmpty()) { // new random, empty medoid
                medoidsIdx.putScalar(i, rg.nextInt(assignments.length()));
                changed = true;
            } else {
                Map<Integer, Double> clusterDistance = new ConcurrentHashMap<>(points_idx.size());         
                // Sumarization of distances for each point of the cluster to the others
                IntStream.range(0, points_idx.size()).forEach(idx -> {
                    double distance = 0;
                    for(int l=0;l<points_idx.size(); l++){
                        distance += Math.abs(distMatrix.getDouble(idx, l));
                    }
                    clusterDistance.put(idx, distance);
                });
                // Check witch clustes is the new medoid
                double minTotalDistance = Double.MAX_VALUE;
                for(int j=0;j<clusterDistance.keySet().size();j++){
                    Double dist = clusterDistance.get(j);
                    if(minTotalDistance > dist){
                        minTotalDistance = dist;
                        medoidsIdx.putScalar(i, j);
                    }
                }
                if(!oldMedoidsIdx.getScalar(i).equals(medoidsIdx.getScalar(i))) {
                    changed = true;
                }
            }
        }
        return changed;
    }
    
    public INDArray getMedoids(){
        if(trained)
            return dataset.get(medoidsIdx);
        return null;
    }

}
