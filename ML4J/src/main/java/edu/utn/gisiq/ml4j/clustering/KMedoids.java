package edu.utn.gisiq.ml4j.clustering;

import edu.utn.gisiq.ml4j.metrics.pairwise.DistanceMetric;
import edu.utn.gisiq.ml4j.metrics.pairwise.Pairwise;
import edu.utn.gisiq.ml4j.random.MersenneTwisterFast;
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
        //This are the fitted medoids
        private INDArray medoids;
    
        
        /**
	 * Creates a new instance of the k-medoids algorithm with the specified
	 * parameters.
	 * 
	 * @param numberOfClusters
	 *            the number of clusters to generate
	 * @param maxIterations
	 *            the maximum number of iteration the algorithm is allowed to
	 *            run
	 * @param DistanceMeasure
	 *            dm the distance metric to use for measuring the distance
	 *            between instances
	 * 
	 */
	public KMedoids(int numberOfClusters, int maxIterations, DistanceMetric dm) {
		super();
		this.numberOfClusters = numberOfClusters;
		this.maxIterations = maxIterations;
		this.dm = dm;
		rg = new MersenneTwisterFast(System.currentTimeMillis());
	}
        
        public void fit(INDArray data) {
		int[] medoidsIdx = new int[numberOfClusters];		
                       
                /**
                 * Step 1: (Select initial medoids)      
                 * 1-1. Calculate the distance between every pair of all objects 
                 * based on the chosen dissimilarity measure.
                 */
                INDArray distMatrix = Pairwise.getDistance(data, dm, true);                
                /**
                 * 1-2. Calculate vj for object j as follows:
                 * v_j=\sum_{i=1}^{n}\frac{d_{ij}}{\sum_{l=1}^{n}d_{il}}
                 */
                INDArray vj = this.vj(distMatrix, data.rows());
                /**
                 * 1-3. Sort vj’s in ascending order. Select k objects having 
                 * the first k smallest values as initial medoids.
                 */
                INDArray[] idx = Nd4j.sortWithIndices(vj, 0, true);                
                for(int i=0;i<numberOfClusters;i++){
                    medoidsIdx[i] = idx  NDArrayIndex.interval(0, numberOfClusters)); 
                }     
                /**
                 * 1-4. Obtain the initial cluster result by assigning each object 
                 * to the nearest medoid.
                 */
                //TODO          
                /**
                 * 1-5. Calculate the sum of distances from all objects to their medoids.
                 */
                //TODO

                /*
                Step 2: (Update medoids)
                Find a new medoid of each cluster, which is the object minimizing 
                the total distance to other objects in its cluster. Update the current 
                medoid in each cluster by replacing with the new medoid.

                Step 3: (Assign objects to medoids)
                3-1. Assign each object to the nearest medoid and obtain the cluster result.
                3-2. Calculate the sum of distance from all objects to their medoids. 
                If the sum is equal to the previous one, then stop the algorithm. 
                Otherwise, go back to the Step 2.

                */
                
                
                

		return output;
	}
        
        private INDArray vj(INDArray dist, int n){
            INDArray out = Nd4j.zeros(n);
            for(int j=0;j<n;j++){                
                for(int i=0;i<n;i++){
                    double sum_dil = 0d;
                    for(int l=0;l<n;l++){
                        sum_dil += dist.getDouble(i, l);
                    }
                    out.putScalar(j, out.getDouble(j) + dist.getDouble(i, j) / sum_dil);                    
                }
            }
            return out;
        }
        
//	public TimeSeriesDataset[] cluster(TimeSeriesDataset data) {
//		TimeSeries[] medoids = new TimeSeries[numberOfClusters];
//		TimeSeriesDataset[] output = new TimeSeriesDataset[numberOfClusters];
//		for (int i = 0; i < numberOfClusters; i++) {
//			int random = rg.nextInt(data.size());
//			medoids[i] = data.point(random);
//		}
//
//		boolean changed = true;
//		int count = 0;
//		while (changed && count < maxIterations) {
//			changed = false;
//			count++;
//			int[] assignment = assign(medoids, data);
//			changed = recalculateMedoids(assignment, medoids, output, data);
//
//		}
//
//		return output;
//
//	}

	/**
	 * Assign all instances from the data set to the medoids.
	 * 
	 * @param medoids candidate medoids
	 * @param data the data to assign to the medoids
	 * @return best cluster indices for each instance in the data set
	 */
	private int[] assign(TimeSeries[] medoids, TimeSeriesDataset data) {
		int[] out = new int[data.size()];
		for (int i = 0; i < data.size(); i++) {
			double bestDistance = dm.measure(data.point(i), medoids[0]);
			int bestIndex = 0;
			for (int j = 1; j < medoids.length; j++) {
				double tmpDistance = dm.measure(data.point(i), medoids[j]);
				if (dm.compare(tmpDistance, bestDistance)) {
					bestDistance = tmpDistance;
					bestIndex = j;
				}
			}
			out[i] = bestIndex;

		}
		return out;

	}

	/**
	 * Return a array with on each position the clusterIndex to which the
	 * Instance on that position in the dataset belongs.
	 * 
	 * @param medoids
	 *            the current set of cluster medoids, will be modified to fit
	 *            the new assignment
	 * @param assigment
	 *            the new assignment of all instances to the different medoids
	 * @param output
	 *            the cluster output, this will be modified at the end of the
	 *            method
	 * @return the
	 */
	private boolean recalculateMedoids(int[] assignment, TimeSeries[] medoids, TimeSeriesDataset[] output, TimeSeriesDataset data) {
		boolean changed = false;
		for (int i = 0; i < numberOfClusters; i++) {
			output[i] = new TimeSeriesDatasetImpl();
			for (int j = 0; j < assignment.length; j++) {
				if (assignment[j] == i) {
					output[i].add(data.point(j));
				}
			}
			if (output[i].size() == 0) { // new random, empty medoid
				medoids[i] = data.point(rg.nextInt(data.size()));
				changed = true;
			} else {
				Instance centroid = DatasetTools.average(output[i]);
				TimeSeries oldMedoid = medoids[i];
				medoids[i] = data.kNearest(1, centroid, dm).iterator().next();
				if (!medoids[i].equals(oldMedoid))
					changed = true;
			}
		}
		return changed;
	}
    
}
