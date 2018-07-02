package edu.utn.gisiq.ml4j.clustering;

/**
 *
 * @author Ezequiel Beccaria
 * @date 27/06/2018  
 */
public class KMedoids {
    final public static int DEF_MAX_ITER = 10;
    /**
    * Stores the indices of the current medoids. Each index,
    * 0 thru k-1, corresponds to the class label for the cluster.
    */
    volatile private int[] medoid_indices;
    /**
    * Upper triangular, M x M matrix denoting distances between records.
    * Is only populated during training phase and then set to null for 
    * garbage collection, as a large-M matrix has a high space footprint: O(N^2).
    * This is only needed during training and then can safely be collected
    * to free up heap space.
    */
    volatile private double[][] dist_mat = null;
    
    
    
}
