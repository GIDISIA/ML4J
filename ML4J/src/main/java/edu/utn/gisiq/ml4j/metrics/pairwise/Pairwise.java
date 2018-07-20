package edu.utn.gisiq.ml4j.metrics.pairwise;

import java.util.List;
import java.util.stream.IntStream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 *
 * @author Ezequiel Beccaria
 * @date 27/06/2018  
 */
public abstract class Pairwise {
    
    public static INDArray getDistance(INDArray a, INDArray b, DistanceMetric distanceMeasure) {
        return pairwise(a, b, distanceMeasure);
    }
    
    public static INDArray getDistanceIncremental(INDArray a, INDArray b, INDArray distanceMatrix, DistanceMetric distanceMeasure) {
        return pairwiseIncremental(a, b, distanceMatrix, distanceMeasure);
    }
    
    public static INDArray getDistance(List<INDArray> a, List<INDArray> b, DistanceMetric distanceMeasure) {
        return pairwise(a, b, distanceMeasure);
    }
    
    public static INDArray getDistanceIncremental(List<INDArray> a, List<INDArray> b, INDArray distanceMatrix, DistanceMetric distanceMeasure) {
        return pairwiseIncremental(a, b, distanceMatrix, distanceMeasure);
    }
    
    public static INDArray getDistance(INDArray a, DistanceMetric distanceMeasure, boolean upperMatrix) {
        return pairwise(a, distanceMeasure, upperMatrix);
    }
    
    public static INDArray getDistanceIncremental(INDArray a, INDArray distanceMatrix, DistanceMetric distanceMeasure, boolean upperMatrix) {
        return pairwiseIncremental(a, distanceMatrix, distanceMeasure, upperMatrix);
    }
    
    public static INDArray getDistance(List<INDArray> a, DistanceMetric distanceMeasure, boolean upperMatrix) {
        return pairwise(a, distanceMeasure, upperMatrix);
    }
    
    public static INDArray getDistanceIncremental(List<INDArray> a, INDArray distanceMatrix, DistanceMetric distanceMeasure, boolean upperMatrix) {
        return pairwiseIncremental(a, distanceMatrix, distanceMeasure, upperMatrix);
    }
     
    /**
     * Method that calculates the distance matrix of all elementes of two 
     * datasets given an instance of distanceMeasure. Each element is a row in
     * the objects a and b.
     * @param a
     * @param b
     * @param distanceMeasure
     * @return 
     */
    private static INDArray pairwise(INDArray a, INDArray b, DistanceMetric distanceMeasure) {
        int n = a.rows();
        int m = b.rows();
        INDArray distances = Nd4j.zeros(n, m);

        for (int i=0;i<n;i++) {
            for (int j=0;j<m;j++) {
                distances.put(i, j, distanceMeasure.distance(a.getRow(i), b.getRow(j)));
            }
        }
    	return distances;
    }
    
    private static INDArray pairwiseIncremental(INDArray a, INDArray b, INDArray initDistMatrix, DistanceMetric distanceMeasure) {
        int n = a.rows();
        int m = b.rows();
        INDArray distances = Nd4j.zeros(n, m);
        
        //Complete de new distance matrix with the old values
        distances.get(NDArrayIndex.interval(0,initDistMatrix.rows()), NDArrayIndex.interval(0,initDistMatrix.columns())).assign(initDistMatrix);

        for (int i=0;i<n;i++) {
            for (int j = i<initDistMatrix.rows()?initDistMatrix.rows():i+1; j<m; j++) {
                distances.put(i, j, distanceMeasure.distance(a.getRow(i), b.getRow(j)));
            }
        }
    	return distances;
    }
    
    /**
     * Method that calculates the distance matrix of all elementes of two 
     * datasets given an instance of distanceMeasure. Each element is a item in
     * the objects a and b.
     * @param a
     * @param b
     * @param distanceMeasure
     * @return 
     */
    private static INDArray pairwise(List<INDArray> a, List<INDArray> b, DistanceMetric distanceMeasure) {
        int n = a.size();
        int m = b.size();
        INDArray distances = Nd4j.zeros(n, m);

        for (int i=0;i<n;i++) {
            for (int j=0;j<m;j++) {
                distances.put(i, j, distanceMeasure.distance(a.get(i), b.get(j)));
            }
        }
    	return distances;
    }
    
    /**
     * Method that calculates the distance matrix incrementally using the 
     * distanceMeasure instance as a measure of the elements in a vs the 
     * elements in b. Given an initial distance matrix, it increases its size 
     * incorporating the new elements passed by parameter in the list 'a'. 
     * The elements in the list must be ordered in the same way they were used 
     * to calculate the initial distance matrix. Being at the end of it the new 
     * elements to incorporate.
     * @param a
     * @param b
     * @param initDistMatrix
     * @param distanceMeasure
     * @return 
     */
    private static INDArray pairwiseIncremental(List<INDArray> a, List<INDArray> b, INDArray initDistMatrix, DistanceMetric distanceMeasure) {
        int n = a.size();
        int m = b.size();
        INDArray distances = Nd4j.zeros(n, m);
        
        //Complete de new distance matrix with the old values
        distances.get(NDArrayIndex.interval(0,initDistMatrix.rows()), NDArrayIndex.interval(0,initDistMatrix.columns())).assign(initDistMatrix);

        for (int i=0;i<n;i++) {
            for (int j = i<initDistMatrix.rows()?initDistMatrix.rows():i+1; j<m; j++) {
                distances.put(i, j, distanceMeasure.distance(a.get(i), b.get(j)));
            }
        }
    	return distances;
    }

    /**
     * Method that calculates the distance matrix of a dataset given an instance
     * of distanceMeasure.
     * @param a
     * @param distanceMeasure
     * @param upper
     * @return 
     */
    private static INDArray pairwise(INDArray a, DistanceMetric distanceMeasure, boolean upper) {
        /*
        * Don't need to check dims, because that happens in each
        * getDistance call. Any non-uniformity should be handled 
        * there.
        */
        final int m = a.rows();
        final INDArray distances = Nd4j.zeros(m, m);
        double dist;

        /*
        * First loop: O(M choose 2). Do computations
        */
        for (int i=0;i<m-1;i++) {
            for (int j=i+1;j<m;j++) {
                dist = distanceMeasure.distance(a.getRow(i), a.getRow(j));
                distances.put(i, j, dist);

                // We want the full matrix
                if (!upper) {
                    distances.put(j, i, dist);
                }
            }
        }

        /*
        *  If we want the full matrix, we need to compute the diagonal.
        *  O(M) - Only the diagonal elements
         */
        if (!upper) {
            for (int i=0;i<m;i++) {
                distances.put(i, i, distanceMeasure.distance(a.getRow(i), a.getRow(i)));                
            }

        }

        return distances;
    }  
    
    /**
     * Method that calculates the distance matrix incrementally using the 
     * distanceMeasure instance as a measure. Given an initial distance matrix, 
     * it increases its size incorporating the new elements passed by parameter 
     * in the list 'a'. The elements in the list must be ordered in the same way
     * they were used to calculate the initial distance matrix. Being at the end
     * of it the new elements to incorporate.
     * @param a
     * @param initDistMatrix
     * @param distanceMeasure
     * @param upper
     * @return 
     */
    private static INDArray pairwiseIncremental(INDArray a, INDArray initDistMatrix, DistanceMetric distanceMeasure, boolean upper) {
        final int m = a.rows();
        final INDArray distances = Nd4j.zeros(m, m);        
        //Complete de new distance matrix with the old values
        distances.get(NDArrayIndex.interval(0,initDistMatrix.rows()), NDArrayIndex.interval(0,initDistMatrix.columns())).assign(initDistMatrix);
        
        IntStream.range(0, m).parallel().forEach(i -> {
            for (int j = i<initDistMatrix.rows()?initDistMatrix.rows():i+1; j<m; j++) {
                double dist = distanceMeasure.distance(a.getRow(i), a.getRow(j));
                distances.put(i, j, dist);
                // We want the full matrix
                if (!upper) {
                    distances.put(j, i, dist);
                }                
            }
        });       
        
        /*
        *  If we want the full matrix, we need to fill the diagonal with zeros.
        *  O(M) - Only the diagonal elements
         */
        if (!upper) {
            IntStream.range(initDistMatrix.rows(), m).parallel().forEach(i -> {
                distances.put(i, i, 0D);          
            });
        }

        return distances;
    }
    
    /**
     * Method that calculates the distance matrix of a dataset given an instance
     * of distanceMeasure.
     * @param a
     * @param distanceMeasure
     * @param upper
     * @return 
     */
    private static INDArray pairwise(List<INDArray> a, DistanceMetric distanceMeasure, boolean upper) {
        /*
        * Don't need to check dims, because that happens in each
        * getDistance call. Any non-uniformity should be handled 
        * there.
        */
        final int m = a.size();
        final INDArray distances = Nd4j.zeros(m, m);        

        /*
        * First loop: O(M choose 2). Do computations
        */

        IntStream.range(0, m).parallel().forEach(i -> {
            for (int j=i+1;j<m;j++) {
                double dist = distanceMeasure.distance(a.get(i), a.get(j));
                distances.put(i, j, dist);
                // We want the full matrix
                if (!upper) {
                    distances.put(j, i, dist);
                }
                System.out.println("Distance for point ("+i+","+j+") calculated");
            }
        });        

        /*
        *  If we want the full matrix, we need to fill the diagonal with zeros.
        *  O(M) - Only the diagonal elements
         */
        if (!upper) {
            IntStream.range(0, m).parallel().forEach(i -> {
                distances.put(i, i, 0D);          
            });
        }

        return distances;
    }  
    
    /**
     * Method that calculates the distance matrix incrementally using the 
     * distanceMeasure instance as a measure. Given an initial distance matrix, 
     * it increases its size incorporating the new elements passed by parameter 
     * in the list 'a'. The elements in the list must be ordered in the same way
     * they were used to calculate the initial distance matrix. Being at the end
     * of it the new elements to incorporate.
     * @param a
     * @param initDistMatrix
     * @param distanceMeasure
     * @param upper
     * @return 
     */
    private static INDArray pairwiseIncremental(List<INDArray> a, INDArray initDistMatrix, DistanceMetric distanceMeasure, boolean upper) {
        final int m = a.size();
        final INDArray distances = Nd4j.zeros(m, m);        
        //Complete de new distance matrix with the old values
        distances.get(NDArrayIndex.interval(0,initDistMatrix.rows()), NDArrayIndex.interval(0,initDistMatrix.columns())).assign(initDistMatrix);
        
        IntStream.range(0, m).parallel().forEach(i -> {
            for (int j = i<initDistMatrix.rows()?initDistMatrix.rows():i+1; j<m; j++) {
                double dist = distanceMeasure.distance(a.get(i), a.get(j));
                distances.put(i, j, dist);
                // We want the full matrix
                if (!upper) {
                    distances.put(j, i, dist);
                }                
            }
        });       
        
        /*
        *  If we want the full matrix, we need to fill the diagonal with zeros.
        *  O(M) - Only the diagonal elements
         */
        if (!upper) {
            IntStream.range(initDistMatrix.rows(), m).parallel().forEach(i -> {
                distances.put(i, i, 0D);          
            });
        }

        return distances;
    }
}
