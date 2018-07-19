package edu.utn.gisiq.ml4j.metrics.pairwise;

import java.util.List;
import java.util.stream.IntStream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Ezequiel Beccaria
 * @date 27/06/2018  
 */
public abstract class Pairwise {
    
    public static INDArray getDistance(INDArray a, INDArray b, DistanceMetric distanceMeasure) {
        return pairwise(a, b, distanceMeasure);
    }
    
    public static INDArray getDistance(List<INDArray> a, List<INDArray> b, DistanceMetric distanceMeasure) {
        return pairwise(a, b, distanceMeasure);
    }
    
    public static INDArray getDistance(INDArray a, DistanceMetric distanceMeasure, boolean upperMatrix) {
        return pairwise(a, distanceMeasure, upperMatrix);
    }
    
    public static INDArray getDistance(List<INDArray> a, DistanceMetric distanceMeasure, boolean upperMatrix) {
        return pairwise(a, distanceMeasure, upperMatrix);
    }
     
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
        *  If we want the full matrix, we need to compute the diagonal.
        *  O(M) - Only the diagonal elements
         */
        if (!upper) {
            IntStream.range(0, m).parallel().forEach(i -> {
                distances.put(i, i, 0D);          
            });
        }

        return distances;
    }  
}
