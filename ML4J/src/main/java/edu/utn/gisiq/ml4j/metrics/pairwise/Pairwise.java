package edu.utn.gisiq.ml4j.metrics.pairwise;

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
    
    public static INDArray getDistance(INDArray a, DistanceMetric distanceMeasure, boolean upperMatrix) {
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
}
