package edu.utn.gisiq.ml4j.metrics.pairwise;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Ezequiel Beccaria
 * @date 27/06/2018  
 */
public class EuclideanDistance implements DistanceMetric{

    @Override
    public double distance(INDArray a, INDArray b) {
        return a.distance2(b);
    }

    @Override
    public double similarity(INDArray a, INDArray b) {
        return 1d/(1+distance(a, b));
    }

    
}
