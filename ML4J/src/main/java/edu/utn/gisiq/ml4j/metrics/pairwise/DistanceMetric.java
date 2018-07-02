package edu.utn.gisiq.ml4j.metrics.pairwise;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Ezequiel Becaría
 * @date 27/06/2018  
 */
public interface DistanceMetric {
    public double distance(final INDArray a, final INDArray b); 
    public double similarity(final INDArray a, final INDArray b); 
    
}
