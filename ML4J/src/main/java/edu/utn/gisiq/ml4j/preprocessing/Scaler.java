package edu.utn.gisiq.ml4j.preprocessing;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Ezequiel Beccaría
 * @date 14/08/2018  
 */
public interface Scaler {
    public void fit(INDArray data);
    public INDArray transform(INDArray data);
    public boolean isFitted();
}
