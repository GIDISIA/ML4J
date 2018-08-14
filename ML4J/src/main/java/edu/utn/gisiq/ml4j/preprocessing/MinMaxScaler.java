package edu.utn.gisiq.ml4j.preprocessing;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Standard scaler for time series
 *
 * @author Ezequiel Beccaría
 */
public class MinMaxScaler implements Scaler {

    private boolean fitted;
    private INDArray min;
    private INDArray max;

    public MinMaxScaler() {
        super();
        fitted = false;
    }

    @Override
    public void fit(INDArray data) {
        if (data != null) {
            min = data.min(0);
            max = data.max(0);

            fitted = true;
        }
    }

    @Override
    public INDArray transform(INDArray data) {
        if (fitted) {
            INDArray denominator = max.subRowVector(min);
            INDArray numerator = data.subRowVector(min);
            return numerator.divRowVector(denominator);    
        } else {
            return null;
        }
    }

    @Override
    public boolean isFitted() {
        return fitted;
    }
}
