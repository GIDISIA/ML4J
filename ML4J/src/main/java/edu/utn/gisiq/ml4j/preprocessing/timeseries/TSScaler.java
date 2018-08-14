package edu.utn.gisiq.ml4j.preprocessing.timeseries;

import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author ezequiel
 */
public interface TSScaler {
    public void fit(List<INDArray> series);
    public List<INDArray> transform(List<INDArray> series);
    public boolean isFitted();
}
