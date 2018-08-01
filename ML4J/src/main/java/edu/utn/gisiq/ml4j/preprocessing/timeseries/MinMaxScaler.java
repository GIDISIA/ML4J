package edu.utn.gisiq.ml4j.preprocessing.timeseries;

import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Standard scaler for time series
 * @author Ezequiel Beccaría
 */
public class MinMaxScaler implements TimeSeriesScaler{
    private boolean fitted;
    private INDArray min;
    private INDArray max;

    public MinMaxScaler() {
        super();
        fitted = false;
    }
    
    /**
     *
     * @param series
     */
    @Override
    public void fit(List<INDArray> series){
        if(series != null && !series.isEmpty()){
            INDArray a = series.get(0);
            for(int i=1;i<series.size();i++){
                a = Nd4j.vstack(a, series.get(i));
            }
            
            min = a.min(0);
            max = a.max(0);
            
            fitted = true;
        }
    }
    
    @Override
    public List<INDArray> transform(List<INDArray> series){                
        if(fitted){
            List<INDArray> transformed = new ArrayList<>();
            for(INDArray serie : series){
                INDArray t = serie.sub(min).div(max.sub(min));
                transformed.add(t);
            }      
            return transformed;
        }else{
            return null;
        }    
    }
    
    @Override
    public boolean isFitted(){
        return fitted;
    }
}
