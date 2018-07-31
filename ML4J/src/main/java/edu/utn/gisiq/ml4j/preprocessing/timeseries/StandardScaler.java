package edu.utn.gisiq.ml4j.preprocessing.timeseries;

import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Standard scaler for time series
 * @author Ezequiel Beccaría
 */
public class StandardScaler {
    private boolean fitted;
    private INDArray mean;
    private INDArray std;

    public StandardScaler() {
        super();
        fitted = false;
    }
    
    public void fit(List<INDArray> series){
        if(series != null && !series.isEmpty()){
            INDArray a = series.get(0);
            for(int i=1;i<series.size();i++){
                a = Nd4j.vstack(a, series.get(i));
            }
            
            mean = a.mean(0);
            std = a.std(0);
            
            fitted = true;
        }
    }
    
    public List<INDArray> transform(List<INDArray> series){                
        if(fitted){
            List<INDArray> transformed = new ArrayList<>();
            for(INDArray serie : series){
                INDArray t = serie.sub(mean).div(std);
                transformed.add(t);
            }      
            return transformed;
        }else{
            return null;
        }    
    }
    
    public boolean isFitted(){
        return fitted;
    }
}
