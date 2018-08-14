package edu.utn.gisiq.ml4j.preprocessing;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Standard scaler for standard dataset
 * @author Ezequiel Beccaría
 */
public class StandardScaler implements Scaler{
    private boolean fitted;
    private INDArray mean;
    private INDArray std;

    public StandardScaler() {
        super();
        fitted = false;
    }

    @Override
    public void fit(INDArray data) {
         if(data != null){
            mean = data.mean(0);
            std = data.std(0);            
            fitted = true;
        }
    }

    @Override
    public INDArray transform(INDArray data) {
        if(fitted){      
            return data.subRowVector(mean).divRowVector(std);    
        }else{
            return null;
        }    
    }
    
    @Override
    public boolean isFitted(){
        return fitted;
    }
}
