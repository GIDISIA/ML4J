package edu.utn.gisiq.ml4j.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.util.ArrayUtil;

/**
 *
 * @author Ezequiel Beccaría
 * @date 11/09/2018  
 */
public class NDArrayUtils {
    /**
     * Remove a vector in @param idx position given @param axis.
     * @param array
     * @param idx
     * @param axis
     * @return 
     */
    public static INDArray remove(INDArray array, int idx, int axis){
        long[] idxs = ArrayUtil.range(0, array.size(axis));
        idxs = ArrayUtil.removeIndex(idxs, idx);
        INDArrayIndex range = new SpecifiedIndex(idxs);
            
        if(axis==1){             
            return array.get(NDArrayIndex.all(), range);
        }
        //default, axis=0
        return array.get(range, NDArrayIndex.all());
    }
}
