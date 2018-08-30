
package edu.utn.gisiq.ml4j.rl.policy;

import org.apache.commons.math3.random.MersenneTwister;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Class that implements ramdom choose action
 * @author Ezequiel Beccaría
 */
public class Ramdom implements Policy{
    private MersenneTwister mt;

    public Ramdom(MersenneTwister mt) {
        this.mt = mt;
    }    
    
    @Override
    public int chooseAction(INDArray qValues, int episode) {
        return mt.nextInt(qValues.rows());
    }

    @Override
    public void finishedEpisode() {    
    }
    
}
