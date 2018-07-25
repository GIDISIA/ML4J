
package edu.utn.gisiq.ml4j.rl.policy;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author ezequiel
 */
public interface Policy {
    /**
     * Method that return the index of the List given a implemented policy choose
     * algorithm.
     * @param qValues
     * @return 
     */
    public int chooseAction(INDArray qValues);
}
