
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
     * @param episode
     * @return 
     */
    public int chooseAction(INDArray qValues, int episode);
    
    /**
     * Method that return the index of the List given a implemented policy choose
     * algorithm.
     * Action masked with 0 in maskQValues are ignored
     * @param qValues
     * @param maskQValues
     * @return 
     */
    public int chooseAction(INDArray qValues, INDArray maskQValues, int allowedActionCount, int episode);
    
    /**
     * Method used for calculations after the ending of the episode.
     * This method can be used, for example, to update epsilon value after the 
     * ending of the episode in e-greedy kind of policy choosing algorithms.
     */
    public void finishedEpisode();
}
