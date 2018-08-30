package edu.utn.gisiq.ml4j.rl.policy;

import edu.utn.gisiq.ml4j.random.MersenneTwisterFast;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.factory.Nd4j;

/**
 * e-greedy implementation for policy choosing.
 * For fixed epsilon, must set stepEpsion equals to zero.
 * @author Ezequiel Beccaria
 */
public class EGreedy implements Policy{
    private final MersenneTwisterFast mt;
    private final double epsilonMin;
    private final double epsilonDecay;
    private double epsilon;

    public EGreedy(double initialEpsilon, double epsilonMin, double epsilonDecay) {
        this.epsilon = initialEpsilon;
        this.epsilonMin = epsilonMin;
        this.epsilonDecay = epsilonDecay;
        mt = new MersenneTwisterFast();
    }
    
    @Override
    public int chooseAction(INDArray qValues, int episode) {
        int rta = -1;
        int idx = Nd4j.getExecutioner().execAndReturn(new IMax(qValues)).getFinalResult();
        double e = Math.max(epsilonMin, Math.min(epsilon, 1.0 - Math.log10((episode + 1) * epsilonDecay)));
        
        if(mt.nextDouble()<=e){
            //explore
            rta = mt.nextInt(qValues.length()-1);
            if(rta == idx)
                rta++; //if rta==idx increment rta by one to avoid best action
        }else{
            //explote
            rta = idx;
        }
            
        return rta;
    }
    
    @Override
    public void finishedEpisode() {
        // Update episilon value after episode ending
        if(epsilon > epsilonMin)
            epsilon *= epsilonDecay;        
    }
    
}
