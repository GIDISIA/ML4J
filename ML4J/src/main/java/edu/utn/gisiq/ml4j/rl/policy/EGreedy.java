package edu.utn.gisiq.ml4j.rl.policy;

import edu.utn.gisiq.ml4j.random.MersenneTwisterFast;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.factory.Nd4j;

/**
 * e-greedy implementation for policy choosing.
 * For fixed epsilon, must set stepEpsion equals to zero.
 * @author Ezequiel Beccaria
 */
public class EGreedy implements Policy{
    private final MersenneTwisterFast mt;
    private final double finalEpsilon;
    private final double stepEpsilon;
    private double currentEpsilon;

    public EGreedy(double initialEpsilon, double finalEpsilon, double stepEpsilon) {
        this.currentEpsilon = initialEpsilon;
        this.finalEpsilon = finalEpsilon;
        this.stepEpsilon = stepEpsilon;
        mt = new MersenneTwisterFast();
    }
    
    @Override
    public int chooseAction(INDArray qValues) {
        int rta = -1;
        int idx = Nd4j.getExecutioner().execAndReturn(new IAMax(qValues)).getFinalResult();
        
        if(mt.nextDouble()<=currentEpsilon){
            //explore
            rta = mt.nextInt(qValues.length()-1);
            if(rta == idx)
                rta++; //if rta==idx increment rta by one to avoid best action
        }else{
            //explote
            rta = idx;
        }
        
        currentEpsilon = nextEpsilon();
            
        return rta;
    }
    
    private double nextEpsilon(){
        if(stepEpsilon > 0 && currentEpsilon < finalEpsilon) return currentEpsilon+stepEpsilon; 
        if(stepEpsilon < 0 && currentEpsilon > finalEpsilon) return currentEpsilon+stepEpsilon; 
        return currentEpsilon; //fixed epsilon
    }
    
}
