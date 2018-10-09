package edu.utn.gisiq.ml4j.rl.policy;

import edu.utn.gisiq.ml4j.random.MersenneTwisterFast;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 * e-greedy implementation for policy choosing.
 * For fixed epsilon, must set stepEpsion equals to zero.
 * @author Ezequiel Beccaria
 */
public class EGreedyExponentialDecay implements Policy{
    private final MersenneTwisterFast mt;
    private final double epsilonMin;
    private final double epsilonDecay;
    private double epsilon;

    public EGreedyExponentialDecay(double initialEpsilon, double epsilonMin, double epsilonDecay) {
        this.epsilon = initialEpsilon;
        this.epsilonMin = epsilonMin;
        this.epsilonDecay = epsilonDecay;
        mt = new MersenneTwisterFast();
    }
    
    @Override
    public int chooseAction(INDArray qValues, int episode) {
        int rta = -1;
        int idx = Nd4j.getExecutioner().execAndReturn(new IMax(qValues)).getFinalResult();
        double e = Math.max(epsilonMin, epsilon*Math.exp(-epsilonDecay*episode));
        
        if(mt.nextDouble()<=e){
            //explore
            rta = mt.nextInt(qValues.toDoubleVector().length-1);            
        }else{
            //explote
            rta = idx;
        }
            
        return rta;
    }
    
    @Override
    public int chooseAction(INDArray qValues, INDArray maskQValues, int allowedActionCount, int episode) {
        BooleanIndexing.replaceWhere(maskQValues, qValues, Conditions.equals(1));
        int idx = Nd4j.getExecutioner().execAndReturn(new IMax(maskQValues)).getFinalResult();
        double e = Math.max(epsilonMin, epsilon*Math.exp(-epsilonDecay*episode));
        
        if(mt.nextDouble()<=e){
            //explore
            idx = mt.nextInt(allowedActionCount-1); 
            int count = 0;
            for(int i=0;i<idx;i++){
                if(count == idx)
                    return i;
                if(maskQValues.getDouble(0, i)!=0D)
                    count++;                
            }                
        }
        //explote
        return idx;
    }
    
    @Override
    public void finishedEpisode() {
    }
    
}
