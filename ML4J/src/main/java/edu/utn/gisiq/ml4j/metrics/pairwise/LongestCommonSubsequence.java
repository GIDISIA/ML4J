package edu.utn.gisiq.ml4j.metrics.pairwise;

import edu.utn.gisiq.ml4j.exception.ArraysOfDifferentSizeException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import static org.paukov.combinatorics.CombinatoricsFactory.createPermutationWithRepetitionGenerator;
import static org.paukov.combinatorics.CombinatoricsFactory.createVector;
import org.paukov.combinatorics.Generator;
import org.paukov.combinatorics.ICombinatoricsVector;


/**
 * Class that implements the techniques described in Vlachos et. al. (2002).
 * @author Ezequiel Beccaría
 */
public class LongestCommonSubsequence implements DistanceMetric{
    private double delta;
    private double epsilon;
    private double beta;
    private boolean distance2;

    /**
     * Constructor. Delta, beta and epsilon variables must be provided.
     * @param delta is the time error allowed to compare two point of time series
     * @param epsilon is the error allowed to compare two point of time series per dimension
     * @param beta is AS2 distances aproximation constant
     */
    public LongestCommonSubsequence(double delta, double epsilon, double beta, boolean distance2) {
        this.delta = delta;
        this.epsilon = epsilon;
        this.beta = beta;
        this.distance2 = distance2;
    }
    
    /**
     * Method that calc Longest Common Subsequence for two time series of n 
     * dimentions.
     * @param A
     * @param B
     * @return 
     */
    private double lcss(INDArray A, INDArray B) throws Exception{
        if(A == null || B == null) return 0;
        int n = A.rows();
        int m = B.rows();
        if(Math.abs(n-m) <= delta && epsilonValidation(A.getRow(n-1), B.getRow(m-1)))
            return 1 + lcss(head(A), head(B));
        return Math.max(lcss(head(A),B), lcss(A,head(B)));        
    }
    
    /**
     * Method that retun the head of a time serie.
     * This is head(X)=X[n-2]
     * @param X
     * @return 
     */
    private INDArray head(INDArray X){
        int n = X.rows()-1;
        if(n<1)            
            return null;
        return X.get(NDArrayIndex.interval(0,n), NDArrayIndex.all());
    }
    
    /**
     * Method that validate if each component of the last point of each 
     * time series is not *epsilon* far away each other. 
     * @param A
     * @param B
     * @return 
     */
    private boolean epsilonValidation(INDArray A, INDArray B) throws Exception{
        if(!(A.isVector()&&B.isVector()))
            throw new Exception("A and B must be a vector.");
        boolean flag = true;
        for(int i=0;i<A.length();i++){
            if(Math.abs(A.getDouble(i)-B.getDouble(i)) >= epsilon){
                flag = false;
                break;
            }
        }        
        return flag;
    }
    
    /**
     * Vlachos' S1 metric calculation
     * @param A
     * @param B
     * @return 
     */
    private double s1(INDArray A, INDArray B){
        try {
            return lcss(A,B)/Math.min(A.rows(), B.rows());
        } catch (Exception ex) {
            Logger.getLogger(LongestCommonSubsequence.class.getName()).log(Level.SEVERE, null, ex);
            return 0d;
        }
    }
    
    /**
     * Method that make a translation over a time serie array
     * @param X
     * @param translateValues
     * @return
     * @throws ArraysOfDifferentSizeException 
     */
    private INDArray translate(INDArray X, Double[] translateValues) throws ArraysOfDifferentSizeException {
        if(X.getRow(0).length() != translateValues.length)
            throw new ArraysOfDifferentSizeException();
        INDArray XX = Nd4j.zeros(X.rows(), translateValues.length);
        for(int i=0;i<XX.rows();i++){
            for(int j=0;j<XX.columns();j++){
                XX.putScalar(i, j, X.getDouble(i, j)+translateValues[j]);                
            }
        }
        return XX;
    }
    
    /**
     * Return max Double value in the collection.
     * @param values
     * @return 
     */
    private double maxValue(Collection<Double> values){
        double max = Double.MIN_VALUE;
        for(Double val : values)
            if(val>max)
                max = val;
        return max;
    }
    
    /**
     * Vlachos' aproximation of S2 metric calculation.
     * If parallel=true uses all CPUs to parallelize s1 metrics translations
     * @param A
     * @param B
     * @param parallel 
     * @return 
     */
    private double as2(final INDArray A, final INDArray B, boolean parallel) {
        int dim = A.columns();
        int n = A.rows();
        int m = B.rows();
        
        //Make search interval quantiles
        List<Double> paramInterval = new ArrayList<Double>();
        double finalVaule = beta*(n+m)/2;
        double initialVaule = -finalVaule;
        double increment =  (finalVaule-initialVaule)/(4*delta/beta);
        for(double i=initialVaule; i<=finalVaule; i+=increment)
            paramInterval.add(i);
        
        //Make a permutation with repetitions 
        ICombinatoricsVector<Double> vector = createVector(paramInterval);
        Generator<Double> gen = createPermutationWithRepetitionGenerator(vector, dim);
        if(!parallel){
            // Serial execution
            //Make a vector for store results
            List<Double> s1Val = new ArrayList<>(paramInterval.size());
            for (ICombinatoricsVector<Double> perm : gen) {
                try {
                    s1Val.add(s1(A, translate(B, perm.getVector().toArray(new Double[perm.getVector().size()]))));
                } catch (Exception ex) {
                    Logger.getLogger(LongestCommonSubsequence.class.getName()).log(Level.SEVERE, null, ex);
                }
             } 
            //Get max value
            return maxValue(s1Val);
        }else{
            // Parallel execution
            final Queue<Double> s1Val = new ConcurrentLinkedQueue<>();
            gen.forEach((item)-> {
                try {
                    s1Val.add(s1(A, translate(B, item.getVector().toArray(new Double[item.getVector().size()]))));
                } catch (Exception ex) {
                    Logger.getLogger(LongestCommonSubsequence.class.getName()).log(Level.SEVERE, null, ex);
                }
            });
            //Get max value
            return maxValue(s1Val);
        }
    }
    
    /**
     * Vlachos' D1 distance metric
     * @param A
     * @param B
     * @return 
     */
    public double d1(INDArray A, INDArray B){
        return 1d-s1(A, B);
    }
    
    /**
     * Vlachos' D2 distance metric
     * @param A
     * @param B
     * @return 
     */
    public double d2(INDArray A, INDArray B, boolean parallel){
        return 1d-as2(A, B, parallel);
    }

    @Override
    public double distance(INDArray a, INDArray b) {
        if(distance2)
            return this.d2(a, b, true);
        return this.d1(a, b);        
    }

    @Override
    public double similarity(INDArray a, INDArray b) {
        if(distance2)
            return this.as2(a, b, true);
        return this.s1(a, b);        
    }
}
