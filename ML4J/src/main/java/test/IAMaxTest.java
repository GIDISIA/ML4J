/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author ezequiel
 */
public class IAMaxTest {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        INDArray r = Nd4j.create(new double[]{0,0,-48,5,0});
        int idxMin = Nd4j.getExecutioner().execAndReturn(new IMax(r.dup())).getFinalResult(); //Index of max value
        int idxMax = Nd4j.getExecutioner().execAndReturn(new IAMax(r.dup())).getFinalResult(); //index of max absolute value
        System.out.println("min:"+idxMin);
        System.out.println("max:"+idxMax);
    }
    
}
