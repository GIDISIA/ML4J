package edu.utn.gisiq.ml4j.exception;

/**
 *
 * @author ezequiel
 * @date 19/06/2018  
 */
public class ArraysOfDifferentSizeException extends RuntimeException {

    public ArraysOfDifferentSizeException() {
        super("The arrays do not have the same size.");
    }
    
}
