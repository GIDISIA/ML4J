package test;


/**
 *
 * @author ezequiel
 */
public class TestRecursionVsIteration {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        new TestRecursionVsIteration();
    }

    public TestRecursionVsIteration() {
        int n = 1000;
        long start = System.currentTimeMillis();
        int  rta1 = sumAllIteration(n);
        long finish = System.currentTimeMillis();
        System.out.println(finish-start);
        start = System.currentTimeMillis();
        int rta2  = sumAllRecursion(n);
        finish = System.currentTimeMillis();
        System.out.println(finish-start);        
        System.out.println(rta1);
        System.out.println(rta2);
    }
    
    
    
    public int sumAllIteration(int n){
        int result = 0;
        for(int i=0;i<=n;i++)
            result += i;
        return result;
    }
    
    public int sumAllRecursion(int n){
        return n==0?0:n+sumAllRecursion(n-1);
    }   
}
