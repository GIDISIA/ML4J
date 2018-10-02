package edu.utn.gisiq.ml4j.trees;

/**
 *
 * @author ezequiel
 */
public class TestDoubleSegmentTree {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        new TestDoubleSegmentTree();
    }

    public TestDoubleSegmentTree() {
        double[] array = new double[]{1D,3D,3D,0.001,7D,9D,11D,0.02,0.0001};
        
        DoubleSegmentTree st = new DoubleSegmentTree(array);
        
        System.out.println(st.rMinQ(0, st.size()));
        System.out.println(st.rsq(0, st.size()));
        System.out.println(st.rsqIdx(10.87));
        
        st = new DoubleSegmentTree(array.length);
        for(int i=0;i<array.length;i++)
            st.update(i, i, array[i]);
        
        System.out.println(st.rMinQ(0, st.size()));
        System.out.println(st.rsq(0, st.size()));
        System.out.println(st.rsqIdx(10.87));
        
        
    }
    
    
    
}
