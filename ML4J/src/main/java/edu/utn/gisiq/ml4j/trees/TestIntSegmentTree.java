package edu.utn.gisiq.ml4j.trees;

/**
 *
 * @author ezequiel
 */
public class TestIntSegmentTree {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        new TestIntSegmentTree();
    }

    public TestIntSegmentTree() {
        int[] array = new int[]{2,3,5,7,1,9,11};
        
        IntSegmentTree st = new IntSegmentTree(array);
        
        System.out.println(st.rMinQ(0, st.size()));
        System.out.println(st.rsq(0, st.size()));
        System.out.println(st.rsqIdx(22));
        
        st = new IntSegmentTree(array.length);
        for(int i=0;i<array.length;i++)
            st.update(i, i, array[i]);
        
        System.out.println(st.rMinQ(0, st.size()));
        System.out.println(st.rsq(0, st.size()));
        System.out.println(st.rsqIdx(22));
        
        
    }
    
    
    
}
