package edu.utn.gisiq.ml4j.trees;

/**
 *
 * @author ezequiel
 */
public class TestSegmentTree {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        new TestSegmentTree();
    }

    public TestSegmentTree() {
        int[] array = new int[]{2,3,5,7,1,9,11};
        
        SegmentTree st = new SegmentTree(array);
        
        System.out.println(st.rMinQ(0, st.size()));
        System.out.println(st.rMinQIdx(0, st.size()));
        System.out.println(st.rsq(0, st.size()));
        System.out.println(st.rsqIdx(22));
        
        st = new SegmentTree(array.length);
        for(int i=0;i<array.length;i++)
            st.update(i, i, array[i]);
        
        System.out.println(st.rMinQ(0, st.size()));
        System.out.println(st.rMinQIdx(0, st.size()));
        System.out.println(st.rsq(0, st.size()));
        System.out.println(st.rsqIdx(22));
        
        
    }
    
    
    
}
