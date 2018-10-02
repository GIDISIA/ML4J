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
        int[] array = new int[]{1,3,5,7,9,11};
        
        SegmentTree st = new SegmentTree(array);
        
        System.out.println(st.rMinQ(0, st.size()));
        System.out.println(st.rMinQIdx(0, st.size()));
        System.out.println(st.rsq(0, st.size()));
        System.out.println(st.rsqIdx(22));
        
        st = new SegmentTree(array);
        st.update(0, 0, 1);
        st.update(1, 1, 3);
        st.update(2, 2, 5);
        st.update(3, 3, 7);
        st.update(4, 4, 9);
        st.update(5, 5, 11);
        
        System.out.println(st.rMinQ(0, st.size()));
        System.out.println(st.rMinQIdx(0, st.size()));
        System.out.println(st.rsq(0, st.size()));
        System.out.println(st.rsqIdx(22));
        
        
    }
    
    
    
}
