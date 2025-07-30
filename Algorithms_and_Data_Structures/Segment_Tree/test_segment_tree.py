"""
Unit Tests for Segment Tree
"""

import unittest
from segment_tree import SegmentTree

class TestSegmentTree(unittest.TestCase):
    def setUp(self):
        self.arr = [1, 3, 5, 7, 9, 11]
        self.st = SegmentTree(self.arr)

    def test_query_entire_range(self):
        self.assertEqual(self.st.query(0, 5), sum(self.arr))

    def test_query_subranges(self):
        self.assertEqual(self.st.query(0, 2), 1 + 3 + 5)
        self.assertEqual(self.st.query(1, 4), 3 + 5 + 7 + 9)
        self.assertEqual(self.st.query(3, 5), 7 + 9 + 11)

    def test_update(self):
        self.st.update(2, 10)  # arr[2] = 10
        self.assertEqual(self.st.query(0, 2), 1 + 3 + 10)
        self.st.update(0, 0)
        self.assertEqual(self.st.query(0, 1), 0 + 3)

    def test_invalid_update(self):
        with self.assertRaises(IndexError):
            self.st.update(-1, 5)
        with self.assertRaises(IndexError):
            self.st.update(6, 5)

    def test_invalid_query(self):
        with self.assertRaises(IndexError):
            self.st.query(-1, 2)
        with self.assertRaises(IndexError):
            self.st.query(0, 6)
        with self.assertRaises(IndexError):
            self.st.query(3, 2)

    def test_single_element(self):
        st = SegmentTree([42])
        self.assertEqual(st.query(0, 0), 42)
        st.update(0, 100)
        self.assertEqual(st.query(0, 0), 100)

    def test_large_array(self):
        import random
        arr = [random.randint(1, 100) for _ in range(1000)]
        st = SegmentTree(arr)
        for _ in range(10):
            l = random.randint(0, 999)
            r = random.randint(l, 999)
            self.assertEqual(st.query(l, r), sum(arr[l:r+1]))

if __name__ == "__main__":
    unittest.main()
