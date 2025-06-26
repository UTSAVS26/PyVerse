"""
Unit Tests for Union-Find (Disjoint Set Union)
"""

import unittest
from union_find import UnionFind

class TestUnionFind(unittest.TestCase):
    def setUp(self):
        self.uf = UnionFind(6)

    def test_initial_state(self):
        self.assertEqual(self.uf.set_count(), 6)
        for i in range(6):
            self.assertEqual(self.uf.find(i), i)

    def test_union_and_connected(self):
        self.uf.union(0, 1)
        self.assertTrue(self.uf.connected(0, 1))
        self.assertEqual(self.uf.set_count(), 5)
        self.uf.union(1, 2)
        self.assertTrue(self.uf.connected(0, 2))
        self.assertEqual(self.uf.set_count(), 4)
        self.uf.union(3, 4)
        self.assertTrue(self.uf.connected(3, 4))
        self.assertEqual(self.uf.set_count(), 3)
        self.uf.union(2, 3)
        self.assertTrue(self.uf.connected(0, 4))
        self.assertEqual(self.uf.set_count(), 2)

    def test_redundant_union(self):
        self.uf.union(0, 1)
        self.uf.union(1, 2)
        self.assertFalse(self.uf.union(0, 2))  # Already connected

    def test_groups(self):
        self.uf.union(0, 1)
        self.uf.union(2, 3)
        self.uf.union(4, 5)
        groups = [set(g) for g in self.uf.get_groups()]
        self.assertIn(set([0, 1]), groups)
        self.assertIn(set([2, 3]), groups)
        self.assertIn(set([4, 5]), groups)

    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            UnionFind(0)
        with self.assertRaises(ValueError):
            UnionFind(-5)

    def test_str_and_repr(self):
        s = str(self.uf)
        r = repr(self.uf)
        self.assertIn("UnionFind", s)
        self.assertIn("UnionFind", r)

if __name__ == "__main__":
    unittest.main()
