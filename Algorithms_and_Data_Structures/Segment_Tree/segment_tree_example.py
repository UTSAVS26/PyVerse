"""
Segment Tree Usage Examples

Demonstrates how to use the SegmentTree class for range sum queries and point updates.
"""

from segment_tree import SegmentTree

def basic_demo():
    print("=== Segment Tree Basic Demo ===")
    arr = [2, 4, 5, 7, 8, 9]
    st = SegmentTree(arr)
    print("Original array:", arr)

    # Range sum queries
    print("Sum of [0, 2]:", st.query(0, 2))  # 2 + 4 + 5 = 11
    print("Sum of [1, 4]:", st.query(1, 4))  # 4 + 5 + 7 + 8 = 24
    print("Sum of [3, 5]:", st.query(3, 5))  # 7 + 8 + 9 = 24

    # Point update
    st.update(2, 10)  # arr[2] = 10
    print("After updating index 2 to 10:")
    print("Sum of [0, 2]:", st.query(0, 2))  # 2 + 4 + 10 = 16

def advanced_demo():
    print("=== Segment Tree Advanced Demo ===")
    arr = [1] * 10
    st = SegmentTree(arr)
    print("Original array:", arr)

    # Update multiple points
    st.update(0, 5)
    st.update(9, 7)
    print("After updates: [0]=5, [9]=7")
    print("Sum of [0, 9]:", st.query(0, 9))  # 5 + 1*8 + 7 = 20
    print("Sum of [1, 8]:", st.query(1, 8))  # 1*8 = 8

def performance_demo():
    print("=== Segment Tree Performance Demo ===")
    import random, time
    n = 100000
    arr = [random.randint(1, 100) for _ in range(n)]
    st = SegmentTree(arr)

    # Timing queries
    start = time.time()
    total = 0
    for _ in range(1000):
        l = random.randint(0, n - 100)
        r = l + random.randint(0, 99)
        total += st.query(l, r)
    end = time.time()
    print(f"1000 random queries took {end - start:.4f} seconds.")

if __name__ == "__main__":
    basic_demo()
    print()
    advanced_demo()
    print()
    performance_demo()
