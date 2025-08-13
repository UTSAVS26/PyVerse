# A city's skyline is the outer contour of the silhouette formed by all the buildings in that city when viewed from a distance. Given the locations and heights of all the buildings, return the skyline formed by these buildings collectively.
# The geometric information of each building is given in the array buildings where buildings[i] = [lefti, righti, heighti]:


class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        # for the same x, (x, -H) should be in front of (x, 0)
        # For Example 2, we should process (2, -3) then (2, 0), as there's no height change
        x_height_right_tuples = sorted([(L, -H, R) for L, R, H in buildings] + [(R, 0, "doesn't matter") for _, R, _ in buildings])   
        # (0, float('inf')) is always in max_heap, so max_heap[0] is always valid
        result, max_heap = [[0, 0]], [(0, float('inf'))]
        for x, negative_height, R in x_height_right_tuples:
            while x >= max_heap[0][1]:
                # reduce max height up to date, i.e. only consider max height in the right side of line x
                heapq.heappop(max_heap)
            if negative_height:
                # Consider each height, as it may be the potential max height
                heapq.heappush(max_heap, (negative_height, R))
            curr_max_height = -max_heap[0][0]
            if result[-1][1] != curr_max_height:
                result.append([x, curr_max_height])
        return result[1:] 

