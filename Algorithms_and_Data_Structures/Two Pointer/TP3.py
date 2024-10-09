
# Fruit Into Baskets

class TP3:
    def totalFruits(self, a: List[int]) -> int:
        n = len(a)
        max_len = 0
        for i in range(n):
            st = set() 
            for j in range(i, n):
                st.add(a[j])
                if len(st) <= 2:
                    max_len = max(max_len, j - i + 1)
                else:
                    break
        return max_len

solution = Solution()

a = [1, 2, 1, 2, 3]

result = solution.totalFruits(a)
print("The maximum length of fruits in two baskets is:", result)