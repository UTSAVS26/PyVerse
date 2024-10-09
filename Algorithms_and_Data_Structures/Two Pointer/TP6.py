
# count-number-of-nice-subarrays

from collections import Counter

class TP6:
    def numberOfSubarrays(self, a: List[int], k: int) -> int:
        result, c, q = 0, Counter([0]), 0
        for v in a:
            q += v%2
            c[q] += 1
            result += c[q-k]

        return result

tp6 = TP6()

a = [1, 1, 0, 1, 0]
k = 2

result = tp6.numberOfSubarrays(a, k)
print("The number of nice subarrays is:", result)