import sys
from typing import List
from string import ascii_lowercase

def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

class Solution:
    def countConsistentStrings(self, allowed: str, words: List[str]) -> int:
        
        ans = 0
        for word in words:
            for c in ascii_lowercase:
                if c in word and c not in allowed:
                    break
            else:
                ans += 1
        return ans

# sol = Solution()
# print(sol.countConsistentStrings('cad', ["cc","acd","b","ba","bac","bad","ac","d"]))
