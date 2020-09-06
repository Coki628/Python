"""
・ここまで自力AC！全完！！
・素直に考えるDP
・何回通ってもいいとかいう複雑設定で、制約も小さいのでDPを疑う。
・依存性を保証して遷移させるには何を基準にしたらいいか考えたら、燃料だった。
　街は何回も行き来するけど、燃料は回復しないので。
・それが分かれば、距離に応じた燃料を減らしながら遷移させていけばいい。
・計算量がが200*100*100の200万で3重ループするので、python通るか心配だったけど大丈夫だった。
　りとこはpypy使えないんだなぁ。。
"""

from typing import List

import sys

def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

class Solution:
    def countRoutes(self, loc: List[int], s: int, t: int, F: int) -> int:
        N = len(loc)

        dp = list2d(F+1, N, 0)
        dp[0][s] = 1
        for i in range(F+1):
            for j in range(N):
                for k in range(N):
                    if j == k:
                        continue
                    dist = abs(loc[j] - loc[k])
                    if i+dist <= F:
                        dp[i+dist][k] += dp[i][j]
                        dp[i+dist][k] %= MOD
        ans = 0
        for i in range(F+1):
            ans += dp[i][t]
            ans %= MOD
        return ans

# locations = [1,2,3]
# start = 0
# finish = 2
# fuel = 40 
# sol = Solution()
# print(sol.countRoutes(locations, start, finish, fuel))
