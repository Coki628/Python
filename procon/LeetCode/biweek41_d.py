"""
参考：https://twitter.com/laycrs/status/1337789410805391361
・自力ならず。。
・最善になりうる遷移だけやるDP
・自力は貪欲で死んだり、遷移生やしすぎて死んだりして終了。。
・実は最適になりうるのは最も伸ばすか、その直前の港の区切り位置かの2択。この遷移だけやる。
"""

import sys
from typing import List
from itertools import accumulate
from bisect import bisect_left, bisect_right

def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

class Solution:
    def boxDelivering(self, PW: List[List[int]], portsCount: int, maxBoxes: int, maxWeight: int) -> int:
        N = len(PW)
        PW.insert(0, (0, 0))
        P, W = zip(*PW)

        # 港の区切り位置と重さを高速に取得できるように累積和を用意
        sep = [0] * (N+1)
        for i in range(1, N+1):
            if P[i-1] != P[i]:
                sep[i] = 1
        sep = list(accumulate(sep))
        accw = list(accumulate(W))

        dp = [INF] * (N+1)
        dp[0] = 0
        for i in range(N):
            # 一番伸ばせる位置：min(重さで行ける位置, 個数で行ける位置, 終点)
            j = min(bisect_right(accw, accw[i]+maxWeight, lo=i)-1, i+maxBoxes, N)
            # 増えるコストは、経由で移動する港の数 +2(行き帰り)
            dp[j] = min(dp[j], dp[i] + sep[j]-sep[i+1]+2)
            # jの直近にある港の区切り位置の手前
            k = bisect_left(sep, sep[j]) - 1
            dp[k] = min(dp[k], dp[i] + sep[k]-sep[i+1]+2)
        ans = dp[N]
        return ans

# sol = Solution()
# print(sol.boxDelivering([[2,4],[2,5],[3,1],[3,2],[3,7],[3,1],[4,4],[1,3],[5,2]], 5, 5, 7))
