"""
参考：https://twitter.com/koboshi_kyopro/status/1266048921316372480
・DP解
・i個目の値まで見た時の区間和最大値(num出現前/済)、でやる。
・これは配列生成毎回やってるけどAC0.9秒で通った。分からん。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10 ** 10
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

nums = sorted(set(A), reverse=1)
ans = -INF
for num in nums:
    dp0 = [-INF] * (N+1)
    dp1 = [-INF] * (N+1)
    for i in range(N):
        if A[i] < num:
            dp0[i+1] = max(dp0[i+1], dp0[i] + A[i], A[i])
            dp1[i+1] = max(dp1[i+1], dp1[i] + A[i], A[i])
        elif A[i] == num:
            dp1[i+1] = max(dp1[i+1], dp0[i] + A[i], A[i])
            dp1[i+1] = max(dp1[i+1], dp1[i] + A[i], A[i])
    ans = max(ans, max(dp1) - num)
print(ans)
