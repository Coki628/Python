"""
・さくっと自力AC
・奇数番目と偶数番目でそれぞれ和を変数に持ちながらやればいい。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7
EPS = 10 ** -10

N = INT()

evsm = odsm = 0
dp = [0] * N
dp[0] = 1
evsm += 1
for i in range(1, N):
    if i % 2 == 1:
        dp[i] = (i+1) * evsm
        dp[i] %= MOD
        odsm += dp[i]
        odsm %= MOD
    else:
        dp[i] = (i+1) * odsm
        dp[i] %= MOD
        evsm += dp[i]
        evsm %= MOD
ans = dp[N-1]
print(ans)
