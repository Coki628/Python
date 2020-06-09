"""
・自力AC！
・当時(1年ちょっと前)は自力全く通せなかったぽいのでいい感じ。
・総和が10*5以下の制約があるので、配列が大きい時は各要素が小さく遷移先が多くないと思われる。
・それを踏まえて、2次元目をdictにしてDPする。
"""

import sys
from collections import defaultdict

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
INF = 10 ** 19
MOD = 10 ** 9 + 7

N, K = MAP()
A = LIST(N)

M = 0
for a in A:
    M |= a
if K > M:
    print(0)
    exit()

dp = [defaultdict(int) for i in range(N+1)]
dp[0][0] = 1
for i, a in enumerate(A):
    for j in dp[i].keys():
        dp[i+1][j] += dp[i][j]
        dp[i+1][j] %= MOD
        dp[i+1][j^a] += dp[i][j]
        dp[i+1][j^a] %= MOD
ans = dp[N][K]
print(ans)
