"""
・さくっと自力AC
・GCD系、部分和DP応用
・部分和DPで足す代わりにGCDやる感じ。
・値の範囲は大きいけど、値の取りうる種類数は少ないのでdictで管理すれば大丈夫。
"""

import sys
from math import gcd
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
EPS = 10 ** -10

N = INT()
A = LIST()

dp = [defaultdict(int) for i in range(N+1)]
dp[0][0] = 1
for i, a in enumerate(A):
    for k in dp[i].keys():
        dp[i+1][k] += dp[i][k]
        dp[i+1][gcd(k, a)] += dp[i][k]
ans = dp[N][1]
print(ans)
