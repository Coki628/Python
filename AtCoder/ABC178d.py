"""
・自力AC！
・個数制限なし部分和DPみたいなやつ
・3~Nまでの数をいくらでも使えるので、0~Nまで順番に遷移元として見て行って、
　各値から3~Nを全部遷移させてみる。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N = INT()

dp = [0] * (N+1)
dp[0] = 1
for i in range(N+1):
    for j in range(3, N+1):
        if i+j <= N:
            dp[i+j] += dp[i]
            dp[i+j] %= MOD
ans = dp[N]
print(ans)
