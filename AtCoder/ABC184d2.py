"""
・期待値DP、メモ化再帰
・自力ひとまず解いたのは確率DPしてからガチャガチャしたけど、
　みんな期待値DP一発で解いてたっぽいのでやってみた。
・遷移とかはEDPC_Jを参考に、よりシンプルにした感じ。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

a, b, c = MAP()

N = 100
memo = list3d(N+1, N+1, N+1, -1)
def rec(i, j, k):
    if memo[i][j][k] != -1:
        return memo[i][j][k]
    # いずれかが100枚に達したらそこで終了
    if i == N or j == N or k == N:
        return 0
    total = i + j + k
    # 今回の操作1回
    res = 1
    if i+1 <= N:
        # 金貨を取る
        res += rec(i+1, j, k) * i/total
    if j+1 <= N:
        # 銀貨を取る
        res += rec(i, j+1, k) * j/total
    if k+1 <= N:
        # 銅貨を取る
        res += rec(i, j, k+1) * k/total
    memo[i][j][k] = res
    return res

ans = rec(a, b, c)
print(ans)
