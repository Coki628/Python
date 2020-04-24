# -*- coding: utf-8 -*-

"""
参考：https://www.youtube.com/watch?v=aDDV_WBwzTM
　　　https://img.atcoder.jp/abc145/editorial.pdf
・自力ならず。
・2次元+N回遷移でN^3系のDP(iroha2019day1g,colocon2018qual_dとかと同じ感じ)
・まず操作なしでかかるコストは、列の左側面が見えている数とする。
・で、ある列を操作する時は必ず左隣に合わせる、とすると、列が1つ消えたみたいになる。
・すると、K列消してN-K列残した時に得られる最小コストは、のDPが作れる。
・遷移は遷移元位置より手前の全てから遷移させて最善を取る。
・300^3=2700万でpythonTLE,pypyAC0.5秒。
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

N, K = MAP()
M = N - K
A = [0] + LIST()

# dp[i][j] := 列をi個使って、最後に使ったのがj列目の時の最小コスト
dp = list2d(M+1, N+1, INF)
dp[0][0] = 0
for i in range(1, M+1):
    for j, h1 in enumerate(A):
        for k in range(0, j):
            h2 = A[k]
            dp[i][j] = min(dp[i][j], dp[i-1][k] + max(h1 - h2, 0))
print(min(dp[M]))
