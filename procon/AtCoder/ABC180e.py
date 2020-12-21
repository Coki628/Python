"""
・自力AC！ここまで結構いい感じに早かった。
・bitDP、TSP
・コストの算出がちょっと独特だけど、そこ整えたら後はほぼTSP貼るだけ。
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
xyz = []
for i in range(N):
    x, y, z = MAP()
    xyz.append((x, y, z))

def calc(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + max(0, b[2]-a[2])

G = list2d(N, N, INF)
for i in range(N):
    for j in range(i+1, N):
        G[i][j] = calc(xyz[i], xyz[j])
        G[j][i] = calc(xyz[j], xyz[i])

dp = list2d(1<<N, N, INF)
dp[1][0] = 0
for bit in range(1, (1<<N)-1):
    for i in range(N):
        if not bit>>i&1: continue
        for j in range(N):
            if bit>>j&1: continue
            dp[bit+(1<<j)][j] = min(dp[bit+(1<<j)][j], dp[bit][i] + G[i][j])

mn = INF
for end in range(N):
    mn = min(mn, dp[(1<<N)-1][end] + G[end][0])
if mn == INF:
    print(-1)
else:
    print(mn)
