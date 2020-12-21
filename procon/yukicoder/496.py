"""
・自力AC！
・グリッド、2次元でナップザックっぽくDP
・ちょっと考察悩んだ。正解の方針に辿り着けてよかった。
・クリスタルでのワープ分はナップザックっぽく最善を前計算しておく。
　そこに後から愚直に歩いた分を計算してゴールまでのコストを計ればOK。
・公式解見たけど、まあ同時にやってもよかったね。だいたいやりたいことは同じ。
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
INF = 10**18
MOD = 10**9 + 7
EPS = 10**-10

gx, gy, N, F = MAP()
xyc = []
for i in range(N):
    x, y, c = MAP()
    xyc.append((x, y, c))

dp = list3d(N+1, gx+1, gy+1, INF)
dp[0][0][0] = 0
for i, (cx, cy, c) in enumerate(xyc):
    for j in range(gx+1):
        for k in range(gy+1):
            dp[i+1][j][k] = min(dp[i+1][j][k], dp[i][j][k])
            if j+cx <= gx and k+cy <= gy:
                dp[i+1][j+cx][k+cy] = min(dp[i+1][j+cx][k+cy], dp[i][j][k] + c)

ans = INF
for i in range(gx+1):
    for j in range(gy+1):
        if dp[N][i][j] == INF:
            continue
        ans = min(ans, dp[N][i][j] + abs(i-gx)*F + abs(j-gy)*F)
print(ans)
