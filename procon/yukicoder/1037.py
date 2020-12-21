"""
・結構さくっと自力AC！
・ナップザック系2次元DP
・dp[i][j] := i個目のガソリンスタンドに着いて、残り燃料がjの時の最小コスト
　にして、始点と終点もiに含める。あとは燃料を補充しないかするで2つの遷移を作る。
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

N, M, L = MAP()
XVW = [(0, 0, 0)]
for i in range(N):
    x, v, w = MAP()
    XVW.append((x, v, w))
XVW.append((L, 0, 0))
N += 1

# dp[i][j] := i個目のガソリンスタンドに着いて、残り燃料がjの時の最小コスト
dp = list2d(N+1, M+1, INF)
dp[0][M] = 0
for i, (x, v, w) in enumerate(XVW[:-1]):
    # 次の場所までの距離d
    d = XVW[i+1][0] - x
    for j in range(M+1):
        # ここで補給しない
        if j-d >= 0:
            dp[i+1][j-d] = min(dp[i+1][j-d], dp[i][j])
        # ここで補給する
        if min(j+v, M)-d >= 0:
            dp[i+1][min(j+v, M)-d] = min(dp[i+1][min(j+v, M)-d], dp[i][j] + w)

ans = min(dp[N])
if ans == INF:
    print(-1)
else:
    print(ans)
