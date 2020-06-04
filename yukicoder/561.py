"""
・さくっと自力AC
・いわゆる耳DP
・今いるのが東京か京都かの状態を持っておけばいい。
・制約、10万のNでもよさそうだけど、なんで100なんだろー。
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

N, D = MAP()
TK = []
for i in range(N):
    t, k = MAP()
    TK.append((t, k))

# dp0[i] := i日目まで見て、東京にいる時の最大値
dp0 = [-INF] * (N+1)
# dp1[i] := i日目まで見て、京都にいる時の最大値
dp1 = [-INF] * (N+1)
dp0[0] = 0
for i, (t, k) in enumerate(TK):
    dp0[i+1] = max(dp0[i+1], dp0[i] + t)
    dp0[i+1] = max(dp0[i+1], dp1[i] + t - D)
    dp1[i+1] = max(dp1[i+1], dp1[i] + k)
    dp1[i+1] = max(dp1[i+1], dp0[i] + k - D)
ans = max(dp0[N], dp1[N])
print(ans)
