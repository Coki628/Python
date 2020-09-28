"""
・自力AC！
・状態2つ、次が+か-かを持ってDPする。
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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    N, Q = MAP()
    A = LIST()

    dp0 = [0] * (N+1)
    dp1 = [0] * (N+1)
    for i in range(N):
        dp1[i+1] = max(dp1[i+1], dp1[i], dp0[i] + A[i])
        dp0[i+1] = max(dp0[i+1], dp0[i], dp1[i] - A[i])
    ans = max(dp0[N], dp1[N])
    print(ans)
