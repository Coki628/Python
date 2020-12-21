"""
・自力AC
・状態持つ1次元DP
・最初貪欲やろうとしてコケたので、正解の考察にもっと素早く辿り着けるようにしたい。
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

N = INT()
A = LIST()

# dp0[i] := i番目の餅まで見ての最大値(直前をずんだにしてない)
dp0 = [-INF] * (N+1)
# dp1[i] := i番目の餅まで見ての最大値(直前をずんだにした)
dp1 = [-INF] * (N+1)
dp0[0] = 0
for i in range(N):
    dp0[i+1] = max(dp0[i+1], dp0[i])
    dp0[i+1] = max(dp0[i+1], dp1[i])
    dp1[i+1] = max(dp1[i+1], dp0[i] + 1)
    # 連続でずんだにする遷移は、同じ数が続く時だけ
    if i == 0 or A[i-1] == A[i]:
        dp1[i+1] = max(dp1[i+1], dp1[i] + 1)
ans = max(dp0[N], dp1[N])
print(ans)
