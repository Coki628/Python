"""
・自力AC！
・ビット、累積和
・題意を満たすようなペアは、最上位ビットが両方立ってるもの。他のビットは関係なくなる。
・なので最上位ビット毎にまとめて、後から累積和取ってペア数を確認、でOK。
"""

import sys
from itertools import accumulate

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
    N = INT()
    A = LIST()

    B = list2d(32, N, 0)
    for i in range(N):
        B[A[i].bit_length()-1][i] = 1
    for k in range(32):
        B[k] = list(accumulate(B[k][::-1]))[::-1]

    ans = 0
    for i in range(N):
        ans += B[A[i].bit_length()-1][i] - 1
    print(ans)
