"""
・なんとか自力AC
・三角関数
・角度が等間隔で斜辺が常に1の時の底辺を求めることになる。cosを使える。
"""

import sys
from math import cos, radians

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    M = N // 2

    each = 90 / M
    ans = 1
    dig = each
    for i in range(M-1):
        ans += cos(radians(dig)) * 2
        dig += each
    print(ans)
