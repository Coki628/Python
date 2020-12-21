# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/titia_til/status/1248660009791021057
・コンテスト中はTLE(制約1秒)、C++でAC。
・30万の入力
・titiaさんのツイートから、readlineにbufferつけたら通った。
"""

import sys
from itertools import accumulate

def input(): return sys.stdin.buffer.readline().strip()
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

out = []
for _ in range(INT()):
    N = INT()
    AB = []
    for i in range(N):
        a, b = MAP()
        AB.append((a, b))

    need = [0] * N
    for i in range(N-1, -1, -1):
        a, b = AB[i]
        na, nb = AB[(i+1)%N]
        if b < na:
            need[(i+1)%N] = na - b

    sm = sum(need)
    ans = INF
    for i, (a, b) in enumerate(AB):
        ans = min(ans, a + sm - need[i])
    out.append(str(ans))
print('\n'.join(out))
