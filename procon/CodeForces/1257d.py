# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/contest/1257/submission/64821375
・自力WAから、比較的方向性が近そうだったtitiaさんのを参考に修正。
・がしかしこれでもまだWA。。
・詰める貪欲つらい。
"""

import sys
from bisect import bisect_left

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

ans = []
for _ in range(INT()):
    N = INT()
    A = LIST()
    M = INT()
    PS = []
    for i in range(M):
        p, s = MAP()
        PS.append((p, s))

    PS.sort()
    for i in range(M-1, 0, -1):
        if PS[i][1] > PS[i-1][1]:
            PS[i-1] = (PS[i-1][0], PS[i][1])
    P, _ = zip(*PS)
    
    day = 1
    scnt = 0
    cntmax = INF
    for a in A:
        idx = bisect_left(P, a)
        if idx == M:
            ans.append(str(-1))
            break
        p, s = PS[idx]
        cntmax = min(cntmax, s)
        if cntmax >= scnt + 1:
            scnt += 1
        else:
            day += 1
            scnt = 1
    else:
        ans.append(str(day))

print('\n'.join(ans))
