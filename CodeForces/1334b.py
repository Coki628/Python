# -*- coding: utf-8 -*-

import sys
from itertools import accumulate

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
    N, X = MAP()
    A = LIST()

    B = []
    cnt = 0
    ans = 0
    for a in A:
        if a >= X:
            cnt += a - X
            ans += 1
        else:
            B.append(a)
    B.sort(reverse=1)
    
    for b in B:
        if cnt - (X - b) >= 0:
            cnt -= (X - b)
            ans += 1
        else:
            break
    print(ans)