# -*- coding: utf-8 -*-

import sys
from collections import defaultdict

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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    S = input()

    D = defaultdict(lambda: -INF)
    mn = INF
    ans = ()
    cur = (0, 0)
    D[cur] = 0
    for i, s in enumerate(S):
        x, y = cur
        if s == 'L':
            cur = (x-1, y)
        elif s == 'R':
            cur = (x+1, y)
        elif s == 'D':
            cur = (x, y-1)
        else:
            cur = (x, y+1)
        dist = i - D[cur]
        if dist < mn:
            mn = dist
            ans = (D[cur]+1, i+1)
        D[cur] = i + 1
    if ans:
        print(*ans)
    else:
        print(-1)
