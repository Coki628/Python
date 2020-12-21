# -*- coding: utf-8 -*-

"""
・10万クエリ系
・ランレングス圧縮とdequeとCounterで効率良くシミュレーション
"""

import sys
from collections import deque, Counter

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

Q = INT()

que = deque()
for _ in range(Q):
    cmd, *args = input().split()
    if cmd == '1':
        c, x = args
        x = int(x)
        que.append((c, x))
    else:
        d, = args
        d = int(d)
        C = Counter()
        while que and d - que[0][1] >= 0:
            c, x = que.popleft()
            d -= x
            C[c] += x
        if que and d > 0:
            c, x = que[0]
            x -= d
            C[c] += d
            que[0] = (c, x)
        ans = 0
        for v in C.values():
            ans += v**2
        print(ans)
