# -*- coding: utf-8 -*-

"""
・すんなり自力AC
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

M = INT()
tgt = []
dx, dy = MAP()
for i in range(M-1):
    x, y = MAP()
    # 最初の星との相対位置で座標を持つ
    tgt.append((x - dx, y - dy))

N = INT()
XY = []
S = set()
for i in range(N):
    x, y = MAP()
    XY.append((x, y))
    S.add((x, y))

# 基準の位置として全ての座標を試す
for x, y in XY:
    for x2, y2 in tgt:
        if (x + x2, y + y2) not in S:
            break
    else:
        # 必要な全ての座標に星が存在すればここが答え
        ans = (x - dx, y - dy)
        print(*ans)
        exit()
