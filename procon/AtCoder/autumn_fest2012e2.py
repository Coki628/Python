# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/tomerun/together-14867665
・実験用
"""

import sys
from collections import deque

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

N = INT()
YX = []
for i in range(N):
    x, y = MAP()
    YX.append((y, x))

# グリッドBFS
directions = ((1,0),(-1,0),(0,1),(0,-1))
que = deque([(0, 0, 0)])
S = [set() for i in range(11)]
while que:
    h, w, c = que.popleft()
    # このターンでこの座標は訪問済
    if (h, w) in S[c]:
        continue
    S[c].add((h, w))
    # 10ターンで終了
    if c == 10:
        continue
    for dh, dw in directions:
        # 現在のターンに応じた距離を移動する
        h2 = h + dh * (c+1)
        w2 = w + dw * (c+1)
        que.append((h2, w2, c+1))

li = [[]] * 11
for i, s in enumerate(S):
    li[i] = sorted(s)
print()
