# -*- coding: utf-8 -*-

"""
・蟻本演習2-1-2、自力AC
・DFS
・DFSはいつも再帰で書いてたけど、練習のためにスタックでやってみた。
・やっぱり再帰よりちょっと速い。python0.8秒 => 0.7秒。あとなぜかメモリ消費も再帰より少ない。性能面では再帰良いとこなし。
・pypyだともっとひどい。pypyTLE => 0.4秒。pypyはwhileでも回せるなら再帰させない方が良さそう。。
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
INF = float('inf')
MOD = 10 ** 9 + 7

H, W = MAP()
directions = ((1,0),(-1,0),(0,1),(0,-1))
grid = list2d(H+2, W+2, '#')
sh = sw = gh = gw = 0
for i in range(1, H+1):
    row = list(input())
    for j in range(1, W+1):
        grid[i][j] = row[j-1]
        if grid[i][j] == 's':
            sh, sw = i, j
        elif grid[i][j] == 'g':
            gh, gw = i, j

dist = list2d(H+2, W+2, INF)
def dfs(sh, sw):
    stack = [(0, sh, sw)]
    while stack:
        cur, h, w = stack.pop()
        if dist[h][w] != INF:
            continue
        dist[h][w] = cur
        for d in directions:
            h2 = h + d[0]
            w2 = w + d[1]
            if grid[h2][w2] != '#':
                stack.append((cur+1, h2, w2))
    return

dfs(sh, sw)
if dist[gh][gw] != INF:
    Yes()
else:
    No()
